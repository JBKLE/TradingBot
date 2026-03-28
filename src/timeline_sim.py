"""Offline timeline simulation – runs DQN on historical data at max speed.

Reads candles from simLastCharts.db (or any DB with price_history table),
runs DQN inference minute-by-minute, opens/closes trades, and stores
results in the same DB's sim_trades table.

Speed: ~500-2000 minutes/second (GPU), no API calls, no sleep.
"""
import asyncio
import logging
import os
from datetime import datetime
from typing import Callable

import aiosqlite
import numpy as np
import torch

from . import config
from .ai_analyzer import (
    ACTION_SIZE,
    ACTIONS,
    ASSETS,
    ASSET_INDEX,
    MAX_WINDOW,
    DuelingDQN,
    _atr,
    _ema,
    _get_latest_model_path,
    _rsi,
    _scale_confidence,
)
from .fetch_history import HISTORY_DB_PATH

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_CONFIDENCE_THRESHOLD = 8  # 1-10 scale
SL_PCT = 0.003  # from DQN training
TP_PCT = 0.005


# ── Timeline Simulator ───────────────────────────────────────────────────────

class TimelineSimulator:
    """Runs DQN offline on historical candles at maximum speed."""

    def __init__(
        self,
        db_path: str = HISTORY_DB_PATH,
        confidence_threshold: int = DEFAULT_CONFIDENCE_THRESHOLD,
        models_dir: str | None = None,
    ) -> None:
        self.db_path = db_path
        self.confidence_threshold = confidence_threshold
        self._models_dir = models_dir or config.AI_MODELS_DIR
        self._device = self._resolve_device()
        self._net: DuelingDQN | None = None
        self._cancelled = False

    @staticmethod
    def _resolve_device() -> torch.device:
        if config.DQN_DEVICE == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(config.DQN_DEVICE)

    def cancel(self) -> None:
        """Signal the simulation to stop after the current minute."""
        self._cancelled = True

    def _load_model(self) -> DuelingDQN:
        if self._net is not None:
            return self._net
        model_path = _get_latest_model_path(self._models_dir)
        logger.info("Timeline sim: loading model %s (device=%s)", model_path, self._device)
        net = DuelingDQN(ACTION_SIZE).to(self._device)
        ckpt = torch.load(model_path, map_location=self._device, weights_only=True)
        net.load_state_dict(ckpt["policy_net"])
        net.eval()
        self._net = net
        return net

    def _build_state(
        self,
        asset: str,
        ohlc_window: np.ndarray,
        has_position: bool = False,
        position_direction: float = 0.0,
        unrealised_r: float = 0.0,
    ) -> np.ndarray:
        """Build state vector from a sliding window of OHLC data.

        ohlc_window: shape (n, 4) with columns [open, high, low, close]
        """
        avail = len(ohlc_window)
        state_ohlcv = np.zeros((MAX_WINDOW, 5), dtype=np.float32)

        if avail > 0:
            opens = ohlc_window[:, 0]
            highs = ohlc_window[:, 1]
            lows = ohlc_window[:, 2]
            closes = ohlc_window[:, 3]
            volumes = np.zeros(avail, dtype=np.float64)

            ref = float(closes[-1])
            v_mean = 1e-8

            rows = np.column_stack([
                opens / ref - 1,
                highs / ref - 1,
                lows / ref - 1,
                closes / ref - 1,
                volumes / v_mean - 1,
            ]).astype(np.float32)
            state_ohlcv[MAX_WINDOW - avail:] = np.clip(rows, -5.0, 5.0)

            rsi = (_rsi(closes) - 50.0) / 50.0
            atr_val = _atr(highs, lows, closes)
            atr_pct = float(np.clip(atr_val / (closes[-1] + 1e-8), 0, 0.05) / 0.05)
            ema20_r = float(np.clip(
                _ema(closes, min(20, avail)) / (closes[-1] + 1e-8) - 1, -0.1, 0.1,
            ) / 0.1)
            ema50_r = float(np.clip(
                _ema(closes, min(50, avail)) / (closes[-1] + 1e-8) - 1, -0.1, 0.1,
            ) / 0.1)
        else:
            rsi, atr_pct, ema20_r, ema50_r = 0.0, 0.0, 0.0, 0.0

        indicators = np.array([rsi, atr_pct, ema20_r, ema50_r], dtype=np.float32)

        asset_oh = np.zeros(4, dtype=np.float32)
        if asset in ASSET_INDEX:
            asset_oh[ASSET_INDEX[asset]] = 1.0

        if has_position:
            pos = np.array([1.0, position_direction, float(np.clip(unrealised_r, -3, 3)), 0.0], dtype=np.float32)
        else:
            pos = np.zeros(4, dtype=np.float32)

        return np.concatenate([state_ohlcv.flatten(), indicators, asset_oh, pos])

    @torch.no_grad()
    def _infer_batch(self, states: list[np.ndarray]) -> list[tuple[int, float]]:
        """Run DQN inference on multiple states at once.

        Returns list of (action_id, softmax_confidence).
        """
        net = self._load_model()
        batch = torch.FloatTensor(np.array(states)).to(self._device)
        q_values = net(batch).cpu().numpy()

        results = []
        for q in q_values:
            action = int(q.argmax())
            softmax = float(np.exp(q - q.max()) / np.exp(q - q.max()).sum())
            # Use the max softmax value
            probs = np.exp(q - q.max())
            probs = probs / probs.sum()
            softmax_conf = float(probs.max())
            results.append((action, softmax_conf))
        return results

    async def run(
        self,
        assets: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        progress_callback: Callable | None = None,
    ) -> dict:
        """Run the full timeline simulation.

        Args:
            assets: Which assets to simulate (None = all in DB)
            start_date: ISO date start (None = from earliest data)
            end_date: ISO date end (None = to latest data)
            progress_callback: fn(current_minute, total_minutes, open_trades, closed_trades)

        Returns:
            Summary dict with trade stats, equity curve, etc.
        """
        self._cancelled = False
        self._load_model()

        target_assets = assets or list(config.WATCHLIST.keys())

        # Clear old sim_trades from this DB
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM sim_trades")
            await db.commit()

        # Load ALL candles for each asset into memory (fast numpy arrays)
        asset_data: dict[str, np.ndarray] = {}
        asset_timestamps: dict[str, list[str]] = {}

        for asset in target_assets:
            candles, timestamps = await self._load_asset_candles(asset, start_date, end_date)
            if len(candles) > 0:
                asset_data[asset] = candles
                asset_timestamps[asset] = timestamps
                logger.info("Timeline sim: %s loaded %d candles", asset, len(candles))

        if not asset_data:
            return {"error": "Keine Kerzen in der DB gefunden", "trades": 0}

        # Find the global timeline (union of all timestamps)
        all_ts = set()
        for ts_list in asset_timestamps.values():
            all_ts.update(ts_list)
        timeline = sorted(all_ts)
        total_minutes = len(timeline)

        if total_minutes == 0:
            return {"error": "Leerer Zeitstrahl", "trades": 0}

        # Build timestamp-to-index maps for each asset
        asset_ts_index: dict[str, dict[str, int]] = {}
        for asset in asset_data:
            asset_ts_index[asset] = {ts: i for i, ts in enumerate(asset_timestamps[asset])}

        # ── Simulation state ─────────────────────────────────────
        open_trades: dict[str, dict] = {}  # asset -> trade info
        closed_trades: list[dict] = []
        trade_id_counter = 0

        logger.info(
            "Timeline simulation starting: %d assets, %d minutes, threshold=%d",
            len(asset_data), total_minutes, self.confidence_threshold,
        )

        for minute_idx, current_ts in enumerate(timeline):
            if self._cancelled:
                logger.info("Timeline simulation cancelled at minute %d", minute_idx)
                break

            # ── Step 1: Get current candle data for each asset ────
            for asset in list(asset_data.keys()):
                ts_idx_map = asset_ts_index[asset]
                if current_ts not in ts_idx_map:
                    continue

                candle_idx = ts_idx_map[current_ts]
                candles = asset_data[asset]
                current_candle = candles[candle_idx]
                current_high = float(current_candle[1])
                current_low = float(current_candle[2])
                current_close = float(current_candle[3])

                # ── Step 2: Evaluate open trade for this asset ────
                if asset in open_trades:
                    trade = open_trades[asset]
                    direction = trade["direction"]

                    hit_sl = False
                    hit_tp = False

                    if direction == "BUY":
                        if current_low <= trade["sl_price"]:
                            hit_sl = True
                        elif current_high >= trade["tp_price"]:
                            hit_tp = True
                    else:  # SELL
                        if current_high >= trade["sl_price"]:
                            hit_sl = True
                        elif current_low <= trade["tp_price"]:
                            hit_tp = True

                    if hit_sl or hit_tp:
                        exit_price = trade["sl_price"] if hit_sl else trade["tp_price"]
                        if direction == "BUY":
                            pnl = exit_price - trade["entry_price"]
                        else:
                            pnl = trade["entry_price"] - exit_price

                        risk = abs(trade["entry_price"] - trade["sl_price"])
                        r_multiple = pnl / risk if risk > 0 else 0.0

                        trade["exit_timestamp"] = current_ts
                        trade["exit_price"] = exit_price
                        trade["status"] = "closed_tp" if hit_tp else "closed_sl"
                        trade["pnl"] = pnl
                        trade["r_multiple"] = r_multiple
                        closed_trades.append(trade)
                        del open_trades[asset]

                # ── Step 3: DQN inference (only after 500 candles) ────
                if candle_idx < MAX_WINDOW:
                    continue

                if asset in open_trades:
                    continue  # already have a trade for this asset

                # Build state from sliding window
                window_start = max(0, candle_idx - MAX_WINDOW + 1)
                window = candles[window_start:candle_idx + 1]

                state = self._build_state(asset, window)
                states_batch = [state]
                results = self._infer_batch(states_batch)
                action_id, softmax_conf = results[0]
                confidence = _scale_confidence(softmax_conf)

                # ── Step 4: Open trade if confident enough ────
                if action_id in (1, 2) and confidence >= self.confidence_threshold:
                    direction = "BUY" if action_id == 1 else "SELL"
                    trade_id_counter += 1

                    if direction == "BUY":
                        sl_price = current_close * (1 - SL_PCT)
                        tp_price = current_close * (1 + TP_PCT)
                    else:
                        sl_price = current_close * (1 + SL_PCT)
                        tp_price = current_close * (1 - TP_PCT)

                    open_trades[asset] = {
                        "id": trade_id_counter,
                        "asset": asset,
                        "direction": direction,
                        "sl_variant": "dqn_timeline",
                        "entry_timestamp": current_ts,
                        "entry_price": current_close,
                        "sl_price": sl_price,
                        "tp_price": tp_price,
                        "confidence": confidence,
                    }

            # Progress callback every 1000 minutes
            if progress_callback and (minute_idx % 1000 == 0 or minute_idx == total_minutes - 1):
                progress_callback(minute_idx + 1, total_minutes, len(open_trades), len(closed_trades))

        # Close any remaining open trades at last price
        for asset, trade in open_trades.items():
            candles = asset_data[asset]
            last_close = float(candles[-1, 3])
            if trade["direction"] == "BUY":
                pnl = last_close - trade["entry_price"]
            else:
                pnl = trade["entry_price"] - last_close
            risk = abs(trade["entry_price"] - trade["sl_price"])
            r_multiple = pnl / risk if risk > 0 else 0.0

            trade["exit_timestamp"] = timeline[-1]
            trade["exit_price"] = last_close
            trade["status"] = "closed_end"
            trade["pnl"] = pnl
            trade["r_multiple"] = r_multiple
            closed_trades.append(trade)

        # ── Save trades to DB ─────────────────────────────────────
        await self._save_trades(closed_trades)

        # ── Build summary ─────────────────────────────────────────
        return self._build_summary(closed_trades, total_minutes, timeline)

    async def _load_asset_candles(
        self,
        asset: str,
        start_date: str | None,
        end_date: str | None,
    ) -> tuple[np.ndarray, list[str]]:
        """Load all candles for an asset into numpy array + timestamp list."""
        conditions = ["asset = ?"]
        params: list = [asset]

        if start_date:
            conditions.append("timestamp >= ?")
            params.append(start_date + "T00:00:00")
        if end_date:
            conditions.append("timestamp <= ?")
            params.append(end_date + "T23:59:59")

        where = " AND ".join(conditions)
        query = f"SELECT timestamp, open, high, low, close FROM price_history WHERE {where} ORDER BY timestamp ASC"

        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()

        if not rows:
            return np.array([]), []

        timestamps = [r[0] for r in rows]
        candles = np.array([[r[1], r[2], r[3], r[4]] for r in rows], dtype=np.float64)
        return candles, timestamps

    async def _save_trades(self, trades: list[dict]) -> None:
        """Write all closed trades to sim_trades table."""
        if not trades:
            return

        rows = [
            (
                t["asset"],
                t["direction"],
                t["sl_variant"],
                t["entry_timestamp"],
                t["entry_price"],
                t["sl_price"],
                t["tp_price"],
                t["exit_timestamp"],
                t["exit_price"],
                t["status"],
                t["pnl"],
                t["r_multiple"],
            )
            for t in trades
        ]

        async with aiosqlite.connect(self.db_path) as db:
            await db.executemany(
                """INSERT INTO sim_trades
                   (asset, direction, sl_variant, entry_timestamp, entry_price,
                    sl_price, tp_price, exit_timestamp, exit_price, status, pnl, r_multiple)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            await db.commit()

        logger.info("Timeline sim: saved %d trades to DB", len(trades))

    @staticmethod
    def _build_summary(trades: list[dict], total_minutes: int, timeline: list[str]) -> dict:
        """Build summary statistics from closed trades."""
        if not trades:
            return {
                "total_minutes": total_minutes,
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "total_pnl_points": 0.0,
                "avg_r_multiple": 0.0,
                "start_ts": timeline[0] if timeline else "",
                "end_ts": timeline[-1] if timeline else "",
                "per_asset": {},
            }

        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]
        total_pnl = sum(t["pnl"] for t in trades)
        avg_r = sum(t["r_multiple"] for t in trades) / len(trades) if trades else 0

        # Per-asset breakdown
        per_asset: dict[str, dict] = {}
        for t in trades:
            a = t["asset"]
            if a not in per_asset:
                per_asset[a] = {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0}
            per_asset[a]["trades"] += 1
            per_asset[a]["pnl"] += t["pnl"]
            if t["pnl"] > 0:
                per_asset[a]["wins"] += 1
            else:
                per_asset[a]["losses"] += 1

        for a in per_asset:
            pa = per_asset[a]
            pa["win_rate"] = pa["wins"] / pa["trades"] * 100 if pa["trades"] > 0 else 0

        return {
            "total_minutes": total_minutes,
            "trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(trades) * 100 if trades else 0,
            "total_pnl_points": round(total_pnl, 4),
            "avg_r_multiple": round(avg_r, 3),
            "start_ts": timeline[0] if timeline else "",
            "end_ts": timeline[-1] if timeline else "",
            "per_asset": per_asset,
            "trade_list": [
                {
                    "asset": t["asset"],
                    "direction": t["direction"],
                    "entry_ts": t["entry_timestamp"],
                    "exit_ts": t["exit_timestamp"],
                    "entry_price": t["entry_price"],
                    "exit_price": t["exit_price"],
                    "pnl": round(t["pnl"], 4),
                    "r_multiple": round(t["r_multiple"], 3),
                    "status": t["status"],
                    "confidence": t.get("confidence", 0),
                }
                for t in trades
            ],
        }
