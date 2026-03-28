"""Offline timeline simulation – runs DQN on historical data at max speed.

Reads candles from simLastCharts.db, runs DQN inference minute-by-minute,
opens/closes trades, saves results to sim_trades table for AI training.

Key optimisations:
- All 4 assets batched into one forward pass per minute
- await asyncio.sleep(0) every YIELD_EVERY minutes → event loop stays alive
- Trades saved incrementally every FLUSH_EVERY minutes (cancel-safe)
"""
import asyncio
import logging
from typing import Callable

import aiosqlite
import numpy as np
import torch

from . import config
from .ai_analyzer import (
    ACTION_SIZE,
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

DEFAULT_CONFIDENCE_THRESHOLD = 8
SL_PCT = 0.003
TP_PCT = 0.005
YIELD_EVERY = 200    # yield to event loop every N minutes
FLUSH_EVERY = 2000   # save trades to DB every N minutes (partial results on cancel)


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
        # Live stats for progress polling
        self.current_minute = 0
        self.total_minutes = 0
        self.open_trade_count = 0
        self.closed_trade_count = 0

    @staticmethod
    def _resolve_device() -> torch.device:
        if config.DQN_DEVICE == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(config.DQN_DEVICE)

    def cancel(self) -> None:
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
        """Build 2512-dim state vector from sliding OHLC window (n, 4)."""
        avail = len(ohlc_window)
        state_ohlcv = np.zeros((MAX_WINDOW, 5), dtype=np.float32)

        if avail > 0:
            opens  = ohlc_window[:, 0]
            highs  = ohlc_window[:, 1]
            lows   = ohlc_window[:, 2]
            closes = ohlc_window[:, 3]

            ref = float(closes[-1]) or 1.0
            rows = np.column_stack([
                opens  / ref - 1,
                highs  / ref - 1,
                lows   / ref - 1,
                closes / ref - 1,
                np.zeros(avail, dtype=np.float32),   # volume = 0
            ]).astype(np.float32)
            state_ohlcv[MAX_WINDOW - avail:] = np.clip(rows, -5.0, 5.0)

            rsi     = (_rsi(closes) - 50.0) / 50.0
            atr_val = _atr(highs, lows, closes)
            atr_pct = float(np.clip(atr_val / (ref + 1e-8), 0, 0.05) / 0.05)
            ema20_r = float(np.clip(_ema(closes, min(20, avail)) / ref - 1, -0.1, 0.1) / 0.1)
            ema50_r = float(np.clip(_ema(closes, min(50, avail)) / ref - 1, -0.1, 0.1) / 0.1)
        else:
            rsi = atr_pct = ema20_r = ema50_r = 0.0

        indicators = np.array([rsi, atr_pct, ema20_r, ema50_r], dtype=np.float32)
        asset_oh   = np.zeros(4, dtype=np.float32)
        if asset in ASSET_INDEX:
            asset_oh[ASSET_INDEX[asset]] = 1.0

        pos = np.array(
            [1.0, position_direction, float(np.clip(unrealised_r, -3, 3)), 0.0],
            dtype=np.float32,
        ) if has_position else np.zeros(4, dtype=np.float32)

        return np.concatenate([state_ohlcv.flatten(), indicators, asset_oh, pos])

    @torch.no_grad()
    def _infer_batch(self, states: list[np.ndarray]) -> list[tuple[int, float]]:
        """Single GPU/CPU forward pass for all states at once."""
        net = self._load_model()
        batch = torch.FloatTensor(np.stack(states)).to(self._device)
        q_values = net(batch).cpu().numpy()

        results = []
        for q in q_values:
            action = int(q.argmax())
            probs  = np.exp(q - q.max())
            probs /= probs.sum()
            results.append((action, float(probs.max())))
        return results

    async def run(
        self,
        assets: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        progress_callback: Callable | None = None,
    ) -> dict:
        """Run full timeline simulation. Yields to event loop every YIELD_EVERY minutes."""
        self._cancelled = False
        self._load_model()

        target_assets = assets or list(config.WATCHLIST.keys())

        # Clear old sim_trades from this DB
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM sim_trades WHERE sl_variant = 'dqn_timeline'")
            await db.commit()

        # Load all candles into RAM
        asset_data:      dict[str, np.ndarray]   = {}
        asset_timestamps: dict[str, list[str]]    = {}

        for asset in target_assets:
            candles, timestamps = await self._load_asset_candles(asset, start_date, end_date)
            if len(candles) > 0:
                asset_data[asset]       = candles
                asset_timestamps[asset] = timestamps

        if not asset_data:
            return {"error": "Keine Kerzen in der DB gefunden", "trades": 0}

        # Build global timeline + per-asset ts→index maps
        all_ts = set()
        for ts_list in asset_timestamps.values():
            all_ts.update(ts_list)
        timeline = sorted(all_ts)
        self.total_minutes = len(timeline)

        if self.total_minutes == 0:
            return {"error": "Leerer Zeitstrahl", "trades": 0}

        asset_ts_index: dict[str, dict[str, int]] = {
            asset: {ts: i for i, ts in enumerate(asset_timestamps[asset])}
            for asset in asset_data
        }

        open_trades:   dict[str, dict] = {}  # asset → trade
        closed_trades: list[dict]      = []
        pending_save:  list[dict]      = []  # unsaved closed trades
        trade_id = 0

        logger.info(
            "Timeline simulation: %d assets | %d minutes | threshold=%d | device=%s",
            len(asset_data), self.total_minutes, self.confidence_threshold, self._device,
        )

        for minute_idx, current_ts in enumerate(timeline):
            self.current_minute = minute_idx + 1

            # ── Yield to event loop periodically ─────────────────
            if minute_idx % YIELD_EVERY == 0:
                self.open_trade_count   = len(open_trades)
                self.closed_trade_count = len(closed_trades)
                if progress_callback:
                    progress_callback(
                        minute_idx + 1, self.total_minutes,
                        len(open_trades), len(closed_trades),
                    )
                await asyncio.sleep(0)

            # ── Cancel check ──────────────────────────────────────
            if self._cancelled:
                logger.info("Timeline simulation cancelled at minute %d", minute_idx)
                break

            # ── Flush trades to DB incrementally ─────────────────
            if pending_save and minute_idx % FLUSH_EVERY == 0:
                await self._save_trades(pending_save)
                pending_save.clear()

            # ── Collect states for all assets at this timestamp ───
            states_to_infer: list[tuple[str, np.ndarray]] = []

            for asset in list(asset_data.keys()):
                ts_idx_map = asset_ts_index[asset]
                if current_ts not in ts_idx_map:
                    continue

                candle_idx     = ts_idx_map[current_ts]
                candles        = asset_data[asset]
                current_candle = candles[candle_idx]
                c_high         = float(current_candle[1])
                c_low          = float(current_candle[2])
                c_close        = float(current_candle[3])

                # ── Evaluate open trade ───────────────────────────
                if asset in open_trades:
                    tr = open_trades[asset]
                    hit_sl = hit_tp = False

                    if tr["direction"] == "BUY":
                        hit_sl = c_low  <= tr["sl_price"]
                        hit_tp = c_high >= tr["tp_price"]
                    else:
                        hit_sl = c_high >= tr["sl_price"]
                        hit_tp = c_low  <= tr["tp_price"]

                    if hit_sl or hit_tp:
                        exit_px  = tr["sl_price"] if hit_sl else tr["tp_price"]
                        pnl      = (exit_px - tr["entry_price"]) if tr["direction"] == "BUY" else (tr["entry_price"] - exit_px)
                        risk     = abs(tr["entry_price"] - tr["sl_price"])
                        r_mult   = pnl / risk if risk > 0 else 0.0

                        tr.update({
                            "exit_timestamp": current_ts,
                            "exit_price":     exit_px,
                            "status":         "closed_tp" if hit_tp else "closed_sl",
                            "pnl":            pnl,
                            "r_multiple":     r_mult,
                        })
                        closed_trades.append(tr)
                        pending_save.append(tr)
                        del open_trades[asset]

                # ── Queue state for inference (only if no open trade) ─
                if asset not in open_trades and candle_idx >= MAX_WINDOW:
                    window_start = candle_idx - MAX_WINDOW + 1
                    window       = candles[window_start:candle_idx + 1]
                    states_to_infer.append((asset, window, c_close))

            # ── Batch inference for all assets at this minute ─────
            if states_to_infer:
                states = [self._build_state(a, w) for a, w, _ in states_to_infer]
                results = self._infer_batch(states)

                for (asset, window, c_close), (action_id, softmax_conf) in zip(states_to_infer, results):
                    confidence = _scale_confidence(softmax_conf)

                    if action_id in (1, 2) and confidence >= self.confidence_threshold:
                        direction = "BUY" if action_id == 1 else "SELL"
                        trade_id += 1
                        sl = c_close * (1 - SL_PCT) if direction == "BUY" else c_close * (1 + SL_PCT)
                        tp = c_close * (1 + TP_PCT) if direction == "BUY" else c_close * (1 - TP_PCT)
                        open_trades[asset] = {
                            "id":              trade_id,
                            "asset":           asset,
                            "direction":       direction,
                            "sl_variant":      "dqn_timeline",
                            "entry_timestamp": current_ts,
                            "entry_price":     c_close,
                            "sl_price":        sl,
                            "tp_price":        tp,
                            "confidence":      confidence,
                        }

        # ── Close remaining open trades at last available price ───
        for asset, tr in open_trades.items():
            candles  = asset_data[asset]
            last_px  = float(candles[-1, 3])
            pnl      = (last_px - tr["entry_price"]) if tr["direction"] == "BUY" else (tr["entry_price"] - last_px)
            risk     = abs(tr["entry_price"] - tr["sl_price"])
            tr.update({
                "exit_timestamp": timeline[-1],
                "exit_price":     last_px,
                "status":         "closed_end",
                "pnl":            pnl,
                "r_multiple":     pnl / risk if risk > 0 else 0.0,
            })
            closed_trades.append(tr)
            pending_save.append(tr)

        # ── Final flush ───────────────────────────────────────────
        if pending_save:
            await self._save_trades(pending_save)

        self.open_trade_count   = 0
        self.closed_trade_count = len(closed_trades)

        if progress_callback:
            progress_callback(self.total_minutes, self.total_minutes, 0, len(closed_trades))

        return self._build_summary(closed_trades, self.total_minutes, timeline)

    # ── DB helpers ────────────────────────────────────────────────────────────

    async def _load_asset_candles(
        self,
        asset: str,
        start_date: str | None,
        end_date: str | None,
    ) -> tuple[np.ndarray, list[str]]:
        conditions = ["asset = ?"]
        params: list = [asset]
        if start_date:
            conditions.append("timestamp >= ?")
            params.append(start_date + "T00:00:00")
        if end_date:
            conditions.append("timestamp <= ?")
            params.append(end_date + "T23:59:59")

        where = " AND ".join(conditions)
        query = (
            f"SELECT timestamp, open, high, low, close "
            f"FROM price_history WHERE {where} ORDER BY timestamp ASC"
        )
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(query, params) as cur:
                rows = await cur.fetchall()

        if not rows:
            return np.array([]), []

        timestamps = [r[0] for r in rows]
        candles    = np.array([[r[1], r[2], r[3], r[4]] for r in rows], dtype=np.float64)
        return candles, timestamps

    async def _save_trades(self, trades: list[dict]) -> None:
        if not trades:
            return
        rows = [
            (
                t["asset"], t["direction"], t["sl_variant"],
                t["entry_timestamp"], t["entry_price"],
                t["sl_price"],        t["tp_price"],
                t.get("exit_timestamp"), t.get("exit_price"),
                t.get("status", "closed_end"),
                t.get("pnl"),         t.get("r_multiple"),
            )
            for t in trades
        ]
        async with aiosqlite.connect(self.db_path) as db:
            await db.executemany(
                """INSERT OR IGNORE INTO sim_trades
                   (asset, direction, sl_variant, entry_timestamp, entry_price,
                    sl_price, tp_price, exit_timestamp, exit_price, status, pnl, r_multiple)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            await db.commit()

    @staticmethod
    def _build_summary(trades: list[dict], total_minutes: int, timeline: list[str]) -> dict:
        if not trades:
            return {
                "total_minutes": total_minutes, "trades": 0,
                "wins": 0, "losses": 0, "win_rate": 0.0,
                "total_pnl_points": 0.0, "avg_r_multiple": 0.0,
                "start_ts": timeline[0] if timeline else "",
                "end_ts":   timeline[-1] if timeline else "",
                "per_asset": {}, "trade_list": [],
            }

        wins   = [t for t in trades if (t.get("pnl") or 0) > 0]
        losses = [t for t in trades if (t.get("pnl") or 0) <= 0]
        pnls   = [t.get("pnl") or 0 for t in trades]
        rs     = [t.get("r_multiple") or 0 for t in trades]

        per_asset: dict[str, dict] = {}
        for t in trades:
            a = t["asset"]
            if a not in per_asset:
                per_asset[a] = {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0}
            per_asset[a]["trades"] += 1
            per_asset[a]["pnl"]    += t.get("pnl") or 0
            if (t.get("pnl") or 0) > 0:
                per_asset[a]["wins"]   += 1
            else:
                per_asset[a]["losses"] += 1
        for a in per_asset:
            pa = per_asset[a]
            pa["win_rate"] = pa["wins"] / pa["trades"] * 100 if pa["trades"] > 0 else 0

        return {
            "total_minutes":    total_minutes,
            "trades":           len(trades),
            "wins":             len(wins),
            "losses":           len(losses),
            "win_rate":         len(wins) / len(trades) * 100 if trades else 0,
            "total_pnl_points": round(sum(pnls), 4),
            "avg_r_multiple":   round(sum(rs) / len(rs), 3) if rs else 0,
            "start_ts":         timeline[0]  if timeline else "",
            "end_ts":           timeline[-1] if timeline else "",
            "per_asset":        per_asset,
            "trade_list": [
                {
                    "asset":       t["asset"],
                    "direction":   t["direction"],
                    "entry_ts":    t["entry_timestamp"],
                    "exit_ts":     t.get("exit_timestamp", ""),
                    "entry_price": t["entry_price"],
                    "exit_price":  t.get("exit_price", 0),
                    "pnl":         round(t.get("pnl") or 0, 4),
                    "r_multiple":  round(t.get("r_multiple") or 0, 3),
                    "status":      t.get("status", ""),
                    "confidence":  t.get("confidence", 0),
                }
                for t in trades
            ],
        }
