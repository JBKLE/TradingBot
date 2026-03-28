"""Offline timeline simulation – runs DQN on historical data at max speed.

The heavy computation runs in a ThreadPoolExecutor so the FastAPI event loop
(and dashboard) stay fully responsive throughout. Cancel works instantly.

Trades are saved incrementally to sim_trades (sl_variant='dqn_timeline')
for use with: python train_rl.py --db history
"""
import asyncio
import logging
import sqlite3
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
    calculate_trade_financials,
    DEFAULT_SPREADS,
)
from .fetch_history import HISTORY_DB_PATH

logger = logging.getLogger(__name__)

DEFAULT_CONFIDENCE_THRESHOLD = 8
SL_PCT = 0.003
TP_PCT = 0.005
PROGRESS_EVERY = 500   # update progress counters every N minutes
FLUSH_EVERY    = 2000  # flush closed trades to DB every N minutes


class TimelineSimulator:
    """Runs DQN offline on historical candles. Heavy work in thread pool."""

    def __init__(
        self,
        db_path: str = HISTORY_DB_PATH,
        confidence_threshold: int = DEFAULT_CONFIDENCE_THRESHOLD,
        models_dir: str | None = None,
        # Financial simulation parameters (None = no financial tracking)
        capital: float | None = None,
        risk_pct: float | None = None,
        leverage: int | None = None,
        eur_usd: float = 1.08,
    ) -> None:
        self.db_path = db_path
        self.confidence_threshold = confidence_threshold
        self._models_dir = models_dir or config.AI_MODELS_DIR
        self._device = self._resolve_device()
        self._net: DuelingDQN | None = None
        self._cancelled = False
        # Financial params
        self.capital   = capital
        self.risk_pct  = risk_pct
        self.leverage  = leverage
        self.eur_usd   = eur_usd
        self.fin_enabled = capital is not None and risk_pct is not None and leverage is not None
        # Live stats – written from worker thread, read from API thread (GIL-safe)
        self.current_minute:   int = 0
        self.total_minutes:    int = 0
        self.open_trade_count: int = 0
        self.closed_trade_count: int = 0
        self.current_capital: float = capital or 0.0

    @staticmethod
    def _resolve_device() -> torch.device:
        if config.DQN_DEVICE == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(config.DQN_DEVICE)

    def cancel(self) -> None:
        """Thread-safe cancel flag (simple bool, GIL-protected)."""
        self._cancelled = True

    # ── Model ─────────────────────────────────────────────────────────────────

    def _load_model(self) -> DuelingDQN:
        if self._net is not None:
            return self._net
        path = _get_latest_model_path(self._models_dir)
        logger.info("Timeline sim: loading model %s (device=%s)", path, self._device)
        net = DuelingDQN(ACTION_SIZE).to(self._device)
        ckpt = torch.load(path, map_location=self._device, weights_only=True)
        net.load_state_dict(ckpt["policy_net"])
        net.eval()
        self._net = net
        return net

    # ── State builder ─────────────────────────────────────────────────────────

    def _build_state(
        self,
        asset: str,
        window: np.ndarray,           # shape (n, 4): open/high/low/close
        has_position: bool = False,
        position_dir: float = 0.0,
        unrealised_r: float = 0.0,
    ) -> np.ndarray:
        avail      = len(window)
        ohlcv_buf  = np.zeros((MAX_WINDOW, 5), dtype=np.float32)

        if avail > 0:
            opens  = window[:, 0]
            highs  = window[:, 1]
            lows   = window[:, 2]
            closes = window[:, 3]
            ref    = float(closes[-1]) or 1.0

            rows = np.column_stack([
                opens  / ref - 1,
                highs  / ref - 1,
                lows   / ref - 1,
                closes / ref - 1,
                np.zeros(avail),
            ]).astype(np.float32)
            ohlcv_buf[MAX_WINDOW - avail:] = np.clip(rows, -5.0, 5.0)

            rsi     = (_rsi(closes) - 50.0) / 50.0
            atr_v   = _atr(highs, lows, closes)
            atr_pct = float(np.clip(atr_v / (ref + 1e-8), 0, 0.05) / 0.05)
            e20     = float(np.clip(_ema(closes, min(20, avail)) / ref - 1, -0.1, 0.1) / 0.1)
            e50     = float(np.clip(_ema(closes, min(50, avail)) / ref - 1, -0.1, 0.1) / 0.1)
        else:
            rsi = atr_pct = e20 = e50 = 0.0

        indicators = np.array([rsi, atr_pct, e20, e50], dtype=np.float32)
        asset_oh   = np.zeros(4, dtype=np.float32)
        if asset in ASSET_INDEX:
            asset_oh[ASSET_INDEX[asset]] = 1.0
        pos = (
            np.array([1.0, position_dir, float(np.clip(unrealised_r, -3, 3)), 0.0], dtype=np.float32)
            if has_position else np.zeros(4, dtype=np.float32)
        )
        return np.concatenate([ohlcv_buf.flatten(), indicators, asset_oh, pos])

    # ── Batch inference ───────────────────────────────────────────────────────

    @torch.no_grad()
    def _infer_batch(self, states: list[np.ndarray]) -> list[tuple[int, float]]:
        """One forward pass for all states at once."""
        net   = self._load_model()
        batch = torch.FloatTensor(np.stack(states)).to(self._device)
        qs    = net(batch).cpu().numpy()
        out   = []
        for q in qs:
            action = int(q.argmax())
            probs  = np.exp(q - q.max())
            probs /= probs.sum()
            out.append((action, float(probs.max())))
        return out

    # ── Async entry point ─────────────────────────────────────────────────────

    async def run(
        self,
        assets: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        progress_callback: Callable | None = None,
    ) -> dict:
        """Load candles async, then run heavy simulation in thread pool."""
        self._cancelled = False
        self._load_model()

        target_assets = assets or list(config.WATCHLIST.keys())

        # Clear previous timeline trades
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM sim_trades WHERE sl_variant = 'dqn_timeline'")
            await db.commit()

        # Load all candles into RAM (async DB reads)
        asset_candles:    dict[str, np.ndarray] = {}
        asset_timestamps: dict[str, list[str]]  = {}
        for asset in target_assets:
            candles, timestamps = await self._load_asset_candles(asset, start_date, end_date)
            if len(candles) > 0:
                asset_candles[asset]    = candles
                asset_timestamps[asset] = timestamps
                logger.info("Timeline sim: %s loaded %d candles", asset, len(candles))

        if not asset_candles:
            return {"error": "Keine Kerzen in der DB gefunden", "trades": 0}

        # Build global timeline
        all_ts   = set().union(*(set(ts) for ts in asset_timestamps.values()))
        timeline = sorted(all_ts)
        self.total_minutes = len(timeline)

        if not timeline:
            return {"error": "Leerer Zeitstrahl", "trades": 0}

        logger.info(
            "Timeline simulation: %d assets | %d minutes | threshold=%d | device=%s",
            len(asset_candles), self.total_minutes, self.confidence_threshold, self._device,
        )

        # ── Run heavy loop in thread – event loop stays free ──────────────────
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._simulate_sync,
            asset_candles, asset_timestamps, timeline, progress_callback,
            self.capital, self.risk_pct, self.leverage, self.eur_usd,
        )
        return result

    # ── Synchronous simulation (runs in thread) ───────────────────────────────

    def _simulate_sync(
        self,
        asset_candles:    dict[str, np.ndarray],
        asset_timestamps: dict[str, list[str]],
        timeline:         list[str],
        progress_callback: Callable | None,
        capital:   float | None = None,
        risk_pct:  float | None = None,
        leverage:  int   | None = None,
        eur_usd:   float = 1.08,
    ) -> dict:
        asset_ts_index = {
            asset: {ts: i for i, ts in enumerate(asset_timestamps[asset])}
            for asset in asset_candles
        }

        fin_enabled = capital is not None and risk_pct is not None and leverage is not None
        running_capital = capital or 0.0
        peak_capital    = running_capital
        max_drawdown    = 0.0
        equity_curve: list[float] = [running_capital] if fin_enabled else []

        open_trades:  dict[str, dict] = {}
        closed_trades: list[dict]     = []
        pending_save:  list[dict]     = []
        trade_id = 0
        total    = len(timeline)
        margin_call = False

        for minute_idx, current_ts in enumerate(timeline):
            self.current_minute = minute_idx + 1

            if self._cancelled:
                logger.info("Timeline sim cancelled at minute %d", minute_idx)
                break

            # Progress + DB flush
            if minute_idx % PROGRESS_EVERY == 0:
                self.open_trade_count   = len(open_trades)
                self.closed_trade_count = len(closed_trades)
                if progress_callback:
                    progress_callback(minute_idx + 1, total, len(open_trades), len(closed_trades))

            if pending_save and minute_idx % FLUSH_EVERY == 0:
                self._save_trades_sync(pending_save)
                pending_save.clear()

            # ── Per-asset: evaluate open trade + queue inference ──────────
            states_to_infer: list[tuple[str, np.ndarray, float]] = []

            for asset, candles in asset_candles.items():
                ts_map = asset_ts_index[asset]
                if current_ts not in ts_map:
                    continue

                ci      = ts_map[current_ts]
                candle  = candles[ci]
                c_high  = float(candle[1])
                c_low   = float(candle[2])
                c_close = float(candle[3])

                # Evaluate open trade
                if asset in open_trades:
                    tr  = open_trades[asset]
                    buy = tr["direction"] == "BUY"
                    hit_sl = (c_low <= tr["sl_price"]) if buy else (c_high >= tr["sl_price"])
                    hit_tp = (c_high >= tr["tp_price"]) if buy else (c_low <= tr["tp_price"])

                    if hit_sl or hit_tp:
                        exit_px = tr["sl_price"] if hit_sl else tr["tp_price"]
                        pnl     = (exit_px - tr["entry_price"]) if buy else (tr["entry_price"] - exit_px)
                        risk    = abs(tr["entry_price"] - tr["sl_price"])
                        r_mult  = pnl / risk if risk > 0 else 0.0

                        fin: dict | None = None
                        if fin_enabled:
                            fin = calculate_trade_financials(
                                asset=asset,
                                direction=tr["direction"],
                                entry_price=tr["entry_price"],
                                exit_pnl=pnl,
                                sl_price=tr["sl_price"],
                                capital=running_capital,
                                risk_pct=risk_pct,
                                leverage=leverage,
                                eur_usd=eur_usd,
                            )
                            running_capital += fin["netto_pnl_eur"]
                            self.current_capital = running_capital
                            equity_curve.append(running_capital)
                            if running_capital > peak_capital:
                                peak_capital = running_capital
                            dd = (peak_capital - running_capital) / peak_capital * 100 if peak_capital > 0 else 0
                            if dd > max_drawdown:
                                max_drawdown = dd

                        tr.update({
                            "exit_timestamp": current_ts,
                            "exit_price":     exit_px,
                            "status":         "closed_tp" if hit_tp else "closed_sl",
                            "pnl":            pnl,
                            "r_multiple":     r_mult,
                            "financial":      fin,
                            "capital_after":  running_capital if fin_enabled else None,
                        })
                        closed_trades.append(tr)
                        pending_save.append(tr)
                        del open_trades[asset]

                        # Margin call: capital gone negative → stop simulation
                        if fin_enabled and running_capital <= 0:
                            margin_call = True

                # Queue for inference (only if no open trade + enough history)
                if asset not in open_trades and ci >= MAX_WINDOW:
                    window = candles[ci - MAX_WINDOW + 1: ci + 1]
                    states_to_infer.append((asset, window, c_close))

            if margin_call:
                logger.info("Margin call at minute %d — capital: %.2f EUR", minute_idx, running_capital)
                break

            # ── Batch inference for all queued assets ─────────────────────
            if states_to_infer:
                states  = [self._build_state(a, w) for a, w, _ in states_to_infer]
                results = self._infer_batch(states)

                for (asset, _w, c_close), (action_id, softmax_conf) in zip(states_to_infer, results):
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

        # Close any remaining open trades at last price
        for asset, tr in open_trades.items():
            last_px = float(asset_candles[asset][-1, 3])
            buy     = tr["direction"] == "BUY"
            pnl     = (last_px - tr["entry_price"]) if buy else (tr["entry_price"] - last_px)
            risk    = abs(tr["entry_price"] - tr["sl_price"])

            fin = None
            if fin_enabled:
                fin = calculate_trade_financials(
                    asset=asset,
                    direction=tr["direction"],
                    entry_price=tr["entry_price"],
                    exit_pnl=pnl,
                    sl_price=tr["sl_price"],
                    capital=running_capital,
                    risk_pct=risk_pct,
                    leverage=leverage,
                    eur_usd=eur_usd,
                )
                running_capital += fin["netto_pnl_eur"]
                self.current_capital = running_capital
                equity_curve.append(running_capital)

            tr.update({
                "exit_timestamp": timeline[-1],
                "exit_price":     last_px,
                "status":         "closed_end",
                "pnl":            pnl,
                "r_multiple":     pnl / risk if risk > 0 else 0.0,
                "financial":      fin,
                "capital_after":  running_capital if fin_enabled else None,
            })
            closed_trades.append(tr)
            pending_save.append(tr)

        if pending_save:
            self._save_trades_sync(pending_save)

        self.open_trade_count   = 0
        self.closed_trade_count = len(closed_trades)
        if progress_callback:
            progress_callback(total, total, 0, len(closed_trades))

        # Build financial summary extras
        fin_summary: dict = {}
        if fin_enabled:
            total_return = (running_capital - capital) / capital * 100 if capital else 0
            fin_summary = {
                "start_capital":   round(capital, 2),
                "end_capital":     round(running_capital, 2),
                "total_return_pct": round(total_return, 2),
                "max_drawdown_pct": round(max_drawdown, 2),
                "margin_call":     margin_call,
                "equity_curve":    [round(v, 2) for v in equity_curve],
            }

        return self._build_summary(closed_trades, total, timeline, fin_summary)

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

    def _save_trades_sync(self, trades: list[dict]) -> None:
        """Synchronous sqlite3 write – called from worker thread."""
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
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executemany(
            """INSERT OR IGNORE INTO sim_trades
               (asset, direction, sl_variant, entry_timestamp, entry_price,
                sl_price, tp_price, exit_timestamp, exit_price, status, pnl, r_multiple)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        conn.commit()
        conn.close()

    @staticmethod
    def _build_summary(trades: list[dict], total_minutes: int, timeline: list[str], fin_summary: dict | None = None) -> dict:
        if not trades:
            return {
                "total_minutes": total_minutes, "trades": 0,
                "wins": 0, "losses": 0, "win_rate": 0.0,
                "total_pnl_points": 0.0, "avg_r_multiple": 0.0,
                "start_ts": timeline[0] if timeline else "",
                "end_ts":   timeline[-1] if timeline else "",
                "per_asset": {}, "trade_list": [],
            }

        pnls = [t.get("pnl") or 0 for t in trades]
        rs   = [t.get("r_multiple") or 0 for t in trades]
        wins = sum(1 for p in pnls if p > 0)

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
        for stats in per_asset.values():
            stats["win_rate"] = stats["wins"] / stats["trades"] * 100 if stats["trades"] > 0 else 0

        result = {
            "total_minutes":    total_minutes,
            "trades":           len(trades),
            "wins":             wins,
            "losses":           len(trades) - wins,
            "win_rate":         wins / len(trades) * 100,
            "total_pnl_points": round(sum(pnls), 4),
            "avg_r_multiple":   round(sum(rs) / len(rs), 3) if rs else 0,
            "start_ts":         timeline[0],
            "end_ts":           timeline[-1],
            "per_asset":        per_asset,
            "financial":        fin_summary or {},
            "trade_list": [
                {
                    "asset":       t["asset"],
                    "direction":   t["direction"],
                    "entry_ts":    t["entry_timestamp"],
                    "exit_ts":     t.get("exit_timestamp", ""),
                    "entry_price": t["entry_price"],
                    "exit_price":  t.get("exit_price", 0),
                    "pnl":            round(t.get("pnl") or 0, 4),
                    "r_multiple":     round(t.get("r_multiple") or 0, 3),
                    "status":         t.get("status", ""),
                    "confidence":     t.get("confidence", 0),
                    "netto_pnl_eur":  round(t["financial"]["netto_pnl_eur"], 2) if t.get("financial") else None,
                    "capital_after":  round(t["capital_after"], 2) if t.get("capital_after") is not None else None,
                    "lot_size":       round(t["financial"]["lot_size"], 4) if t.get("financial") else None,
                    "margin_eur":     round(t["financial"]["margin_eur"], 2) if t.get("financial") else None,
                }
                for t in trades
            ],
        }
        return result
