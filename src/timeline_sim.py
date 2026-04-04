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
    ASSET_INDEX,
    MODEL_VERSIONS,
    ModelVersionConfig,
    DuelingDQN,
    _atr,
    _bollinger_width,
    _ema,
    _get_latest_model_path,
    _macd_histogram,
    _rsi,
    _scale_confidence,
    calculate_trade_financials,
    parse_model_filename,
    DEFAULT_SPREADS,
)
from .fetch_history import HISTORY_DB_PATH

logger = logging.getLogger(__name__)


def _holding_nights(entry_ts: str, exit_ts: str) -> int:
    """Berechnet die Anzahl Uebernachtungen zwischen Entry und Exit (22:00 UTC)."""
    from datetime import datetime
    try:
        e = datetime.fromisoformat(entry_ts)
        x = datetime.fromisoformat(exit_ts)
        return max(0, (x.date() - e.date()).days)
    except Exception:
        return 0


DEFAULT_CONFIDENCE_THRESHOLD = 8
DEFAULT_SL_PCT = 0.003
DEFAULT_TP_PCT = 0.005
PROGRESS_EVERY = 500   # update progress counters every N minutes
FLUSH_EVERY    = 2000  # flush closed trades to DB every N minutes


class TimelineSimulator:
    """Runs DQN offline on historical candles. Heavy work in thread pool."""

    def __init__(
        self,
        db_path: str = HISTORY_DB_PATH,
        confidence_threshold: int = DEFAULT_CONFIDENCE_THRESHOLD,
        confidence_levels: list[int] | None = None,
        min_q_spread: float = 0.0,
        model_path: str | None = None,
        models_dir: str | None = None,
        # Financial simulation parameters (None = no financial tracking)
        capital: float | None = None,
        risk_pct: float | None = None,
        leverage: int | None = None,
        # SL/TP percentages
        sl_pct: float = DEFAULT_SL_PCT,
        tp_pct: float = DEFAULT_TP_PCT,
        # Output DB for trades (None = same as db_path)
        output_db_path: str | None = None,
    ) -> None:
        self.db_path = db_path
        self.output_db_path = output_db_path or db_path
        self.confidence_levels = confidence_levels or list(range(confidence_threshold, 11))
        self.min_q_spread = min_q_spread
        self._explicit_model_path = model_path  # wenn gesetzt, wird genau dieses Modell geladen
        self._models_dir = models_dir or config.AI_MODELS_DIR
        self._device = self._resolve_device()
        self._net: DuelingDQN | None = None
        self._vcfg: ModelVersionConfig = MODEL_VERSIONS["v1"]  # Default, wird bei _load_model aktualisiert
        self._cancelled = False
        self.model_path: str = ""   # gesetzt sobald Modell geladen ist
        # Financial params
        self.capital   = capital
        self.risk_pct  = risk_pct
        self.leverage  = leverage
        self.sl_pct    = sl_pct
        self.tp_pct    = tp_pct
        self.fin_enabled = capital is not None and risk_pct is not None and leverage is not None
        # Live stats – written from worker thread, read from API thread (GIL-safe)
        self.current_minute:   int = 0
        self.total_minutes:    int = 0
        self.open_trade_count: int = 0
        self.closed_trade_count: int = 0
        self.current_capital: float = capital or 0.0
        # Live snapshots for dashboard (updated every PROGRESS_EVERY minutes)
        self.equity_snap:        list = []   # sampled equity curve values
        self.open_trades_snap:   list = []   # current open trades
        self.closed_trades_snap: list = []   # last ~100 closed trades

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
        path = self._explicit_model_path or _get_latest_model_path(self._models_dir)
        self.model_path = path

        # Version aus Dateinamen erkennen
        info = parse_model_filename(path)
        version = info.get("version") or "v1"
        self._vcfg = MODEL_VERSIONS.get(version, MODEL_VERSIONS["v1"])

        logger.info(
            "Timeline sim: loading model %s (version=%s, device=%s)",
            path, self._vcfg.version, self._device,
        )

        ckpt = torch.load(path, map_location=self._device, weights_only=True)
        sd = ckpt["policy_net"]

        # Auto-detect Architektur aus Checkpoint (wie in DQNAnalyzer)
        from .ai_analyzer import DQNAnalyzer
        detected = DQNAnalyzer._detect_version_from_checkpoint(sd)
        if detected and detected != self._vcfg.version:
            logger.warning(
                "Timeline sim: Checkpoint ist '%s', nicht '%s' – korrigiere",
                detected, self._vcfg.version,
            )
            self._vcfg = MODEL_VERSIONS[detected]

        # Action-Size Feinabgleich
        adv_key = "advantage_stream.2.weight"
        if adv_key in sd:
            ckpt_actions = sd[adv_key].shape[0]
            if ckpt_actions != self._vcfg.action_size:
                from dataclasses import replace
                matching = [v for v in MODEL_VERSIONS.values() if v.action_size == ckpt_actions]
                if matching:
                    self._vcfg = replace(
                        self._vcfg,
                        action_size=ckpt_actions,
                        action_map=matching[0].action_map,
                        actions=matching[0].actions,
                    )

        net = DuelingDQN(self._vcfg).to(self._device)
        net.load_state_dict(sd)
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
        # v5 extras (ignored for v1/v2)
        steps_norm: float = 0.0,
        peak_pnl_pct: float = 0.0,
        drawdown_from_peak: float = 0.0,
    ) -> np.ndarray:
        vcfg = self._vcfg
        max_window = vcfg.max_window
        avail = min(len(window), max_window)
        ohlcv_buf = np.zeros((max_window, 5), dtype=np.float32)

        if avail > 0:
            w = window[-avail:]
            opens  = w[:, 0]
            highs  = w[:, 1]
            lows   = w[:, 2]
            closes = w[:, 3]
            ref    = float(closes[-1]) or 1.0

            rows = np.column_stack([
                opens  / ref - 1,
                highs  / ref - 1,
                lows   / ref - 1,
                closes / ref - 1,
                np.zeros(avail),
            ]).astype(np.float32)
            ohlcv_buf[max_window - avail:] = np.clip(rows, -5.0, 5.0)

            rsi     = (_rsi(closes) - 50.0) / 50.0
            atr_v   = _atr(highs, lows, closes)
            atr_pct = float(np.clip(atr_v / (ref + 1e-8), 0, 0.05) / 0.05)
            e20     = float(np.clip(_ema(closes, min(20, avail)) / ref - 1, -0.1, 0.1) / 0.1)
            e50     = float(np.clip(_ema(closes, min(50, avail)) / ref - 1, -0.1, 0.1) / 0.1)

            ind_list = [rsi, atr_pct, e20, e50]
            # v2+: MACD + Bollinger
            if vcfg.n_indicators >= 6:
                ind_list.append(_macd_histogram(closes))
                ind_list.append(_bollinger_width(closes))
        else:
            ind_list = [0.0] * vcfg.n_indicators

        indicators = np.array(ind_list, dtype=np.float32)
        asset_oh   = np.zeros(4, dtype=np.float32)
        if asset in ASSET_INDEX:
            asset_oh[ASSET_INDEX[asset]] = 1.0

        pos_size = vcfg.position_size
        if not has_position:
            pos = np.zeros(pos_size, dtype=np.float32)
        elif pos_size == 6:
            # v5: [in_pos, dir, unreal_pct, steps_norm, peak_pnl_pct, drawdown]
            pos = np.array([
                1.0, position_dir,
                float(np.clip(unrealised_r, -5, 5)),
                steps_norm,
                float(np.clip(peak_pnl_pct, -5, 5)),
                float(np.clip(drawdown_from_peak, 0, 5)),
            ], dtype=np.float32)
        else:
            pos = np.array([1.0, position_dir, float(np.clip(unrealised_r, -3, 3)), 0.0], dtype=np.float32)

        return np.concatenate([ohlcv_buf.flatten(), indicators, asset_oh, pos])

    # ── Batch inference ───────────────────────────────────────────────────────

    @torch.no_grad()
    def _infer_batch(self, states: list[np.ndarray]) -> list[tuple[int, float, np.ndarray]]:
        """One forward pass for all states at once. Returns (action, softmax_conf, q_values)."""
        net   = self._load_model()
        batch = torch.FloatTensor(np.stack(states)).to(self._device)
        qs    = net(batch).cpu().numpy()
        out   = []
        for q in qs:
            action = int(q.argmax())
            probs  = np.exp(q - q.max())
            probs /= probs.sum()
            out.append((action, float(probs.max()), q.copy()))
        return out

    def _extract_q_fields(self, q: np.ndarray) -> dict:
        """Extract q_buy, q_sell, q_close, q_spread from raw Q-values."""
        rev = {v: k for k, v in self._vcfg.action_map.items()}
        q_buy   = float(q[rev["BUY"]])   if "BUY"   in rev else None
        q_sell  = float(q[rev["SELL"]])   if "SELL"  in rev else None
        q_close = float(q[rev["CLOSE"]]) if "CLOSE" in rev else None
        sorted_q = np.sort(q)[::-1]
        q_spread = float(sorted_q[0] - sorted_q[1]) if len(sorted_q) >= 2 else 0.0
        return {"q_buy": q_buy, "q_sell": q_sell, "q_close": q_close, "q_spread": q_spread}

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

        # Ensure sim_trades table exists in output DB and clear previous run
        async with aiosqlite.connect(self.output_db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS sim_trades (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    asset           TEXT,
                    direction       TEXT,
                    sl_variant      TEXT,
                    entry_timestamp TEXT,
                    entry_price     REAL,
                    sl_price        REAL,
                    tp_price        REAL,
                    exit_timestamp  TEXT,
                    exit_price      REAL,
                    status          TEXT,
                    pnl             REAL,
                    r_multiple      REAL,
                    confidence      REAL,
                    close_confidence REAL
                )
            """)
            # Migrate: add columns if missing (table may pre-date these columns)
            for col, ctype in [
                ("confidence", "REAL"), ("close_confidence", "REAL"),
                ("q_buy", "REAL"), ("q_sell", "REAL"), ("q_close", "REAL"),
                ("q_spread", "REAL"), ("rsi_entry", "REAL"), ("atr_pct_entry", "REAL"),
                ("q_buy_exit", "REAL"), ("q_sell_exit", "REAL"), ("q_close_exit", "REAL"),
                ("peak_pnl", "REAL"), ("peak_timestamp", "TEXT"), ("steps", "INTEGER"),
            ]:
                try:
                    await db.execute(f"ALTER TABLE sim_trades ADD COLUMN {col} {ctype}")
                except Exception:
                    pass  # column already exists
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
            "Timeline simulation: %d assets | %d minutes | levels=%s | q_spread>=%.2f | device=%s",
            len(asset_candles), self.total_minutes, self.confidence_levels, self.min_q_spread, self._device,
        )

        # ── Run heavy loop in thread – event loop stays free ──────────────────
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._simulate_sync,
            asset_candles, asset_timestamps, timeline, progress_callback,
            self.capital, self.risk_pct, self.leverage, self.sl_pct, self.tp_pct,
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
        sl_pct:    float = DEFAULT_SL_PCT,
        tp_pct:    float = DEFAULT_TP_PCT,
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
                # Live snapshots – copied here (GIL-safe list replacement)
                if fin_enabled and equity_curve:
                    eq = equity_curve
                    step = max(1, len(eq) // 300)
                    self.equity_snap = eq[::step]
                self.open_trades_snap = [
                    {k: v for k, v in t.items() if k != "financial"}
                    for t in open_trades.values()
                ]
                self.closed_trades_snap = [
                    {k: v for k, v in t.items() if k != "financial"}
                    for t in closed_trades[-100:]
                ]

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

                    # Peak-PnL + steps tracking
                    tr["steps"] = tr.get("steps", 0) + 1
                    # High/Low-based peak for trade analytics
                    current_peak = (c_high - tr["entry_price"]) if buy else (tr["entry_price"] - c_low)
                    if current_peak > tr.get("peak_pnl", 0.0):
                        tr["peak_pnl"] = current_peak
                        tr["peak_timestamp"] = current_ts
                    # Close-based peak for v5 model state (training uses close, not high/low)
                    close_pnl_pct = (c_close - tr["entry_price"]) / (tr["entry_price"] + 1e-8) * 100.0
                    if not buy:
                        close_pnl_pct = -close_pnl_pct
                    if close_pnl_pct > tr.get("_peak_close_pct", 0.0):
                        tr["_peak_close_pct"] = close_pnl_pct

                    hit_sl = (c_low <= tr["sl_price"]) if buy else (c_high >= tr["sl_price"])
                    hit_tp = (c_high >= tr["tp_price"]) if buy else (c_low <= tr["tp_price"])

                    if hit_sl or hit_tp:
                        exit_px = tr["sl_price"] if hit_sl else tr["tp_price"]
                        pnl     = (exit_px - tr["entry_price"]) if buy else (tr["entry_price"] - exit_px)
                        risk    = abs(tr["entry_price"] - tr["sl_price"])
                        r_mult  = pnl / risk if risk > 0 else 0.0

                        fin: dict | None = None
                        if fin_enabled:
                            nights = _holding_nights(tr["entry_timestamp"], current_ts)
                            fin = calculate_trade_financials(
                                asset=asset,
                                direction=tr["direction"],
                                entry_price=tr["entry_price"],
                                exit_pnl=pnl,
                                sl_price=tr["sl_price"],
                                capital=running_capital,
                                risk_pct=risk_pct,
                                leverage=leverage,
                                holding_nights=nights,
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

                # Queue for inference (auch bei offener Position – Modell kann CLOSE entscheiden)
                if ci >= self._vcfg.max_window:
                    window = candles[ci - self._vcfg.max_window + 1: ci + 1]
                    # Position-Info fuer den State
                    has_pos = asset in open_trades
                    pos_dir = 0.0
                    unreal  = 0.0
                    steps_n = 0.0
                    peak_pct = 0.0
                    dd_pct   = 0.0
                    if has_pos:
                        tr = open_trades[asset]
                        pos_dir = 1.0 if tr["direction"] == "BUY" else -1.0
                        if self._vcfg.position_encoding == "pct":
                            unreal = (c_close - tr["entry_price"]) / (tr["entry_price"] + 1e-8) * 100.0 * pos_dir
                        else:
                            risk = abs(tr["entry_price"] - tr["sl_price"])
                            raw = (c_close - tr["entry_price"]) * pos_dir
                            unreal = raw / (risk + 1e-8)
                        # v5 extras (close-based peak, matching training environment)
                        if self._vcfg.position_size == 6:
                            steps_n = min(tr.get("steps", 0) / 120.0, 1.0)
                            peak_pct = tr.get("_peak_close_pct", 0.0)
                            dd_pct = max(0.0, peak_pct - unreal)
                    states_to_infer.append((asset, window, c_close, has_pos, pos_dir, unreal, steps_n, peak_pct, dd_pct))

            if margin_call:
                logger.info("Margin call at minute %d — capital: %.2f EUR", minute_idx, running_capital)
                break

            # ── Batch inference for all queued assets ─────────────────────
            if states_to_infer:
                states = [
                    self._build_state(
                        a, w, has_position=hp, position_dir=pd, unrealised_r=ur,
                        steps_norm=sn, peak_pnl_pct=pp, drawdown_from_peak=dp,
                    )
                    for a, w, _, hp, pd, ur, sn, pp, dp in states_to_infer
                ]
                results = self._infer_batch(states)

                for (asset, _w, c_close, *_rest), (action_id, softmax_conf, q_raw) in zip(states_to_infer, results):
                    confidence = _scale_confidence(softmax_conf)
                    bot_action = self._vcfg.action_map.get(action_id, "HOLD")
                    qf = self._extract_q_fields(q_raw)

                    # ── CLOSE: offene Position am Markt schliessen ───────
                    if bot_action == "CLOSE" and asset in open_trades:
                        tr  = open_trades[asset]
                        buy = tr["direction"] == "BUY"
                        pnl  = (c_close - tr["entry_price"]) if buy else (tr["entry_price"] - c_close)
                        risk = abs(tr["entry_price"] - tr["sl_price"])
                        r_mult = pnl / risk if risk > 0 else 0.0

                        fin_close: dict | None = None
                        if fin_enabled:
                            nights = _holding_nights(tr["entry_timestamp"], current_ts)
                            fin_close = calculate_trade_financials(
                                asset=asset,
                                direction=tr["direction"],
                                entry_price=tr["entry_price"],
                                exit_pnl=pnl,
                                sl_price=tr["sl_price"],
                                capital=running_capital,
                                risk_pct=risk_pct,
                                leverage=leverage,
                                holding_nights=nights,
                            )
                            running_capital += fin_close["netto_pnl_eur"]
                            self.current_capital = running_capital
                            equity_curve.append(running_capital)
                            if running_capital > peak_capital:
                                peak_capital = running_capital
                            dd = (peak_capital - running_capital) / peak_capital * 100 if peak_capital > 0 else 0
                            if dd > max_drawdown:
                                max_drawdown = dd

                        tr.update({
                            "exit_timestamp":   current_ts,
                            "exit_price":       c_close,
                            "status":           "closed_dqn",
                            "pnl":              pnl,
                            "r_multiple":       r_mult,
                            "close_confidence": confidence,
                            "q_buy_exit":       qf["q_buy"],
                            "q_sell_exit":      qf["q_sell"],
                            "q_close_exit":     qf["q_close"],
                            "financial":        fin_close,
                            "capital_after":    running_capital if fin_enabled else None,
                        })
                        closed_trades.append(tr)
                        pending_save.append(tr)
                        del open_trades[asset]

                        if fin_enabled and running_capital <= 0:
                            margin_call = True
                        continue

                    # ── BUY/SELL: neuen Trade eroeffnen (nur wenn flat) ──
                    if bot_action in ("BUY", "SELL") and asset not in open_trades \
                            and confidence in self.confidence_levels \
                            and qf["q_spread"] >= self.min_q_spread:
                        direction = bot_action
                        trade_id += 1
                        sl = c_close * (1 - sl_pct) if direction == "BUY" else c_close * (1 + sl_pct)
                        tp = c_close * (1 + tp_pct) if direction == "BUY" else c_close * (1 - tp_pct)

                        # RSI / ATR% from window
                        _closes = _w[:, 3]
                        _highs  = _w[:, 1]
                        _lows   = _w[:, 2]
                        _ref    = float(_closes[-1]) or 1.0
                        rsi_entry   = round(float(_rsi(_closes)), 4)
                        atr_pct_entry = round(float(np.clip(_atr(_highs, _lows, _closes) / (_ref + 1e-8), 0, 0.05) / 0.05), 4)

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
                            "q_buy":           qf["q_buy"],
                            "q_sell":          qf["q_sell"],
                            "q_close":         qf["q_close"],
                            "q_spread":        qf["q_spread"],
                            "rsi_entry":       rsi_entry,
                            "atr_pct_entry":   atr_pct_entry,
                            "peak_pnl":        0.0,
                            "peak_timestamp":  current_ts,
                            "_peak_close_pct": 0.0,
                            "steps":           0,
                        }

        # Close any remaining open trades at last price
        for asset, tr in open_trades.items():
            last_px = float(asset_candles[asset][-1, 3])
            buy     = tr["direction"] == "BUY"
            pnl     = (last_px - tr["entry_price"]) if buy else (tr["entry_price"] - last_px)
            risk    = abs(tr["entry_price"] - tr["sl_price"])

            fin = None
            if fin_enabled:
                nights = _holding_nights(tr["entry_timestamp"], timeline[-1])
                fin = calculate_trade_financials(
                    asset=asset,
                    direction=tr["direction"],
                    entry_price=tr["entry_price"],
                    exit_pnl=pnl,
                    sl_price=tr["sl_price"],
                    capital=running_capital,
                    risk_pct=risk_pct,
                    leverage=leverage,
                    holding_nights=nights,
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
                t.get("confidence"),  t.get("close_confidence"),
                t.get("q_buy"),       t.get("q_sell"),       t.get("q_close"),
                t.get("q_spread"),    t.get("rsi_entry"),    t.get("atr_pct_entry"),
                t.get("q_buy_exit"),  t.get("q_sell_exit"),  t.get("q_close_exit"),
                t.get("peak_pnl"),    t.get("peak_timestamp"), t.get("steps"),
            )
            for t in trades
        ]
        conn = sqlite3.connect(self.output_db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executemany(
            """INSERT OR IGNORE INTO sim_trades
               (asset, direction, sl_variant, entry_timestamp, entry_price,
                sl_price, tp_price, exit_timestamp, exit_price, status, pnl, r_multiple,
                confidence, close_confidence,
                q_buy, q_sell, q_close, q_spread, rsi_entry, atr_pct_entry,
                q_buy_exit, q_sell_exit, q_close_exit,
                peak_pnl, peak_timestamp, steps)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                       ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                    "sl_price":    t.get("sl_price", 0),
                    "tp_price":    t.get("tp_price", 0),
                    "pnl":            round(t.get("pnl") or 0, 4),
                    "r_multiple":     round(t.get("r_multiple") or 0, 3),
                    "status":         t.get("status", ""),
                    "confidence":       t.get("confidence", 0),
                    "close_confidence": t.get("close_confidence"),
                    "q_buy":          t.get("q_buy"),
                    "q_sell":         t.get("q_sell"),
                    "q_close":        t.get("q_close"),
                    "q_spread":       round(t["q_spread"], 4) if t.get("q_spread") is not None else None,
                    "rsi_entry":      t.get("rsi_entry"),
                    "atr_pct_entry":  t.get("atr_pct_entry"),
                    "q_buy_exit":     t.get("q_buy_exit"),
                    "q_sell_exit":    t.get("q_sell_exit"),
                    "q_close_exit":   t.get("q_close_exit"),
                    "peak_pnl":       round(t["peak_pnl"], 4) if t.get("peak_pnl") is not None else None,
                    "peak_timestamp": t.get("peak_timestamp"),
                    "steps":          t.get("steps"),
                    "netto_pnl_eur":  round(t["financial"]["netto_pnl_eur"], 2) if t.get("financial") else None,
                    "capital_after":  round(t["capital_after"], 2) if t.get("capital_after") is not None else None,
                    "lot_size":       round(t["financial"]["lot_size"], 4) if t.get("financial") else None,
                    "margin_eur":     round(t["financial"]["margin_eur"], 2) if t.get("financial") else None,
                }
                for t in trades
            ],
        }
        return result
