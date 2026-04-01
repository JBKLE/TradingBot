"""DQN-basierte Marktanalyse – ersetzt die Claude-API komplett.

Lädt das neueste .pt-Modell aus dem models/-Ordner und führt
Inferenz für alle Watchlist-Assets durch. Das Interface
(AnalysisResult) bleibt identisch zum alten MarketAnalyzer.

State-Vektor wird aus der price_history-Tabelle (simulation.db)
gebaut – identisch zum Training in TradeAI/predict.py.
"""
import glob
import logging
import os
from datetime import datetime
from typing import Optional

import aiosqlite
import numpy as np
import torch
import torch.nn.functional as F

from . import config
from .models import (
    AnalysisResult,
    AssetOutlook,
    BestOpportunity,
    Direction,
    EscalationResult,
    MarketData,
    PriceBar,
    PositionInfo,
    Recommendation,
    RecheckInfo,
)

logger = logging.getLogger(__name__)

# ── Konstanten (identisch zu TradeAI/rl_agent/environment.py) ─────────────────
MAX_WINDOW = 100
ASSETS = ["GOLD", "SILVER", "OIL_CRUDE", "NATURALGAS"]
ASSET_INDEX = {a: i for i, a in enumerate(ASSETS)}
STATE_SIZE = MAX_WINDOW * 5 + 4 + 4 + 4  # 512
ACTION_SIZE = 4
ACTIONS = {0: "HOLD", 1: "BUY", 2: "SELL", 3: "CLOSE"}


# ── Modell-Architektur (identisch zu TradeAI/rl_agent/model.py) ───────────────

class DuelingDQN(torch.nn.Module):
    CONTEXT_SIZE = 4 + 4 + 4  # indicators + asset-one-hot + position = 12

    def __init__(self, action_size: int = ACTION_SIZE):
        super().__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv1d(5, 32, kernel_size=7, stride=2, padding=3),
            torch.nn.ReLU(),
            torch.nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(8),
        )
        cnn_out = 128 * 8
        shared_in = cnn_out + self.CONTEXT_SIZE
        self.shared = torch.nn.Sequential(
            torch.nn.Linear(shared_in, 512),
            torch.nn.LayerNorm(512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.LayerNorm(256),
            torch.nn.ReLU(),
        )
        self.value_stream = torch.nn.Sequential(
            torch.nn.Linear(256, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
        )
        self.advantage_stream = torch.nn.Sequential(
            torch.nn.Linear(256, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        candles = x[:, : MAX_WINDOW * 5].view(-1, MAX_WINDOW, 5)
        candles = candles.permute(0, 2, 1)
        context = x[:, MAX_WINDOW * 5:]
        cnn_out = self.cnn(candles).flatten(1)
        feat = self.shared(torch.cat([cnn_out, context], dim=1))
        value = self.value_stream(feat)
        adv = self.advantage_stream(feat)
        return value + adv - adv.mean(dim=1, keepdim=True)


# ── Hilfsfunktionen (identisch zu TradeAI/predict.py) ─────────────────────────

def _rsi(closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    d = np.diff(closes[-(period + 1):])
    g = np.where(d > 0, d, 0.0).mean()
    l = np.where(d < 0, -d, 0.0).mean()
    return 100.0 if l == 0 else 100.0 - 100.0 / (1.0 + g / l)


def _atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int = 14) -> float:
    if len(c) < 2:
        return 0.0
    n = min(len(c), period + 1)
    tr = np.maximum(
        h[-n:][1:] - l[-n:][1:],
        np.maximum(np.abs(h[-n:][1:] - c[-n:][:-1]),
                   np.abs(l[-n:][1:] - c[-n:][:-1])),
    )
    return float(tr.mean())


def _ema(closes: np.ndarray, period: int) -> float:
    if len(closes) == 0:
        return 0.0
    k = 2.0 / (period + 1)
    e = closes[0]
    for p in closes[1:]:
        e = p * k + e * (1 - k)
    return float(e)


def _scale_confidence(softmax_conf: float) -> int:
    """Softmax-Konfidenz (0.25–1.0) auf 1–10 Skala mappen."""
    scaled = (softmax_conf - 0.25) / 0.75 * 9.0 + 1.0
    return max(1, min(10, round(scaled)))


# ── Finanzrechnung fuer Backtest ────────────────────────────────────────────

# Typische Capital.com Spreads (in Asset-Preis-Einheiten)
DEFAULT_SPREADS = {
    "GOLD": 0.30,
    "SILVER": 0.020,
    "OIL_CRUDE": 0.030,
    "NATURALGAS": 0.005,
}


def calculate_trade_financials(
    asset: str,
    direction: str,
    entry_price: float,
    exit_pnl: float,
    sl_price: float,
    capital: float,
    risk_pct: float,
    leverage: int,
    spread: float | None = None,
    eur_usd: float = 1.08,
) -> dict:
    """Berechnet die finanzielle Auswirkung eines Trades.

    Args:
        asset: Asset-Key (GOLD, SILVER, ...)
        direction: BUY oder SELL
        entry_price: Einstiegspreis
        exit_pnl: P&L in Preis-Einheiten (positiv = Gewinn)
        sl_price: Stop-Loss-Preis
        capital: Aktuelles Kapital in EUR
        risk_pct: Risiko pro Trade (z.B. 0.01 = 1%)
        leverage: Hebel (z.B. 20)
        spread: Spread in Preis-Einheiten (None = Default)
        eur_usd: EUR/USD Kurs fuer Umrechnung

    Returns:
        Dict mit lot_size, position_value_eur, margin_eur,
        brutto_pnl_eur, spread_cost_eur, netto_pnl_eur
    """
    if spread is None:
        spread = DEFAULT_SPREADS.get(asset, 0.03)

    # SL-Distanz in Preis-Einheiten
    sl_distance = abs(entry_price - sl_price)
    if sl_distance == 0:
        sl_distance = entry_price * 0.003  # fallback 0.3%

    # Risikobetrag in EUR
    risk_amount_eur = capital * risk_pct

    # Lotgroesse: wie viele Einheiten kann ich kaufen, sodass SL = risk_amount
    # P&L in USD = lot_size * price_change_usd
    # risk_amount_eur * eur_usd = lot_size * sl_distance
    lot_size = (risk_amount_eur * eur_usd) / sl_distance

    # Positionswert in USD und EUR
    position_value_usd = lot_size * entry_price
    position_value_eur = position_value_usd / eur_usd

    # Margin (gebundenes Kapital)
    margin_eur = position_value_eur / leverage

    # Brutto-P&L: lot_size * exit_pnl (in USD), dann in EUR
    brutto_pnl_usd = lot_size * exit_pnl
    brutto_pnl_eur = brutto_pnl_usd / eur_usd

    # Spread-Kosten (einmal beim Einstieg)
    spread_cost_usd = lot_size * spread
    spread_cost_eur = spread_cost_usd / eur_usd

    # Netto-P&L
    netto_pnl_eur = brutto_pnl_eur - spread_cost_eur

    # Margin-Call Check
    margin_call = margin_eur > capital

    return {
        "lot_size": round(lot_size, 4),
        "position_value_eur": round(position_value_eur, 2),
        "margin_eur": round(margin_eur, 2),
        "brutto_pnl_eur": round(brutto_pnl_eur, 2),
        "spread_cost_eur": round(spread_cost_eur, 2),
        "netto_pnl_eur": round(netto_pnl_eur, 2),
        "margin_call": margin_call,
    }


def _get_latest_model_path(models_dir: str) -> str:
    """Liefert den Pfad zur neuesten .pt-Datei im models/-Ordner."""
    candidates = glob.glob(os.path.join(models_dir, "*.pt"))
    if not candidates:
        raise FileNotFoundError(f"Kein .pt-Modell in {models_dir} gefunden.")
    return max(candidates, key=os.path.getmtime)


# ── DQN Analyzer ──────────────────────────────────────────────────────────────

class DQNAnalyzer:
    """Ersetzt MarketAnalyzer – nutzt das eigene DQN-Modell statt Claude API."""

    def __init__(self, models_dir: str | None = None) -> None:
        self._models_dir = models_dir or config.AI_MODELS_DIR
        self._device = self._resolve_device()
        self._net: DuelingDQN | None = None
        self._model_path: str | None = None

    @staticmethod
    def _resolve_device() -> torch.device:
        if config.DQN_DEVICE == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(config.DQN_DEVICE)

    def _load_model(self) -> DuelingDQN:
        """Lädt das neueste Modell (cached, reload bei neuerem .pt)."""
        latest = _get_latest_model_path(self._models_dir)
        if self._net is not None and self._model_path == latest:
            return self._net

        logger.info("Lade DQN-Modell: %s (device=%s)", latest, self._device)
        net = DuelingDQN(ACTION_SIZE).to(self._device)
        ckpt = torch.load(latest, map_location=self._device, weights_only=True)
        net.load_state_dict(ckpt["policy_net"])
        net.eval()
        self._net = net
        self._model_path = latest
        return net

    # ── State-Vektor aus DB (identisch zu TradeAI/predict.py) ───────────────

    @staticmethod
    async def _load_candles_from_db(
        asset: str,
        before_timestamp: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Liest die letzten 100 Kerzen aus price_history (simulation.db).

        Args:
            asset: Asset-Key (z.B. "GOLD")
            before_timestamp: Wenn gesetzt, nur Kerzen VOR diesem Zeitpunkt (ISO-Format).
                              Fuer Backtest: Kerzen vor dem Trade-Entry laden.
        """
        async with aiosqlite.connect(config.SIM_DB_PATH) as db:
            if before_timestamp:
                query = (
                    "SELECT open, high, low, close FROM price_history "
                    "WHERE asset = ? AND timestamp <= ? "
                    "ORDER BY timestamp DESC LIMIT ?"
                )
                params = (asset, before_timestamp, MAX_WINDOW)
            else:
                query = (
                    "SELECT open, high, low, close FROM price_history "
                    "WHERE asset = ? ORDER BY timestamp DESC LIMIT ?"
                )
                params = (asset, MAX_WINDOW)
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()

        if not rows:
            return (
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
            )

        # Rows kommen DESC – umdrehen zu aelteste zuerst
        rows = rows[::-1]
        opens = np.array([r[0] for r in rows], dtype=np.float64)
        highs = np.array([r[1] for r in rows], dtype=np.float64)
        lows = np.array([r[2] for r in rows], dtype=np.float64)
        closes = np.array([r[3] for r in rows], dtype=np.float64)
        return opens, highs, lows, closes

    def _build_state_from_arrays(
        self,
        asset: str,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        open_position: PositionInfo | None = None,
    ) -> tuple[np.ndarray, float]:
        """Baut den State-Vektor aus numpy-Arrays (DB oder PriceBar)."""
        avail = len(closes)
        ohlcv = np.zeros((MAX_WINDOW, 5), dtype=np.float32)
        current_price = 0.0

        if avail > 0:
            # Volume ist in der DB immer 0 (Capital.com MINUTE hat kein Volume)
            volumes = np.zeros(avail, dtype=np.float64)

            current_price = float(closes[-1])
            ref = current_price
            v_mean = float(volumes.mean()) + 1e-8

            rows = np.column_stack([
                opens / ref - 1,
                highs / ref - 1,
                lows / ref - 1,
                closes / ref - 1,
                volumes / v_mean - 1,
            ]).astype(np.float32)
            ohlcv[MAX_WINDOW - avail:] = np.clip(rows, -5.0, 5.0)

            # Indikatoren
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

        # Asset one-hot
        asset_oh = np.zeros(4, dtype=np.float32)
        asset_oh[ASSET_INDEX[asset]] = 1.0

        # Position
        if open_position is None:
            pos = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        else:
            risk = open_position.entry_price * config.DQN_SL_PCT
            if open_position.direction == Direction.BUY:
                unreal_r = (current_price - open_position.entry_price) / risk
                direction = 1.0
            else:
                unreal_r = (open_position.entry_price - current_price) / risk
                direction = -1.0
            pos = np.array(
                [1.0, direction, float(np.clip(unreal_r, -3, 3)), 0.0],
                dtype=np.float32,
            )

        state = np.concatenate([ohlcv.flatten(), indicators, asset_oh, pos])
        return state, current_price

    def _build_state(
        self,
        asset: str,
        price_bars: list[PriceBar],
        open_position: PositionInfo | None = None,
    ) -> tuple[np.ndarray, float]:
        """Baut den State-Vektor aus PriceBar-Objekten (Fallback fuer API-Endpunkte)."""
        avail = min(len(price_bars), MAX_WINDOW)
        bars = price_bars[-avail:]
        opens = np.array([b.open for b in bars], dtype=np.float64)
        highs = np.array([b.high for b in bars], dtype=np.float64)
        lows = np.array([b.low for b in bars], dtype=np.float64)
        closes = np.array([b.close for b in bars], dtype=np.float64)
        return self._build_state_from_arrays(asset, opens, highs, lows, closes, open_position)

    # ── Inferenz ──────────────────────────────────────────────────────────────

    def _infer(self, state: np.ndarray, current_price: float, asset: str) -> dict:
        """Fuehrt DQN-Inferenz auf einem fertigen State-Vektor aus."""
        net = self._load_model()
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self._device)
        with torch.no_grad():
            q = net(state_t).squeeze(0).cpu().numpy()

        action = int(q.argmax())
        softmax_conf = float(F.softmax(torch.FloatTensor(q), dim=0).max())
        confidence = _scale_confidence(softmax_conf)

        # SL/TP fuer neuen Eintritt
        sl = tp = rr = None
        if action == 1:  # BUY
            sl = current_price * (1 - config.DQN_SL_PCT)
            tp = current_price * (1 + config.DQN_TP_PCT)
            rr = config.DQN_TP_PCT / config.DQN_SL_PCT
        elif action == 2:  # SELL
            sl = current_price * (1 + config.DQN_SL_PCT)
            tp = current_price * (1 - config.DQN_TP_PCT)
            rr = config.DQN_TP_PCT / config.DQN_SL_PCT

        return {
            "asset": asset,
            "action": ACTIONS[action],
            "action_id": action,
            "current_price": current_price,
            "sl": sl,
            "tp": tp,
            "risk_reward_ratio": rr,
            "confidence": confidence,
            "softmax_confidence": softmax_conf,
            "q_values": {ACTIONS[i]: round(float(q[i]), 4) for i in range(ACTION_SIZE)},
        }

    def _get_signal(
        self,
        asset: str,
        price_bars: list[PriceBar],
        open_position: PositionInfo | None = None,
    ) -> dict:
        """DQN-Inferenz fuer ein Asset (aus PriceBar-Objekten)."""
        state, current_price = self._build_state(asset, price_bars, open_position)
        return self._infer(state, current_price, asset)

    async def get_signal_from_db(
        self,
        asset: str,
        open_position: PositionInfo | None = None,
    ) -> dict:
        """DQN-Inferenz fuer ein Asset (State aus price_history DB)."""
        opens, highs, lows, closes = await self._load_candles_from_db(asset)
        if len(closes) == 0:
            return {
                "asset": asset, "action": "HOLD", "action_id": 0,
                "current_price": 0.0, "sl": None, "tp": None,
                "risk_reward_ratio": None, "confidence": 1,
                "softmax_confidence": 0.25,
                "q_values": {ACTIONS[i]: 0.0 for i in range(ACTION_SIZE)},
            }
        state, current_price = self._build_state_from_arrays(
            asset, opens, highs, lows, closes, open_position,
        )
        return self._infer(state, current_price, asset)

    async def get_all_signals(
        self,
        open_positions: list[PositionInfo],
    ) -> list[dict]:
        """DQN-Inferenz fuer alle 4 Assets (State aus DB). Fuer unified_tick."""
        signals = []
        for asset in ASSETS:
            epic = config.WATCHLIST.get(asset, {}).get("epic", asset)
            open_pos = next((p for p in open_positions if p.epic == epic), None)
            sig = await self.get_signal_from_db(asset, open_pos)
            signals.append(sig)
            logger.debug(
                "DQN %s: %s (Conf: %d/10, Q=%s)",
                asset, sig["action"], sig["confidence"], sig["q_values"],
            )
        return signals

    async def backtest_trade(
        self,
        asset: str,
        entry_timestamp: str,
        trade_direction: str,
        trade_entry_price: float,
        trade_result_pnl: float,
        with_position: bool = False,
        position_direction: str | None = None,
        position_entry_price: float | None = None,
        # Finanzparameter (optional)
        capital: float | None = None,
        risk_pct: float | None = None,
        leverage: int | None = None,
        eur_usd: float = 1.08,
    ) -> dict:
        """Backtest: DQN-Inferenz zum Zeitpunkt eines historischen Trades.

        Laedt 500 Kerzen VOR entry_timestamp aus der DB und laesst die KI
        blind entscheiden. Vergleicht dann mit dem echten Trade-Ergebnis.

        Args:
            asset: Asset-Key (z.B. "GOLD")
            entry_timestamp: ISO-Zeitpunkt des Trade-Entries
            trade_direction: "BUY" oder "SELL" (echter Trade)
            trade_entry_price: Entry-Preis des echten Trades
            trade_result_pnl: P/L des echten Trades
            with_position: Wenn True, wird Position-Info in den State eingebaut
            position_direction: Richtung der offenen Position (fuer with_position)
            position_entry_price: Entry-Preis der offenen Position
            capital: Kapital in EUR (None = keine Finanzrechnung)
            risk_pct: Risiko pro Trade (z.B. 0.01 = 1%)
            leverage: Hebel (z.B. 20)
            eur_usd: EUR/USD Kurs
        """
        opens, highs, lows, closes = await self._load_candles_from_db(
            asset, before_timestamp=entry_timestamp,
        )

        candle_count = len(closes)
        if candle_count == 0:
            return {
                "error": f"Keine Kerzen vor {entry_timestamp} fuer {asset} in der DB",
                "candle_count": 0,
            }

        # Position-Info optional einbauen
        open_pos = None
        if with_position and position_direction and position_entry_price:
            epic = config.WATCHLIST.get(asset, {}).get("epic", asset)
            open_pos = PositionInfo(
                deal_id="backtest",
                epic=epic,
                direction=Direction(position_direction),
                size=1.0,
                entry_price=position_entry_price,
                current_price=float(closes[-1]),
                stop_loss=0.0,
                take_profit=0.0,
                profit_loss=0.0,
                profit_loss_pct=0.0,
            )

        state, current_price = self._build_state_from_arrays(
            asset, opens, highs, lows, closes, open_pos,
        )
        signal = self._infer(state, current_price, asset)

        # Bewertung: DQN vs. echter Trade
        dqn_action = signal["action"]
        trade_won = trade_result_pnl > 0

        if dqn_action == trade_direction:
            verdict = "MATCH"
        elif dqn_action == "HOLD":
            verdict = "BESSER" if not trade_won else "MISS"
        elif dqn_action in ("BUY", "SELL") and dqn_action != trade_direction:
            verdict = "BESSER" if not trade_won else "MISS"
        elif dqn_action == "CLOSE":
            verdict = "BESSER" if not trade_won else "MISS"
        else:
            verdict = "UNKLAR"

        result = {
            "asset": asset,
            "entry_timestamp": entry_timestamp,
            "candle_count": candle_count,
            "current_price_at_entry": current_price,
            # Echter Trade
            "trade_direction": trade_direction,
            "trade_entry_price": trade_entry_price,
            "trade_pnl": trade_result_pnl,
            "trade_won": trade_won,
            # DQN-Entscheidung
            "dqn_action": dqn_action,
            "dqn_confidence": signal["confidence"],
            "dqn_softmax": signal["softmax_confidence"],
            "dqn_q_values": signal["q_values"],
            "dqn_sl": signal["sl"],
            "dqn_tp": signal["tp"],
            # Bewertung
            "verdict": verdict,
            "with_position": with_position,
        }

        # Finanzrechnung (optional)
        if capital is not None and risk_pct is not None and leverage is not None:
            sl_price = config.DQN_SL_PCT * trade_entry_price
            if trade_direction == "BUY":
                sl_abs = trade_entry_price - sl_price
            else:
                sl_abs = trade_entry_price + sl_price

            result["financial"] = calculate_trade_financials(
                asset=asset,
                direction=trade_direction,
                entry_price=trade_entry_price,
                exit_pnl=trade_result_pnl,
                sl_price=sl_abs,
                capital=capital,
                risk_pct=risk_pct,
                leverage=leverage,
                eur_usd=eur_usd,
            )

        return result

    async def analyze_market(
        self,
        market_data: dict[str, MarketData],
        account_balance: float,
        open_positions: list[PositionInfo],
        market_context: object = None,
        indicators: dict[str, dict] | None = None,
        performance_stats: dict | None = None,
        recent_lessons: list[dict] | None = None,
    ) -> AnalysisResult:
        """Führt DQN-Analyse für alle Assets aus – gleiche Signatur wie MarketAnalyzer."""
        model_path = _get_latest_model_path(self._models_dir)
        model_name = os.path.basename(model_path)
        logger.info(
            "DQN-Analyse gestartet (%d Assets, model=%s, device=%s)",
            len(market_data), model_name, self._device,
        )

        signals: list[dict] = []
        for asset_key, data in market_data.items():
            if asset_key not in ASSET_INDEX:
                logger.warning("Asset %s nicht im DQN-Modell – übersprungen", asset_key)
                continue

            bars = data.price_history
            if not bars:
                logger.warning("Keine Preisdaten für %s", asset_key)
                continue

            # Offene Position für dieses Asset finden
            epic = config.WATCHLIST.get(asset_key, {}).get("epic", asset_key)
            open_pos = next(
                (p for p in open_positions if p.epic == epic),
                None,
            )

            sig = self._get_signal(asset_key, bars, open_pos)
            signals.append(sig)
            logger.info(
                "DQN-Signal %s: %s (Confidence: %d/10, softmax=%.3f, Q=%s)",
                asset_key, sig["action"], sig["confidence"],
                sig["softmax_confidence"],
                sig["q_values"],
            )

        if not signals:
            return self._fallback_wait("Keine Signale – keine Preisdaten verfügbar")

        # Bestes BUY/SELL-Signal auswählen
        trade_signals = [s for s in signals if s["action"] in ("BUY", "SELL")]

        if trade_signals:
            best = max(trade_signals, key=lambda s: s["confidence"])
        else:
            # Kein BUY/SELL – WAIT
            best_hold = max(signals, key=lambda s: s["confidence"])
            return self._build_wait_result(signals, best_hold, model_name)

        # Recommendation bestimmen
        if best["confidence"] >= config.MIN_CONFIDENCE_SCORE:
            recommendation = Recommendation.TRADE
            wait_reason = None
        else:
            recommendation = Recommendation.WAIT
            wait_reason = (
                f"Bestes Signal {best['asset']} {best['action']} hat Confidence "
                f"{best['confidence']}/10 (Minimum: {config.MIN_CONFIDENCE_SCORE})"
            )

        # Other assets
        other_assets = []
        for sig in signals:
            if sig["asset"] == best["asset"]:
                continue
            if sig["action"] == "BUY":
                outlook = "bullish"
            elif sig["action"] == "SELL":
                outlook = "bearish"
            else:
                outlook = "neutral"
            other_assets.append(AssetOutlook(
                asset=sig["asset"],
                outlook=outlook,
                confidence=sig["confidence"],
                note=f"Q-Werte: {sig['q_values']}",
            ))

        # Recheck-Info wenn WAIT aber vielversprechend
        recheck = None
        if recommendation == Recommendation.WAIT and 5 <= best["confidence"] <= 7:
            recheck = RecheckInfo(
                worthy=True,
                asset=best["asset"],
                direction=Direction(best["action"]) if best["action"] in ("BUY", "SELL") else Direction.NONE,
                trigger_condition=f"DQN-Confidence steigt über {config.MIN_CONFIDENCE_SCORE}",
                recheck_in_minutes=config.RECHECK_DEFAULT_MINUTES,
                current_confidence=best["confidence"],
                expected_confidence_if_trigger=config.MIN_CONFIDENCE_SCORE,
            )

        direction = Direction.BUY if best["action"] == "BUY" else Direction.SELL
        opportunity = BestOpportunity(
            asset=best["asset"],
            direction=direction,
            confidence=best["confidence"],
            reasoning=(
                f"DQN-Signal: {best['action']} | Q-Werte: {best['q_values']} | "
                f"Softmax-Confidence: {best['softmax_confidence']:.3f}"
            ),
            entry_price=best["current_price"],
            stop_loss=best["sl"],
            take_profit=best["tp"],
            risk_reward_ratio=best["risk_reward_ratio"] or 0.0,
        )

        # Market summary
        summary_parts = [f"DQN-Analyse ({model_name}):"]
        for sig in signals:
            summary_parts.append(f"{sig['asset']}={sig['action']}({sig['confidence']}/10)")
        market_summary = " | ".join(summary_parts)

        analysis = AnalysisResult(
            date=datetime.now().strftime("%Y-%m-%d"),
            market_summary=market_summary,
            best_opportunity=opportunity,
            other_assets=other_assets,
            recommendation=recommendation,
            wait_reason=wait_reason,
            recheck=recheck,
            tokens_used=0,
            cost_usd=0.0,
        )
        logger.info(
            "DQN-Analyse: %s | %s %s (Confidence: %d, RR: %.2f)",
            recommendation.value,
            opportunity.asset,
            opportunity.direction.value,
            opportunity.confidence,
            opportunity.risk_reward_ratio,
        )
        return analysis

    async def escalate_position(
        self,
        trade,
        escalation_reason: str,
        current_price: float,
        profit_loss: float,
        profit_loss_pct: float,
    ) -> EscalationResult:
        """DQN-basierte Positionsbewertung – ersetzt Claude-Eskalation."""
        epic = trade.epic
        asset = trade.asset
        if asset not in ASSET_INDEX:
            return EscalationResult(
                action="HOLD",
                reasoning=f"Asset {asset} nicht im DQN-Modell",
                urgency="low",
            )

        # Minimale PriceBar-Liste aus Trade-Daten bauen (nur aktueller Preis)
        bar = PriceBar(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
            open=current_price,
            high=current_price,
            low=current_price,
            close=current_price,
        )

        open_pos = PositionInfo(
            deal_id=trade.deal_id or "",
            epic=epic,
            direction=trade.direction,
            size=trade.position_size,
            entry_price=trade.entry_price,
            current_price=current_price,
            stop_loss=trade.stop_loss,
            take_profit=trade.take_profit,
            profit_loss=profit_loss,
            profit_loss_pct=profit_loss_pct,
        )

        sig = self._get_signal(asset, [bar], open_pos)

        if sig["action"] == "CLOSE":
            action = "CLOSE"
            urgency = "high"
        elif sig["action"] in ("BUY", "SELL") and sig["action"] != trade.direction.value:
            action = "CLOSE"
            urgency = "high"
        else:
            action = "HOLD"
            urgency = "low"

        return EscalationResult(
            action=action,
            reasoning=(
                f"DQN-Signal: {sig['action']} (Confidence: {sig['confidence']}/10) | "
                f"Q-Werte: {sig['q_values']} | Eskalationsgrund: {escalation_reason}"
            ),
            urgency=urgency,
        )

    async def recheck_opportunities(
        self,
        rechecks: list,
        market_data: dict,
        indicators: dict,
    ) -> list[dict]:
        """DQN-basierter Recheck – prüft ob Setups jetzt reif sind."""
        results = []
        for rc in rechecks:
            if rc.asset not in market_data:
                results.append({
                    "is_ready": False,
                    "retry_worthy": False,
                    "reasoning": f"Keine Marktdaten für {rc.asset}",
                })
                continue

            data = market_data[rc.asset]
            bars = data.price_history
            if not bars:
                results.append({
                    "is_ready": False,
                    "retry_worthy": False,
                    "reasoning": "Keine Preisdaten",
                })
                continue

            sig = self._get_signal(rc.asset, bars)

            is_trade = sig["action"] in ("BUY", "SELL")
            direction_match = (
                is_trade and Direction(sig["action"]) == rc.direction
            )
            confident = sig["confidence"] >= config.MIN_CONFIDENCE_SCORE

            if direction_match and confident:
                results.append({
                    "asset": rc.asset,
                    "is_ready": True,
                    "confidence": sig["confidence"],
                    "entry_price": sig["current_price"],
                    "stop_loss": sig["sl"],
                    "take_profit": sig["tp"],
                    "risk_reward_ratio": sig["risk_reward_ratio"] or 0.0,
                    "reasoning": f"DQN bestätigt: {sig['action']} mit Confidence {sig['confidence']}",
                    "retry_worthy": False,
                    "retry_in_minutes": 0,
                })
            else:
                still_worthy = is_trade and sig["confidence"] >= 4
                results.append({
                    "asset": rc.asset,
                    "is_ready": False,
                    "confidence": sig["confidence"],
                    "reasoning": (
                        f"DQN: {sig['action']} (Confidence: {sig['confidence']}/10) "
                        f"– {'Richtung passt nicht' if is_trade and not direction_match else 'Noch nicht confident genug'}"
                    ),
                    "retry_worthy": still_worthy and rc.recheck_count + 1 < rc.max_rechecks,
                    "retry_in_minutes": config.RECHECK_DEFAULT_MINUTES,
                })

        return results

    async def generate_summary(
        self,
        trades: list,
        balance: float,
        performance_stats: dict,
        period: str = "Tages",
    ) -> dict:
        """Generiert eine regelbasierte Zusammenfassung (ohne LLM)."""
        total_pl = sum(t.profit_loss or 0 for t in trades)
        wins = sum(1 for t in trades if (t.profit_loss or 0) > 0)
        losses = sum(1 for t in trades if (t.profit_loss or 0) < 0)
        win_rate = (wins / len(trades) * 100) if trades else 0

        if total_pl > 0:
            rating = "good"
        elif total_pl < 0:
            rating = "poor"
        else:
            rating = "neutral"

        highlights = []
        issues = []
        if wins > 0:
            highlights.append(f"{wins} profitable Trades")
        if win_rate >= 60:
            highlights.append(f"Win-Rate {win_rate:.0f}%")
        if losses > 0:
            issues.append(f"{losses} Verlust-Trades")
        streak = performance_stats.get("current_loss_streak", 0)
        if streak >= 2:
            issues.append(f"Verlustserie: {streak} in Folge")

        return {
            "summary": (
                f"{period}-Bilanz: {len(trades)} Trades, P/L: {total_pl:+.2f} EUR, "
                f"Win-Rate: {win_rate:.0f}%, Balance: {balance:.2f} EUR"
            ),
            "highlights": highlights,
            "issues": issues,
            "recommendations": [],
            "overall_rating": rating,
        }

    async def review_trade(self, trade, price_bars_after: list | None = None) -> dict:
        """Regelbasiertes Trade-Review (ohne LLM)."""
        pl = trade.profit_loss or 0
        entry_quality = "good" if pl > 0 else ("bad" if pl < 0 else "neutral")

        return {
            "entry_quality": entry_quality,
            "entry_quality_explanation": f"P/L: {pl:+.2f} EUR",
            "sl_quality": "good",
            "sl_quality_explanation": "Regelbasierte Bewertung",
            "market_condition": "unknown",
            "what_happened_after": "Regelbasierte Analyse – kein LLM verfügbar",
            "lesson_learned": (
                f"Trade {trade.asset} {trade.direction.value}: "
                f"{'Gewinn' if pl > 0 else 'Verlust'} von {pl:+.2f} EUR"
            ),
            "would_trade_again": pl > 0,
            "improvement_suggestions": [],
        }

    def _build_wait_result(
        self, signals: list[dict], best: dict, model_name: str,
    ) -> AnalysisResult:
        """Baut ein WAIT-Ergebnis wenn kein BUY/SELL-Signal vorliegt."""
        other_assets = []
        for sig in signals:
            if sig["action"] == "BUY":
                outlook = "bullish"
            elif sig["action"] == "SELL":
                outlook = "bearish"
            else:
                outlook = "neutral"
            other_assets.append(AssetOutlook(
                asset=sig["asset"],
                outlook=outlook,
                confidence=sig["confidence"],
                note=f"Q-Werte: {sig['q_values']}",
            ))

        summary_parts = [f"DQN-Analyse ({model_name}):"]
        for sig in signals:
            summary_parts.append(f"{sig['asset']}={sig['action']}({sig['confidence']}/10)")

        return AnalysisResult(
            date=datetime.now().strftime("%Y-%m-%d"),
            market_summary=" | ".join(summary_parts),
            best_opportunity=BestOpportunity(
                asset=best["asset"],
                direction=Direction.NONE,
                confidence=best["confidence"],
                reasoning=f"DQN empfiehlt {best['action']} – kein Trade-Signal | Q: {best['q_values']}",
                entry_price=best["current_price"],
                stop_loss=0.0,
                take_profit=0.0,
                risk_reward_ratio=0.0,
            ),
            other_assets=other_assets,
            recommendation=Recommendation.WAIT,
            wait_reason=f"Kein BUY/SELL-Signal – alle Assets auf {best['action']}",
            tokens_used=0,
            cost_usd=0.0,
        )

    @staticmethod
    def _fallback_wait(reason: str) -> AnalysisResult:
        return AnalysisResult(
            date=datetime.now().strftime("%Y-%m-%d"),
            market_summary=f"DQN-Analyse fehlgeschlagen: {reason}",
            best_opportunity=BestOpportunity(
                asset="GOLD",
                direction=Direction.NONE,
                confidence=1,
                reasoning=reason,
                entry_price=0.0,
                stop_loss=0.0,
                take_profit=0.0,
                risk_reward_ratio=0.0,
            ),
            recommendation=Recommendation.WAIT,
            wait_reason=reason,
            tokens_used=0,
            cost_usd=0.0,
        )
