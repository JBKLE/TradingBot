"""DQN-basierte Marktanalyse – ersetzt die Claude-API komplett.

Lädt ein .pt-Modell aus dem models/-Ordner und führt
Inferenz für alle Watchlist-Assets durch. Das Interface
(AnalysisResult) bleibt identisch zum alten MarketAnalyzer.

Unterstützt mehrere Modellversionen (v1, v2) mit unterschiedlichen
Architekturen. Die Version wird aus dem Dateinamen geparst
(z.B. GOLD_v1_run1.pt) oder kann manuell gesetzt werden.

State-Vektor wird aus der price_history-Tabelle (simulation.db)
gebaut – identisch zum Training in TradeAI.
"""
import glob
import logging
import os
import re
from dataclasses import dataclass, field, replace
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

# ── Assets ────────────────────────────────────────────────────────────────────
ASSETS = ["GOLD", "SILVER", "OIL_CRUDE", "NATURALGAS"]
ASSET_INDEX = {a: i for i, a in enumerate(ASSETS)}


# ── Modellversions-Konfiguration ──────────────────────────────────────────────

@dataclass(frozen=True)
class ModelVersionConfig:
    """Alle architektur-relevanten Parameter einer Modellversion."""
    version: str
    max_window: int
    n_indicators: int
    action_size: int
    actions: dict[int, str]
    state_size: int
    context_size: int          # n_indicators + 4 (asset) + 4 (position)
    dropout: float
    cnn_layers: int            # 3 oder 4
    cnn_pool_size: int         # AdaptiveAvgPool1d output
    sl_pct: float
    tp_pct: float
    # Action-Mapping auf den einheitlichen Bot-Standard (HOLD/BUY/SELL/CLOSE)
    action_map: dict[int, str] = field(default_factory=dict)
    # Position-Encoding: "r_multiple" (v2) oder "pct" (v1)
    position_encoding: str = "r_multiple"


MODEL_VERSIONS: dict[str, ModelVersionConfig] = {
    "v1": ModelVersionConfig(
        version="v1",
        max_window=100,
        n_indicators=4,
        action_size=3,
        actions={0: "BUY", 1: "SELL", 2: "CANCEL"},
        state_size=512,
        context_size=12,       # 4 + 4 + 4
        dropout=0.0,
        cnn_layers=4,
        cnn_pool_size=8,
        sl_pct=0.003,
        tp_pct=0.005,
        action_map={0: "BUY", 1: "SELL", 2: "CLOSE"},
        position_encoding="pct",
    ),
    "v2": ModelVersionConfig(
        version="v2",
        max_window=50,
        n_indicators=6,
        action_size=4,
        actions={0: "HOLD", 1: "BUY", 2: "SELL", 3: "CLOSE"},
        state_size=264,
        context_size=14,       # 6 + 4 + 4
        dropout=0.15,
        cnn_layers=3,
        cnn_pool_size=4,
        sl_pct=0.003,
        tp_pct=0.005,
        action_map={0: "HOLD", 1: "BUY", 2: "SELL", 3: "CLOSE"},
        position_encoding="r_multiple",
    ),
}

# Einheitliche Aktionen für den Bot (unabhängig von Modellversion)
BOT_ACTIONS = {0: "HOLD", 1: "BUY", 2: "SELL", 3: "CLOSE"}


# ── Dateinamen-Parser ─────────────────────────────────────────────────────────

# Patterns: GOLD_v1_run1.pt, GOLD_dqn_v1.pt, GOLD_v2_run3.pt
_FILENAME_PATTERNS = [
    re.compile(r"^(?P<asset>[A-Z_]+)_v(?P<version>\d+)_run\d+\.pt$"),
    re.compile(r"^(?P<asset>[A-Z_]+)_dqn_v\d+\.pt$"),      # v1-Stil aus TradeAI
    re.compile(r"^(?P<asset>[A-Z_]+)_v(?P<version>\d+)\.pt$"),
]


def parse_model_filename(filename: str) -> dict[str, str | None]:
    """Parst Asset und Version aus dem Modell-Dateinamen.

    Returns:
        {"asset": "GOLD"|None, "version": "v1"|None, "parsed": True|False}
    """
    basename = os.path.basename(filename)
    for pat in _FILENAME_PATTERNS:
        m = pat.match(basename)
        if m:
            groups = m.groupdict()
            asset = groups.get("asset")
            ver_num = groups.get("version")
            # v1-Stil: GOLD_dqn_v1.pt → version immer v1
            if ver_num is None and "_dqn_" in basename:
                ver_num = "1"
            version = f"v{ver_num}" if ver_num else None
            return {"asset": asset, "version": version, "parsed": True}
    return {"asset": None, "version": None, "parsed": False}


def get_model_version_config(version: str) -> ModelVersionConfig:
    """Gibt die Konfiguration für eine Modellversion zurück."""
    if version not in MODEL_VERSIONS:
        raise ValueError(
            f"Unbekannte Modellversion '{version}'. "
            f"Verfügbar: {list(MODEL_VERSIONS.keys())}"
        )
    return MODEL_VERSIONS[version]


def list_available_models(models_dir: str | None = None) -> list[dict]:
    """Listet alle .pt-Modelle mit geparsten Infos auf."""
    d = models_dir or config.AI_MODELS_DIR
    candidates = glob.glob(os.path.join(d, "*.pt"))
    result = []
    for path in sorted(candidates, key=os.path.getmtime, reverse=True):
        info = parse_model_filename(path)
        basename = os.path.basename(path)
        info["filename"] = basename
        info["name"] = basename          # Alias fuer statisches Dashboard
        info["path"] = path
        info["size_mb"] = round(os.path.getsize(path) / 1024 / 1024, 1)
        info["size_kb"] = round(os.path.getsize(path) / 1024, 1)
        result.append(info)
    return result


# ── Modell-Architektur (parametrisiert fuer v1/v2) ───────────────────────────

def _build_cnn_v1() -> torch.nn.Sequential:
    """CNN fuer v1: 4 Layer, 100 Candles → 128×8 = 1024."""
    return torch.nn.Sequential(
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


def _build_cnn_v2() -> torch.nn.Sequential:
    """CNN fuer v2: 3 Layer, 50 Candles → 128×4 = 512."""
    return torch.nn.Sequential(
        torch.nn.Conv1d(5, 32, kernel_size=5, stride=2, padding=2),
        torch.nn.ReLU(),
        torch.nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool1d(4),
    )


class DuelingDQN(torch.nn.Module):
    """Dueling DQN – unterstuetzt v1 und v2 Architektur."""

    def __init__(self, vcfg: ModelVersionConfig):
        super().__init__()
        self._max_window = vcfg.max_window
        self._context_size = vcfg.context_size

        # CNN je nach Version
        if vcfg.cnn_layers == 4:
            self.cnn = _build_cnn_v1()
            cnn_out = 128 * vcfg.cnn_pool_size  # 1024
        else:
            self.cnn = _build_cnn_v2()
            cnn_out = 128 * vcfg.cnn_pool_size  # 512

        # Shared FC (mit optionalem Dropout fuer v2)
        shared_in = cnn_out + vcfg.context_size
        layers: list[torch.nn.Module] = [
            torch.nn.Linear(shared_in, 512),
            torch.nn.LayerNorm(512),
            torch.nn.ReLU(),
        ]
        if vcfg.dropout > 0:
            layers.append(torch.nn.Dropout(vcfg.dropout))
        layers += [
            torch.nn.Linear(512, 256),
            torch.nn.LayerNorm(256),
            torch.nn.ReLU(),
        ]
        if vcfg.dropout > 0:
            layers.append(torch.nn.Dropout(vcfg.dropout))
        self.shared = torch.nn.Sequential(*layers)

        # Dueling Streams
        self.value_stream = torch.nn.Sequential(
            torch.nn.Linear(256, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
        )
        self.advantage_stream = torch.nn.Sequential(
            torch.nn.Linear(256, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, vcfg.action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        candles = x[:, : self._max_window * 5].view(-1, self._max_window, 5)
        candles = candles.permute(0, 2, 1)
        context = x[:, self._max_window * 5:]
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


def _macd_histogram(closes: np.ndarray) -> float:
    """MACD-Histogramm (EMA12 - EMA26 - Signal9), normiert durch Preis."""
    if len(closes) < 27:
        return 0.0
    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    macd_line = ema12 - ema26
    # Signal: EMA9 der MACD-Linie (Approximation mit den letzten 35 Werten)
    n = min(len(closes), 35)
    macd_vals = []
    for i in range(n):
        e12 = _ema(closes[:len(closes) - n + i + 1], 12)
        e26 = _ema(closes[:len(closes) - n + i + 1], 26)
        macd_vals.append(e12 - e26)
    signal = macd_vals[0]
    k9 = 2.0 / 10.0
    for v in macd_vals[1:]:
        signal = v * k9 + signal * (1.0 - k9)
    ref = closes[-1] + 1e-8
    return float(np.clip((macd_line - signal) / ref, -0.01, 0.01) / 0.01)


def _bollinger_width(closes: np.ndarray, period: int = 20) -> float:
    """Relative Bollinger-Bandbreite (2*std / SMA), normiert auf [0, 1]."""
    if len(closes) < period:
        return 0.0
    window = closes[-period:]
    sma = float(window.mean())
    std = float(window.std())
    raw = (2.0 * std) / (sma + 1e-8)
    return float(np.clip(raw, 0, 0.1) / 0.1)


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

# Capital.com Overnight-Finanzierungsrate (pro Nacht, auf Positionswert)
# Typisch 0.007%–0.01% je nach Asset (Annualized ~2.5-3.6%)
OVERNIGHT_RATES = {
    "GOLD": 0.00008,
    "SILVER": 0.00008,
    "OIL_CRUDE": 0.0001,
    "NATURALGAS": 0.0001,
}

# Max SL-Distanz fuer Lot-Sizing (% vom Entry-Preis).
# Wenn SL effektiv deaktiviert (>5%), wird auf diesen Wert gekappt.
LOT_SIZE_MAX_SL_PCT = 0.02  # 2%


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
    holding_nights: int = 0,
) -> dict:
    """Berechnet die finanzielle Auswirkung eines Trades (alles in EUR).

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
        holding_nights: Anzahl Uebernacht-Halteperioden

    Returns:
        Dict mit lot_size, position_value_eur, margin_eur,
        brutto_pnl_eur, spread_cost_eur, overnight_cost_eur, netto_pnl_eur
    """
    if spread is None:
        spread = DEFAULT_SPREADS.get(asset, 0.03)

    # SL-Distanz in Preis-Einheiten
    sl_distance = abs(entry_price - sl_price)
    if sl_distance == 0:
        sl_distance = entry_price * 0.003  # fallback 0.3%

    # Cap: wenn SL effektiv deaktiviert (> 5% vom Preis), auf 2% begrenzen
    max_sl = entry_price * LOT_SIZE_MAX_SL_PCT
    if sl_distance > entry_price * 0.05:
        sl_distance = max_sl

    # Risikobetrag in EUR
    risk_amount_eur = capital * risk_pct

    # Lotgroesse: wie viele Einheiten kann ich kaufen, sodass SL = risk_amount
    lot_size = risk_amount_eur / sl_distance

    # Positionswert und Margin in EUR
    position_value_eur = lot_size * entry_price
    margin_eur = position_value_eur / leverage

    # Brutto-P&L in EUR
    brutto_pnl_eur = lot_size * exit_pnl

    # Spread-Kosten (einmal beim Einstieg)
    spread_cost_eur = lot_size * spread

    # Overnight-Finanzierungskosten
    overnight_rate = OVERNIGHT_RATES.get(asset, 0.00008)
    overnight_cost_eur = position_value_eur * overnight_rate * holding_nights

    # Netto-P&L
    netto_pnl_eur = brutto_pnl_eur - spread_cost_eur - overnight_cost_eur

    # Margin-Call Check
    margin_call = margin_eur > capital

    return {
        "lot_size": round(lot_size, 4),
        "position_value_eur": round(position_value_eur, 2),
        "margin_eur": round(margin_eur, 2),
        "brutto_pnl_eur": round(brutto_pnl_eur, 2),
        "spread_cost_eur": round(spread_cost_eur, 2),
        "overnight_cost_eur": round(overnight_cost_eur, 2),
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
    """Ersetzt MarketAnalyzer – nutzt das eigene DQN-Modell statt Claude API.

    Unterstuetzt v1 und v2 Modelle. Die Version wird aus dem Dateinamen
    geparst oder kann manuell gesetzt werden.
    """

    def __init__(self, models_dir: str | None = None) -> None:
        self._models_dir = models_dir or config.AI_MODELS_DIR
        self._device = self._resolve_device()
        self._net: DuelingDQN | None = None
        self._model_path: str | None = None
        self._vcfg: ModelVersionConfig = MODEL_VERSIONS["v1"]  # Default

        # Manuelle Overrides (None = auto-detect aus Dateinamen)
        self._override_version: str | None = None
        self._override_asset: str | None = None
        self._override_model_file: str | None = None

    @staticmethod
    def _resolve_device() -> torch.device:
        if config.DQN_DEVICE == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(config.DQN_DEVICE)

    # ── Modellauswahl ─────────────────────────────────────────────────────

    def select_model(
        self,
        filename: str | None = None,
        version: str | None = None,
        asset: str | None = None,
    ) -> dict:
        """Modell auswaehlen. Gibt geparste Infos zurueck.

        Args:
            filename: Konkreter Dateiname (z.B. "GOLD_v1_run1.pt").
                      Wenn None, wird das neueste .pt verwendet.
            version:  Manuelle Version (z.B. "v1", "v2").
                      Ueberschreibt auto-detect aus Dateinamen.
            asset:    Manuelles Asset (z.B. "GOLD").
                      Ueberschreibt auto-detect aus Dateinamen.

        Returns:
            Dict mit model_file, version, asset, parsed, auto_detected.
        """
        self._override_model_file = filename
        self._override_version = version
        self._override_asset = asset
        # Cache invalidieren → naechster _load_model() laedt neu
        self._net = None
        self._model_path = None

        # Info-Dict aufbauen
        model_path = self._resolve_model_path()
        info = parse_model_filename(model_path)
        effective_version = version or info.get("version") or "v1"
        effective_asset = asset or info.get("asset")

        if effective_version in MODEL_VERSIONS:
            self._vcfg = MODEL_VERSIONS[effective_version]
        else:
            logger.warning("Unbekannte Version '%s', fallback auf v1", effective_version)
            self._vcfg = MODEL_VERSIONS["v1"]

        result = {
            "model_file": os.path.basename(model_path),
            "version": effective_version,
            "asset": effective_asset,
            "parsed": info.get("parsed", False),
            "auto_detected": version is None and asset is None,
            "config": {
                "max_window": self._vcfg.max_window,
                "n_indicators": self._vcfg.n_indicators,
                "action_size": self._vcfg.action_size,
                "actions": self._vcfg.action_map,
                "state_size": self._vcfg.state_size,
                "sl_pct": self._vcfg.sl_pct,
                "tp_pct": self._vcfg.tp_pct,
            },
        }
        logger.info(
            "Modell ausgewaehlt: %s (version=%s, asset=%s)",
            result["model_file"], effective_version, effective_asset,
        )
        return result

    def get_current_model_info(self) -> dict:
        """Gibt Infos zum aktuell geladenen/ausgewaehlten Modell zurueck."""
        path = self._resolve_model_path()
        info = parse_model_filename(path)
        return {
            "model_file": os.path.basename(path),
            "version": self._vcfg.version,
            "filename_version": info.get("version"),
            "asset": self._override_asset or info.get("asset"),
            "config": {
                "max_window": self._vcfg.max_window,
                "n_indicators": self._vcfg.n_indicators,
                "action_size": self._vcfg.action_size,
                "actions": self._vcfg.action_map,
                "state_size": self._vcfg.state_size,
                "sl_pct": self._vcfg.sl_pct,
                "tp_pct": self._vcfg.tp_pct,
            },
        }

    @staticmethod
    def _detect_version_from_checkpoint(sd: dict) -> str | None:
        """Erkennt die Modellversion anhand der Checkpoint-Gewichte.

        v1: 4 CNN-Layer (cnn.6.weight existiert), shared_in=1036
        v2: 3 CNN-Layer (kein cnn.6.weight), shared_in=526
        """
        has_cnn6 = "cnn.6.weight" in sd
        shared_in = sd.get("shared.0.weight")
        if shared_in is None:
            return None
        shared_width = shared_in.shape[1]

        if has_cnn6 and shared_width >= 1024:
            return "v1"
        if not has_cnn6 and shared_width < 1024:
            return "v2"
        # Ambig – anhand CNN-Kernel-Groesse unterscheiden
        cnn0 = sd.get("cnn.0.weight")
        if cnn0 is not None:
            kernel = cnn0.shape[2]
            if kernel == 7:
                return "v1"
            if kernel == 5:
                return "v2"
        return None

    def _resolve_model_path(self) -> str:
        """Bestimmt den Modellpfad (Override oder neueste Datei)."""
        if self._override_model_file:
            path = os.path.join(self._models_dir, self._override_model_file)
            if os.path.isfile(path):
                return path
            # Eventuell absoluter Pfad
            if os.path.isfile(self._override_model_file):
                return self._override_model_file
            logger.warning(
                "Override-Modell '%s' nicht gefunden, fallback auf neuestes",
                self._override_model_file,
            )
        return _get_latest_model_path(self._models_dir)

    def _load_model(self) -> DuelingDQN:
        """Laedt das Modell mit der korrekten Architektur (cached)."""
        latest = self._resolve_model_path()
        if self._net is not None and self._model_path == latest:
            return self._net

        # Version aus Dateinamen bestimmen (falls kein Override)
        if self._override_version:
            version = self._override_version
        else:
            info = parse_model_filename(latest)
            version = info.get("version") or "v1"

        if version in MODEL_VERSIONS:
            self._vcfg = MODEL_VERSIONS[version]
        else:
            logger.warning("Unbekannte Version '%s', fallback auf v1", version)
            self._vcfg = MODEL_VERSIONS["v1"]

        logger.info(
            "Lade DQN-Modell: %s (version=%s, device=%s, state=%d, actions=%d)",
            latest, self._vcfg.version, self._device,
            self._vcfg.state_size, self._vcfg.action_size,
        )
        ckpt = torch.load(latest, map_location=self._device, weights_only=True)
        sd = ckpt["policy_net"]

        # ── Auto-detect Architektur aus Checkpoint ───────────────────────
        detected_version = self._detect_version_from_checkpoint(sd)
        if detected_version and detected_version != self._vcfg.version:
            logger.warning(
                "Checkpoint-Architektur ist '%s', nicht '%s' – korrigiere automatisch",
                detected_version, self._vcfg.version,
            )
            self._vcfg = MODEL_VERSIONS[detected_version]

        # Action-Size Feinabgleich (falls Checkpoint nicht exakt passt)
        adv_key = "advantage_stream.2.weight"
        if adv_key in sd:
            ckpt_actions = sd[adv_key].shape[0]
            if ckpt_actions != self._vcfg.action_size:
                logger.warning(
                    "Checkpoint hat %d Actions, Config erwartet %d – passe an",
                    ckpt_actions, self._vcfg.action_size,
                )
                matching = [v for v in MODEL_VERSIONS.values() if v.action_size == ckpt_actions]
                if matching:
                    matched = matching[0]
                    self._vcfg = replace(
                        self._vcfg,
                        action_size=ckpt_actions,
                        action_map=matched.action_map,
                        actions=matched.actions,
                    )
                else:
                    generic_map = {i: f"ACTION_{i}" for i in range(ckpt_actions)}
                    self._vcfg = replace(
                        self._vcfg,
                        action_size=ckpt_actions,
                        action_map=generic_map,
                        actions=generic_map,
                    )

        net = DuelingDQN(self._vcfg).to(self._device)
        net.load_state_dict(ckpt["policy_net"])
        net.eval()
        self._net = net
        self._model_path = latest
        return net

    # ── State-Vektor aus DB (identisch zu TradeAI/predict.py) ───────────────

    async def _load_candles_from_db(
        self,
        asset: str,
        before_timestamp: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Liest Kerzen aus price_history (simulation.db).

        Laedt genug Kerzen fuer Indikatoren (mind. 50 extra fuer EMA50/MACD).

        Args:
            asset: Asset-Key (z.B. "GOLD")
            before_timestamp: Wenn gesetzt, nur Kerzen VOR diesem Zeitpunkt.
        """
        # Mehr laden als max_window, damit Indikatoren genug History haben
        load_count = self._vcfg.max_window + 50
        async with aiosqlite.connect(config.SIM_DB_PATH) as db:
            if before_timestamp:
                query = (
                    "SELECT open, high, low, close FROM price_history "
                    "WHERE asset = ? AND timestamp <= ? "
                    "ORDER BY timestamp DESC LIMIT ?"
                )
                params = (asset, before_timestamp, load_count)
            else:
                query = (
                    "SELECT open, high, low, close FROM price_history "
                    "WHERE asset = ? ORDER BY timestamp DESC LIMIT ?"
                )
                params = (asset, load_count)
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
        """Baut den State-Vektor aus numpy-Arrays (DB oder PriceBar).

        Versionsbewusst: v1 = 512 dim, v2 = 264 dim.
        """
        vcfg = self._vcfg
        max_window = vcfg.max_window
        avail = min(len(closes), max_window)
        ohlcv = np.zeros((max_window, 5), dtype=np.float32)
        current_price = 0.0

        if avail > 0:
            # Nur die letzten max_window Kerzen verwenden
            o = opens[-avail:]
            h = highs[-avail:]
            l = lows[-avail:]
            c = closes[-avail:]
            volumes = np.zeros(avail, dtype=np.float64)

            current_price = float(c[-1])
            ref = current_price
            v_mean = float(volumes.mean()) + 1e-8

            rows = np.column_stack([
                o / ref - 1,
                h / ref - 1,
                l / ref - 1,
                c / ref - 1,
                volumes / v_mean - 1,
            ]).astype(np.float32)
            ohlcv[max_window - avail:] = np.clip(rows, -5.0, 5.0)

            # Basis-Indikatoren (v1 + v2)
            rsi = (_rsi(closes) - 50.0) / 50.0
            atr_val = _atr(highs, lows, closes)
            atr_pct = float(np.clip(atr_val / (closes[-1] + 1e-8), 0, 0.05) / 0.05)
            ema20_r = float(np.clip(
                _ema(closes, min(20, len(closes))) / (closes[-1] + 1e-8) - 1, -0.1, 0.1,
            ) / 0.1)
            ema50_r = float(np.clip(
                _ema(closes, min(50, len(closes))) / (closes[-1] + 1e-8) - 1, -0.1, 0.1,
            ) / 0.1)

            ind_list = [rsi, atr_pct, ema20_r, ema50_r]

            # v2: zusaetzlich MACD und Bollinger Width
            if vcfg.n_indicators >= 6:
                ind_list.append(_macd_histogram(closes))
                ind_list.append(_bollinger_width(closes))
        else:
            ind_list = [0.0] * vcfg.n_indicators

        indicators = np.array(ind_list, dtype=np.float32)

        # Asset one-hot
        asset_oh = np.zeros(4, dtype=np.float32)
        asset_oh[ASSET_INDEX[asset]] = 1.0

        # Position (Encoding je nach Version)
        if open_position is None:
            pos = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        else:
            sl_pct = vcfg.sl_pct
            if open_position.direction == Direction.BUY:
                direction = 1.0
                if vcfg.position_encoding == "pct":
                    # v1: unrealised_pct = (price-entry)/entry * 100
                    unreal = (current_price - open_position.entry_price) / (open_position.entry_price + 1e-8) * 100.0
                    unreal = float(np.clip(unreal, -5, 5))
                else:
                    # v2: unrealised_r = PnL / SL-Distanz
                    risk = open_position.entry_price * sl_pct
                    unreal = float(np.clip(
                        (current_price - open_position.entry_price) / (risk + 1e-8), -3, 3,
                    ))
            else:
                direction = -1.0
                if vcfg.position_encoding == "pct":
                    unreal = (open_position.entry_price - current_price) / (open_position.entry_price + 1e-8) * 100.0
                    unreal = float(np.clip(unreal, -5, 5))
                else:
                    risk = open_position.entry_price * sl_pct
                    unreal = float(np.clip(
                        (open_position.entry_price - current_price) / (risk + 1e-8), -3, 3,
                    ))
            pos = np.array([1.0, direction, unreal, 0.0], dtype=np.float32)

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
        """Fuehrt DQN-Inferenz auf einem fertigen State-Vektor aus.

        Mappt versionsspezifische Aktionen auf einheitliche Bot-Aktionen.
        """
        vcfg = self._vcfg
        net = self._load_model()
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self._device)
        with torch.no_grad():
            q = net(state_t).squeeze(0).cpu().numpy()

        raw_action = int(q.argmax())
        softmax_conf = float(F.softmax(torch.FloatTensor(q), dim=0).max())
        confidence = _scale_confidence(softmax_conf)

        # Action-Mapping: versionsspezifisch → einheitlich (HOLD/BUY/SELL/CLOSE)
        bot_action = vcfg.action_map[raw_action]

        # SL/TP fuer neuen Eintritt (aus Versions-Config)
        sl_pct = vcfg.sl_pct
        tp_pct = vcfg.tp_pct
        sl = tp = rr = None
        if bot_action == "BUY":
            sl = current_price * (1 - sl_pct)
            tp = current_price * (1 + tp_pct)
            rr = tp_pct / sl_pct
        elif bot_action == "SELL":
            sl = current_price * (1 + sl_pct)
            tp = current_price * (1 - tp_pct)
            rr = tp_pct / sl_pct

        # Q-Werte mit einheitlichen Action-Labels
        q_labels = {}
        for i in range(vcfg.action_size):
            label = vcfg.action_map[i]
            q_labels[label] = round(float(q[i]), 4)

        return {
            "asset": asset,
            "action": bot_action,
            "action_id": raw_action,
            "current_price": current_price,
            "sl": sl,
            "tp": tp,
            "risk_reward_ratio": rr,
            "confidence": confidence,
            "softmax_confidence": softmax_conf,
            "q_values": q_labels,
            "model_version": vcfg.version,
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
        # _load_model() aufrufen um _vcfg sicher zu initialisieren
        self._load_model()
        opens, highs, lows, closes = await self._load_candles_from_db(asset)
        if len(closes) == 0:
            vcfg = self._vcfg
            return {
                "asset": asset, "action": "HOLD", "action_id": 0,
                "current_price": 0.0, "sl": None, "tp": None,
                "risk_reward_ratio": None, "confidence": 1,
                "softmax_confidence": 0.25,
                "q_values": {vcfg.action_map[i]: 0.0 for i in range(vcfg.action_size)},
                "model_version": vcfg.version,
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
