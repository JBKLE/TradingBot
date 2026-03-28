"""Technische Indikatoren – pure Python, keine externen Libraries."""
from __future__ import annotations

import logging
import math
from typing import Optional

from .models import PriceBar

logger = logging.getLogger(__name__)


# ── ATR ────────────────────────────────────────────────────────────────────────

def calculate_atr(bars: list[PriceBar], period: int = 14) -> Optional[float]:
    """
    Average True Range – Maß für Volatilität.

    True Range = max(high - low, |high - prev_close|, |low - prev_close|)
    ATR = Simple Moving Average der True Ranges über `period` Bars.

    Returns None wenn zu wenig Bars vorhanden sind (< period + 1).
    """
    if len(bars) < period + 1:
        logger.debug("calculate_atr: zu wenig Bars (%d), brauche %d", len(bars), period + 1)
        return None

    true_ranges: list[float] = []
    for i in range(1, len(bars)):
        high = bars[i].high
        low = bars[i].low
        prev_close = bars[i - 1].close
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        true_ranges.append(tr)

    # ATR = SMA der letzten `period` True Ranges
    recent_trs = true_ranges[-period:]
    return sum(recent_trs) / len(recent_trs)


def calculate_average_atr(bars: list[PriceBar], period: int = 14, lookback: int = 50) -> Optional[float]:
    """
    Durchschnitts-ATR über einen längeren Zeitraum (für Volatilitäts-Gate).

    Berechnet ATR für jedes Fenster und gibt den Durchschnitt zurück.
    Returns None wenn zu wenig Bars vorhanden sind.
    """
    if len(bars) < period + lookback:
        # Fallback: einfach aktuellen ATR zurückgeben
        return calculate_atr(bars, period)

    atrs: list[float] = []
    for start in range(len(bars) - lookback - period, len(bars) - period):
        window = bars[start:start + period + 1]
        atr = calculate_atr(window, period)
        if atr is not None:
            atrs.append(atr)

    if not atrs:
        return None
    return sum(atrs) / len(atrs)


# ── RSI ────────────────────────────────────────────────────────────────────────

def calculate_rsi(bars: list[PriceBar], period: int = 14) -> Optional[float]:
    """
    Relative Strength Index (0-100).

    Nutzt Wilder's Smoothing (exponentiell gewichteter Durchschnitt).
    Returns None wenn zu wenig Bars vorhanden sind (< period + 1).
    """
    if len(bars) < period + 1:
        logger.debug("calculate_rsi: zu wenig Bars (%d), brauche %d", len(bars), period + 1)
        return None

    closes = [b.close for b in bars]
    gains: list[float] = []
    losses: list[float] = []

    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))

    # Initialer Durchschnitt über erste `period` Werte
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    # Wilder's Smoothing für den Rest
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


# ── EMA ────────────────────────────────────────────────────────────────────────

def calculate_ema(bars: list[PriceBar], period: int) -> list[float]:
    """
    Exponential Moving Average über Close-Preise.

    Returns leere Liste wenn zu wenig Bars vorhanden sind (< period).
    Die Liste hat die gleiche Länge wie `bars` ab dem ersten vollständigen EMA.
    """
    if len(bars) < period:
        logger.debug("calculate_ema: zu wenig Bars (%d), brauche %d", len(bars), period)
        return []

    closes = [b.close for b in bars]
    multiplier = 2.0 / (period + 1)

    # Seed: SMA der ersten `period` Werte
    ema = sum(closes[:period]) / period
    result = [ema]

    for close in closes[period:]:
        ema = (close - ema) * multiplier + ema
        result.append(ema)

    return result


def _ema_of_values(values: list[float], period: int) -> list[float]:
    """EMA auf einer rohen Float-Liste (intern genutzt für MACD)."""
    if len(values) < period:
        return []
    multiplier = 2.0 / (period + 1)
    ema = sum(values[:period]) / period
    result = [ema]
    for v in values[period:]:
        ema = (v - ema) * multiplier + ema
        result.append(ema)
    return result


# ── Bollinger Bands ────────────────────────────────────────────────────────────

def calculate_bollinger_bands(
    bars: list[PriceBar],
    period: int = 20,
    std_dev: float = 2.0,
) -> Optional[dict]:
    """
    Bollinger Bands.

    Returns {"upper": float, "middle": float, "lower": float, "width": float}
    oder None wenn zu wenig Bars vorhanden sind (< period).
    """
    if len(bars) < period:
        logger.debug("calculate_bollinger_bands: zu wenig Bars (%d), brauche %d", len(bars), period)
        return None

    closes = [b.close for b in bars[-period:]]
    middle = sum(closes) / period

    variance = sum((c - middle) ** 2 for c in closes) / period
    std = math.sqrt(variance)

    upper = middle + std_dev * std
    lower = middle - std_dev * std

    return {
        "upper": upper,
        "middle": middle,
        "lower": lower,
        "width": upper - lower,
    }


# ── MACD ───────────────────────────────────────────────────────────────────────

def calculate_macd(
    bars: list[PriceBar],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Optional[dict]:
    """
    MACD (Moving Average Convergence/Divergence).

    Returns {"macd": float, "signal": float, "histogram": float}
    oder None wenn zu wenig Bars vorhanden sind.
    """
    min_bars = slow + signal
    if len(bars) < min_bars:
        logger.debug("calculate_macd: zu wenig Bars (%d), brauche %d", len(bars), min_bars)
        return None

    closes = [b.close for b in bars]

    fast_ema = _ema_of_values(closes, fast)
    slow_ema = _ema_of_values(closes, slow)

    if not fast_ema or not slow_ema:
        return None

    # Aligne: slow_ema ist kürzer → schneide fast_ema an
    # fast_ema beginnt bei Index (fast-1), slow_ema bei Index (slow-1)
    offset = slow - fast  # slow_ema ist `offset` Schritte kürzer
    aligned_fast = fast_ema[offset:]  # gleiche Länge wie slow_ema

    if len(aligned_fast) != len(slow_ema):
        min_len = min(len(aligned_fast), len(slow_ema))
        aligned_fast = aligned_fast[-min_len:]
        slow_ema = slow_ema[-min_len:]

    macd_line = [f - s for f, s in zip(aligned_fast, slow_ema)]

    signal_line = _ema_of_values(macd_line, signal)
    if not signal_line:
        return None

    # Letzter Wert
    macd_val = macd_line[-1]
    signal_val = signal_line[-1]
    histogram = macd_val - signal_val

    return {
        "macd": macd_val,
        "signal": signal_val,
        "histogram": histogram,
    }


# ── Support / Resistance ───────────────────────────────────────────────────────

def find_support_resistance(
    bars: list[PriceBar],
    lookback: int = 20,
) -> dict:
    """
    Findet die 2-3 relevantesten Support- und Resistance-Levels.

    Methode: Swing Highs/Lows im Lookback-Fenster identifizieren,
    dann nahe beieinander liegende Levels clustern (< 0.5% Abstand).

    Returns {"support": list[float], "resistance": list[float]}
    """
    result: dict[str, list[float]] = {"support": [], "resistance": []}

    window = bars[-lookback:] if len(bars) >= lookback else bars
    if len(window) < 3:
        return result

    swing_highs: list[float] = []
    swing_lows: list[float] = []

    for i in range(1, len(window) - 1):
        prev_high = window[i - 1].high
        curr_high = window[i].high
        next_high = window[i + 1].high
        if curr_high >= prev_high and curr_high >= next_high:
            swing_highs.append(curr_high)

        prev_low = window[i - 1].low
        curr_low = window[i].low
        next_low = window[i + 1].low
        if curr_low <= prev_low and curr_low <= next_low:
            swing_lows.append(curr_low)

    def cluster(levels: list[float], tolerance_pct: float = 0.005) -> list[float]:
        if not levels:
            return []
        sorted_levels = sorted(levels)
        clusters: list[list[float]] = []
        current_cluster = [sorted_levels[0]]
        for level in sorted_levels[1:]:
            ref = current_cluster[0]
            if ref > 0 and abs(level - ref) / ref < tolerance_pct:
                current_cluster.append(level)
            else:
                clusters.append(current_cluster)
                current_cluster = [level]
        clusters.append(current_cluster)
        # Cluster-Repräsentant = Durchschnitt
        return [sum(c) / len(c) for c in clusters]

    # Nächste Levels: Support unter aktuellem Kurs, Resistance darüber
    current_close = window[-1].close
    all_supports = cluster(swing_lows)
    all_resistances = cluster(swing_highs)

    # Sortieren: Support = nächste levels unter Kurs (absteigend), max 3
    supports = sorted([l for l in all_supports if l < current_close], reverse=True)[:3]
    resistances = sorted([l for l in all_resistances if l > current_close])[:3]

    result["support"] = supports
    result["resistance"] = resistances
    return result


# ── calculate_all ──────────────────────────────────────────────────────────────

def calculate_all(bars: list[PriceBar]) -> dict:
    """
    Berechnet alle Indikatoren und gibt ein Summary-Dict zurück.
    Dieses Dict wird in der Analyse verwendet.

    Struktur:
    {
        "atr": float | None,
        "avg_atr": float | None,
        "atr_ratio": float | None,   # aktueller ATR / Durchschnitts-ATR
        "rsi": float | None,
        "ema_20": float | None,
        "ema_50": float | None,
        "bollinger": dict | None,
        "macd": dict | None,
        "support_resistance": dict,
    }
    """
    if not bars:
        return {}

    atr = calculate_atr(bars, period=14)
    avg_atr = calculate_average_atr(bars, period=14, lookback=50)
    atr_ratio = (atr / avg_atr) if (atr and avg_atr and avg_atr > 0) else None

    rsi = calculate_rsi(bars, period=14)

    ema_20_list = calculate_ema(bars, period=20)
    ema_50_list = calculate_ema(bars, period=50)
    ema_20 = ema_20_list[-1] if ema_20_list else None
    ema_50 = ema_50_list[-1] if ema_50_list else None

    bollinger = calculate_bollinger_bands(bars, period=20, std_dev=2.0)
    macd = calculate_macd(bars, fast=12, slow=26, signal=9)
    sr = find_support_resistance(bars, lookback=20)

    return {
        "atr": atr,
        "avg_atr": avg_atr,
        "atr_ratio": atr_ratio,
        "rsi": rsi,
        "ema_20": ema_20,
        "ema_50": ema_50,
        "bollinger": bollinger,
        "macd": macd,
        "support_resistance": sr,
    }
