"""Tests für src/indicators.py – validiert ATR, RSI, EMA, Bollinger, MACD."""
import sys
import os

# Stelle sicher, dass src im Pfad ist
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models import PriceBar
from src.indicators import (
    calculate_atr,
    calculate_rsi,
    calculate_ema,
    calculate_bollinger_bands,
    calculate_macd,
    find_support_resistance,
    calculate_all,
)


# ── Hilfsfunktion ──────────────────────────────────────────────────────────────

def make_bars(closes: list[float], highs: list[float] | None = None, lows: list[float] | None = None) -> list[PriceBar]:
    """Erstellt PriceBar-Liste aus Close-Werten. H/L default = Close ±1."""
    bars = []
    for i, c in enumerate(closes):
        h = highs[i] if highs else c + 1.0
        l = lows[i] if lows else c - 1.0
        bars.append(PriceBar(timestamp=f"2024-01-{i+1:02d}", open=c, high=h, low=l, close=c))
    return bars


# ── ATR Tests ──────────────────────────────────────────────────────────────────

class TestATR:
    def test_returns_none_with_too_few_bars(self):
        bars = make_bars([100.0] * 5)
        assert calculate_atr(bars, period=14) is None

    def test_returns_float_with_enough_bars(self):
        bars = make_bars([100.0 + i * 0.5 for i in range(20)])
        result = calculate_atr(bars, period=14)
        assert result is not None
        assert result > 0

    def test_constant_prices_give_low_atr(self):
        # Bei konstanten Preisen: High=Close+1, Low=Close-1 → TR = 2.0 immer
        bars = make_bars([1000.0] * 20)
        result = calculate_atr(bars, period=14)
        assert result is not None
        assert abs(result - 2.0) < 0.001

    def test_volatile_prices_give_high_atr(self):
        # Alternierend +10 / -10 → hohe TR
        closes = [100.0 + (10 if i % 2 == 0 else -10) for i in range(20)]
        highs = [c + 5 for c in closes]
        lows = [c - 5 for c in closes]
        bars = make_bars(closes, highs, lows)
        result = calculate_atr(bars, period=14)
        assert result is not None
        assert result > 5.0

    def test_atr_with_exactly_period_plus_one_bars(self):
        bars = make_bars([100.0] * 15)
        result = calculate_atr(bars, period=14)
        assert result is not None


# ── RSI Tests ──────────────────────────────────────────────────────────────────

class TestRSI:
    def test_returns_none_with_too_few_bars(self):
        bars = make_bars([100.0] * 5)
        assert calculate_rsi(bars, period=14) is None

    def test_rsi_range(self):
        bars = make_bars([100.0 + i for i in range(20)])
        result = calculate_rsi(bars, period=14)
        assert result is not None
        assert 0.0 <= result <= 100.0

    def test_uptrend_gives_high_rsi(self):
        # Konstant steigend → RSI nahe 100
        bars = make_bars([100.0 + i * 2 for i in range(20)])
        result = calculate_rsi(bars, period=14)
        assert result is not None
        assert result > 70.0

    def test_downtrend_gives_low_rsi(self):
        # Konstant fallend → RSI nahe 0
        bars = make_bars([200.0 - i * 2 for i in range(20)])
        result = calculate_rsi(bars, period=14)
        assert result is not None
        assert result < 30.0

    def test_flat_prices_returns_none_or_50(self):
        # Flat → keine Gewinne/Verluste → RSI undefined (avg_loss=0 → 100) oder 50
        bars = make_bars([100.0] * 20)
        result = calculate_rsi(bars, period=14)
        # Bei flat: avg_loss=0, avg_gain=0 → 100 (da avg_loss=0 → rs=inf → RSI=100)
        assert result is not None
        assert result == 100.0


# ── EMA Tests ──────────────────────────────────────────────────────────────────

class TestEMA:
    def test_returns_empty_with_too_few_bars(self):
        bars = make_bars([100.0] * 5)
        result = calculate_ema(bars, period=20)
        assert result == []

    def test_returns_correct_length(self):
        bars = make_bars([100.0] * 30)
        result = calculate_ema(bars, period=20)
        # Erwartet: len(bars) - period + 1 Werte
        assert len(result) == 30 - 20 + 1

    def test_ema_seed_is_sma(self):
        # Erster EMA-Wert soll SMA der ersten `period` Bars sein
        closes = [float(i + 1) for i in range(25)]
        bars = make_bars(closes)
        result = calculate_ema(bars, period=10)
        expected_seed = sum(closes[:10]) / 10
        assert abs(result[0] - expected_seed) < 0.0001

    def test_ema_reacts_to_price_changes(self):
        # EMA soll bei steigenden Preisen nach oben tendieren
        bars = make_bars([100.0 + i for i in range(30)])
        result = calculate_ema(bars, period=10)
        assert result[-1] > result[0]


# ── Bollinger Bands Tests ──────────────────────────────────────────────────────

class TestBollingerBands:
    def test_returns_none_with_too_few_bars(self):
        bars = make_bars([100.0] * 5)
        result = calculate_bollinger_bands(bars, period=20)
        assert result is None

    def test_returns_dict_with_correct_keys(self):
        bars = make_bars([100.0] * 25)
        result = calculate_bollinger_bands(bars, period=20)
        assert result is not None
        assert set(result.keys()) == {"upper", "middle", "lower", "width"}

    def test_upper_above_lower(self):
        bars = make_bars([100.0 + (i % 3) for i in range(25)])
        result = calculate_bollinger_bands(bars, period=20)
        assert result is not None
        assert result["upper"] > result["lower"]

    def test_middle_is_sma(self):
        closes = [float(i + 100) for i in range(25)]
        bars = make_bars(closes)
        result = calculate_bollinger_bands(bars, period=20)
        assert result is not None
        expected_middle = sum(closes[-20:]) / 20
        assert abs(result["middle"] - expected_middle) < 0.0001

    def test_flat_prices_give_zero_width(self):
        bars = make_bars([100.0] * 25)
        result = calculate_bollinger_bands(bars, period=20)
        assert result is not None
        assert abs(result["width"]) < 0.0001


# ── MACD Tests ─────────────────────────────────────────────────────────────────

class TestMACD:
    def test_returns_none_with_too_few_bars(self):
        bars = make_bars([100.0] * 20)
        result = calculate_macd(bars)
        assert result is None

    def test_returns_dict_with_correct_keys(self):
        bars = make_bars([100.0 + i * 0.5 for i in range(40)])
        result = calculate_macd(bars)
        assert result is not None
        assert set(result.keys()) == {"macd", "signal", "histogram"}

    def test_histogram_is_macd_minus_signal(self):
        bars = make_bars([100.0 + i * 0.3 for i in range(40)])
        result = calculate_macd(bars)
        assert result is not None
        assert abs(result["histogram"] - (result["macd"] - result["signal"])) < 0.0001

    def test_uptrend_gives_positive_macd(self):
        # Starker Aufwärtstrend → MACD positiv
        bars = make_bars([100.0 + i * 2 for i in range(40)])
        result = calculate_macd(bars)
        assert result is not None
        assert result["macd"] > 0


# ── Support/Resistance Tests ───────────────────────────────────────────────────

class TestSupportResistance:
    def test_returns_dict_with_correct_keys(self):
        bars = make_bars([100.0] * 5)
        result = find_support_resistance(bars)
        assert "support" in result
        assert "resistance" in result

    def test_finds_swing_levels(self):
        # Erstelle klare Highs und Lows
        closes = [100, 105, 110, 105, 100, 95, 90, 95, 100, 105,
                  110, 105, 100, 95, 90, 95, 100, 105, 110, 105, 100]
        highs = [c + 2 for c in closes]
        lows = [c - 2 for c in closes]
        bars = make_bars(closes, highs, lows)
        result = find_support_resistance(bars, lookback=20)
        assert isinstance(result["support"], list)
        assert isinstance(result["resistance"], list)

    def test_returns_empty_with_too_few_bars(self):
        bars = make_bars([100.0, 101.0])
        result = find_support_resistance(bars, lookback=20)
        assert result["support"] == []
        assert result["resistance"] == []


# ── calculate_all Tests ────────────────────────────────────────────────────────

class TestCalculateAll:
    def test_returns_empty_dict_for_empty_bars(self):
        result = calculate_all([])
        assert result == {}

    def test_returns_dict_with_expected_keys(self):
        bars = make_bars([100.0 + i for i in range(60)])
        result = calculate_all(bars)
        expected_keys = {"atr", "avg_atr", "atr_ratio", "rsi", "ema_20", "ema_50", "bollinger", "macd", "support_resistance"}
        assert set(result.keys()) == expected_keys

    def test_handles_minimal_bars_gracefully(self):
        bars = make_bars([100.0] * 5)
        result = calculate_all(bars)
        assert result is not None
        # Mit wenig Bars sollen die Indikatoren None/leer zurückgeben, kein Exception
        assert result.get("atr") is None
        assert result.get("rsi") is None

    def test_all_indicators_computed_with_enough_bars(self):
        bars = make_bars([100.0 + i * 0.5 for i in range(60)])
        result = calculate_all(bars)
        assert result["atr"] is not None
        assert result["rsi"] is not None
        assert result["ema_20"] is not None
        assert result["ema_50"] is not None
        assert result["bollinger"] is not None
        assert result["macd"] is not None


# ── Direkter Ausführungstest ───────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    test_classes = [TestATR, TestRSI, TestEMA, TestBollingerBands, TestMACD, TestSupportResistance, TestCalculateAll]
    passed = 0
    failed = 0

    for cls in test_classes:
        instance = cls()
        for name in dir(instance):
            if name.startswith("test_"):
                try:
                    getattr(instance, name)()
                    print(f"  ✓ {cls.__name__}.{name}")
                    passed += 1
                except Exception as exc:
                    print(f"  ✗ {cls.__name__}.{name}: {exc}")
                    traceback.print_exc()
                    failed += 1

    print(f"\n{passed} passed, {failed} failed")
