"""Configuration and environment variables for the trading bot."""
import os
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()

# ── Capital.com API ────────────────────────────────────────────────────────────
CAPITAL_API_KEY: str = os.getenv("CAPITAL_API_KEY", "")
CAPITAL_EMAIL: str = os.getenv("CAPITAL_EMAIL", "")
CAPITAL_PASSWORD: str = os.getenv("CAPITAL_PASSWORD", "")
CAPITAL_DEMO: bool = os.getenv("CAPITAL_DEMO", "true").lower() == "true"

CAPITAL_BASE_URL: str = (
    "https://demo-api-capital.backend-capital.com"
    if CAPITAL_DEMO
    else "https://api-capital.backend-capital.com"
)

# ── Anthropic Claude API ───────────────────────────────────────────────────────
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")

# ── Trading parameters ─────────────────────────────────────────────────────────
MAX_OPEN_POSITIONS: int = int(os.getenv("MAX_OPEN_POSITIONS", "1"))
MAX_RISK_PER_TRADE_PCT: float = float(os.getenv("MAX_RISK_PER_TRADE_PCT", "2.0"))   # War: 5.0 – 5% bei 900 EUR war zu aggressiv
MIN_CONFIDENCE_SCORE: int = int(os.getenv("MIN_CONFIDENCE_SCORE", "8"))             # War: 7 – Claude vergibt 7 zu leicht
MIN_RISK_REWARD_RATIO: float = float(os.getenv("MIN_RISK_REWARD_RATIO", "1.8"))     # War: 1.5
ACCOUNT_BALANCE_LIMIT: float = float(os.getenv("ACCOUNT_BALANCE_LIMIT", "500.0"))

# Kill-switch – set TRADING_ENABLED=false to stop all trade execution
TRADING_ENABLED: bool = os.getenv("TRADING_ENABLED", "true").lower() == "true"

# ── Watchlist – Capital.com epic codes ────────────────────────────────────────
WATCHLIST: dict[str, dict[str, str]] = {
    "GOLD": {"epic": "GOLD", "name": "Gold", "category": "precious_metals"},
    "SILVER": {"epic": "SILVER", "name": "Silver", "category": "precious_metals"},
    "OIL_CRUDE": {"epic": "OIL_CRUDE", "name": "Crude Oil WTI", "category": "energy"},
    "NATURALGAS": {"epic": "NATURALGAS", "name": "Natural Gas", "category": "energy"},
}

# ── Schedule ───────────────────────────────────────────────────────────────────
ANALYSIS_SCHEDULE: str = os.getenv("ANALYSIS_SCHEDULE", "0 8,12,16 * * 1-5")
TRADE_WINDOW_START: str = os.getenv("TRADE_WINDOW_START", "09:00")
TRADE_WINDOW_END: str = os.getenv("TRADE_WINDOW_END", "20:00")
INTRADAY_CLOSE_TIME: str = os.getenv("INTRADAY_CLOSE_TIME", "21:30")
TIMEZONE: str = os.getenv("TIMEZONE", "Europe/Berlin")
TZ = ZoneInfo(TIMEZONE)

# ── Database / paths ──────────────────────────────────────────────────────────
DATA_DIR: str = os.getenv("DATA_DIR", "/app/data")
DB_PATH: str = os.path.join(DATA_DIR, "trades.db")
LOG_DIR: str = os.path.join(DATA_DIR, "logs")

# ── Notifications ──────────────────────────────────────────────────────────────
NTFY_TOPIC: str = os.getenv("NTFY_TOPIC", "")
NTFY_SERVER: str = os.getenv("NTFY_SERVER", "https://ntfy.sh")

# ── News-Analyse ──────────────────────────────────────────────────────────────
NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")
NEWS_FETCH_INTERVAL_SECONDS: int = int(os.getenv("NEWS_FETCH_INTERVAL_HOURS", "4")) * 3600
NEWS_KEYWORDS: list[str] = [
    "oil", "gold", "silver", "gas", "commodities",
    "fed", "ecb", "tariff", "sanctions", "opec",
    "inflation", "recession",
]

# ── Volatilitäts-Gate ──────────────────────────────────────────────────────────
ATR_LOOKBACK_PERIOD: int = int(os.getenv("ATR_LOOKBACK_PERIOD", "14"))
MAX_ATR_MULTIPLIER: float = float(os.getenv("MAX_ATR_MULTIPLIER", "2.0"))
# Wenn aktueller ATR > Durchschnitts-ATR × MAX_ATR_MULTIPLIER → kein Trade

# Overtrading-Schutz
MAX_TRADES_PER_DAY: int = int(os.getenv("MAX_TRADES_PER_DAY", "1"))
COOLDOWN_AFTER_LOSS_MINUTES: int = int(os.getenv("COOLDOWN_AFTER_LOSS_MINUTES", "120"))

# Stop-Loss Konfiguration (ATR-basiert)
SL_ATR_MULTIPLIER: float = float(os.getenv("SL_ATR_MULTIPLIER", "1.5"))
TP_ATR_MULTIPLIER: float = float(os.getenv("TP_ATR_MULTIPLIER", "2.5"))
USE_ATR_BASED_SL: bool = os.getenv("USE_ATR_BASED_SL", "true").lower() == "true"

# Slippage-Limits pro Asset (absolute Werte in Asset-Waehrung)
MAX_SLIPPAGE_ABS: dict[str, float] = {
    "GOLD": 3.0,
    "SILVER": 0.5,
    "OIL_CRUDE": 1.0,
    "NATURALGAS": 0.1,
}
MAX_SLIPPAGE_PCT_DEFAULT: float = float(os.getenv("MAX_SLIPPAGE_PCT", "0.5"))

# Max-Drawdown Schutz
MAX_DAILY_DRAWDOWN_PCT: float = float(os.getenv("MAX_DAILY_DRAWDOWN_PCT", "5.0"))
MAX_WEEKLY_DRAWDOWN_PCT: float = float(os.getenv("MAX_WEEKLY_DRAWDOWN_PCT", "10.0"))

# ── Position Monitor ──────────────────────────────────────────────────────────
TRAIL_TRIGGER_PCT: float = float(os.getenv("TRAIL_TRIGGER_PCT", "1.0"))
TRAIL_DISTANCE_PCT: float = float(os.getenv("TRAIL_DISTANCE_PCT", "0.5"))
TRAIL_MIN_STEP: float = float(os.getenv("TRAIL_MIN_STEP", "5.0"))
ESCALATION_MAX_PER_DAY: int = int(os.getenv("ESCALATION_MAX_PER_DAY", "2"))

# ── API rate limiting ──────────────────────────────────────────────────────────
CAPITAL_MAX_REQUESTS_PER_HOUR: int = 1000
CAPITAL_SESSION_TTL_SECONDS: int = 600  # 10 minutes

# ── Recheck-Konfiguration ────────────────────────────────────────────────
RECHECK_MAX_PER_IDEA: int = int(os.getenv("RECHECK_MAX_PER_IDEA", "3"))
RECHECK_DEFAULT_MINUTES: int = int(os.getenv("RECHECK_DEFAULT_MINUTES", "60"))
RECHECK_MIN_MINUTES: int = int(os.getenv("RECHECK_MIN_MINUTES", "15"))
RECHECK_MAX_MINUTES: int = int(os.getenv("RECHECK_MAX_MINUTES", "180"))
RECHECK_EXPIRE_TIME: str = os.getenv("RECHECK_EXPIRE_TIME", "20:00")

# ── Trading Style ──────────────────────────────────────────────────────────────
# "swing"   → Tageskerzen, 30 Bars, mittelfristige Trends (Standard)
# "intraday" → Stundenkerzen, 48 Bars, kurze Trades ohne Overnight-Gebühren
TRADING_STYLE: str = os.getenv("TRADING_STYLE", "swing").lower()

# ── Price history ──────────────────────────────────────────────────────────────
PRICE_HISTORY_DAYS: int = 7
_default_resolution = "HOUR" if TRADING_STYLE == "intraday" else "DAY"
_default_bars = 48 if TRADING_STYLE == "intraday" else 30
PRICE_HISTORY_RESOLUTION: str = os.getenv("PRICE_HISTORY_RESOLUTION", _default_resolution)
PRICE_HISTORY_MAX_BARS: int = int(os.getenv("PRICE_HISTORY_MAX_BARS", str(_default_bars)))
