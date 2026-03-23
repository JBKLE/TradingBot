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
MAX_RISK_PER_TRADE_PCT: float = float(os.getenv("MAX_RISK_PER_TRADE_PCT", "5.0"))
MIN_CONFIDENCE_SCORE: int = int(os.getenv("MIN_CONFIDENCE_SCORE", "7"))
MIN_RISK_REWARD_RATIO: float = float(os.getenv("MIN_RISK_REWARD_RATIO", "1.5"))
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
TIMEZONE: str = os.getenv("TIMEZONE", "Europe/Berlin")
TZ = ZoneInfo(TIMEZONE)

# ── Database / paths ──────────────────────────────────────────────────────────
DATA_DIR: str = os.getenv("DATA_DIR", "/app/data")
DB_PATH: str = os.path.join(DATA_DIR, "trades.db")
LOG_DIR: str = os.path.join(DATA_DIR, "logs")

# ── Notifications ──────────────────────────────────────────────────────────────
NTFY_TOPIC: str = os.getenv("NTFY_TOPIC", "")
NTFY_SERVER: str = os.getenv("NTFY_SERVER", "https://ntfy.sh")

# ── Position Monitor ──────────────────────────────────────────────────────────
TRAIL_TRIGGER_PCT: float = float(os.getenv("TRAIL_TRIGGER_PCT", "1.0"))
TRAIL_DISTANCE_PCT: float = float(os.getenv("TRAIL_DISTANCE_PCT", "0.5"))
TRAIL_MIN_STEP: float = float(os.getenv("TRAIL_MIN_STEP", "5.0"))
ESCALATION_MAX_PER_DAY: int = int(os.getenv("ESCALATION_MAX_PER_DAY", "2"))

# ── Trade confirmation ─────────────────────────────────────────────────────────
TRADE_CONFIRMATION_ENABLED: bool = os.getenv("TRADE_CONFIRMATION_ENABLED", "false").lower() == "true"
TRADE_CONFIRMATION_TIMEOUT_MINUTES: int = int(os.getenv("TRADE_CONFIRMATION_TIMEOUT_MINUTES", "15"))
NTFY_CONFIRM_TOPIC: str = os.getenv("NTFY_CONFIRM_TOPIC", "")

# ── API rate limiting ──────────────────────────────────────────────────────────
CAPITAL_MAX_REQUESTS_PER_HOUR: int = 1000
CAPITAL_SESSION_TTL_SECONDS: int = 600  # 10 minutes

# ── Price history ──────────────────────────────────────────────────────────────
PRICE_HISTORY_DAYS: int = 7
PRICE_HISTORY_RESOLUTION: str = "DAY"
PRICE_HISTORY_MAX_BARS: int = 30
