"""Configuration and environment variables for the trading bot."""
import os
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()

# ── Settings Schema (fuer Dashboard-UI) ─────────────────────────────────────
SETTINGS_SCHEMA: list[dict] = [
    # Kill-Switch
    {"key": "TRADING_ENABLED", "label": "Trading aktiv", "group": "Kill-Switch",
     "type": "bool", "description": "Master-Schalter fuer Trade-Ausfuehrung"},
    # Trading
    {"key": "DQN_SL_PCT", "label": "DQN Stop-Loss (%)", "group": "Trading",
     "type": "float", "min": 0.001, "max": 0.05, "step": 0.001,
     "description": "Stop-Loss Prozentsatz aus DQN-Training"},
    {"key": "DQN_TP_PCT", "label": "DQN Take-Profit (%)", "group": "Trading",
     "type": "float", "min": 0.001, "max": 0.05, "step": 0.001,
     "description": "Take-Profit Prozentsatz aus DQN-Training"},
    # Trailing Stop
    {"key": "TRAIL_TRIGGER_PCT", "label": "Trail Trigger (%)", "group": "Trailing Stop",
     "type": "float", "min": 0.1, "max": 5.0, "step": 0.1},
    {"key": "TRAIL_DISTANCE_PCT", "label": "Trail Distance (%)", "group": "Trailing Stop",
     "type": "float", "min": 0.1, "max": 5.0, "step": 0.1},
    {"key": "TRAIL_MIN_STEP", "label": "Trail Min Step (abs)", "group": "Trailing Stop",
     "type": "float", "min": 1.0, "max": 50.0, "step": 1.0},
    # Schedule
    {"key": "ANALYSIS_SCHEDULE", "label": "Analyse-Schedule (Cron)", "group": "Schedule",
     "type": "str", "description": "Cron-Syntax, z.B. '0 8,12,16 * * 1-5'"},
    # Recheck
    {"key": "RECHECK_MAX_PER_IDEA", "label": "Max Rechecks pro Idee", "group": "Recheck",
     "type": "int", "min": 1, "max": 10},
    {"key": "RECHECK_DEFAULT_MINUTES", "label": "Recheck Interval (Min)", "group": "Recheck",
     "type": "int", "min": 15, "max": 360},
    # Sonstiges
    {"key": "ESCALATION_MAX_PER_DAY", "label": "Max Eskalationen pro Tag", "group": "Sonstiges",
     "type": "int", "min": 0, "max": 10},
    {"key": "NTFY_TOPIC", "label": "Notification Topic", "group": "Sonstiges",
     "type": "str", "description": "ntfy.sh Topic fuer Push-Benachrichtigungen"},
]

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

# ── DQN-Modell Konfiguration ──────────────────────────────────────────────────
AI_MODELS_DIR: str = os.getenv("AI_MODELS_DIR", os.path.join(os.path.dirname(__file__), "..", "models"))
DQN_SL_PCT: float = float(os.getenv("DQN_SL_PCT", "0.003"))   # 0.3 % – aus Training
DQN_TP_PCT: float = float(os.getenv("DQN_TP_PCT", "0.005"))   # 0.5 % – aus Training
DQN_DEVICE: str = os.getenv("DQN_DEVICE", "auto")              # "auto" | "cpu" | "cuda"

# ── Backtest Finanz-Defaults ─────────────────────────────────────────────────
BACKTEST_CAPITAL: float = float(os.getenv("BACKTEST_CAPITAL", "1000.0"))
BACKTEST_RISK_PCT: float = float(os.getenv("BACKTEST_RISK_PCT", "0.01"))
BACKTEST_LEVERAGE: int = int(os.getenv("BACKTEST_LEVERAGE", "20"))

# ── Trading parameters ─────────────────────────────────────────────────────────
MAX_RISK_PER_TRADE_PCT: float = float(os.getenv("MAX_RISK_PER_TRADE_PCT", "2.0"))
MIN_CONFIDENCE_SCORE: int = int(os.getenv("MIN_CONFIDENCE_SCORE", "8"))
MIN_CLOSE_CONFIDENCE_SCORE: int = int(os.getenv("MIN_CLOSE_CONFIDENCE_SCORE", "1"))
# Per-direction confidence thresholds (dashboard-settable)
CONF_BUY: int = int(os.getenv("CONF_BUY", "8"))
CONF_SELL: int = int(os.getenv("CONF_SELL", "8"))
CONF_BUY_CLOSE: int = int(os.getenv("CONF_BUY_CLOSE", "1"))
CONF_SELL_CLOSE: int = int(os.getenv("CONF_SELL_CLOSE", "1"))

# Kill-switch – set TRADING_ENABLED=false to stop all trade execution
TRADING_ENABLED: bool = os.getenv("TRADING_ENABLED", "false").lower() == "true"

# Bot-aktive Assets (wird vom Dashboard-UI gesetzt)
BOT_ACTIVE_ASSETS: list[str] | None = None  # None = alle

# Letzte Bot-Signale (vom unified_tick gesetzt, vom Dashboard gelesen)
BOT_LAST_SIGNALS: list[dict] | None = None

# ── Watchlist – Capital.com epic codes ────────────────────────────────────────
WATCHLIST: dict[str, dict[str, str]] = {
    "GOLD": {"epic": "GOLD", "name": "Gold", "category": "precious_metals"},
    "SILVER": {"epic": "SILVER", "name": "Silver", "category": "precious_metals"},
    "OIL_CRUDE": {"epic": "OIL_CRUDE", "name": "Crude Oil WTI", "category": "energy"},
    "NATURALGAS": {"epic": "NATURALGAS", "name": "Natural Gas", "category": "energy"},
}

# ── Schedule ───────────────────────────────────────────────────────────────────
ANALYSIS_SCHEDULE: str = os.getenv("ANALYSIS_SCHEDULE", "0 8,12,16 * * 1-5")
TIMEZONE: str = os.getenv("TIMEZONE", "Europe/Berlin")
TZ = ZoneInfo(TIMEZONE)

# ── Database / paths ──────────────────────────────────────────────────────────
DATA_DIR: str = os.getenv("DATA_DIR", "/app/data")
DB_PATH: str = os.path.join(DATA_DIR, "trades.db")
LOG_DIR: str = os.path.join(DATA_DIR, "logs")

# ── Simulation engine ─────────────────────────────────────────────────────────
SIM_ENABLED: bool = os.getenv("SIM_ENABLED", "false").lower() == "true"
SIM_DB_PATH: str = os.path.join(DATA_DIR, "simulation.db")

# ── Notifications ──────────────────────────────────────────────────────────────
NTFY_TOPIC: str = os.getenv("NTFY_TOPIC", "")
NTFY_SERVER: str = os.getenv("NTFY_SERVER", "https://ntfy.sh")


# ── Slippage-Limits pro Asset (absolute Werte in Asset-Waehrung) ─────────────
MAX_SLIPPAGE_ABS: dict[str, float] = {
    "GOLD": 3.0,
    "SILVER": 0.5,
    "OIL_CRUDE": 1.0,
    "NATURALGAS": 0.1,
}
MAX_SLIPPAGE_PCT_DEFAULT: float = float(os.getenv("MAX_SLIPPAGE_PCT", "0.5"))

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

# ── Price history ─────────────────────────────────────────────────────────────
PRICE_HISTORY_RESOLUTION: str = os.getenv("PRICE_HISTORY_RESOLUTION", "DAY")
PRICE_HISTORY_MAX_BARS: int = int(os.getenv("PRICE_HISTORY_MAX_BARS", "30"))


def reload() -> None:
    """Re-read .env and update all module-level config variables."""
    load_dotenv(override=True)

    g = globals()

    g["CAPITAL_DEMO"] = os.getenv("CAPITAL_DEMO", "true").lower() == "true"
    g["CAPITAL_BASE_URL"] = (
        "https://demo-api-capital.backend-capital.com"
        if g["CAPITAL_DEMO"]
        else "https://api-capital.backend-capital.com"
    )
    g["AI_MODELS_DIR"] = os.getenv("AI_MODELS_DIR", os.path.join(os.path.dirname(__file__), "..", "models"))
    g["DQN_SL_PCT"] = float(os.getenv("DQN_SL_PCT", "0.003"))
    g["DQN_TP_PCT"] = float(os.getenv("DQN_TP_PCT", "0.005"))
    g["DQN_DEVICE"] = os.getenv("DQN_DEVICE", "auto")
    g["BACKTEST_CAPITAL"] = float(os.getenv("BACKTEST_CAPITAL", "1000.0"))
    g["BACKTEST_RISK_PCT"] = float(os.getenv("BACKTEST_RISK_PCT", "0.01"))
    g["BACKTEST_LEVERAGE"] = int(os.getenv("BACKTEST_LEVERAGE", "20"))

    g["MAX_RISK_PER_TRADE_PCT"] = float(os.getenv("MAX_RISK_PER_TRADE_PCT", "2.0"))
    g["MIN_CONFIDENCE_SCORE"] = int(os.getenv("MIN_CONFIDENCE_SCORE", "8"))
    g["MIN_CLOSE_CONFIDENCE_SCORE"] = int(os.getenv("MIN_CLOSE_CONFIDENCE_SCORE", "1"))
    g["CONF_BUY"] = int(os.getenv("CONF_BUY", "8"))
    g["CONF_SELL"] = int(os.getenv("CONF_SELL", "8"))
    g["CONF_BUY_CLOSE"] = int(os.getenv("CONF_BUY_CLOSE", "1"))
    g["CONF_SELL_CLOSE"] = int(os.getenv("CONF_SELL_CLOSE", "1"))
    g["TRADING_ENABLED"] = os.getenv("TRADING_ENABLED", "true").lower() == "true"

    g["ANALYSIS_SCHEDULE"] = os.getenv("ANALYSIS_SCHEDULE", "0 8,12,16 * * 1-5")
    g["TIMEZONE"] = os.getenv("TIMEZONE", "Europe/Berlin")
    g["TZ"] = ZoneInfo(g["TIMEZONE"])

    g["NTFY_TOPIC"] = os.getenv("NTFY_TOPIC", "")
    g["NTFY_SERVER"] = os.getenv("NTFY_SERVER", "https://ntfy.sh")

    g["TRAIL_TRIGGER_PCT"] = float(os.getenv("TRAIL_TRIGGER_PCT", "1.0"))
    g["TRAIL_DISTANCE_PCT"] = float(os.getenv("TRAIL_DISTANCE_PCT", "0.5"))
    g["TRAIL_MIN_STEP"] = float(os.getenv("TRAIL_MIN_STEP", "5.0"))
    g["ESCALATION_MAX_PER_DAY"] = int(os.getenv("ESCALATION_MAX_PER_DAY", "2"))

    g["RECHECK_MAX_PER_IDEA"] = int(os.getenv("RECHECK_MAX_PER_IDEA", "3"))
    g["RECHECK_DEFAULT_MINUTES"] = int(os.getenv("RECHECK_DEFAULT_MINUTES", "60"))
    g["RECHECK_MIN_MINUTES"] = int(os.getenv("RECHECK_MIN_MINUTES", "15"))
    g["RECHECK_MAX_MINUTES"] = int(os.getenv("RECHECK_MAX_MINUTES", "180"))
    g["RECHECK_EXPIRE_TIME"] = os.getenv("RECHECK_EXPIRE_TIME", "20:00")

    g["PRICE_HISTORY_RESOLUTION"] = os.getenv("PRICE_HISTORY_RESOLUTION", "DAY")
    g["PRICE_HISTORY_MAX_BARS"] = int(os.getenv("PRICE_HISTORY_MAX_BARS", "30"))
