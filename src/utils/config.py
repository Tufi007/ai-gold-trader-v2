# src/utils/config.py
import os
from dotenv import load_dotenv
from loguru import logger
from pathlib import Path

# Load .env file
BASE_DIR = Path(__file__).resolve().parent.parent.parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH, override=True)

# =============================
# ✅ General Paths
# =============================
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, MODELS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =============================
# ✅ Trading Parameters
# =============================
BASE_SYMBOL = os.getenv("BASE_SYMBOL", "XAUUSD")
TIMEFRAMES = os.getenv("TIMEFRAMES", "1m,5m,15m,1h,4h,1d").split(",")
MIN_ROWS = int(os.getenv("MIN_ROWS", "2000"))

# =============================
# ✅ MetaTrader 5 Configuration
# =============================
MT5_ACCOUNT = os.getenv("MT5_ACCOUNT", "")
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER = os.getenv("MT5_SERVER", "")
MT5_TERMINAL_PATH = os.getenv("MT5_TERMINAL_PATH", "")
MT5_ENABLED = all([MT5_ACCOUNT, MT5_PASSWORD, MT5_SERVER, MT5_TERMINAL_PATH])

# =============================
# ✅ TradingView Configuration
# =============================
TRADINGVIEW_USERNAME = os.getenv("TRADINGVIEW_USERNAME", "")
TRADINGVIEW_PASSWORD = os.getenv("TRADINGVIEW_PASSWORD", "")

# =============================
# ✅ Telegram Notifications
# =============================
TELEGRAM_NOTIFY = os.getenv("TELEGRAM_NOTIFY", "false").lower() == "true"
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# =============================
# ✅ Validate Environment
# =============================
def validate_env():
    """Validate the most critical configuration values."""
    if not TRADINGVIEW_USERNAME or not TRADINGVIEW_PASSWORD:
        logger.warning("⚠️ TradingView credentials missing. Limited functionality may apply.")
    if not MT5_ENABLED:
        logger.warning("⚠️ MT5 login not configured. Using no-login data mode.")
    else:
        logger.success("✅ MT5 configuration is ready.")
    logger.success("✅ Configuration environment loaded successfully.")

validate_env()

# =============================
# ✅ Exported Constants
# =============================
__all__ = [
    "BASE_SYMBOL",
    "TIMEFRAMES",
    "MIN_ROWS",
    "RAW_DIR",
    "PROCESSED_DIR",
    "MODELS_DIR",
    "LOGS_DIR",
    "MT5_ACCOUNT",
    "MT5_PASSWORD",
    "MT5_SERVER",
    "MT5_TERMINAL_PATH",
    "MT5_ENABLED",
    "TRADINGVIEW_USERNAME",
    "TRADINGVIEW_PASSWORD",
    "TELEGRAM_NOTIFY",
    "TELEGRAM_TOKEN",
    "TELEGRAM_CHAT_ID"
]
