# src/utils/config.py
import os
from dotenv import load_dotenv
from loguru import logger
from pathlib import Path
import torch

# =============================
# ✅ Base & Load environment
# =============================
BASE_DIR = Path(__file__).resolve().parent.parent.parent
ENV_PATH = BASE_DIR / ".env"

if not ENV_PATH.exists():
    logger.error(f"❌ .env file not found at {ENV_PATH}")
else:
    load_dotenv(dotenv_path=ENV_PATH, override=True)
    logger.info(f"✅ Loaded environment from {ENV_PATH}")

# =============================
# ✅ Directories (consistent names)
# =============================
# Keep Path objects for convenience elsewhere
DATA_DIR = BASE_DIR / os.getenv("DATA_DIR", "data")
RAW_DIR = BASE_DIR / os.getenv("RAW_DIR", "data/raw")
PROCESSED_DIR = BASE_DIR / os.getenv("PROCESSED_DIR", "data/processed")
MODELS_DIR = BASE_DIR / os.getenv("MODELS_DIR", "models")
LOGS_DIR = BASE_DIR / os.getenv("LOGS_DIR", "logs")
PREDICTIONS_DIR = BASE_DIR / os.getenv("PREDICTIONS_DIR", "predictions")

for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, MODELS_DIR, LOGS_DIR, PREDICTIONS_DIR]:
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.warning(f"Could not create dir {d}: {e}")

# Also provide str aliases for legacy imports (if any code expects string paths)
DATA_DIR_STR = str(DATA_DIR)
RAW_DIR_STR = str(RAW_DIR)
PROCESSED_DIR_STR = str(PROCESSED_DIR)

# =============================
# ✅ Trading Configuration
# =============================
BASE_SYMBOL = os.getenv("BASE_SYMBOL", "XAUUSD=X")
TIMEFRAMES = [t.strip() for t in os.getenv("TIMEFRAMES", "5m,15m,30m,1h,4h,1d").split(",")]
MIN_ROWS = int(os.getenv("MIN_ROWS", "2000"))

# =============================
# ✅ MetaTrader 5
# =============================
MT5_ACCOUNT = os.getenv("MT5_ACCOUNT", "")
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER = os.getenv("MT5_SERVER", "")
MT5_TERMINAL_PATH = os.getenv("MT5_TERMINAL_PATH", "")
MT5_ENABLED = all([MT5_ACCOUNT, MT5_PASSWORD, MT5_SERVER, MT5_TERMINAL_PATH])

# =============================
# ✅ TradingView credentials (optional)
# =============================
TRADINGVIEW_USERNAME = os.getenv("TV_USER", "")
TRADINGVIEW_PASSWORD = os.getenv("TV_PASS", "")

# =============================
# ✅ Telegram
# =============================
TELEGRAM_NOTIFY = os.getenv("TELEGRAM_NOTIFY", "false").lower() == "true"
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# =============================
# ✅ ML/Model config
# =============================
USE_TRANSFORMER_FOR_NEWS = os.getenv("USE_TRANSFORMER_FOR_NEWS", "false").lower() == "true"
TRANSFORMER_MODEL_NAME = os.getenv("TRANSFORMER_MODEL_NAME", "yjernite/distilroberta-base-sentiment")
MODEL_BACKEND = os.getenv("MODEL_BACKEND", "xgboost")
USE_CUDA = torch.cuda.is_available()

# =============================
# ✅ Risk Management
# =============================
ACCOUNT_BALANCE = float(os.getenv("ACCOUNT_BALANCE", "1000.0"))
TRADE_RISK_PCT = float(os.getenv("TRADE_RISK_PCT", "1.0"))
SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "1.5"))
TP_ATR_MULT = float(os.getenv("TP_ATR_MULT", "3.0"))
THR_LONG = float(os.getenv("THR_LONG", "0.62"))
THR_SHORT = float(os.getenv("THR_SHORT", "0.38"))
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))

# =============================
# ✅ Validation
# =============================
def validate_env():
    if not TRADINGVIEW_USERNAME or not TRADINGVIEW_PASSWORD:
        logger.warning("⚠️ TradingView credentials missing. Using no-login mode for tvDatafeed (if used).")
    if not MT5_ENABLED:
        logger.warning("⚠️ MT5 not configured — fallback to TradingView/YFinance.")
    if USE_CUDA:
        logger.success("✅ CUDA GPU detected and active.")
    else:
        logger.warning("⚠️ CUDA not active. Running on CPU.")
    logger.success("✅ Environment validated successfully.")

validate_env()

__all__ = [
    "BASE_DIR", "DATA_DIR", "RAW_DIR", "PROCESSED_DIR", "MODELS_DIR", "LOGS_DIR", "PREDICTIONS_DIR",
    "DATA_DIR_STR", "RAW_DIR_STR", "PROCESSED_DIR_STR",
    "BASE_SYMBOL", "TIMEFRAMES", "MIN_ROWS",
    "MT5_ACCOUNT", "MT5_PASSWORD", "MT5_SERVER", "MT5_TERMINAL_PATH", "MT5_ENABLED",
    "TRADINGVIEW_USERNAME", "TRADINGVIEW_PASSWORD",
    "TELEGRAM_NOTIFY", "TELEGRAM_TOKEN", "TELEGRAM_CHAT_ID",
    "USE_TRANSFORMER_FOR_NEWS", "TRANSFORMER_MODEL_NAME",
    "MODEL_BACKEND", "USE_CUDA",
    "ACCOUNT_BALANCE", "TRADE_RISK_PCT", "SL_ATR_MULT", "TP_ATR_MULT",
    "THR_LONG", "THR_SHORT", "ATR_PERIOD"
]
