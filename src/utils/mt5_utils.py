# src/utils/mt5_utils.py
import importlib
from loguru import logger
from src.utils.config import MT5_ACCOUNT, MT5_PASSWORD, MT5_SERVER, MT5_TERMINAL_PATH

MT5_AVAILABLE = False
_mt5 = None

try:
    _mt5 = importlib.import_module("MetaTrader5")
    MT5_AVAILABLE = True
except Exception:
    logger.info("MetaTrader5 library not available on this platform.")

def initialize_mt5(login: int = None, password: str = None, server: str = None, path: str = None) -> bool:
    if not MT5_AVAILABLE:
        return False
    try:
        p = path or MT5_TERMINAL_PATH or None
        ok = _mt5.initialize(path=p) if p else _mt5.initialize()
        if not ok:
            logger.error("mt5.initialize failed: %s", _mt5.last_error())
            return False
        acct = int(login or MT5_ACCOUNT) if (login or MT5_ACCOUNT) else None
        if acct and (password or MT5_PASSWORD) and (server or MT5_SERVER):
            logged = _mt5.login(acct, password=(password or MT5_PASSWORD), server=(server or MT5_SERVER))
            if not logged:
                logger.error("mt5.login failed: %s", _mt5.last_error())
                _mt5.shutdown()
                return False
            logger.success("MT5 logged in: %s", acct)
        else:
            logger.info("MT5 initialized without login.")
        return True
    except Exception as e:
        logger.exception("MT5 initialize error: %s", e)
        return False

def shutdown_mt5():
    if MT5_AVAILABLE and _mt5:
        try:
            _mt5.shutdown()
            logger.info("MT5 shutdown.")
        except Exception as e:
            logger.warning("MT5 shutdown error: %s", e)

def fetch_rates_mt5(symbol: str, timeframe_constant, n: int = 500):
    # only call when MT5_AVAILABLE
    import pandas as pd
    rates = _mt5.copy_rates_from_pos(symbol, timeframe_constant, 0, n)
    if rates is None:
        logger.error("mt5.copy_rates_from_pos returned None: %s", _mt5.last_error())
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df['datetime'] = pd.to_datetime(df['time'], unit='s')
    df = df.set_index('datetime')
    # convert columns to typical names
    df = df.rename(columns={'open':'open','high':'high','low':'low','close':'close','tick_volume':'volume'})
    return df[['open','high','low','close','volume']]
