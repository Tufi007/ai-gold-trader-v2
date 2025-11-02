import os
import pandas as pd
import yfinance as yf
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from src.utils.config import RAW_DIR, BASE_SYMBOL, TIMEFRAMES
from src.utils.file_utils import raw_filepath, save_df_csv
from src.utils.mt5_utils import MT5_AVAILABLE, fetch_rates_mt5

load_dotenv()

# =========================
# TradingView setup
# =========================
TV_AVAILABLE = False
try:
    from src.tvdatafeed.tvdatafeed import TvDatafeed, Interval
    TV_USER = os.getenv("TV_USER")
    TV_PASS = os.getenv("TV_PASS")
    if TV_USER and TV_PASS:
        tv = TvDatafeed(TV_USER, TV_PASS)
        logger.success(f"✅ Logged into TradingView as {TV_USER}")
    else:
        tv = TvDatafeed()
        logger.warning("⚠️ Using tvDatafeed in NOLOGIN mode (limited data). Add TV_USER & TV_PASS in .env")
    TV_AVAILABLE = True
except Exception as e:
    logger.warning(f"⚠️ tvDatafeed not available: {e}")

# Mapping of timeframes
INTERVAL_MAP = {
    "5m": {"tv": Interval.in_5_minute, "yf": "5m"},
    "15m": {"tv": Interval.in_15_minute, "yf": "15m"},
    "30m": {"tv": Interval.in_30_minute, "yf": "30m"},
    "1h": {"tv": Interval.in_1_hour, "yf": "60m"},
    "4h": {"tv": Interval.in_4_hour, "yf": "4h"},
    "1d": {"tv": Interval.in_daily, "yf": "1d"}
}

# =========================
# Fetch functions
# =========================
def fetch_tf_tv(symbol: str, tf: str, n_bars: int = 2000) -> pd.DataFrame:
    """Fetch data from TradingView"""
    iv = INTERVAL_MAP.get(tf, {}).get("tv")
    if iv is None:
        raise ValueError(f"Unsupported TF for tvDatafeed: {tf}")
    df = tv.get_hist(symbol=symbol, exchange='OANDA', interval=iv, n_bars=n_bars)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns=str.lower)
    if "datetime" in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
    else:
        df.index = pd.to_datetime(df.index)
    return df[['open', 'high', 'low', 'close', 'volume']]


def fetch_tf_yf(symbol: str, tf: str, period: str = None) -> pd.DataFrame:
    """Fetch data from Yahoo Finance"""
    interval = INTERVAL_MAP.get(tf, {}).get("yf", tf)
    period = period or ("7d" if tf in ("5m","15m","30m") else "730d")
    logger.info(f"yfinance fetch {symbol} {interval} period={period}")
    df = yf.download(symbol, interval=interval, period=period, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns=str.lower)
    df.index = pd.to_datetime(df.index)
    if 'volume' not in df.columns:
        df['volume'] = 0
    return df[['open', 'high', 'low', 'close', 'volume']]


def fetch_timeframe(symbol: str, tf: str, n_bars: int = 2000):
    """Try MT5 → TradingView → yfinance"""
    path = raw_filepath(symbol, tf)
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            logger.info(f"Using cached raw {path}")
            return df
        except Exception:
            logger.warning(f"Cached file corrupt, will refetch: {path}")

    if MT5_AVAILABLE:
        try:
            df = fetch_rates_mt5(symbol, timeframe_constant=None, n=n_bars)
            if df is not None and not df.empty:
                save_df_csv(df, path)
                return df
        except Exception as e:
            logger.debug(f"MT5 failed for {symbol} {tf}: {e}")

    if TV_AVAILABLE:
        try:
            df = fetch_tf_tv(symbol, tf, n_bars=n_bars)
            if not df.empty:
                save_df_csv(df, path)
                return df
        except Exception as e:
            logger.warning(f"TV fetch failed for {symbol} {tf}: {e}")

    try:
        df = fetch_tf_yf(symbol, tf)
        if not df.empty:
            save_df_csv(df, path)
            return df
    except Exception as e:
        logger.error(f"yfinance failed: {e}")

    logger.error(f"No provider returned data for {symbol} {tf}")
    return pd.DataFrame()


def fetch_multi_timeframes(symbol: str = None, timeframes: list = None, n_bars: int = 2000, max_workers: int = 4):
    """Fetch multiple timeframes concurrently"""
    symbol = symbol or BASE_SYMBOL
    timeframes = timeframes or TIMEFRAMES
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch_timeframe, symbol, tf, n_bars): tf for tf in timeframes}
        for fut in as_completed(futures):
            tf = futures[fut]
            try:
                df = fut.result()
                if df is None or df.empty:
                    logger.warning(f"No data for tf={tf}")
                    continue
                df = df.rename(columns=str.lower)
                df.index = pd.to_datetime(df.index)
                results[tf] = df.sort_index()
                logger.info(f"Fetched {len(df)} rows for {tf}")
            except Exception as e:
                logger.exception(f"Failed to fetch tf={tf}: {e}")
    return results
