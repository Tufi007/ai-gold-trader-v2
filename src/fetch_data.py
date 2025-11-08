# src/fetch_data.py
"""
Production-grade fetcher for AI-GOLD-TRADER-V2

Strategy:
  1) MT5 (if available and configured)
  2) TradingView via tvDatafeed (login from env/config)
  3) yfinance fallback (several symbol fallbacks)

Features:
  - Uses configuration from src.utils.config (BASE_SYMBOL, RAW_DIR, TIMEFRAMES, etc.)
  - Concurrent multi-timeframe fetching with ThreadPoolExecutor
  - Retries + exponential backoff for network calls
  - Detailed logging about which env keys are set and which provider succeeded
  - Saves both timestamped backup and canonical cache CSV
"""

from __future__ import annotations
import os
import time
import math
import traceback
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
import yfinance as yf
from loguru import logger

# try to import project config (reads .env etc.)
try:
    from src.utils.config import (
        RAW_DIR,
        BASE_SYMBOL,
        TIMEFRAMES,
        TRADINGVIEW_USERNAME,
        TRADINGVIEW_PASSWORD,
        MT5_ENABLED,
        CACHE_MAX_MINUTES,
    )
except Exception:
    # robust fallback if import fails (should normally not happen)
    logger.warning("Could not import src.utils.config — falling back to defaults.")
    from dotenv import load_dotenv
    load_dotenv()
    RAW_DIR = os.getenv("RAW_DIR", "data/raw")
    BASE_SYMBOL = os.getenv("BASE_SYMBOL", "XAUUSD=X")
    TIMEFRAMES = [t.strip() for t in os.getenv("TIMEFRAMES", "5m,15m,30m,1h,4h").split(",")]
    TRADINGVIEW_USERNAME = os.getenv("TRADINGVIEW_USERNAME", "")
    TRADINGVIEW_PASSWORD = os.getenv("TRADINGVIEW_PASSWORD", "")
    MT5_ENABLED = False
    CACHE_MAX_MINUTES = int(os.getenv("CACHE_MAX_MINUTES", "10"))

# Optional helper imports (your repo may already have these; we attempt to import)
try:
    from src.utils.file_utils import raw_filepath, save_df_csv  # type: ignore
except Exception:
    raw_filepath = None
    save_df_csv = None

# optional MT5 utils
MT5_AVAILABLE = False
if MT5_ENABLED:
    try:
        from src.utils.mt5_utils import fetch_rates_mt5  # type: ignore
        MT5_AVAILABLE = True
    except Exception:
        logger.warning("MT5 enabled in config but src.utils.mt5_utils import failed. MT5 disabled.")

# tvdatafeed optional import (TradingView)
TV_AVAILABLE = False
try:
    # library name differs sometimes; user has tvdatafeed/tvDatafeed earlier — try both
    try:
        # tvdatafeed package used earlier in this repo as src.tvdatafeed
        from src.tvdatafeed.tvdatafeed import TvDatafeed, Interval  # type: ignore
        TV_AVAILABLE = True
    except Exception:
        # try pip-installed package name
        from tvDatafeed import TvDatafeed, Interval  # type: ignore
        TV_AVAILABLE = True
except Exception:
    TvDatafeed = None  # type: ignore
    Interval = None  # type: ignore
    TV_AVAILABLE = False

# ensure RAW_DIR exists
RAW_DIR_PATH = Path(RAW_DIR)
RAW_DIR_PATH.mkdir(parents=True, exist_ok=True)

# Mapping for yfinance and tvdatafeed intervals
INTERVAL_MAP = {
    "1m": {"yf": "1m", "tv": "in_1_minute"},
    "5m": {"yf": "5m", "tv": "in_5_minute"},
    "15m": {"yf": "15m", "tv": "in_15_minute"},
    "30m": {"yf": "30m", "tv": "in_30_minute"},
    "1h": {"yf": "60m", "tv": "in_1_hour"},
    "2h": {"yf": "60m", "tv": "in_2_hour"},
    "3h": {"yf": "60m", "tv": "in_3_hour"},
    "4h": {"yf": "240m", "tv": "in_4_hour"},
    "1d": {"yf": "1d", "tv": "in_1_day"},
}

# fallback symbol list for yfinance
SYMBOL_FALLBACKS = [
    BASE_SYMBOL,
    os.getenv("BASE_SYMBOL_ALT1", "XAUUSD=X"),
    os.getenv("BASE_SYMBOL_ALT2", "GC=F"),
]

# small helpers
def retry_backoff(fn, attempts=3, initial_delay=1.0, factor=2.0, allowed_exceptions=(Exception,), log_prefix=""):
    """Generic retry wrapper with exponential backoff (returns function result or raises last)."""
    delay = initial_delay
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            if attempt > 1:
                logger.debug(f"{log_prefix} retry attempt {attempt}/{attempts} after {delay:.1f}s")
                time.sleep(delay)
                delay *= factor
            return fn()
        except allowed_exceptions as e:
            last_exc = e
            logger.debug(f"{log_prefix} attempt {attempt} failed: {e}")
            # continue to next attempt
    # after attempts exhausted
    logger.debug(f"{log_prefix} all {attempts} attempts failed.")
    raise last_exc

def is_cache_expired(path: Path, max_minutes: int = CACHE_MAX_MINUTES) -> bool:
    if not path.exists():
        return True
    age_minutes = (time.time() - path.stat().st_mtime) / 60.0
    return age_minutes > max_minutes

def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, index_col=0)
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()]
        return df
    except Exception as e:
        logger.warning(f"Cache read failed for {path}: {e}")
        return pd.DataFrame()

def safe_save_df(df: pd.DataFrame, path: Path) -> None:
    try:
        # save backup with timestamp first
        ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
        backup = path.with_name(f"{path.stem}_{ts}{path.suffix}")
        df.to_csv(backup)
        df.to_csv(path)
        logger.info(f"Saved cache {path} (+backup {backup.name})")
    except Exception as e:
        logger.warning(f"Could not save cache {path}: {e}")

# --- PROVIDERS ---------------------------------------------------------------

def fetch_from_mt5(symbol: str, tf: str, n_bars: int = 2000) -> pd.DataFrame:
    if not MT5_AVAILABLE:
        return pd.DataFrame()
    def _call():
        logger.debug(f"MT5: requesting {symbol} {tf} {n_bars} bars")
        df = fetch_rates_mt5(symbol, timeframe_constant=None, n=n_bars)
        if df is None or df.empty:
            raise ValueError("MT5 returned empty")
        return df
    try:
        df = retry_backoff(_call, attempts=2, initial_delay=1.0, log_prefix="MT5")
        logger.success(f"MT5 fetch OK for {symbol} {tf} ({len(df)} rows)")
        return df
    except Exception as e:
        logger.debug(f"MT5 fetch failed for {symbol} {tf}: {e}")
        return pd.DataFrame()

def _tv_interval_obj(tf: str):
    """Return the Interval object attribute for tvdatafeed (handles different names)."""
    if not TV_AVAILABLE or Interval is None:
        return None
    tv_key = INTERVAL_MAP.get(tf, {}).get("tv")
    if not tv_key:
        return None
    # Interval may be a class with attributes like in_5_minute etc. Use getattr safely.
    try:
        return getattr(Interval, tv_key)
    except Exception:
        # sometimes Interval members have slightly different names; try a substring match
        for attr in dir(Interval):
            if tv_key.lower() in attr.lower():
                return getattr(Interval, attr)
    return None

def fetch_from_tradingview(symbol: str, tf: str, n_bars: int = 2000, tv_instance: Optional[Any] = None) -> pd.DataFrame:
    if not TV_AVAILABLE or TvDatafeed is None:
        logger.debug("TradingView provider not available in environment.")
        return pd.DataFrame()

    iv = _tv_interval_obj(tf)
    if iv is None:
        logger.debug(f"No TV interval mapping for timeframe '{tf}'")
        return pd.DataFrame()

    def _call():
        tv = tv_instance if tv_instance is not None else TvDatafeed(TRADINGVIEW_USERNAME, TRADINGVIEW_PASSWORD) if TRADINGVIEW_USERNAME else TvDatafeed()
        # log login status (best effort)
        try:
            if hasattr(tv, "login_status"):
                logger.info(f"TradingView login_status: {getattr(tv, 'login_status')}")
        except Exception:
            pass
        logger.debug(f"TV: getting history for {symbol} tf={tf} iv={iv} n_bars={n_bars}")
        df = tv.get_hist(symbol=symbol, exchange='OANDA', interval=iv, n_bars=n_bars)
        if df is None or df.empty:
            raise ValueError("TradingView returned no data")
        # normalize df
        df = df.rename(columns=str.lower)
        if "datetime" in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime')
        else:
            df.index = pd.to_datetime(df.index)
        for col in ["open", "high", "low", "close"]:
            if col not in df.columns:
                raise ValueError(f"Missing col {col} from TradingView data")
        if 'volume' not in df.columns:
            df['volume'] = 0
        return df[["open", "high", "low", "close", "volume"]]

    try:
        df = retry_backoff(_call, attempts=3, initial_delay=0.8, factor=2.0, log_prefix="TV")
        logger.success(f"TradingView fetch OK for {symbol} {tf} ({len(df)} rows)")
        return df
    except Exception as e:
        logger.debug(f"TradingView fetch failed for {symbol} {tf}: {e}")
        return pd.DataFrame()

def fetch_from_yfinance(symbol: str, tf: str, period: Optional[str] = None) -> pd.DataFrame:
    interval = INTERVAL_MAP.get(tf, {}).get("yf", tf)
    period = period or ("7d" if tf in ("1m", "5m", "15m", "30m") else "730d")
    def _call():
        logger.debug(f"yfinance download {symbol} interval={interval} period={period}")
        df = yf.download(symbol, interval=interval, period=period, progress=False)
        if df is None or df.empty:
            raise ValueError("yfinance returned empty")
        df = df.rename(columns=str.lower)
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()]
        if 'volume' not in df.columns:
            df['volume'] = 0
        return df[[c for c in ["open","high","low","close","volume"] if c in df.columns]]
    try:
        df = retry_backoff(_call, attempts=2, initial_delay=0.5, log_prefix="YF")
        logger.success(f"yfinance fetch OK for {symbol} {tf} ({len(df)} rows)")
        return df
    except Exception as e:
        logger.debug(f"yfinance fetch failed for {symbol} {tf}: {e}")
        return pd.DataFrame()

# --- Orchestration -----------------------------------------------------------

def fetch_one(symbol: str, tf: str, n_bars: int = 2000, tv_instance: Optional[Any] = None, force_fetch: bool = False) -> pd.DataFrame:
    """
    Fetch single timeframe for symbol using:
     - cache when available (unless force_fetch)
     - MT5 -> TradingView -> yfinance (with fallback symbols)
    """
    cache_path = RAW_DIR_PATH / f"{symbol}_{tf}.csv"
    if (not force_fetch) and (not is_cache_expired(cache_path)):
        df_cached = safe_read_csv(cache_path)
        if not df_cached.empty:
            logger.info(f"Using cached {cache_path} (age < {CACHE_MAX_MINUTES} minutes)")
            return df_cached

    # 1) MT5
    if MT5_AVAILABLE:
        try:
            df = fetch_from_mt5(symbol, tf, n_bars=n_bars)
            if not df.empty:
                safe_save_df(df, cache_path)
                return df
        except Exception:
            logger.debug("MT5 provider had an unexpected error:\n" + traceback.format_exc())

    # 2) TradingView
    tv_logged_in = None
    if TV_AVAILABLE:
        try:
            if tv_instance is None:
                # attempt to create one (this logs in)
                tv_logged_in = TvDatafeed(TRADINGVIEW_USERNAME, TRADINGVIEW_PASSWORD) if TRADINGVIEW_USERNAME else TvDatafeed()
            else:
                tv_logged_in = tv_instance
        except Exception as e:
            logger.warning(f"Could not create TvDatafeed instance: {e}")
            tv_logged_in = None

        if tv_logged_in is not None:
            df = fetch_from_tradingview(symbol, tf, n_bars=n_bars, tv_instance=tv_logged_in)
            if not df.empty:
                safe_save_df(df, cache_path)
                return df

    # 3) yfinance fallback (try symbol list)
    for sym in SYMBOL_FALLBACKS:
        if not sym:
            continue
        df = fetch_from_yfinance(sym, tf)
        if not df.empty:
            # if symbol was fallback and different, still store using canonical symbol name in cache filename
            safe_save_df(df, cache_path)
            return df

    logger.error(f"All providers failed for {symbol} {tf}")
    return pd.DataFrame()

def fetch_multi_timeframes(symbol: Optional[str] = None, timeframes: Optional[list] = None,
                           n_bars: int = 2000, max_workers: int = 4, force_fetch: bool = False) -> Dict[str, pd.DataFrame]:
    symbol = symbol or BASE_SYMBOL
    # timeframes may be stored as comma-separated in config
    if timeframes is None:
        # if TIMEFRAMES is a mapping already (like dict), preserve its keys
        if isinstance(TIMEFRAMES, (list, tuple)):
            tfs = list(TIMEFRAMES)
        elif isinstance(TIMEFRAMES, dict):
            tfs = list(TIMEFRAMES.keys())
        else:
            tfs = [t.strip() for t in str(TIMEFRAMES).split(",")]
    else:
        tfs = timeframes

    results: Dict[str, pd.DataFrame] = {}
    logger.info(f"Starting multi-timeframe fetch for {symbol}: {tfs} (force_fetch={force_fetch})")
    # create a single tv instance to re-use login for all threads (tvdatafeed may not be thread-safe,
    # so we will pass it but still be defensive)
    tv_instance = None
    if TV_AVAILABLE:
        try:
            tv_instance = TvDatafeed(TRADINGVIEW_USERNAME, TRADINGVIEW_PASSWORD) if TRADINGVIEW_USERNAME else TvDatafeed()
            logger.info(f"TradingView instance created (username set={bool(TRADINGVIEW_USERNAME)})")
        except Exception as e:
            logger.warning(f"Could not create TradingView instance: {e}")
            tv_instance = None

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch_one, symbol, tf, n_bars, tv_instance, force_fetch): tf for tf in tfs}
        for fut in as_completed(futures):
            tf = futures[fut]
            try:
                df = fut.result()
                if df is None or df.empty:
                    logger.warning(f"No data for {symbol} {tf}")
                    continue
                # ensure canonical column names + index type
                df = df.rename(columns=str.lower)
                df.index = pd.to_datetime(df.index, errors="coerce")
                df = df[~df.index.isna()]
                results[tf] = df.sort_index()
                logger.success(f"Fetched {symbol} {tf}: {len(df)} rows (latest={df.index[-1] if len(df)>0 else 'None'})")
            except Exception:
                logger.exception(f"Failed fetching timeframe {tf}:\n{traceback.format_exc()}")

    return results

# --- CLI/run demo -----------------------------------------------------------

if __name__ == "__main__":
    # print which env keys are set to help debugging
    logger.info("=== Fetcher starting: environment check ===")
    logger.info(f"RAW_DIR: {RAW_DIR_PATH}")
    logger.info(f"BASE_SYMBOL: {BASE_SYMBOL}")
    logger.info(f"TIMEFRAMES: {TIMEFRAMES}")
    logger.info(f"TRADINGVIEW_USERNAME set: {bool(TRADINGVIEW_USERNAME)}")
    logger.info(f"MT5_ENABLED config: {MT5_ENABLED}, MT5_AVAILABLE runtime: {MT5_AVAILABLE}")
    logger.info(f"TV_AVAILABLE runtime: {TV_AVAILABLE}")
    logger.info("==========================================")

    # fetch everything once (force fetch)
    res = fetch_multi_timeframes(force_fetch=True, max_workers=4, n_bars=2000)
    if not res:
        logger.error("❌ No data fetched at all.")
    else:
        for tf, df in res.items():
            logger.info(f"Summary: {tf} -> {len(df)} rows, latest={df.index[-1] if not df.empty else 'None'}")
    logger.info("🏁 fetch_data.py finished.")
