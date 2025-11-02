"""
fetch_data.py
Advanced multi-timeframe data fetcher and merger for AI-GOLD-TRADER v2.

Features:
- Fetches data from TradingView or Yahoo Finance.
- Caches to /data/raw/<symbol>_<tf>.csv for reuse.
- Computes indicators per-timeframe.
- Handles missing data and time gaps.
- Auto-suffixes timeframe columns to avoid duplicates.
"""

import os
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
from datetime import datetime, timedelta
import yfinance as yf

# Local imports
from src.utils.config import (
    RAW_DATA_DIR,
    BASE_SYMBOL,
    TIMEFRAMES,
    validate_env,
)

# Validate environment
validate_env()


# =========================
# TradingView / Yahoo fetchers
# =========================
def fetch_from_yf(symbol: str, tf: str, n_bars: int = 2000):
    """Download data via Yahoo Finance."""
    try:
        interval_map = {
            "1m": "1m", "5m": "5m", "15m": "15m",
            "30m": "30m", "1h": "60m", "4h": "240m",
            "1d": "1d"
        }
        interval = interval_map.get(tf, "1h")
        period_map = {
            "1m": "7d", "5m": "60d", "15m": "60d",
            "30m": "60d", "1h": "730d", "4h": "730d",
            "1d": "max"
        }
        period = period_map.get(tf, "1y")

        logger.debug(f"Fetching {symbol} ({tf}) from yfinance...")
        data = yf.download(
            symbol, interval=interval, period=period, progress=False
        )
        if data.empty:
            raise ValueError("Empty DataFrame received")

        data = data.reset_index()
        data = data.rename(
            columns={
                "Datetime": "datetime",
                "Date": "datetime",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )
        data["datetime"] = pd.to_datetime(data["datetime"])
        data = data.set_index("datetime").sort_index()
        return data.tail(n_bars)

    except Exception as e:
        logger.exception(f"Yahoo fetch failed for {symbol} {tf}: {e}")
        return pd.DataFrame()


# =========================
# Local cache handling
# =========================
def load_or_fetch(symbol: str, tf: str, force: bool = False, n_bars: int = 2000):
    """Load cached data or fetch fresh if needed."""
    path = os.path.join(RAW_DATA_DIR, f"{symbol.replace('/', '_')}_{tf}.csv")

    if os.path.exists(path) and not force:
        try:
            df = pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")
            if not df.empty:
                logger.info(f"📂 Loaded cached data: {path}")
                return df
        except Exception as e:
            logger.warning(f"Failed to load cached {tf}: {e}")

    df = fetch_from_yf(symbol, tf, n_bars)
    if not df.empty:
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        df.to_csv(path)
        logger.info(f"💾 Cached {symbol} {tf} → {path}")
    return df


# =========================
# Multi-timeframe wrapper (cleaned + merge-ready)
# =========================
def fetch_multi_timeframes(symbol: str = None, timeframes: list = None, n_bars: int = 2000,
                           max_workers: int = 4, force_fetch: bool = False):
    symbol = symbol or BASE_SYMBOL
    timeframes = timeframes or TIMEFRAMES
    results = {}

    logger.info(f"📡 Fetching {symbol} for {len(timeframes)} timeframes (force_fetch={force_fetch})")

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(load_or_fetch, symbol, tf, force_fetch, n_bars): tf for tf in timeframes}
        for fut in as_completed(futures):
            tf = futures[fut]
            try:
                df = fut.result()
                if df is None or df.empty:
                    logger.warning(f"No data returned for tf={tf}")
                    continue

                # ✅ Normalize and add TF suffix to prevent duplicate columns
                df = df.rename(columns=str.lower)
                df = df.add_suffix(f"_{tf}")
                df = df.rename(columns={f"datetime_{tf}": "datetime"})
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()

                results[tf] = df
                logger.info(f"✅ {symbol} {tf} → {len(df)} rows")
            except Exception as e:
                logger.exception(f"Failed to fetch tf={tf}: {e}")

    if not results:
        logger.error("❌ No timeframe data fetched successfully.")
        return pd.DataFrame()

    # ✅ Merge all timeframes side-by-side using the timestamp
    merged = pd.concat(results.values(), axis=1, join="inner")

    # ✅ Drop duplicate columns if somehow any survived
    merged = merged.loc[:, ~merged.columns.duplicated()]

    # ✅ Sanity check logging
    if merged.columns.duplicated().any():
        logger.warning(f"⚠️ Duplicate columns found even after cleaning: {merged.columns[merged.columns.duplicated()].tolist()}")
    else:
        logger.debug(f"🧩 Final merged columns: {len(merged.columns)} total")

    return merged


# =========================
# Run directly
# =========================
if __name__ == "__main__":
    df = fetch_multi_timeframes(force_fetch=False)
    print(df.tail())
