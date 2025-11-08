"""
fetch_data_tv.py
----------------
A clean, standalone TradingView data fetcher for AI-GOLD-TRADER-V2.

✅ Logs in as taufeeqr346
✅ Fetches XAUUSD (OANDA) for 1m, 5m, 15m, 1h, 4h
✅ Saves CSVs to /data/raw/
✅ Auto-retries login on failure
✅ Rich logging
"""

import os
import time
import traceback
from datetime import datetime
import pandas as pd
from tvDatafeed import TvDatafeed, Interval
from loguru import logger

# ======== CONFIG ========
USERNAME = "taufeeqr346"
PASSWORD = "Taufeeq@goldtraderlinkpassword@121"  # <-- Replace with your actual TradingView password
SYMBOL = "XAUUSD"
EXCHANGE = "OANDA"
RAW_DIR = "data/raw"
N_BARS = 1000  # how many candles to fetch per timeframe

TIMEFRAMES = {
    "1m": Interval.in_1_minute,
    "5m": Interval.in_5_minute,
    "15m": Interval.in_15_minute,
    "1h": Interval.in_1_hour,
    "4h": Interval.in_4_hour,
}

# ======== SETUP ========
os.makedirs(RAW_DIR, exist_ok=True)
logger.remove()
logger.add(lambda msg: print(msg, end=""))
logger.add("logs/fetch_data_tv.log", rotation="1 MB", level="INFO")


def connect_tv(max_retries=3):
    """Try to connect to TradingView with retry logic."""
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"🔐 Logging into TradingView as {USERNAME} (Attempt {attempt})...")
            tv = TvDatafeed(USERNAME, PASSWORD)
            logger.success("✅ Connected to TradingView successfully.")
            return tv
        except Exception as e:
            logger.error(f"⚠️ Login failed (Attempt {attempt}): {e}")
            time.sleep(3)
    logger.critical("❌ Failed to connect to TradingView after multiple attempts.")
    raise RuntimeError("TradingView login failed.")


def fetch_all_timeframes(tv):
    """Fetch and save all timeframe data for the configured symbol."""
    for name, interval in TIMEFRAMES.items():
        try:
            logger.info(f"📡 Fetching {SYMBOL} ({EXCHANGE}) {name} data...")
            df = tv.get_hist(
                symbol=SYMBOL,
                exchange=EXCHANGE,
                interval=interval,
                n_bars=N_BARS
            )
            if df is None or df.empty:
                logger.warning(f"⚠️ No data returned for {name}. Skipping...")
                continue

            df.reset_index(inplace=True)
            file_path = os.path.join(RAW_DIR, f"{SYMBOL}_{name}.csv")
            df.to_csv(file_path, index=False)
            logger.success(f"✅ Saved {name} data → {file_path}")

        except Exception as e:
            logger.error(f"❌ Error fetching {name}: {e}")
            logger.debug(traceback.format_exc())
            time.sleep(2)


def main():
    logger.info("🚀 Starting TradingView Data Fetcher (AI-GOLD-TRADER-V2)")
    try:
        tv = connect_tv()
        fetch_all_timeframes(tv)
        logger.success("🏁 Fetch complete for all timeframes.")
    except Exception as e:
        logger.critical(f"💥 Fatal error: {e}")
        logger.debug(traceback.format_exc())


if __name__ == "__main__":
    main()
