import os
import numpy as np
import pandas as pd
import pytz
from datetime import datetime
from loguru import logger
from src.fetch_data import fetch_multi_timeframes
from src.utils.indicators import compute_indicators
from src.utils.file_utils import processed_path, save_df_csv
from src.utils.config import TIMEFRAMES, MIN_ROWS, TELEGRAM_NOTIFY, TELEGRAM_CHAT_ID
from src.utils.telegram_bot import send_telegram_message

def ist_now_str():
    """Return current IST time as string"""
    ist = pytz.timezone("Asia/Kolkata")
    return datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")

def align_and_merge(feat_frames: dict, base_tf: str):
    """Align and merge multi-timeframe feature dataframes"""
    base_index = feat_frames[base_tf].index
    reindexed = {}
    for tf, feats in feat_frames.items():
        r = feats.reindex(base_index, method="ffill")
        reindexed[tf] = r
    merged_all = pd.concat(reindexed.values(), axis=1)
    merged_all.dropna(how="all", inplace=True)
    keep_mask = merged_all.notnull().mean(axis=1) >= 0.30
    merged_keep = merged_all[keep_mask]
    if merged_keep.shape[0] < MIN_ROWS:
        merged_keep = merged_all.dropna(axis=0, thresh=int(merged_all.shape[1] * 0.1))
        logger.warning(f"Relaxed threshold; rows now {merged_keep.shape[0]}")
    return merged_keep.fillna(method="ffill").fillna(method="bfill")

def process_data(symbol="XAUUSD", timeframes=None, notify=True):
    """Main data processing pipeline — fetches, processes and saves merged data."""
    try:
        logger.info(f"🔄 Starting processing for {symbol} ...")
        timeframes = timeframes or TIMEFRAMES

        frames = fetch_multi_timeframes(symbol=symbol, timeframes=timeframes)
        if not frames:
            raise RuntimeError("❌ No timeframe data fetched")

        feat_frames = {}
        for tf, df in frames.items():
            feats = compute_indicators(df, suffix=f"_{tf}")
            feat_frames[tf] = feats

        base_tf = min(feat_frames.keys(), key=lambda t: int("".join(filter(str.isdigit, t))))
        merged = align_and_merge(feat_frames, base_tf)
        out = processed_path(symbol)
        save_df_csv(merged, out, index=True)
        logger.success(f"✅ Saved processed data → {out} (rows={merged.shape[0]}, cols={merged.shape[1]})")

        if notify and TELEGRAM_NOTIFY:
            msg = (
                f"📊 *{symbol} Data Processed Successfully*\n"
                f"🕒 {ist_now_str()}\n"
                f"💾 Rows: {merged.shape[0]} | Cols: {merged.shape[1]}"
            )
            send_telegram_message(msg, chat_id=TELEGRAM_CHAT_ID)

        return merged
    except Exception as e:
        logger.exception(f"⚠️ Error while processing {symbol}: {e}")
        if TELEGRAM_NOTIFY:
            send_telegram_message(f"❌ Processing failed for {symbol}\nError: {e}", chat_id=TELEGRAM_CHAT_ID)
        raise
