# src/process_data.py
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

def ist_now_str() -> str:
    ist = pytz.timezone("Asia/Kolkata")
    return datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")

def align_and_merge(feat_frames: dict, base_tf: str) -> pd.DataFrame:
    base_index = feat_frames[base_tf].index
    reindexed = {tf: feats.reindex(base_index, method="ffill") for tf, feats in feat_frames.items()}
    merged_all = pd.concat(reindexed.values(), axis=1)
    merged_all.dropna(how="all", inplace=True)

    keep_mask = merged_all.notnull().mean(axis=1) >= 0.30
    merged_keep = merged_all[keep_mask]

    if merged_keep.shape[0] < MIN_ROWS:
        merged_keep = merged_all.dropna(axis=0, thresh=int(merged_all.shape[1] * 0.1))
        logger.warning(f"⚠️ Relaxed threshold; rows now {merged_keep.shape[0]}")

    # use ff/bfill
    return merged_keep.ffill().bfill()

def process_data(symbol: str = "XAUUSD", timeframes: list | None = None,
                 notify: bool = True, force_fetch: bool = False) -> pd.DataFrame:
    try:
        logger.info(f"🔄 Starting processing for {symbol} ...")
        timeframes = timeframes or TIMEFRAMES
        out_path = processed_path(symbol)

        # caching
        if os.path.exists(out_path) and not force_fetch:
            logger.info(f"📁 Using cached processed data → {out_path}")
            df_cached = pd.read_csv(out_path, index_col=0, parse_dates=True)
            logger.success(f"✅ Loaded cached data ({df_cached.shape[0]} rows, {df_cached.shape[1]} cols)")
            return df_cached

        logger.info(f"🌐 Fetching fresh multi-timeframe data for {symbol} ...")
        frames = fetch_multi_timeframes(symbol=symbol, timeframes=timeframes, force_fetch=force_fetch)

        # defensive checks: frames should be a dict mapping tf->DataFrame
        if frames is None:
            raise RuntimeError("❌ fetch_multi_timeframes returned None")
        if isinstance(frames, dict) and len(frames) == 0:
            raise RuntimeError("❌ No timeframe data fetched — check TradingView/YFinance/symbol")
        if not isinstance(frames, dict):
            # if some other shape returned, attempt to coerce
            raise RuntimeError("❌ Unexpected return type from fetch_multi_timeframes")

        # compute indicators
        feat_frames = {}
        for tf, df in frames.items():
            if df is None or df.empty:
                logger.warning(f"Skipping empty timeframe {tf}")
                continue
            feats = compute_indicators(df, suffix=f"_{tf}")
            # ensure base timeframe keeps OHLC where available
            close_cols_present = [c for c in df.columns if c.lower() in ("open","high","low","close","volume")]
            for c in close_cols_present:
                if c not in feats.columns:
                    feats[c] = df[c]
            feat_frames[tf] = feats
            logger.debug(f"🧮 Indicators computed for {symbol} [{tf}] → {feats.shape}")

        if len(feat_frames) == 0:
            raise RuntimeError("❌ No valid feature frames available after indicator computation")

        # base timeframe selection (smallest timeframe in seconds)
        def tf_rank(t):
            # map '5m'->5, '1h'->60, '4h'->240, '1d'->1440
            num = int("".join(filter(str.isdigit, t)) or 0)
            if 'm' in t:
                return num
            if 'h' in t:
                return num * 60
            if 'd' in t:
                return num * 60 * 24
            return 999999
        base_tf = min(feat_frames.keys(), key=tf_rank)
        merged = align_and_merge(feat_frames, base_tf)

        # target generation
        # find best main_close column: prefer exact 'close' label
        close_cols = [c for c in merged.columns if c.lower() == "close"]
        if not close_cols:
            # fallback: any column that endswith '_close' or contains 'close'
            close_cols = [c for c in merged.columns if "close" in c.lower()]
        if not close_cols:
            raise RuntimeError("❌ No 'close' columns found in merged data.")

        main_close_col = close_cols[0]
        if len(close_cols) > 1:
            logger.warning(f"⚠️ Multiple close columns detected: {close_cols}. Using {main_close_col}")

        merged["future_close"] = merged[main_close_col].shift(-1)
        merged["future_return"] = (merged["future_close"] - merged[main_close_col]) / merged[main_close_col].replace(0, np.nan)
        merged["target"] = (merged["future_return"] > 0.0015).astype(int)
        merged.drop(columns=["future_close"], inplace=True)

        logger.info(
            f"🎯 Target generated using [{main_close_col}] — up moves: {int(merged['target'].sum())} / {len(merged)} "
            f"({merged['target'].mean()*100:.2f}% bullish)"
        )

        # save
        save_df_csv(merged, out_path, index=True)
        logger.success(f"✅ Saved processed data → {out_path} (rows={merged.shape[0]}, cols={merged.shape[1]})")

        # notify
        if notify and TELEGRAM_NOTIFY:
            ratio = merged['target'].mean() * 100 if 'target' in merged else 0.0
            msg = (
                f"📊 *{symbol} Data Processed Successfully*\n"
                f"🕒 {ist_now_str()}\n"
                f"💾 Rows: {merged.shape[0]} | Cols: {merged.shape[1]}\n"
                f"📈 Target Ratio: {ratio:.2f}%\n"
                f"📁 Cached: `{out_path}`"
            )
            send_telegram_message(msg, chat_id=TELEGRAM_CHAT_ID)

        return merged

    except Exception as e:
        logger.exception(f"⚠️ Error while processing {symbol}: {e}")
        if TELEGRAM_NOTIFY:
            send_telegram_message(f"❌ Processing failed for {symbol}\nError: {e}", chat_id=TELEGRAM_CHAT_ID)
        raise
