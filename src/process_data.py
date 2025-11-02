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
    """Return current IST time as a formatted string."""
    ist = pytz.timezone("Asia/Kolkata")
    return datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")


def align_and_merge(feat_frames: dict[str, pd.DataFrame], base_tf: str) -> pd.DataFrame:
    """Align and merge multi-timeframe feature dataframes."""
    base_index = feat_frames[base_tf].index
    reindexed = {tf: feats.reindex(base_index, method="ffill") for tf, feats in feat_frames.items()}
    merged_all = pd.concat(reindexed.values(), axis=1)
    merged_all.dropna(how="all", inplace=True)

    keep_mask = merged_all.notnull().mean(axis=1) >= 0.30
    merged_keep = merged_all[keep_mask]

    if merged_keep.shape[0] < MIN_ROWS:
        merged_keep = merged_all.dropna(axis=0, thresh=int(merged_all.shape[1] * 0.1))
        logger.warning(f"⚠️ Relaxed threshold; rows now {merged_keep.shape[0]}")

    return merged_keep.ffill().bfill()


def process_data(symbol: str = "XAUUSD", timeframes: list[str] | None = None,
                 notify: bool = True, force_fetch: bool = False) -> pd.DataFrame:
    """
    Main data processing pipeline.
    Fetches, processes, merges, labels, and saves data for a given symbol.
    """
    try:
        logger.info(f"🔄 Starting processing for {symbol} ...")
        timeframes = timeframes or TIMEFRAMES
        out_path = processed_path(symbol)

        # ✅ Use cached data if available
        if os.path.exists(out_path) and not force_fetch:
            logger.info(f"📁 Using cached processed data → {out_path}")
            df_cached = pd.read_csv(out_path, index_col=0, parse_dates=True)
            logger.success(f"✅ Loaded cached data ({df_cached.shape[0]} rows, {df_cached.shape[1]} cols)")
            return df_cached

        # ✅ Fetch all timeframes
        logger.info(f"🌐 Fetching fresh multi-timeframe data for {symbol} ...")
        frames = fetch_multi_timeframes(symbol=symbol, timeframes=timeframes)
        if not frames:
            raise RuntimeError("❌ No timeframe data fetched — check TradingView or symbol.")

        # ✅ Compute indicators per timeframe
        feat_frames = {}
        for tf, df in frames.items():
            feats = compute_indicators(df, suffix=f"_{tf}")
            # ✅ Keep OHLC columns for base timeframe
            if "close" in df.columns and tf not in feat_frames:
                feats[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]]
            feat_frames[tf] = feats
            logger.debug(f"🧮 Indicators computed for {symbol} [{tf}] → {feats.shape}")

        # ✅ Determine base timeframe (lowest numeric)
        base_tf = min(feat_frames.keys(), key=lambda t: int("".join(filter(str.isdigit, t)) or "999"))
        merged = align_and_merge(feat_frames, base_tf)

        # === 🎯 Generate target and returns ===
        close_cols = [c for c in merged.columns if c.lower().startswith("close")]
        if len(close_cols) == 0:
            raise ValueError("❌ No 'close' columns found in merged data.")

        # ✅ Pick one 'close' column safely
        if len(close_cols) > 1:
            logger.warning(f"⚠️ Multiple close columns detected: {close_cols}. Resolving automatically.")
            # Prefer the last (usually higher timeframe or merged correctly)
            main_close_col = sorted(close_cols)[-1]
        else:
            main_close_col = close_cols[0]

        # ✅ Ensure we use only a single Series, not a multi-column frame
        merged["future_close"] = merged[main_close_col].shift(-1)
        merged["future_return"] = (merged["future_close"] - merged[main_close_col]) / merged[main_close_col]
        merged["target"] = (merged["future_return"] > 0.0015).astype(int)
        merged.drop(columns=["future_close"], inplace=True)

        logger.info(
            f"🎯 Target generated using [{main_close_col}] — up moves: {merged['target'].sum()} / {len(merged)} "
            f"({merged['target'].mean() * 100:.2f}% bullish)"
        )

        # ✅ Save to cache
        save_df_csv(merged, out_path, index=True)
        logger.success(f"✅ Saved processed data → {out_path} (rows={merged.shape[0]}, cols={merged.shape[1]})")

        # ✅ Telegram notification
        if notify and TELEGRAM_NOTIFY:
            msg = (
                f"📊 *{symbol} Data Processed Successfully*\n"
                f"🕒 {ist_now_str()}\n"
                f"💾 Rows: {merged.shape[0]} | Cols: {merged.shape[1]}\n"
                f"📈 Target Ratio: {merged.get('target', pd.Series()).mean() * 100:.2f}%\n"
                f"📁 Cached: `{out_path}`"
            )
            send_telegram_message(msg, chat_id=TELEGRAM_CHAT_ID)

        return merged

    except Exception as e:
        logger.exception(f"⚠️ Error while processing {symbol}: {e}")
        if TELEGRAM_NOTIFY:
            send_telegram_message(f"❌ Processing failed for {symbol}\nError: {e}", chat_id=TELEGRAM_CHAT_ID)
        raise
