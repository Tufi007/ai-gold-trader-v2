# src/utils/file_utils.py
import os
import pandas as pd
from loguru import logger
from datetime import datetime
from src.utils.config import DATA_DIR, PROCESSED_DIR, MODELS_DIR, RAW_DIR

def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path

def timestamp_str() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def raw_filepath(symbol: str, tf: str) -> str:
    """Return path for raw data file for symbol/timeframe."""
    ensure_dir(f"{RAW_DIR}/dummy.txt")
    return os.path.join(RAW_DIR, f"{symbol}_{tf}.csv")

def processed_path(symbol: str) -> str:
    """Return full path to processed CSV."""
    ensure_dir(f"{PROCESSED_DIR}/dummy.txt")
    return os.path.join(PROCESSED_DIR, f"{symbol}_processed.csv")

def model_paths() -> dict:
    """Return model and scaler paths."""
    ensure_dir(f"{MODELS_DIR}/dummy.txt")
    return {
        "model": os.path.join(MODELS_DIR, f"xgb_{timestamp_str()}.joblib"),
        "scaler": os.path.join(MODELS_DIR, f"scaler_{timestamp_str()}.joblib")
    }

def save_df_csv(df: pd.DataFrame, path: str, index: bool = False):
    """
    Save a DataFrame to CSV (optionally with index).
    Automatically ensures directory exists.
    """
    try:
        ensure_dir(path)
        df.to_csv(path, index=index)
        logger.success(f"💾 Saved CSV → {path} (rows={df.shape[0]}, cols={df.shape[1]})")
    except Exception as e:
        logger.exception(f"❌ Failed to save CSV {path}: {e}")
        raise

def save_df_parquet(df: pd.DataFrame, path: str, index: bool = False):
    """Save a DataFrame to Parquet format."""
    try:
        ensure_dir(path)
        df.to_parquet(path, index=index)
        logger.success(f"💾 Saved Parquet → {path} (rows={df.shape[0]}, cols={df.shape[1]})")
    except Exception as e:
        logger.exception(f"❌ Failed to save Parquet {path}: {e}")
        raise
