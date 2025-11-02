# in src/utils/file_utils.py
import pandas as pd
from pathlib import Path
import os, joblib

def save_df_csv(df, path, index=False):
    """Save a DataFrame safely."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=index)
    return path


def raw_filepath(symbol: str, timeframe: str) -> str:
    """
    Return the path for raw data for a given symbol and timeframe.
    Example: data/raw/XAUUSD_15m.csv
    """
    root = Path(os.getenv("DATA_DIR", "data")) / "raw"
    root.mkdir(parents=True, exist_ok=True)
    return str(root / f"{symbol}_{timeframe}.csv")


def processed_path(symbol):
    """Return processed CSV path for symbol."""
    os.makedirs("data", exist_ok=True)
    return f"data/{symbol}_processed.csv"

def model_paths():
    root = Path(os.getenv("MODELS_DIR", "models"))
    root.mkdir(parents=True, exist_ok=True)
    return {"model": str(root / "xgb_model.joblib"), "scaler": str(root / "scaler.joblib")}

def load_model(path: str):
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)

def load_scaler(path: str):
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Scaler not found: {path}")
    return joblib.load(path)
