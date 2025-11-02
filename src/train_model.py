import os
import joblib
import pandas as pd
import numpy as np
from loguru import logger
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from src.utils.file_utils import model_paths
from src.utils.config import PROCESSED_DIR


def train_xgb(data: pd.DataFrame = None, model_params: dict = None):
    """
    Train an XGBoost model on processed data.

    Args:
        data (pd.DataFrame, optional): Preprocessed dataframe. If None, loads from file.
        model_params (dict, optional): Hyperparameters for XGBClassifier.

    Returns:
        dict: Paths to saved model and scaler, plus test accuracy.
    """
    # --- Load data if not provided ---
    if data is None:
        path = os.path.join(PROCESSED_DIR, "merged_all_timeframes.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Processed file not found: {path}")
        data = pd.read_csv(path, index_col=0)
        logger.info(f"📂 Loaded processed data from {path}")

    if data.empty:
        raise RuntimeError("❌ Processed data is empty, cannot train model")

    # --- Model parameters ---
    if model_params is None:
        model_params = {
            "n_estimators": 300,
            "max_depth": 7,
            "learning_rate": 0.05,
            "use_label_encoder": False,
            "eval_metric": "logloss",
        }

    # --- Create target ---
    ret_cols = [c for c in data.columns if c.startswith("ret1")]
    if not ret_cols:
        raise RuntimeError("❌ No 'ret1' columns found for labeling")

    data["target"] = (data[ret_cols[0]].shift(-1) > 0).astype(int)
    data.dropna(inplace=True)

    X = data.drop(columns=["target"])
    y = data["target"]

    # --- Handle missing values ---
    if X.isnull().values.any():
        logger.warning("⚠️ Missing values detected — filling with column means.")
        X = X.fillna(X.mean())

    # --- Scale features (keep column names) ---
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)

    # --- Split ---
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.15, shuffle=False)
    logger.info(f"🧩 Training shape: {X_train.shape}, Testing shape: {X_test.shape}")

    # --- Train model ---
    model = XGBClassifier(**model_params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # --- Save model & scaler ---
    paths = model_paths()
    joblib.dump(model, paths["model"])
    joblib.dump(scaler, paths["scaler"])

    # --- Evaluate ---
    score = model.score(X_test, y_test)
    logger.success(f"✅ XGB model trained. Accuracy: {score:.4f}")
    logger.info(f"📁 Model saved to {paths['model']}")
    logger.info(f"📁 Scaler saved to {paths['scaler']}")

    return {"model": paths["model"], "scaler": paths["scaler"], "accuracy": float(score)}
