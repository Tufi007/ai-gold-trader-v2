# src/train_model.py
import pandas as pd
import numpy as np
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import joblib
from src.utils.file_utils import model_paths
from src.utils.config import MODELS_DIR

def train_xgb(df: pd.DataFrame, force_retrain=False):
    """
    Train an XGBoost model on processed data.
    Automatically skips empty or invalid datasets.
    """
    logger.info("🤖 Starting XGBoost training...")

    # Ensure target exists
    if "target" not in df.columns:
        logger.error("❌ 'target' column missing in processed data.")
        return None

    # Drop NaNs
    df = df.dropna()
    if df.empty:
        logger.error("❌ Processed DataFrame is empty after dropping NaNs. Skipping training.")
        return None

    # Split features/target
    X = df.drop(columns=["target"])
    y = df["target"]

    # Filter numeric only (non-numeric cause issues in sklearn)
    X = X.select_dtypes(include=[np.number])
    if X.empty:
        logger.error("❌ No numeric columns available for training.")
        return None

    # Drop constant columns
    nunique = X.nunique()
    X = X.loc[:, nunique > 1]

    if X.empty:
        logger.error("❌ All features constant or invalid — nothing to train.")
        return None

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale
    scaler = StandardScaler()
    try:
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    except ValueError as e:
        logger.error(f"❌ Scaling failed: {e}")
        return None

    # Initialize model
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    )

    # Train
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    logger.success(f"✅ XGB trained successfully! Accuracy = {acc:.4f}")

    # Save model + scaler
    paths = model_paths()
    joblib.dump(model, paths["model"])
    joblib.dump(scaler, paths["scaler"])
    logger.success(f"💾 Saved model → {paths['model']}")
    logger.success(f"💾 Saved scaler → {paths['scaler']}")

    return {"acc": acc, "model_path": paths["model"], "scaler_path": paths["scaler"]}
