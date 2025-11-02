import os
import pandas as pd
import joblib
from loguru import logger
from pathlib import Path


def load_model_assets(model_info: dict):
    """
    Load trained model and scaler from disk or from model_info dict.
    """
    model = model_info.get("model")
    scaler = model_info.get("scaler")

    # Load from file paths if not in-memory
    if isinstance(model, str) and Path(model).exists():
        model = joblib.load(model)
    if isinstance(scaler, str) and Path(scaler).exists():
        scaler = joblib.load(scaler)

    if model is None or scaler is None:
        raise ValueError("❌ Model or scaler missing — cannot proceed with prediction.")

    return model, scaler


def predict_multi_timeframes(model_info, processed_data):
    """
    Predict across multiple timeframes or a single merged dataframe.
    Supports flexible input types for production stability.
    """

    logger.info("🔮 Starting predictions across timeframes...")

    # Load trained model and scaler
    model, scaler = load_model_assets(model_info)

    # Handle both single-DF or dict-of-DFs
    if isinstance(processed_data, pd.DataFrame):
        data_dict = {"merged": processed_data}
    elif isinstance(processed_data, dict):
        data_dict = processed_data
    else:
        raise TypeError("❌ processed_data must be DataFrame or dict of DataFrames.")

    predictions = {}

    for tf, df in data_dict.items():
        if df.empty:
            logger.warning(f"⚠️ Skipping empty dataframe for {tf}.")
            continue

        # Prepare features
        X = df.drop(columns=["target"], errors="ignore")
        X_scaled = scaler.transform(X)

        # Predict probabilities or classes
        y_pred = model.predict(X_scaled)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_scaled)[:, 1]
            df["pred_proba"] = y_prob

        df["prediction"] = y_pred
        predictions[tf] = df.tail(5)  # keep last few rows for quick view

        logger.info(f"✅ Predictions complete for {tf} (rows={len(df)})")

    # Optionally save predictions
    output_dir = Path("data/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)

    for tf, df in predictions.items():
        save_path = output_dir / f"pred_{tf}.csv"
        df.to_csv(save_path, index=False)
        logger.success(f"📁 Saved predictions → {save_path}")

    logger.success("🎯 All predictions completed successfully.")
    return predictions
