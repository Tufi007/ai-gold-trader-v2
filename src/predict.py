# src/predict.py
import os
import joblib
import pandas as pd
import numpy as np
from loguru import logger
from src.utils.file_utils import model_paths
from src.utils.config import PREDICTIONS_DIR

def load_models(paths: dict):
    models = {}
    try:
        models['xgb'] = joblib.load(paths['model'])
        logger.info("✅ Loaded XGBoost model")
    except Exception as e:
        logger.warning(f"XGB load failed: {e}")

    # placeholder for LSTM / torch model (optional)
    # if os.path.exists(paths.get('lstm')):
    #    load model

    return models

def predict_multi_timeframes(paths: dict, processed_df: pd.DataFrame):
    """
    Accepts processed df and returns a merged DataFrame with:
    - original OHLC columns (close/high/low)
    - model outputs: prob_pos, pred_label
    """
    if processed_df is None or processed_df.empty:
        raise RuntimeError("No processed data provided for prediction")

    models = load_models(paths)
    xgb = models.get('xgb')

    X = processed_df.copy()
    # remove target if exists
    for col in ['target']:
        if col in X.columns:
            X = X.drop(columns=[col])

    # --- compute XGB preds if available
    preds_df = X.copy()
    if xgb is not None:
        try:
            probs = xgb.predict_proba(X)[:, 1]
            preds_df['prob_pos'] = probs
            preds_df['pred_label'] = (probs >= 0.5).astype(int)
            logger.info(f"✅ XGB predictions computed ({len(probs)} rows)")
        except Exception as e:
            logger.exception(f"XGB prediction failed: {e}")
            preds_df['prob_pos'] = 0.5
            preds_df['pred_label'] = 0
    else:
        preds_df['prob_pos'] = 0.5
        preds_df['pred_label'] = 0

    # --- ensemble placeholder: if other model exists, combine them
    # For now, we just ensure prob_pos in 0..1
    preds_df['prob_pos'] = preds_df['prob_pos'].clip(0.0, 1.0)

    # --- ensure OHLC present: try to restore from processed_df if missing
    for c in ['open','high','low','close','volume']:
        if c not in preds_df.columns and c in processed_df.columns:
            preds_df[c] = processed_df[c]

    # --- final cleanup & save
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    out_path = os.path.join(PREDICTIONS_DIR, "pred_merged.csv")
    try:
        preds_df.to_csv(out_path)
        logger.success(f"📁 Saved predictions → {out_path}")
    except Exception as e:
        logger.warning(f"Could not save predictions to {out_path}: {e}")

    # return as dict of timeframes? For compatibility we return DataFrame
    return preds_df
