# src/signal_generator.py
import pandas as pd
import numpy as np
from loguru import logger
from pathlib import Path

# ======================================================
# Helper: Compute ATR (Average True Range)
# ======================================================
def compute_atr(df, period=14):
    """Compute ATR over DataFrame with columns high, low, close."""
    high = df['high']
    low = df['low']
    close = df['close']

    high_low = (high - low).abs()
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    return tr.rolling(period, min_periods=1).mean()


# ======================================================
# Core: Generate Trading Signals
# ======================================================
def generate_signals_from_model(pred_df: pd.DataFrame,
                                prob_col: str = 'prob_pos',
                                thr_long: float = 0.62,
                                thr_short: float = 0.38,
                                atr_period: int = 14,
                                sl_atr_mult: float = 1.5,
                                tp_atr_mult: float = 3.0) -> pd.DataFrame:
    """
    Generates BUY/SELL/NEUTRAL signals based on model predictions.

    Args:
        pred_df: DataFrame with predicted probabilities and ideally OHLC columns.
        prob_col: Column containing buy probability (0–1 range).
        thr_long: Probability threshold for BUY signal.
        thr_short: Probability threshold for SELL signal.
        atr_period: ATR window period.
        sl_atr_mult: Stop-loss multiplier (ATR-based).
        tp_atr_mult: Take-profit multiplier (ATR-based).

    Returns:
        DataFrame with appended columns:
            ['signal', 'confidence', 'atr', 'sl', 'tp']
    """
    df = pred_df.copy()

    # --------------------------------------------------
    # 1️⃣ Validate / Repair Missing Probability Column
    # --------------------------------------------------
    if prob_col not in df.columns:
        logger.warning(f"'{prob_col}' not found in predictions; defaulting to 0.5")
        df[prob_col] = 0.5

    # --------------------------------------------------
    # 2️⃣ Ensure OHLC Data Exists (critical for ATR/SL/TP)
    # --------------------------------------------------
    required_cols = ['close', 'high', 'low']
    missing_cols = [c for c in required_cols if c not in df.columns]

    if missing_cols:
        logger.warning(f"Missing OHLC columns: {missing_cols}. Attempting to restore from cached data...")
        cached_path = Path("data/XAUUSD_processed.csv")
        if cached_path.exists():
            try:
                cached = pd.read_csv(cached_path)
                for c in missing_cols:
                    if c in cached.columns:
                        df[c] = cached[c].tail(len(df)).values
                        logger.success(f"✅ Restored '{c}' from cached data.")
                    else:
                        df[c] = np.nan
                        logger.error(f"❌ '{c}' not found in cached data. Filled with NaN.")
            except Exception as e:
                logger.error(f"⚠️ Could not load cached data for OHLC recovery: {e}")
                for c in missing_cols:
                    df[c] = np.nan
        else:
            logger.error("❌ Cached file not found. Filling missing OHLC columns with NaN.")
            for c in missing_cols:
                df[c] = np.nan

    # --------------------------------------------------
    # 3️⃣ Determine Trading Signal (BUY / SELL / NEUTRAL)
    # --------------------------------------------------
    df['signal'] = 'NEUTRAL'
    df.loc[df[prob_col] >= thr_long, 'signal'] = 'BUY'
    df.loc[df[prob_col] <= thr_short, 'signal'] = 'SELL'

    # Confidence = normalized distance from neutral (0.5)
    df['confidence'] = df[prob_col].apply(lambda p: float(abs(p - 0.5) * 2))

    # --------------------------------------------------
    # 4️⃣ Compute ATR (fallback to volatility if OHLC missing)
    # --------------------------------------------------
    try:
        if {'high', 'low', 'close'}.issubset(df.columns):
            df['atr'] = compute_atr(df[['high', 'low', 'close']], period=atr_period)
        else:
            raise KeyError("Missing OHLC columns for ATR.")
    except Exception as e:
        logger.warning(f"⚠️ ATR computation failed ({e}). Using simple volatility fallback.")
        if 'close' in df.columns:
            df['atr'] = df['close'].pct_change().abs().rolling(atr_period, min_periods=1).mean() * df['close']
        else:
            df['atr'] = np.nan

    # --------------------------------------------------
    # 5️⃣ Compute Stop-Loss and Take-Profit Levels
    # --------------------------------------------------
    def compute_levels(row):
        if pd.isna(row['close']) or pd.isna(row['atr']):
            return pd.Series([np.nan, np.nan])

        if row['signal'] == 'BUY':
            sl = row['close'] - sl_atr_mult * row['atr']
            tp = row['close'] + tp_atr_mult * row['atr']
            return pd.Series([sl, tp])

        elif row['signal'] == 'SELL':
            sl = row['close'] + sl_atr_mult * row['atr']
            tp = row['close'] - tp_atr_mult * row['atr']
            return pd.Series([sl, tp])

        return pd.Series([np.nan, np.nan])

    df[['sl', 'tp']] = df.apply(compute_levels, axis=1)

    logger.success(f"🎯 Generated {len(df)} signals. BUY={sum(df['signal']=='BUY')}, SELL={sum(df['signal']=='SELL')}")
    return df


# ======================================================
# Helper: Get Latest Non-Neutral Signal
# ======================================================
def pick_latest_signal(df_signals: pd.DataFrame):
    """Return dictionary of the most recent non-neutral signal, or None."""
    if df_signals is None or df_signals.empty:
        logger.warning("⚠️ No signal data available.")
        return None

    non_neutral = df_signals[df_signals['signal'] != 'NEUTRAL']
    if non_neutral.empty:
        logger.info("ℹ️ No actionable signals found (all NEUTRAL).")
        return None

    last = non_neutral.iloc[-1]
    logger.info(f"📈 Latest signal: {last['signal']} @ {last.get('close', 'N/A')}")

    return {
        "time": last.name if last.name is not None else None,
        "signal": str(last['signal']),
        "confidence": float(last['confidence']),
        "price": float(last['close']) if pd.notnull(last['close']) else None,
        "sl": float(last['sl']) if pd.notnull(last['sl']) else None,
        "tp": float(last['tp']) if pd.notnull(last['tp']) else None,
        "atr": float(last['atr']) if pd.notnull(last['atr']) else None,
    }
