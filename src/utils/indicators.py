# src/utils/indicators.py
"""
Production-grade technical indicator library.

compute_indicators(df, suffix="") -> pd.DataFrame
- df must have (lowercase) columns: open, high, low, close, volume
- suffix is appended to column names (e.g. "_5m", "_1h")
- returns DataFrame indexed same as input with numeric columns (many NaNs at series head)
"""
from __future__ import annotations
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Optional libs (ta, finta). If not present, fallback to pandas implementations
try:
    from ta.volatility import AverageTrueRange, BollingerBands
    from ta.trend import EMAIndicator, ADXIndicator, MACD, SMAIndicator
    from ta.momentum import RSIIndicator, StochasticOscillator
    TA_LIB_AVAILABLE = True
except Exception:
    TA_LIB_AVAILABLE = False
    logger.debug("ta library not available; using pandas fallbacks where possible.")

try:
    from finta import TA as FINTA
    FINTA_AVAILABLE = True
except Exception:
    FINTA_AVAILABLE = False
    logger.debug("finta not available; fallback to pandas implementations where possible.")


def _safe_series(series) -> pd.Series:
    """Ensure numeric pandas Series and copy to avoid mutating inputs."""
    s = pd.to_numeric(series, errors="coerce").copy()
    return s


def _ema(series: pd.Series, span: int) -> pd.Series:
    try:
        if TA_LIB_AVAILABLE:
            return EMAIndicator(series, window=span).ema_indicator()
    except Exception as e:
        logger.debug("ta.EMA failed: %s", e)
    # fallback
    return series.ewm(span=span, adjust=False).mean()


def _sma(series: pd.Series, window: int) -> pd.Series:
    try:
        if TA_LIB_AVAILABLE:
            return SMAIndicator(series, window=window).sma_indicator()
    except Exception:
        pass
    return series.rolling(window=window, min_periods=1).mean()


def _atr_df(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    try:
        if TA_LIB_AVAILABLE:
            return AverageTrueRange(high, low, close, window=window).average_true_range()
    except Exception:
        # manual ATR
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=window, min_periods=1).mean()


def _adx_df(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14):
    """Return adx, pdi, ndi as tuple of Series"""
    if TA_LIB_AVAILABLE:
        try:
            adx = ADXIndicator(high, low, close, window=window)
            return adx.adx(), adx.adx_pos(), adx.adx_neg()
        except Exception as e:
            logger.debug("ta.ADX failed: %s", e)
    # fallback approx: use finta if available
    if FINTA_AVAILABLE:
        try:
            adx = FINTA.ADX(pd.concat([high, low, close], axis=1).rename(columns={0:"open",1:"high",2:"low"}))
            # finta's ADX output may be different; as a last resort return NaN
        except Exception:
            pass
    return pd.Series(np.nan, index=close.index), pd.Series(np.nan, index=close.index), pd.Series(np.nan, index=close.index)


def _macd_df(close: pd.Series):
    try:
        if TA_LIB_AVAILABLE:
            macd = MACD(close)
            return macd.macd(), macd.macd_signal(), macd.macd_diff()
    except Exception as e:
        logger.debug("ta.MACD failed: %s", e)
    # fallback
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    diff = macd_line - signal
    return macd_line, signal, diff


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    try:
        if TA_LIB_AVAILABLE:
            return RSIIndicator(close, window=window).rsi()
    except Exception as e:
        logger.debug("ta.RSI failed: %s", e)
    # fallback: manual RSI
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/window, adjust=False).mean()
    ma_down = down.ewm(alpha=1/window, adjust=False).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_window=14, d_window=3):
    try:
        if TA_LIB_AVAILABLE:
            st = StochasticOscillator(close=close, high=high, low=low, window=k_window, smooth_window=d_window)
            return st.stoch(), st.stoch_signal()
    except Exception:
        pass
    low_min = low.rolling(k_window, min_periods=1).min()
    high_max = high.rolling(k_window, min_periods=1).max()
    k = 100 * ((close - low_min) / (high_max - low_min + 1e-12))
    d = k.rolling(d_window, min_periods=1).mean()
    return k, d


def _cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    if FINTA_AVAILABLE:
        try:
            # finta expects a DataFrame with Open/High/Low/Close/Volume
            tmp = pd.DataFrame({"open": close, "high": high, "low": low, "close": close})
            return FINTA.CCI(tmp, window)
        except Exception:
            pass
    # fallback
    tp = (high + low + close) / 3.0
    tp_sma = tp.rolling(window, min_periods=1).mean()
    mad = (tp - tp_sma).abs().rolling(window, min_periods=1).mean()
    cci = (tp - tp_sma) / (0.015 * mad.replace(0, np.nan))
    return cci


def _on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
    # OBV: add volume when close up else subtract when down
    direction = np.sign(close.diff().fillna(0))
    obv = (direction * volume).fillna(0).cumsum()
    return obv


def _supertrend(high: pd.Series, low: pd.Series, close: pd.Series, atr_period: int = 10, multiplier: float = 3.0):
    """
    SuperTrend implementation:
    - compute ATR
    - compute basic upper/lower band: (high+low)/2 +/- multiplier*ATR
    - final bands: if current upper < previous final upper and close_prev > prev_final_upper -> ...
    Returns: supertrend (trend value), supertrend_direction (1 for up, -1 for down)
    """
    atr = _atr_df(high, low, close, window=atr_period)
    hl2 = (high + low) / 2.0
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)

    final_upper = upperband.copy()
    final_lower = lowerband.copy()

    for i in range(1, len(close)):
        # final upper cannot be lower than previous final upper unless price closed above it
        if (upperband.iat[i] < final_upper.iat[i - 1]) or (close.iat[i - 1] > final_upper.iat[i - 1]):
            final_upper.iat[i] = upperband.iat[i]
        else:
            final_upper.iat[i] = final_upper.iat[i - 1]

        if (lowerband.iat[i] > final_lower.iat[i - 1]) or (close.iat[i - 1] < final_lower.iat[i - 1]):
            final_lower.iat[i] = lowerband.iat[i]
        else:
            final_lower.iat[i] = final_lower.iat[i - 1]

    # determine trend
    trend = pd.Series(1, index=close.index)  # default up
    for i in range(1, len(close)):
        if close.iat[i] > final_upper.iat[i - 1]:
            trend.iat[i] = 1
        elif close.iat[i] < final_lower.iat[i - 1]:
            trend.iat[i] = -1
        else:
            trend.iat[i] = trend.iat[i - 1]

    # supertrend value is final_lower for uptrend and final_upper for downtrend
    st_value = pd.Series(np.where(trend == 1, final_lower, final_upper), index=close.index)
    return st_value, trend


def compute_indicators(df: pd.DataFrame, suffix: Optional[str] = "") -> pd.DataFrame:
    """
    High-level function to compute a comprehensive set of indicators.
    Returns a DataFrame with suffix appended to each column (e.g. 'rsi_5m' or 'rsi_1h').
    """
    if df is None or df.empty:
        return pd.DataFrame()

    s = f"{suffix}" if suffix and suffix.startswith("_") else (("_" + suffix) if suffix else "")
    out = pd.DataFrame(index=df.index)

    # Ensure lowercase column names for compatibility
    df = df.rename(columns={c: c.lower() for c in df.columns})
    # Coerce to numeric where applicable
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    try:
        high = _safe_series(df["high"]) if "high" in df else pd.Series(np.nan, index=df.index)
        low = _safe_series(df["low"]) if "low" in df else pd.Series(np.nan, index=df.index)
        close = _safe_series(df["close"]) if "close" in df else pd.Series(np.nan, index=df.index)
        open_ = _safe_series(df.get("open", close))
        vol = _safe_series(df.get("volume", pd.Series(0, index=df.index)))
    except Exception as e:
        logger.exception("Failed to prepare series for indicators: %s", e)
        return out

    # ATR
    try:
        out[f"atr{s}"] = _atr_df(high, low, close, window=14)
    except Exception as e:
        logger.debug("ATR failed: %s", e); out[f"atr{s}"] = np.nan

    # ADX, +DI, -DI
    try:
        adx, pdi, ndi = _adx_df(high, low, close, window=14)
        out[f"adx{s}"] = adx
        out[f"pdi{s}"] = pdi
        out[f"ndi{s}"] = ndi
    except Exception as e:
        logger.debug("ADX family failed: %s", e)
        out[[f"adx{s}", f"pdi{s}", f"ndi{s}"]] = np.nan

    # RSI
    try:
        out[f"rsi{s}"] = _rsi(close, window=14)
    except Exception as e:
        logger.debug("RSI failed: %s", e); out[f"rsi{s}"] = np.nan

    # EMAs & SMA
    try:
        out[f"ema12{s}"] = _ema(close, 12)
        out[f"ema26{s}"] = _ema(close, 26)
        out[f"ema_diff{s}"] = out[f"ema12{s}"] - out[f"ema26{s}"]
        out[f"sma50{s}"] = _sma(close, 50)
        out[f"sma200{s}"] = _sma(close, 200)
    except Exception as e:
        logger.debug("EMA/SMA failed: %s", e)
        out[[f"ema12{s}", f"ema26{s}", f"ema_diff{s}", f"sma50{s}", f"sma200{s}"]] = np.nan

    # MACD
    try:
        macd, macd_sig, macd_diff = _macd_df(close)
        out[f"macd{s}"] = macd
        out[f"macd_sig{s}"] = macd_sig
        out[f"macd_diff{s}"] = macd_diff
    except Exception as e:
        logger.debug("MACD failed: %s", e); out[[f"macd{s}", f"macd_sig{s}", f"macd_diff{s}"]] = np.nan

    # Bollinger Bands (20,2) and bandwidth
    try:
        if TA_LIB_AVAILABLE:
            bb = BollingerBands(close, window=20, window_dev=2)
            out[f"bb_mid{s}"] = bb.bollinger_mavg()
            out[f"bb_high{s}"] = bb.bollinger_hband()
            out[f"bb_low{s}"] = bb.bollinger_lband()
            out[f"bb_width{s}"] = (out[f"bb_high{s}"] - out[f"bb_low{s}"]) / (out[f"bb_mid{s}"].replace(0, np.nan))
        else:
            mid = close.rolling(20, min_periods=1).mean()
            std = close.rolling(20, min_periods=1).std()
            out[f"bb_mid{s}"] = mid
            out[f"bb_high{s}"] = mid + 2 * std
            out[f"bb_low{s}"] = mid - 2 * std
            out[f"bb_width{s}"] = (out[f"bb_high{s}"] - out[f"bb_low{s}"]) / (out[f"bb_mid{s}"].replace(0, np.nan))
    except Exception as e:
        logger.debug("BBands failed: %s", e)
        out[[f"bb_mid{s}", f"bb_high{s}", f"bb_low{s}", f"bb_width{s}"]] = np.nan

    # Stochastic
    try:
        st_k, st_d = _stochastic(high, low, close, k_window=14, d_window=3)
        out[f"stoch_k{s}"] = st_k
        out[f"stoch_d{s}"] = st_d
    except Exception as e:
        logger.debug("Stochastic failed: %s", e); out[[f"stoch_k{s}", f"stoch_d{s}"]] = np.nan

    # CCI
    try:
        out[f"cci{s}"] = _cci(high, low, close, window=20)
    except Exception as e:
        logger.debug("CCI failed: %s", e); out[f"cci{s}"] = np.nan

    # OBV
    try:
        out[f"obv{s}"] = _on_balance_volume(close, vol)
    except Exception as e:
        logger.debug("OBV failed: %s", e); out[f"obv{s}"] = np.nan

    # SuperTrend
    try:
        st_val, st_dir = _supertrend(high, low, close, atr_period=10, multiplier=3.0)
        out[f"supertrend{s}"] = st_val
        out[f"super_dir{s}"] = st_dir
    except Exception as e:
        logger.debug("Supertrend failed: %s", e); out[[f"supertrend{s}", f"super_dir{s}"]] = np.nan

    # Simple returns and volatility
    try:
        out[f"ret1{s}"] = close.pct_change(1)
        out[f"ret5{s}"] = close.pct_change(5)
        out[f"vol20{s}"] = vol.rolling(window=20, min_periods=1).mean()
        out[f"hl_range{s}"] = (high - low) / close.replace(0, np.nan)
    except Exception as e:
        logger.debug("Return/vol features failed: %s", e)

    # Replace infinite values and enforce float dtype
    out = out.replace([np.inf, -np.inf], np.nan)
    # ensure float dtype for numeric cols
    for c in out.columns:
        try:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        except Exception:
            pass

    return out
