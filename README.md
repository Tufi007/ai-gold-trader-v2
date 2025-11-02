# AI-GOLD-TRADER v2

Production-ready, multi-timeframe gold trading pipeline.

## Features
- Multi-timeframe data (5m,15m,30m,1h,4h) via TradingView, fallback to yfinance
- Indicator engineering (ATR, ADX, EMA, RSI, MACD, returns)
- Merge/align stacked features per-base timeframe
- Train XGBoost model (optional deep learning)
- Predict latest signal and send Telegram alerts
- MT5 integration on Windows (auto-enable if MetaTrader5 python package is available)

## Quickstart
1. Create and activate venv
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
