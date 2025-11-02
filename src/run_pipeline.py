# src/run_pipeline.py
import os
import argparse
import pandas as pd
from loguru import logger
from src.process_data import process_data
from src.train_model import train_xgb
from src.predict import predict_multi_timeframes
from src.signal_generator import generate_signals_from_model, pick_latest_signal
from src.news_sentiment import fetch_recent_news, summarize_sentiment
from src.utils.telegram_bot import send_telegram_message

def model_paths():
    return {"model": os.path.join("models", "xgb_model.joblib"), "scaler": os.path.join("models", "scaler.joblib")}

def safe_float(v, d=0.0):
    try:
        return float(v)
    except Exception:
        return d

def run_pipeline(force_fetch=False, force_retrain=False, dry_run=False):
    logger.info("🏁 Starting full AI-GOLD-TRADER pipeline...")
    df_processed = process_data("XAUUSD", notify=False, force_fetch=force_fetch)

    train_res = train_xgb(df_processed, force_retrain=force_retrain)
    acc = safe_float(train_res.get("accuracy", 0.0))

    preds = predict_multi_timeframes(model_paths(), df_processed)

    import pandas as pd
    if isinstance(preds, dict):
        logger.warning(f"⚠️ predict_multi_timeframes returned dict with {len(preds)} frames; merging.")
        preds = pd.concat(preds.values(), axis=0).sort_index()
        logger.success(f"✅ Merged prediction frames → shape={preds.shape}")

    signals_df = generate_signals_from_model(preds,
                                             thr_long=float(os.getenv("THR_LONG", 0.62)),
                                             thr_short=float(os.getenv("THR_SHORT", 0.38)),
                                             atr_period=int(os.getenv("ATR_PERIOD", 14)),
                                             sl_atr_mult=float(os.getenv("SL_ATR_MULT", 1.5)),
                                             tp_atr_mult=float(os.getenv("TP_ATR_MULT", 3.0)))
    latest = pick_latest_signal(signals_df)

    keywords = [k.strip().lower() for k in os.getenv("NEWS_KEYWORDS", "gold,xauusd,usd,inflation,cpi,fomc,powell").split(",")]
    lookback = int(os.getenv("NEWS_LOOKBACK_HOURS", 12))
    news_items = fetch_recent_news(keywords, lookback)
    sentiment_summary = summarize_sentiment(news_items)

    sentiment_label = sentiment_summary.get("label", "⚪ Neutral")
    avg_score = safe_float(sentiment_summary.get("avg_score", 0.0))
    news_count = sentiment_summary.get("raw_count", 0)

    ts = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    if latest:
        sig_block = (f"🔔 *{latest['signal']}*  (Confidence: `{latest['confidence']:.2f}`)\n"
                     f"💰 Price: `{latest['price']:.4f}` | ATR: `{latest['atr']:.4f}`\n"
                     f"⛔ SL: `{latest['sl']:.4f}` | 🎯 TP: `{latest['tp']:.4f}`\n"
                     f"🕒 Time: {latest['time']}")
    else:
        sig_block = "⚪ *NEUTRAL* — No actionable trade signal detected."

    message = ("📊 *AI GOLD TRADER — Update*\n\n"
               f"✅ *Model Accuracy:* `{acc:.4f}`\n\n"
               f"{sig_block}\n\n"
               f"📰 *Sentiment:* {sentiment_label} (score `{avg_score:.3f}`) — {news_count} news items\n"
               f"📁 Predictions saved → `data/predictions/pred_merged.csv`\n"
               f"🕒 Run Completed: {ts}")

    if not dry_run:
        send_telegram_message(message)
    logger.success("✅ Pipeline completed successfully.")
    return {"accuracy": acc, "signal": latest, "sentiment": sentiment_summary}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true", help="Force live fetch (ignore cache)")
    p.add_argument("--retrain", action="store_true", help="Force retrain models")
    p.add_argument("--dry", action="store_true", help="Dry run (no Telegram)")
    args = p.parse_args()
    run_pipeline(force_fetch=args.force, force_retrain=args.retrain, dry_run=args.dry)
