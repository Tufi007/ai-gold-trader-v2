import os
from loguru import logger

from src.process_data import process_data
from src.train_model import train_xgb
from src.predict import predict_multi_timeframes
from src.utils.telegram_bot import send_telegram_message
from src.news_sentiment import fetch_recent_news, summarize_sentiment


def model_paths():
    """Return model/scaler paths"""
    return {
        "model": os.path.join("models", "xgb_model.joblib"),
        "scaler": os.path.join("models", "scaler.joblib"),
    }


def safe_float(value, default=0.0):
    """Convert safely to float (for logging and Telegram output)."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def run_pipeline():
    logger.info("🏁 Starting full AI-GOLD-TRADER pipeline...")

    # === STEP 1 & 2: Fetch + Process Data ===
    df_processed = process_data("XAUUSD", notify=False)

    # === STEP 3: Train Model ===
    train_result = train_xgb(df_processed)
    model_path = train_result.get("model")
    scaler_path = train_result.get("scaler")
    acc = safe_float(train_result.get("accuracy", 0.0))

    # === STEP 4: Predict ===
    preds = predict_multi_timeframes(model_paths(), df_processed)

    # === STEP 5: News Sentiment ===
    keywords = [
        k.strip().lower()
        for k in os.getenv(
            "NEWS_KEYWORDS", "gold,XAUUSD,USD,inflation,CPI,FOMC,Powell"
        ).split(",")
    ]
    lookback = int(os.getenv("NEWS_LOOKBACK_HOURS", "12"))

    news_items = fetch_recent_news(keywords, lookback)
    sentiment_summary = summarize_sentiment(news_items)

    sentiment_label = sentiment_summary.get("label", "⚪ Neutral")
    avg_score = safe_float(sentiment_summary.get("avg_score", 0.0))
    news_count = sentiment_summary.get("raw_count", 0)

    # === STEP 6: Telegram Update ===
    message = (
        "📊 *AI Gold Trader Pipeline Update*\n\n"
        f"✅ *Model Accuracy:* `{acc:.4f}`\n"
        f"📰 *Sentiment:* {sentiment_label}\n"
        f"📈 *Avg Sentiment Score:* `{avg_score:.3f}` ({news_count} news items)\n\n"
        "💾 *Predictions saved to:* `data/predictions/pred_merged.csv`"
    )

    send_telegram_message(message)
    logger.success("✅ Pipeline completed successfully.")


def main():
    try:
        run_pipeline()
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        send_telegram_message(f"❌ Pipeline failed:\n`{e}`")


if __name__ == "__main__":
    main()
