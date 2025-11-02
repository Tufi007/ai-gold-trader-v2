import os
from loguru import logger
from src.process_data import process_data
from src.train_model import train_xgb
from src.predict import predict_multi_timeframes
from src.utils.telegram_bot import send_telegram_message
from src.news_sentiment import fetch_recent_news, analyze_sentiment

def model_paths():
    """Return model/scaler paths"""
    return {
        "model": os.path.join("models", "xgb_model.joblib"),
        "scaler": os.path.join("models", "scaler.joblib"),
    }

def run_pipeline():
    logger.info("🏁 Starting full AI-GOLD-TRADER pipeline...")

    # === STEP 1 & 2: Fetch + Process Data ===
    df_processed = process_data("XAUUSD", notify=False)

    # === STEP 3: Train Model ===
    model, scaler, acc = train_xgb(df_processed)

    # === STEP 4: Predict ===
    preds = predict_multi_timeframes(model_paths(), df_processed)

    # === STEP 5: News Sentiment ===
    keywords = os.getenv("NEWS_KEYWORDS", "gold,XAUUSD,USD,inflation,CPI,FOMC,Powell").split(",")
    lookback = int(os.getenv("NEWS_LOOKBACK_HOURS", "12"))
    use_transformer = os.getenv("USE_TRANSFORMER_FOR_NEWS", "false").lower() == "true"

    news_items = fetch_recent_news(keywords, lookback)
    _, sentiment_summary = analyze_sentiment(news_items, use_transformer)

    # === STEP 6: Telegram Update ===
    message = (
        "📊 *AI Gold Trader Update*\n\n"
        f"✅ Model Accuracy: `{acc:.4f}`\n"
        f"📰 Sentiment: {sentiment_summary}\n"
        f"📈 Predictions saved to `data/predictions/pred_merged.csv`"
    )
    send_telegram_message(message)
    logger.success("✅ Pipeline completed successfully.")

def main():
    try:
        run_pipeline()
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        send_telegram_message(f"❌ Pipeline failed: {e}")

if __name__ == "__main__":
    main()
