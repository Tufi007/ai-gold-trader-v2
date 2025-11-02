import os
from datetime import datetime
from loguru import logger

from src.process_data import process_data
from src.train_model import train_xgb
from src.predict import predict_multi_timeframes
from src.utils.file_utils import model_paths
from src.utils.telegram_bot import send_telegram_message
from src.news_sentiment import news_sentiment_summary

# ─────────────────────────────────────────────
# Combine AI + News Sentiment + Volume
# ─────────────────────────────────────────────
def merge_signals(ai_confidence, ai_label, news_score, volume_factor=1.0):
    ai_weight, news_weight = 0.7, 0.3
    sign = 1 if ai_label.lower().startswith("buy") else -1 if ai_label.lower().startswith("sell") else 0
    news_aligned = max(0.0, (news_score * sign + 1) / 2)
    combined = (ai_weight * ai_confidence + news_weight * news_aligned) * volume_factor
    return round(min(max(combined, 0.0), 1.0), 3)

# ─────────────────────────────────────────────
# Build Telegram Signal Message
# ─────────────────────────────────────────────
def compose_telegram_signal(symbol, ai_label, ai_conf, news_summary, strength, extra):
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    msg = [
        f"🤖 *AI-GOLD-TRADER SIGNAL*",
        "━━━━━━━━━━━━━━━━━━━━━━━",
        f"🪙 *Symbol:* {symbol}",
        f"🕒 *Time:* {now}",
        "━━━━━━━━━━━━━━━━━━━━━━━",
        f"📈 *AI Prediction:* {ai_label} ({ai_conf*100:.1f}% confidence)",
        f"📰 *News Sentiment:* {news_summary['label']} ({news_summary['avg_score']:+.2f}) — {news_summary['raw_count']} sources"
    ]

    if extra.get("price"): msg.append(f"💲 *Price:* {extra['price']}")
    if extra.get("trend"): msg.append(f"📊 *Trend:* {extra['trend']}")
    if extra.get("session"): msg.append(f"🌍 *Session:* {extra['session']}")

    bar = "█" * int(strength * 10) + "░" * (10 - int(strength * 10))
    msg.append("━━━━━━━━━━━━━━━━━━━━━━━")
    msg.append(f"💪 *Signal Strength:* `{bar}` ({strength:.2f})")

    headlines = news_summary.get("breakdown", {}).get("details", [])
    if headlines:
        msg.append("🗞️ *Top Headlines:*")
        for h in headlines[:3]:
            msg.append(f"• {h.get('title', '')[:100]} ({h.get('score', 0.0):+.2f})")

    msg.append("━━━━━━━━━━━━━━━━━━━━━━━")
    msg.append("⚙️ *Meta:* AI + News + Volume + Session fusion.")
    msg.append("⚡ Stay disciplined — manage risk ⚡")
    return "\n".join(msg)

# ─────────────────────────────────────────────
# Core Bot Runner
# ─────────────────────────────────────────────
def run_telegram_signal_bot(symbol="XAUUSD"):
    logger.info("🚀 Generating new AI-GOLD-TRADER signal...")

    processed = process_data(symbol=symbol, notify=False)
    paths = model_paths()
    model_path, scaler_path = paths.get("model"), paths.get("scaler")

    model_info = {"model": model_path, "scaler": scaler_path} if os.path.exists(model_path) else train_xgb(processed)
    preds = predict_multi_timeframes(model_info=model_info, processed_data=processed)
    merged_df = preds.get("merged")

    if merged_df is None or merged_df.empty:
        logger.error("❌ No predictions available.")
        return

    last_row = merged_df.iloc[-1]
    ai_label = "BUY" if int(last_row.get("prediction", 0)) == 1 else "SELL" if int(last_row.get("prediction", 0)) == -1 else "NEUTRAL"
    ai_conf = float(last_row.get("pred_proba", 0.5))

    news_summary = news_sentiment_summary()
    news_score = float(news_summary.get("avg_score", 0.0))

    # Volume factor
    volume_factor = 1.0
    try:
        vol, avg_vol = float(last_row["volume"]), float(merged_df["volume"].tail(200).mean())
        if vol > 1.5 * avg_vol: volume_factor = 1.15
        elif vol < 0.6 * avg_vol: volume_factor = 0.9
    except Exception: pass

    strength = merge_signals(ai_conf, ai_label, news_score, volume_factor)

    # Trend + Session context
    extra = {}
    try:
        c, sma20 = float(last_row["close"]), float(last_row.get("close_20", c))
        extra["trend"] = "⬆️ Uptrend" if c > sma20 else "⬇️ Downtrend" if c < sma20 else "➡️ Sideways"
        extra["price"] = c
    except Exception:
        pass

    hour = datetime.utcnow().hour
    extra["session"] = "🗽 New York" if 13 <= hour < 22 else "💹 London" if 7 <= hour < 16 else "🈺 Tokyo" if 0 <= hour < 9 else "🇦🇺 Sydney"

    msg = compose_telegram_signal(symbol, ai_label, ai_conf, news_summary, strength, extra)
    send_telegram_message(msg)
    logger.success("✅ Signal successfully sent to Telegram.")

if __name__ == "__main__":
    run_telegram_signal_bot()
