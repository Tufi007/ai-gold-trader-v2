import os
import requests
from datetime import datetime
import pytz
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def format_prediction_message(predictions, current_price):
    """Formats the final Telegram message for all timeframes."""
    ist_time = datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y-%m-%d | %H:%M:%S IST")

    header = f"🏁 AI-GOLD-TRADER UPDATE [{ist_time}]\n\n"
    price_info = f"📈 Current Price (XAU/USD): ${current_price:.2f}\n\n"
    body = "🕒 Predictions:\n"

    for tf, pred in predictions.items():
        direction = "🟢 BUY" if pred['signal'] == "BUY" else "🔴 SELL"
        conf = pred.get("confidence", 0)
        body += f"• {tf}: {direction} (Conf: {conf:.1f}%)\n"

    # Bias summary
    buy_count = sum(1 for v in predictions.values() if v['signal'] == "BUY")
    sell_count = sum(1 for v in predictions.values() if v['signal'] == "SELL")

    if buy_count > sell_count:
        summary = "\n📊 Summary: Market bias leaning 🟢 Bullish.\n"
    elif sell_count > buy_count:
        summary = "\n📊 Summary: Market bias leaning 🔴 Bearish.\n"
    else:
        summary = "\n📊 Summary: Neutral / Mixed signals ⚖️\n"

    interval = os.getenv("AUTO_RUN_INTERVAL_MINUTES", "30")
    footer = f"\nNext auto-run in {interval} minutes ⏱️"

    return header + price_info + body + summary + footer


def send_telegram_message(message):
    """Sends a formatted message to Telegram bot."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram credentials missing in .env file")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}

    try:
        response = requests.post(url, data=data)
        response.raise_for_status()
        print("✅ Telegram message sent.")
    except Exception as e:
        print(f"❌ Telegram send failed: {e}")
