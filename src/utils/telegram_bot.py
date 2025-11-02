import os
import requests
from loguru import logger

def send_telegram_message(message: str, chat_id: str = None):
    """Send formatted Markdown message to Telegram."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        logger.warning("⚠️ Telegram credentials missing. Set TELEGRAM_BOT_TOKEN & TELEGRAM_CHAT_ID in .env")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}

    try:
        res = requests.post(url, data=data, timeout=10)
        if res.status_code == 200:
            logger.info("📨 Telegram message sent successfully.")
        else:
            logger.error(f"❌ Failed [{res.status_code}]: {res.text}")
    except Exception as e:
        logger.error(f"❌ Telegram send error: {e}")
