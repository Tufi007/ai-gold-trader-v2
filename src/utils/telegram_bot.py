# src/utils/telegram_bot.py
import os
import time
import requests
from loguru import logger

TELEGRAM_API = "https://api.telegram.org"

def _escape_markdown(text: str) -> str:
    # Basic MarkdownV2 escaping for common special chars
    if not text:
        return text
    for ch in ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']:
        text = text.replace(ch, f"\\{ch}")
    return text

def send_telegram_message(message: str, chat_id: str = None, parse_mode="MarkdownV2", retries: int = 2):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        logger.warning("⚠️ Telegram credentials missing. Set TELEGRAM_BOT_TOKEN & TELEGRAM_CHAT_ID in .env")
        return False

    url = f"{TELEGRAM_API}/bot{token}/sendMessage"
    safe_msg = _escape_markdown(message) if parse_mode == "MarkdownV2" else message
    data = {"chat_id": chat_id, "text": safe_msg, "parse_mode": parse_mode}

    for i in range(retries + 1):
        try:
            res = requests.post(url, data=data, timeout=10)
            if res.status_code == 200:
                logger.info("📨 Telegram message sent successfully.")
                return True
            else:
                logger.error(f"❌ Telegram failed [{res.status_code}]: {res.text}")
        except Exception as e:
            logger.error(f"❌ Telegram send error: {e}")
        time.sleep(2 ** i)
    return False
