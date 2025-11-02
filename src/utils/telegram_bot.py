import os
import requests
import html
from loguru import logger

def send_telegram_message(message: str, chat_id: str = None):
    """
    Send a formatted Telegram message using safe HTML escaping.
    Prevents Markdown/HTML parse errors and ensures robust logging.
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        logger.warning("⚠️ Telegram credentials missing. Set TELEGRAM_BOT_TOKEN & TELEGRAM_CHAT_ID in .env")
        return

    # Escape unsafe characters to prevent parse issues
    safe_message = html.escape(str(message))

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {
        "chat_id": chat_id,
        "text": safe_message,
        "parse_mode": "HTML",  # safer than Markdown
        "disable_web_page_preview": True,
    }

    try:
        response = requests.post(url, data=data, timeout=10)
        if response.status_code == 200:
            logger.info("📨 Telegram message sent successfully.")
        else:
            logger.error(f"❌ Telegram send failed [{response.status_code}]: {response.text}")
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Telegram network error: {e}")
    except Exception as e:
        logger.error(f"❌ Unexpected Telegram error: {e}")
