import os
import feedparser
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from loguru import logger
from transformers import pipeline

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
KEYWORDS = [k.strip().lower() for k in os.getenv(
    "NEWS_KEYWORDS", "gold,xauusd,usd,inflation,cpi,fomc,powell"
).split(",")]

LOOKBACK_HOURS = int(os.getenv("NEWS_LOOKBACK_HOURS", 12))
USE_TRANSFORMER = os.getenv("USE_TRANSFORMER_FOR_NEWS", "false").lower() == "true"
MODEL_NAME = os.getenv("TRANSFORMER_MODEL_NAME", "distilbert-base-uncased-finetuned-sst-2-english")

NEWS_FEEDS = {
    "TradingView": "https://www.tradingview.com/news/rss/",
    "Investing.com": "https://www.investing.com/rss/news_301.rss",
    "ForexFactory": "https://www.forexfactory.com/ff_calendar_thisweek.xml",
    "MarketWatch": "https://feeds.marketwatch.com/marketwatch/topstories/",
}

# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────
def clean_html(raw_html: str) -> str:
    """Strip HTML tags from text safely."""
    try:
        soup = BeautifulSoup(raw_html, "html.parser")
        return soup.get_text().strip()
    except Exception:
        return raw_html


def fetch_recent_news(keywords=None, hours: int = LOOKBACK_HOURS):
    """Fetch and clean news headlines from multiple RSS feeds."""
    keywords = [k.lower() for k in (keywords or KEYWORDS)]
    news_items = []
    cutoff = datetime.utcnow() - timedelta(hours=hours)

    for source, url in NEWS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:20]:
                title = clean_html(entry.get("title", ""))
                summary = clean_html(entry.get("summary", ""))
                link = entry.get("link", "")
                published = entry.get("published_parsed")

                if published:
                    published_dt = datetime(*published[:6])
                    if published_dt < cutoff:
                        continue

                # Match any keyword
                if not any(k in (title + summary).lower() for k in keywords):
                    continue

                news_items.append({
                    "source": source,
                    "title": title,
                    "summary": summary,
                    "link": link,
                })
        except Exception as e:
            logger.warning(f"⚠️ Failed to fetch from {source}: {e}")

    logger.info(f"📰 Collected {len(news_items)} relevant news items.")
    return news_items


def analyze_sentiment(texts, use_transformer: bool = USE_TRANSFORMER):
    """Analyze sentiment using Transformer or keyword scoring."""
    if not texts:
        return []

    if use_transformer:
        try:
            sentiment_model = pipeline("sentiment-analysis", model=MODEL_NAME)
            return sentiment_model(texts)
        except Exception as e:
            logger.warning(f"⚠️ Transformer analysis failed, falling back to keyword scoring: {e}")

    # Keyword-based sentiment fallback
    positive_words = ["rise", "gain", "optimism", "bullish", "increase", "surge", "boost"]
    negative_words = ["fall", "drop", "bearish", "decline", "weakness", "slump", "fear"]

    def score_text(text: str):
        text = text.lower()
        score = sum(w in text for w in positive_words) - sum(w in text for w in negative_words)
        return {
            "label": "POSITIVE" if score > 0 else "NEGATIVE" if score < 0 else "NEUTRAL",
            "score": abs(score) / 3.0
        }

    return [score_text(t) for t in texts]


def summarize_sentiment(news_items):
    """Compute average sentiment and label summary."""
    if not news_items:
        return {"avg_score": 0.0, "label": "⚪ Neutral", "raw_count": 0, "breakdown": {"details": []}}

    texts = [f"{n['title']} {n['summary']}" for n in news_items]
    results = analyze_sentiment(texts)

    analyzed = []
    scores = []
    for n, r in zip(news_items, results):
        label = r["label"].upper()
        conf = r["score"]
        score = conf if "POS" in label else -conf if "NEG" in label else 0
        n.update({"score": score, "label": label})
        analyzed.append(n)
        scores.append(score)

    avg = sum(scores) / len(scores)
    mood = "🟢 Bullish" if avg > 0.2 else "🔴 Bearish" if avg < -0.2 else "⚪ Neutral"

    logger.success(f"✅ News sentiment summary: {mood} ({avg:.3f}) from {len(news_items)} items.")
    return {
        "avg_score": avg,
        "label": mood,
        "raw_count": len(news_items),
        "breakdown": {"details": analyzed},
    }


def news_sentiment_summary():
    """Public callable for pipeline & Telegram integration."""
    news = fetch_recent_news()
    return summarize_sentiment(news)
