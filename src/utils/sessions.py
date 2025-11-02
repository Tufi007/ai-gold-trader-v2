# src/utils/session.py
from datetime import datetime, time

def market_session_status():
    """Return dict showing which forex sessions are open."""
    now_utc = datetime.utcnow().time()

    sessions = {
        "🗽 New York": (time(13, 0), time(22, 0)),  # 13:00–22:00 UTC
        "💹 London": (time(7, 0), time(16, 0)),
        "🈺 Tokyo": (time(0, 0), time(9, 0)),
        "🇦🇺 Sydney": (time(22, 0), time(7, 0)),
    }

    result = {}
    for name, (start, end) in sessions.items():
        if start < end:
            open_now = start <= now_utc <= end
        else:
            open_now = now_utc >= start or now_utc <= end
        result[name] = "🟢 Open" if open_now else "🔴 Closed"

    return result

if __name__ == "__main__":
    print(market_session_status())
