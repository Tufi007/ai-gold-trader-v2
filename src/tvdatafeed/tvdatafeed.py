import pandas as pd
import time
from enum import Enum

class Interval(Enum):
    in_1_minute = '1m'
    in_5_minute = '5m'
    in_15_minute = '15m'
    in_30_minute = '30m'
    in_1_hour = '1h'
    in_4_hour = '4h'
    in_daily = '1d'
    in_weekly = '1W'
    in_monthly = '1M'

class TvDatafeed:
    def __init__(self, username=None, password=None):
        self.username = username
        self.password = password

    def get_hist(self, symbol, exchange='FX_IDC', interval=Interval.in_1_hour, n_bars=500):
        now = pd.Timestamp.utcnow()
        idx = pd.date_range(end=now, periods=n_bars, freq=interval.value.upper())
        data = pd.DataFrame({
            'time': idx,
            'open': 1.0,
            'high': 1.0,
            'low': 1.0,
            'close': 1.0,
            'volume': 0
        })
        data.set_index('time', inplace=True)
        return data
