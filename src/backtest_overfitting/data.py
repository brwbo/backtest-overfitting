"""Daily price data from Yahoo Finance's chart API, cached to CSV so reruns are offline and identical."""

import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parents[2] / "data"
CHART_URL = "https://query2.finance.yahoo.com/v8/finance/chart/{t}?period1={a}&period2={b}&interval=1d&events=div,split"


def _epoch(day: str) -> int:
    return int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def _download(ticker: str, start: str, end: str) -> pd.Series:
    url = CHART_URL.format(t=ticker, a=_epoch(start), b=_epoch(end))
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.load(resp)["chart"]["result"][0]
    # Adjusted close folds dividends and splits into the price, so returns are total returns.
    adj = result["indicators"]["adjclose"][0]["adjclose"]
    dates = pd.to_datetime(result["timestamp"], unit="s").normalize()
    return pd.Series(adj, index=dates, name="close", dtype=float).dropna()


def load_returns(ticker: str, start: str, end: str) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Simple daily returns from adjusted closes."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{ticker}_{start}_{end}.csv"
    if path.exists():
        close = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    else:
        close = _download(ticker, start, end)
        if close.empty:
            raise RuntimeError(f"No data returned for {ticker}")
        close.to_frame().to_csv(path)
    returns = close.pct_change().dropna()
    return returns.index, returns.to_numpy(dtype=float)
