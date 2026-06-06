"""
returns.py
----------
Fetch post-earnings stock returns using yfinance.
"""

import yfinance as yf
import pandas as pd
from datetime import timedelta


def get_return(ticker: str, call_date: str, days: int = 1) -> float | None:
    """
    Get the stock return `days` trading days after the earnings call date.

    Args:
        ticker:    Stock ticker symbol (e.g. 'AAPL').
        call_date: Date of the earnings call as 'YYYY-MM-DD'.
        days:      How many trading days forward to measure return.

    Returns:
        Return as a float (e.g. 0.02 = +2%), or None if data unavailable.
    """
    start = pd.Timestamp(call_date)
    end = start + timedelta(days=days * 3)  # buffer for weekends/holidays

    hist = yf.Ticker(ticker).history(start=start, end=end, interval="1d")

    if len(hist) <= days:
        return None

    return float(hist["Close"].iloc[days] / hist["Close"].iloc[0] - 1)


def add_returns(df: pd.DataFrame, horizons: list[int] = [1, 5]) -> pd.DataFrame:
    """
    Add return columns to a DataFrame of earnings call records.

    Args:
        df:       DataFrame with 'ticker' and 'call_date' columns.
        horizons: List of forward-return horizons in trading days.

    Returns:
        DataFrame with added return columns (e.g. 'return_1d', 'return_5d').
    """
    df = df.copy()
    for days in horizons:
        col = f"return_{days}d"
        print(f"Fetching {days}-day returns...")
        df[col] = df.apply(
            lambda row: get_return(row["ticker"], row["call_date"], days), axis=1
        )
    return df
