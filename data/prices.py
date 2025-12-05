# data/prices.py

from datetime import datetime, timedelta
from typing import Dict, Optional

import yfinance as yf


def _parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d")


def get_simple_returns(
    ticker: str,
    call_date: str,
    horizons=(1, 3, 5),
) -> Optional[Dict[str, float]]:
    """
    Fetch daily close prices around the earnings call date and compute
    simple returns over given horizons.

    Returns a dict like:
      {
        "return_1d": 0.0123,
        "return_3d": -0.0045,
        "return_5d": 0.0210
      }

    Returns None if prices are unavailable.
    """
    # Parse the call date
    call_dt = _parse_date(call_date)

    # We fetch a window of prices from the call date to call_date + max_horizon + a buffer
    max_h = max(horizons)
    start = call_dt
    end = call_dt + timedelta(days=max_h + 5)  # small buffer for weekends/holidays

    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=False,
    )

    if df.empty:
        return None

    # Align: we assume the "day 0" price is the close on call_date or the next available trading day
    # Get the first row ON or AFTER call_date
    df = df.sort_index()
    df_after_call = df[df.index >= start]

    if df_after_call.empty:
        return None

    # Day 0 close
    p0 = df_after_call["Close"].iloc[0]

    returns: Dict[str, float] = {}

    for h in horizons:
        # target date = first available trading day that is >= call_date + h days
        target_dt = call_dt + timedelta(days=h)
        df_target = df_after_call[df_after_call.index >= target_dt]

        if df_target.empty:
            # No price for that horizon; skip or set to None
            returns[f"return_{h}d"] = None
            continue

        ph = df_target["Close"].iloc[0]

        # Simple return (ph - p0) / p0
        r = (ph - p0) / p0
        returns[f"return_{h}d"] = float(r)

    return returns
