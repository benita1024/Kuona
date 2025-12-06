# api/main.py

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from data.transcripts_store import get_transcript, list_events
from data.prices import get_simple_returns
from nlp.sentiment import compute_features
from math import isnan
from statistics import mean



class EarningsFeatureRow(BaseModel):
    ticker: str
    call_date: str
    token_count: int
    sentiment_mean: float
    sentiment_std: float
    uncertainty_score: float
    return_1d: Optional[float] = None
    return_3d: Optional[float] = None
    return_5d: Optional[float] = None


class EarningsFeaturesResponse(BaseModel):
    features: List[EarningsFeatureRow]


app = FastAPI(
    title="Kuona API",
    description="API-first NLP engine for earnings call features.",
    version="0.1.0",
)


@app.get("/features/earnings", response_model=EarningsFeaturesResponse)
def get_earnings_features(
    ticker: str = Query(..., description="Ticker symbol, e.g. AAPL"),
    call_date: str = Query(..., description="Call date in YYYY-MM-DD format"),
):
    """
    Return basic NLP features and post-earnings returns for a single earnings call.

    For v0, this uses a stubbed in-memory transcript store.
    """
    transcript = get_transcript(ticker, call_date)
    if transcript is None:
        raise HTTPException(
            status_code=404,
            detail=f"No transcript found for {ticker.upper()} on {call_date}",
        )

    features_dict: Dict[str, Any] = compute_features(transcript)
    returns = get_simple_returns(ticker, call_date, horizons=(1, 3, 5)) or {}

    row = EarningsFeatureRow(
        ticker=ticker.upper(),
        call_date=call_date,
        token_count=features_dict["token_count"],
        sentiment_mean=features_dict["sentiment_mean"],
        sentiment_std=features_dict["sentiment_std"],
        uncertainty_score=features_dict["uncertainty_score"],
        return_1d=returns.get("return_1d"),
        return_3d=returns.get("return_3d"),
        return_5d=returns.get("return_5d"),
    )

    return EarningsFeaturesResponse(features=[row])


@app.get("/features/earnings/bulk", response_model=EarningsFeaturesResponse)
def get_bulk_earnings_features(
    tickers: str = Query(
        "",
        description="Comma-separated list of tickers, e.g. 'AAPL,MSFT'. "
                    "If empty, use all available.",
    ),
    start_date: str = Query(
        "",
        description="Start date in YYYY-MM-DD format (inclusive). Optional.",
    ),
    end_date: str = Query(
        "",
        description="End date in YYYY-MM-DD format (inclusive). Optional.",
    ),
):
    """
    Return NLP features and post-earnings returns for multiple earnings call events.

    This is a bulk version of /features/earnings.
    It scans the transcript store for events matching the provided filters.
    """
    # Parse tickers (may be empty)
    ticker_list = [t.strip() for t in tickers.split(",") if t.strip()] or None

    start = start_date or None
    end = end_date or None

    events = list_events(
        tickers=ticker_list,
        start_date=start,
        end_date=end,
    )

    if not events:
        raise HTTPException(
            status_code=404,
            detail="No events found for the given filters.",
        )

    rows: List[EarningsFeatureRow] = []

    for ticker, call_date, transcript in events:
        features_dict: Dict[str, Any] = compute_features(transcript)
        returns = get_simple_returns(ticker, call_date, horizons=(1, 3, 5)) or {}

        row = EarningsFeatureRow(
            ticker=ticker.upper(),
            call_date=call_date,
            token_count=features_dict["token_count"],
            sentiment_mean=features_dict["sentiment_mean"],
            sentiment_std=features_dict["sentiment_std"],
            uncertainty_score=features_dict["uncertainty_score"],
            return_1d=returns.get("return_1d"),
            return_3d=returns.get("return_3d"),
            return_5d=returns.get("return_5d"),
        )
        rows.append(row)

    return EarningsFeaturesResponse(features=rows)

class BacktestBucketStats(BaseModel):
    count: int
    avg_return_3d: Optional[float] = None


class BacktestResult(BaseModel):
    ticker_filter: Optional[List[str]] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    events_count: int
    corr_sentiment_return_3d: Optional[float] = None
    low_sentiment: BacktestBucketStats
    high_sentiment: BacktestBucketStats
    

@app.get("/backtest/earnings", response_model=BacktestResult)
def backtest_earnings(
    tickers: str = Query(
        "",
        description="Comma-separated list of tickers, e.g. 'AAPL,MSFT'. "
                    "If empty, use all available.",
    ),
    start_date: str = Query(
        "",
        description="Start date in YYYY-MM-DD format (inclusive). Optional.",
    ),
    end_date: str = Query(
        "",
        description="End date in YYYY-MM-DD format (inclusive). Optional.",
    ),
):
    """
    Simple event-study style backtest over earnings events.

    For all matching events, this computes:
      - correlation between sentiment_mean and 3-day return
      - average 3d return for low vs high sentiment buckets (split at median)
    """
    ticker_list = [t.strip() for t in tickers.split(",") if t.strip()] or None
    start = start_date or None
    end = end_date or None

    events = list_events(
        tickers=ticker_list,
        start_date=start,
        end_date=end,
    )

    if not events:
        raise HTTPException(
            status_code=404,
            detail="No events found for the given filters.",
        )

    sentiment_vals: List[float] = []
    ret3_vals: List[float] = []

    rows: List[Dict[str, Any]] = []

    for ticker, call_date, transcript in events:
        features_dict: Dict[str, Any] = compute_features(transcript)
        returns = get_simple_returns(ticker, call_date, horizons=(1, 3, 5)) or {}
        r3 = returns.get("return_3d")

        if r3 is None:
            continue  # skip events with missing return

        s = features_dict["sentiment_mean"]

        sentiment_vals.append(s)
        ret3_vals.append(r3)

        rows.append(
            {
                "ticker": ticker.upper(),
                "call_date": call_date,
                "sentiment_mean": s,
                "return_3d": r3,
            }
        )

    if not rows:
        raise HTTPException(
            status_code=400,
            detail="No events had valid 3-day returns for backtest.",
        )

    # Compute correlation (pearson) between sentiment and 3d returns
    n = len(sentiment_vals)
    if n < 2:
        corr = None
    else:
        mean_s = mean(sentiment_vals)
        mean_r = mean(ret3_vals)

        num = sum((s - mean_s) * (r - mean_r) for s, r in zip(sentiment_vals, ret3_vals))
        den_s = sum((s - mean_s) ** 2 for s in sentiment_vals)
        den_r = sum((r - mean_r) ** 2 for r in ret3_vals)

        if den_s == 0 or den_r == 0:
            corr = None
        else:
            corr = num / (den_s**0.5 * den_r**0.5)
            if isnan(corr):
                corr = None

    # Split into low vs high sentiment based on median
    sorted_pairs = sorted(zip(sentiment_vals, ret3_vals), key=lambda x: x[0])
    mid = len(sorted_pairs) // 2
    low_bucket = sorted_pairs[:mid]
    high_bucket = sorted_pairs[mid:]

    def _bucket_stats(bucket: List[tuple]) -> BacktestBucketStats:
        if not bucket:
            return BacktestBucketStats(count=0, avg_return_3d=None)
        returns_only = [r for _, r in bucket]
        return BacktestBucketStats(
            count=len(bucket),
            avg_return_3d=mean(returns_only),
        )

    low_stats = _bucket_stats(low_bucket)
    high_stats = _bucket_stats(high_bucket)

    return BacktestResult(
        ticker_filter=ticker_list,
        start_date=start,
        end_date=end,
        events_count=len(sentiment_vals),
        corr_sentiment_return_3d=corr,
        low_sentiment=low_stats,
        high_sentiment=high_stats,
    )
