# api/main.py

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from data.transcripts_store import get_transcript, list_events
from data.prices import get_simple_returns
from nlp.sentiment import compute_features


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
