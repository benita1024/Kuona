# api/main.py

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Any
from data.prices import get_simple_returns


from data.transcripts_store import get_transcript
from nlp.sentiment import compute_features


class EarningsFeatureRow(BaseModel):
    ticker: str
    call_date: str
    token_count: int
    sentiment_mean: float
    sentiment_std: float
    uncertainty_score: float
    return_1d: float | None = None
    return_3d: float | None = None
    return_5d: float | None = None


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
    Return basic NLP features for a single earnings call transcript.

    For v0, this uses a stubbed in-memory transcript store.
    """
    transcript = get_transcript(ticker, call_date)
    if transcript is None:
        raise HTTPException(
            status_code=404,
            detail=f"No transcript found for {ticker.upper()} on {call_date}",
        )

    features_dict: Dict[str, Any] = compute_features(transcript)
    # Compute simple price returns after the earnings call
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
