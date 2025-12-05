# api/main.py

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Any

from data.transcripts_store import get_transcript
from nlp.sentiment import compute_features


class EarningsFeatureRow(BaseModel):
    ticker: str
    call_date: str
    token_count: int
    sentiment_mean: float
    sentiment_std: float
    uncertainty_score: float


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

    row = EarningsFeatureRow(
        ticker=ticker.upper(),
        call_date=call_date,
        **features_dict,
    )

    return EarningsFeaturesResponse(features=[row])
