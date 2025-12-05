# data/transcripts_store.py

from typing import Optional, Dict, Tuple

# stub to make the API work.
_TRANSCRIPTS: Dict[Tuple[str, str], str] = {
    ("AAPL", "2025-01-28"): """
    Good afternoon and thank you for joining us today.

    We delivered strong results this quarter with solid growth across our product lines.
    However, we are seeing some uncertainty in certain international markets and macro headwinds
    that could create risk going forward.

    Overall, we remain confident in our long-term strategy and our ability to execute.
    """
}


def get_transcript(ticker: str, call_date: str) -> Optional[str]:
    """
    Retrieve a raw transcript string for a given (ticker, call_date).

    Args:
        ticker: Ticker symbol, e.g. "AAPL"
        call_date: Call date in YYYY-MM-DD format

    Returns:
        Transcript text if found, else None.
    """
    key = (ticker.upper(), call_date)
    return _TRANSCRIPTS.get(key)
