# data/transcripts_store.py

from typing import Optional, Dict, Tuple, List
from datetime import datetime

# In a real system, you'd load this from files or a database.
# For now, this is just a stub to make the API work.
_TRANSCRIPTS: Dict[Tuple[str, str], str] = {
    ("AAPL", "2025-01-28"): """
    Good afternoon and thank you for joining us today.

    We delivered strong results this quarter with solid growth across our product lines.
    However, we are seeing some uncertainty in certain international markets and macro headwinds
    that could create risk going forward.

    Overall, we remain confident in our long-term strategy and our ability to execute.
    """,

    ("AAPL", "2024-10-30"): """
    Welcome, everyone.

    This quarter produced mixed results. Services revenue was strong, but we experienced
    softness in hardware demand across several regions. Foreign exchange pressures and
    slower consumer upgrades continue to create challenges.

    That said, we remain focused on long-term innovation and see opportunities ahead
    despite the current headwinds.
    """,

    ("AAPL", "2024-07-30"): """
    Thanks for joining.

    We delivered solid year-over-year growth and continued to see robust demand in
    our flagship devices. Supply chain constraints have improved materially, and customer
    satisfaction remains at record levels.

    While we are aware of potential volatility in the broader macro environment, we are
    confident in our strategic investments going forward.
    """,

    ("AAPL", "2024-04-30"): """
    Good afternoon.

    Our quarterly performance exceeded expectations. Services and wearables saw notable
    growth, while demand for Macs softened slightly. We continue to navigate uncertain
    market conditions, and inflationary pressures remain a factor.

    Even so, our product roadmap positions us well for future opportunities.
    """,

    ("AAPL", "2024-01-31"): """
    Hello everyone.

    Revenue this quarter declined modestly compared to last year, driven primarily by
    weakness in international markets and competitive pricing pressures. Despite these
    challenges, we saw encouraging trends in services adoption and user retention.

    We are cautiously optimistic about the next fiscal year, though risks remain.
    """,
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


def _parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d")


def list_events(
    tickers: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> List[Tuple[str, str, str]]:
    """
    List available (ticker, call_date, transcript_text) events from the store,
    optionally filtered by ticker universe and date range.

    Args:
        tickers: Optional list of tickers to include. If None, include all.
        start_date: Optional start date (YYYY-MM-DD), inclusive.
        end_date: Optional end date (YYYY-MM-DD), inclusive.

    Returns:
        List of (ticker, call_date, transcript_text) tuples.
    """
    # Normalize ticker filter to uppercase
    ticker_set = {t.upper() for t in tickers} if tickers else None

    start_dt = _parse_date(start_date) if start_date else None
    end_dt = _parse_date(end_date) if end_date else None

    results: List[Tuple[str, str, str]] = []

    for (ticker, call_date), text in _TRANSCRIPTS.items():
        if ticker_set and ticker.upper() not in ticker_set:
            continue

        call_dt = _parse_date(call_date)

        if start_dt and call_dt < start_dt:
            continue
        if end_dt and call_dt > end_dt:
            continue

        results.append((ticker, call_date, text))

    # Sort by ticker, then date
    results.sort(key=lambda x: (x[0], x[1]))
    return results
