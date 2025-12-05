# nlp/sentiment.py

import re
from statistics import mean, pstdev
from typing import Dict, List

# Very simple example lexicons.
# In practice, will make these much richer or replace with a real model.
POSITIVE_WORDS = {
    "strong", "growth", "record", "robust", "confident",
    "positive", "improved", "solid", "ahead"
}

NEGATIVE_WORDS = {
    "weak", "miss", "decline", "soft", "headwinds",
    "pressure", "downturn", "negative", "risk"
}

UNCERTAINTY_WORDS = {
    "uncertain", "uncertainty", "risk", "risks", "volatility",
    "might", "could", "may", "headwinds", "challenge", "pressure"
}


def _tokenize(text: str) -> List[str]:
    # Simple word tokenizer: split on non-letters, keep lowercase words
    return [w for w in re.split(r"[^a-zA-Z]+", text.lower()) if w]


def _split_sentences(text: str) -> List[str]:
    # Very crude sentence splitter: split on ., ?, !
    raw_sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in raw_sentences if s.strip()]


def _sentence_sentiment(tokens: List[str]) -> float:
    if not tokens:
        return 0.0
    pos = sum(1 for t in tokens if t in POSITIVE_WORDS)
    neg = sum(1 for t in tokens if t in NEGATIVE_WORDS)
    return (pos - neg) / len(tokens)


def compute_features(text: str) -> Dict[str, float]:
    """
    Compute simple sentiment and uncertainty features for a transcript.

    Returns:
        {
          "token_count": int,
          "sentiment_mean": float,
          "sentiment_std": float,
          "uncertainty_score": float
        }
    """
    sentences = _split_sentences(text)
    sentence_scores: List[float] = []

    all_tokens: List[str] = []

    for sent in sentences:
        tokens = _tokenize(sent)
        all_tokens.extend(tokens)
        if tokens:
            sentence_scores.append(_sentence_sentiment(tokens))

    token_count = len(all_tokens)

    if sentence_scores:
        sentiment_mean = mean(sentence_scores)
        sentiment_std = pstdev(sentence_scores) if len(sentence_scores) > 1 else 0.0
    else:
        sentiment_mean = 0.0
        sentiment_std = 0.0

    # Uncertainty score = fraction of tokens that are in the uncertainty word list
    if token_count > 0:
        uncertainty_hits = sum(1 for t in all_tokens if t in UNCERTAINTY_WORDS)
        uncertainty_score = uncertainty_hits / token_count
    else:
        uncertainty_score = 0.0

    return {
        "token_count": token_count,
        "sentiment_mean": sentiment_mean,
        "sentiment_std": sentiment_std,
        "uncertainty_score": uncertainty_score,
    }
