# Kuona

Kuona is an API-first research tool that transforms earnings call transcripts into structured NLP features and backtestable signals.

The goal: **help analysts, quants and researchers “see” (kuona) what management is really saying** by turning language from earnings calls into numbers you can model.

---

##  What Kuona Does

- Ingests earnings call transcripts (e.g., from company IR pages, EDGAR, or vendors)
- Applies an NLP pipeline to extract:
  - Overall sentiment (lexicon-based in v1)
  - Uncertainty / risk language
  - Basic text statistics (token counts, sentence-level variation)
- Returns a **tidy feature matrix** via a REST API:
  - One row per (ticker, call_date, section)
  - Ready to load into pandas / your modeling stack
- (Later) Joins features with post-earnings returns to support event studies and trading signal research.

Kuona is designed **API-first** and **backtest-first**: everything it does is meant to be programmatically consumed.

---

##  High-Level Architecture

- **API Layer**: FastAPI application exposing endpoints like:
  - `GET /features/earnings` → earnings call feature matrix
- **NLP Layer**:
  - Text cleaning, tokenization
  - Lexicon-based sentiment + uncertainty scoring (v1)
  - Pluggable models (FinBERT, transformers) in later versions
- **Data Layer**:
  - Raw transcripts (for now: local files / in-memory store)
  - Features and metadata (later: PostgreSQL)
  - Price data + returns (later: yfinance / market data providers)


