"""
fetch_data.py: downloads daily price history from Yahoo Finance and loads it
into the warehouse. Needs an internet connection.

    cd src
    python fetch_data.py

Fetching and loading are kept separate (db.py does the loading). yfinance
sometimes rate-limits or changes its response format, and when that happens
only this file should need fixing.
"""

import yfinance as yf
import pandas as pd
from db import get_engine, init_schema, upsert_security, load_prices

# One ticker from each of a few sectors so the sector comparison has something
# to compare, plus SPY as a market benchmark.
TICKERS = {
    "AAPL": {"name": "Apple Inc.", "sector": "Technology"},
    "JPM":  {"name": "JPMorgan Chase", "sector": "Financials"},
    "XOM":  {"name": "Exxon Mobil", "sector": "Energy"},
    "TSLA": {"name": "Tesla Inc.", "sector": "Consumer Discretionary"},
    "SPY":  {"name": "S&P 500 ETF", "sector": "Index"},  # useful as a market benchmark
}

START_DATE = "2020-01-01"
END_DATE = None  # None = up to today


def fetch_ticker(ticker: str) -> pd.DataFrame:
    """Download OHLCV data and reshape it to match our schema's column names."""
    raw = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False, auto_adjust=False)

    if raw.empty:
        raise ValueError(f"No data returned for {ticker}. Check the symbol and your connection")

    # yfinance sometimes returns MultiIndex columns for a single ticker; flatten if so
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw.reset_index()[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    df.columns = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df


def main():
    engine = get_engine()
    init_schema(engine)

    for ticker, meta in TICKERS.items():
        print(f"\nFetching {ticker}...")
        try:
            df = fetch_ticker(ticker)
        except Exception as e:
            print(f"  Failed: {e}")
            continue

        security_id = upsert_security(engine, ticker, meta["name"], meta["sector"])
        load_prices(engine, security_id, df)

    print("\nDone. Database is at data/warehouse.db")


if __name__ == "__main__":
    main()
