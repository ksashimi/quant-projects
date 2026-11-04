"""
queries.py: analytical queries against the warehouse.

The queries are written as plain SQL (window functions and CTEs) instead of
using the ORM, because that's the part worth practising.
"""

import numpy as np
import pandas as pd
from sqlalchemy import text
from db import get_engine


def daily_returns(engine, ticker: str) -> pd.DataFrame:
    """
    Day-over-day % return. LAG() is a window function that reads the previous
    row, so there's no need for a self-join.
    """
    query = text("""
        SELECT
            s.ticker,
            p.date,
            p.close,
            LAG(p.close) OVER (ORDER BY p.date) AS prev_close,
            ROUND(
                (p.close - LAG(p.close) OVER (ORDER BY p.date))
                / LAG(p.close) OVER (ORDER BY p.date) * 100,
            2) AS pct_return
        FROM daily_prices p
        JOIN securities s ON s.security_id = p.security_id
        WHERE s.ticker = :ticker
        ORDER BY p.date
    """)
    return pd.read_sql(query, engine, params={"ticker": ticker})


def rolling_volatility(engine, ticker: str, window: int = 20) -> pd.DataFrame:
    """
    Rolling N-day standard deviation of daily returns (not annualized).

    SQLite has no STDDEV, so SQL computes the rolling variance with window
    frames using the identity  Var(x) = avg(x^2) - avg(x)^2  and pandas takes
    the square root. (SQRT only exists in SQLite builds with math functions
    enabled, so doing it in pandas keeps this portable.) The population
    variance is rescaled by n/(n-1) to give the usual SAMPLE standard
    deviation, matching pandas' rolling().std().

    The first `window` rows are NULL because there is not yet a full window.
    """
    window = int(window)
    if window < 2:
        raise ValueError("window must be at least 2")
    query = text(f"""
        WITH returns AS (
            SELECT
                p.date,
                (p.close - LAG(p.close) OVER (ORDER BY p.date))
                    / LAG(p.close) OVER (ORDER BY p.date) AS ret
            FROM daily_prices p
            JOIN securities s ON s.security_id = p.security_id
            WHERE s.ticker = :ticker
        )
        SELECT
            date,
            ret,
            AVG(ret * ret) OVER w AS mean_sq,
            AVG(ret) OVER w       AS mean_ret,
            COUNT(ret) OVER w     AS n_obs
        FROM returns
        WINDOW w AS (ORDER BY date ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW)
        ORDER BY date
    """)
    df = pd.read_sql(query, engine, params={"ticker": ticker})
    pop_var = (df["mean_sq"] - df["mean_ret"] ** 2).clip(lower=0)  # clip: float error can dip below 0
    sample_var = pop_var * df["n_obs"] / (df["n_obs"] - 1)
    df["rolling_volatility"] = np.sqrt(sample_var).where(df["n_obs"] == window)
    return df[["date", "ret", "rolling_volatility"]]


def sector_comparison(engine) -> pd.DataFrame:
    """
    Average daily return by sector over the full history. This joins the fact
    table to the dimension table and rolls the rows up with GROUP BY.
    """
    query = text("""
        WITH returns AS (
            SELECT
                s.sector,
                (p.close - LAG(p.close) OVER (PARTITION BY p.security_id ORDER BY p.date))
                    / LAG(p.close) OVER (PARTITION BY p.security_id ORDER BY p.date) AS ret
            FROM daily_prices p
            JOIN securities s ON s.security_id = p.security_id
        )
        SELECT
            sector,
            ROUND(AVG(ret) * 100, 4) AS avg_daily_return_pct,
            COUNT(*) AS num_days
        FROM returns
        WHERE ret IS NOT NULL
        GROUP BY sector
        ORDER BY avg_daily_return_pct DESC
    """)
    return pd.read_sql(query, engine)


def moving_average_crossover(engine, ticker: str, short: int = 20, long: int = 50) -> pd.DataFrame:
    """
    Short and long moving averages, with a bullish/bearish flag for whether the
    short one is above the long one. The backtester uses the same idea.
    """
    short, long = int(short), int(long)
    if not 1 <= short < long:
        raise ValueError("need 1 <= short < long")
    query = text(f"""
        WITH prices AS (
            SELECT p.date, p.close
            FROM daily_prices p
            JOIN securities s ON s.security_id = p.security_id
            WHERE s.ticker = :ticker
        ),
        mas AS (
            SELECT
                date,
                close,
                AVG(close) OVER (ORDER BY date ROWS BETWEEN {short - 1} PRECEDING AND CURRENT ROW) AS short_ma,
                AVG(close) OVER (ORDER BY date ROWS BETWEEN {long - 1} PRECEDING AND CURRENT ROW) AS long_ma,
                ROW_NUMBER() OVER (ORDER BY date) AS row_n
            FROM prices
        )
        SELECT
            *,
            CASE
                WHEN row_n < {long} THEN NULL          -- long window not full yet: no valid signal
                WHEN short_ma > long_ma THEN 'bullish'
                ELSE 'bearish'
            END AS signal
        FROM mas
        ORDER BY date
    """)
    return pd.read_sql(query, engine, params={"ticker": ticker})


if __name__ == "__main__":
    # quick smoke test once you've run fetch_data.py
    engine = get_engine()
    print("=== Daily returns (AAPL, last 5) ===")
    print(daily_returns(engine, "AAPL").tail())

    print("\n=== Sector comparison ===")
    print(sector_comparison(engine))
