"""Tests for the warehouse project: schema rules, idempotent loading, and every
analytical query checked against an independent pandas calculation."""

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError

import db
import queries
from make_synthetic_db import build


@pytest.fixture()
def engine(tmp_path, monkeypatch):
    """A fresh, empty warehouse built through the project's own db.py."""
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "w.db")
    eng = db.get_engine()
    db.init_schema(eng)
    return eng


def price_frame(n=60, seed=0, start=100.0):
    rng = np.random.default_rng(seed)
    close = start * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({
        "date": pd.bdate_range("2024-01-02", periods=n).strftime("%Y-%m-%d"),
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "adj_close": close, "volume": 1_000_000,
    })


# ------------------------------------------------------- schema + loading
def test_upsert_security_returns_the_same_id_every_time(engine):
    a = db.upsert_security(engine, "AAPL", "Apple", "Technology")
    b = db.upsert_security(engine, "AAPL", "Apple", "Technology")
    c = db.upsert_security(engine, "JPM", "JPMorgan", "Financials")
    assert a == b and c != a


def test_loading_twice_does_not_duplicate_rows(engine):
    sid = db.upsert_security(engine, "AAPL")
    df = price_frame()
    db.load_prices(engine, sid, df)
    db.load_prices(engine, sid, df)
    with engine.connect() as c:
        assert c.execute(text("SELECT COUNT(*) FROM daily_prices")).scalar() == len(df)


def test_reloading_overlapping_data_updates_values(engine):
    sid = db.upsert_security(engine, "AAPL")
    df = price_frame()
    db.load_prices(engine, sid, df)
    revised = df.copy()
    revised.loc[5, "adj_close"] = 12345.0
    db.load_prices(engine, sid, revised)
    with engine.connect() as c:
        got = c.execute(text("SELECT adj_close FROM daily_prices WHERE date = :d"), {"d": df.loc[5, "date"]}).scalar()
        assert got == 12345.0
        assert c.execute(text("SELECT COUNT(*) FROM daily_prices")).scalar() == len(df)


def test_foreign_keys_are_actually_enforced(engine):
    with pytest.raises(IntegrityError):
        with engine.begin() as c:
            c.execute(text("INSERT INTO daily_prices (security_id, date, open, high, low, close, adj_close, volume) "
                           "VALUES (999, '2024-01-02', 1, 1, 1, 1, 1, 1)"))


def test_deleting_a_security_cascades_to_its_prices(engine):
    keep = db.upsert_security(engine, "JPM")
    gone = db.upsert_security(engine, "AAPL")
    db.load_prices(engine, keep, price_frame(seed=1))
    db.load_prices(engine, gone, price_frame(seed=2))
    with engine.begin() as c:
        c.execute(text("DELETE FROM securities WHERE ticker = 'AAPL'"))
    with engine.connect() as c:
        remaining = c.execute(text("SELECT security_id, COUNT(*) FROM daily_prices GROUP BY security_id")).fetchall()
    assert remaining == [(keep, 60)]


# --------------------------------------------- queries vs independent pandas
@pytest.fixture()
def synthetic(tmp_path):
    path = build(tmp_path / "syn.db", n_days=300, seed=3)
    eng = create_engine(f"sqlite:///{path}")
    px = pd.read_sql(
        "SELECT s.ticker, p.date, p.close FROM daily_prices p JOIN securities s USING(security_id) ORDER BY p.date", eng
    )
    return eng, px


def test_daily_returns_match_pandas(synthetic):
    eng, px = synthetic
    got = queries.daily_returns(eng, "AAPL")
    expected = (px[px.ticker == "AAPL"].close.pct_change() * 100).round(2).to_numpy()
    np.testing.assert_allclose(got["pct_return"].to_numpy(), expected, equal_nan=True, atol=1e-9)


def test_rolling_volatility_is_a_true_20_day_window(synthetic):
    eng, px = synthetic
    got = queries.rolling_volatility(eng, "AAPL", 20)
    ret = px[px.ticker == "AAPL"].close.pct_change()
    expected = ret.rolling(20).std(ddof=1).to_numpy()
    assert np.isnan(got["rolling_volatility"].iloc[:20]).all()          # no full window yet
    np.testing.assert_allclose(got["rolling_volatility"].to_numpy()[20:], expected[20:], rtol=1e-8)
    # regression test for the old bug, which used ALL history instead of the last 20 days
    expanding = ret.expanding().std(ddof=1).to_numpy()
    assert not np.allclose(got["rolling_volatility"].to_numpy()[100:], expanding[100:], rtol=1e-3)


def test_rolling_volatility_validates_window(synthetic):
    with pytest.raises(ValueError):
        queries.rolling_volatility(synthetic[0], "AAPL", 1)


def test_moving_averages_match_pandas_and_have_no_signal_during_warmup(synthetic):
    eng, px = synthetic
    got = queries.moving_average_crossover(eng, "AAPL", short=10, long=40)
    close = px[px.ticker == "AAPL"].close
    np.testing.assert_allclose(got["short_ma"].to_numpy()[9:], close.rolling(10).mean().to_numpy()[9:], rtol=1e-9)
    np.testing.assert_allclose(got["long_ma"].to_numpy()[39:], close.rolling(40).mean().to_numpy()[39:], rtol=1e-9)
    assert got["signal"].iloc[:39].isna().all()
    assert got["signal"].iloc[39:].notna().all()
    full = got.iloc[39:]
    assert ((full["short_ma"] > full["long_ma"]) == (full["signal"] == "bullish")).all()


def test_moving_average_rejects_bad_windows(synthetic):
    with pytest.raises(ValueError):
        queries.moving_average_crossover(synthetic[0], "AAPL", short=50, long=20)


def test_sector_comparison_matches_pandas(synthetic):
    eng, px = synthetic
    got = queries.sector_comparison(eng).set_index("sector")
    sectors = pd.read_sql("SELECT ticker, sector FROM securities", eng).set_index("ticker")["sector"]
    px["ret"] = px.groupby("ticker").close.pct_change()
    px["sector"] = px.ticker.map(sectors)
    expected = px.dropna(subset=["ret"]).groupby("sector")["ret"].mean() * 100
    for sector, value in expected.items():
        assert got.loc[sector, "avg_daily_return_pct"] == pytest.approx(value, abs=5e-5)  # SQL rounds to 4 dp
