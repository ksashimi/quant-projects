"""
db.py: database connection, schema setup and loading.

SQLite because it needs no server or credentials. To use Postgres instead,
change the connection string in get_engine(). SQLAlchemy handles most of the
differences, so the rest of the code should barely change.
"""

from pathlib import Path
from sqlalchemy import create_engine, event, text
import pandas as pd

DB_PATH = Path(__file__).parent.parent / "data" / "warehouse.db"
SCHEMA_PATH = Path(__file__).parent.parent / "sql" / "schema.sql"


def get_engine():
    DB_PATH.parent.mkdir(exist_ok=True)
    engine = create_engine(f"sqlite:///{DB_PATH}")

    # SQLite does NOT enforce foreign keys (or ON DELETE CASCADE) unless every
    # connection turns it on, so do that here for each new connection.
    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, _record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.close()

    return engine


def init_schema(engine):
    """Run schema.sql against the database. Safe to run more than once."""
    schema_sql = SCHEMA_PATH.read_text()
    with engine.begin() as conn:
        for statement in schema_sql.split(";"):
            statement = statement.strip()
            if statement:
                conn.execute(text(statement))
    print(f"Schema initialized at {DB_PATH}")


def upsert_security(engine, ticker: str, company_name: str = None, sector: str = None) -> int:
    """Insert a security if it doesn't exist, return its security_id either way."""
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO securities (ticker, company_name, sector)
                VALUES (:ticker, :name, :sector)
                ON CONFLICT(ticker) DO NOTHING
            """),
            {"ticker": ticker, "name": company_name, "sector": sector},
        )
        result = conn.execute(
            text("SELECT security_id FROM securities WHERE ticker = :ticker"),
            {"ticker": ticker},
        )
        return result.scalar_one()


def load_prices(engine, security_id: int, df: pd.DataFrame):
    """
    Load a DataFrame of OHLCV data for one security.

    This is an upsert. A day that isn't in the table yet is inserted, and a
    day that already exists (same security_id and date) is updated. So loading
    overlapping date ranges never creates duplicates, which matters for
    anything that gets re-run on a schedule.
    """
    df = df.copy()
    df["security_id"] = security_id
    records = df.to_dict(orient="records")

    with engine.begin() as conn:
        for row in records:
            conn.execute(
                text("""
                    INSERT INTO daily_prices
                        (security_id, date, open, high, low, close, adj_close, volume)
                    VALUES
                        (:security_id, :date, :open, :high, :low, :close, :adj_close, :volume)
                    ON CONFLICT(security_id, date) DO UPDATE SET
                        open = excluded.open,
                        high = excluded.high,
                        low = excluded.low,
                        close = excluded.close,
                        adj_close = excluded.adj_close,
                        volume = excluded.volume
                """),
                row,
            )
    print(f"Loaded {len(records)} rows for security_id={security_id}")
