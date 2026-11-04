-- Financial data warehouse schema.
--
-- Two tables. securities has one row per ticker (name, sector). daily_prices
-- has one row per ticker per day. Keeping name and sector out of the price
-- table means a sector change is one update instead of one per day of history.
--
-- UNIQUE(security_id, date) stops the same day being loaded twice, so
-- re-running the fetch script is safe.
--
-- The foreign key has ON DELETE CASCADE, so deleting a security also deletes
-- its prices. SQLite only enforces that when foreign keys are switched on for
-- the connection, which db.py does.

CREATE TABLE IF NOT EXISTS securities (
    security_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker       TEXT NOT NULL UNIQUE,
    company_name TEXT,
    sector       TEXT,
    added_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS daily_prices (
    price_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    security_id  INTEGER NOT NULL,
    date         DATE NOT NULL,
    open         REAL NOT NULL,
    high         REAL NOT NULL,
    low          REAL NOT NULL,
    close        REAL NOT NULL,
    adj_close    REAL NOT NULL,
    volume       INTEGER NOT NULL,
    FOREIGN KEY (security_id) REFERENCES securities(security_id) ON DELETE CASCADE,
    UNIQUE(security_id, date)
);

-- Index for the most common query pattern: "give me all prices for
-- ticker X in date order". Without this, SQLite does a full table scan
-- every time you query one stock's history.
CREATE INDEX IF NOT EXISTS idx_prices_security_date
    ON daily_prices(security_id, date);
