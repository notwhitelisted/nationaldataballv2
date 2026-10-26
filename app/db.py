import json
import sqlite3
from contextlib import contextmanager

from .config import DB_PATH, DEFAULT_SETTINGS

SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS fetched_days (
    sport TEXT NOT NULL,
    day TEXT NOT NULL,
    PRIMARY KEY (sport, day)
);

CREATE TABLE IF NOT EXISTS nba_games (
    game_id TEXT PRIMARY KEY,
    season INTEGER NOT NULL,
    season_type INTEGER NOT NULL,
    date_et TEXT NOT NULL,
    start_utc TEXT NOT NULL,
    home TEXT NOT NULL,
    away TEXT NOT NULL,
    home_score INTEGER,
    away_score INTEGER,
    completed INTEGER NOT NULL DEFAULT 0,
    book_home_margin REAL,   -- sportsbook expected home margin (= -home spread)
    book_total REAL,
    book_home_ml INTEGER,
    book_away_ml INTEGER,
    box_fetched INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_nba_games_date ON nba_games(date_et);

CREATE TABLE IF NOT EXISTS nba_player_games (
    game_id TEXT NOT NULL,
    player_id TEXT NOT NULL,
    name TEXT NOT NULL,
    team TEXT NOT NULL,
    opp TEXT NOT NULL,
    home INTEGER NOT NULL,
    date_et TEXT NOT NULL,
    starter INTEGER NOT NULL,
    min REAL NOT NULL,
    pts INTEGER, reb INTEGER, ast INTEGER, fg3m INTEGER,
    fga INTEGER, fta INTEGER, tov INTEGER, stl INTEGER, blk INTEGER,
    PRIMARY KEY (game_id, player_id)
);
CREATE INDEX IF NOT EXISTS ix_nba_pg_player ON nba_player_games(player_id, date_et);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Every sportsbook price we pull is kept, so line movement can be studied later.
CREATE TABLE IF NOT EXISTS odds_lines (
    sport TEXT NOT NULL,
    event_id TEXT NOT NULL,
    commence_time TEXT NOT NULL,
    home TEXT NOT NULL,
    away TEXT NOT NULL,
    book TEXT NOT NULL,
    market TEXT NOT NULL,      -- h2h | spreads | totals
    outcome TEXT NOT NULL,     -- home | away | over | under
    point REAL,
    price INTEGER NOT NULL,
    fetched_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_odds_lines_fetch ON odds_lines(sport, fetched_at);

CREATE TABLE IF NOT EXISTS odds_props (
    sport TEXT NOT NULL,
    event_id TEXT NOT NULL,
    book TEXT NOT NULL,
    market TEXT NOT NULL,
    player TEXT NOT NULL,
    outcome TEXT NOT NULL,     -- over | under | yes | no
    point REAL,
    price INTEGER NOT NULL,
    fetched_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_odds_props_event ON odds_props(event_id, fetched_at);

CREATE TABLE IF NOT EXISTS bets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    sport TEXT NOT NULL,
    ticker TEXT NOT NULL,
    description TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('yes', 'no')),
    price REAL NOT NULL,          -- dollars per contract paid
    contracts INTEGER NOT NULL,
    fees REAL NOT NULL DEFAULT 0,
    fair_prob REAL,               -- our fair probability for the side we bought, at bet time
    game_start TEXT,              -- ISO UTC; used to freeze the closing price for CLV
    close_price REAL,             -- last observed price for our side before game start
    status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'won', 'lost', 'void')),
    pnl REAL,
    settled_at TEXT,
    notes TEXT
);
"""


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


@contextmanager
def session():
    conn = connect()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# Columns added after the first release; applied to existing databases on startup.
BET_COLUMNS = {
    "mode": "TEXT NOT NULL DEFAULT 'real'",        # real | paper
    "source": "TEXT NOT NULL DEFAULT 'manual'",    # manual | kalshi | auto
    "ext_id": "TEXT",                              # kalshi:<ticker>:<side> for imported positions
    "market_type": "TEXT",
    "edge": "REAL",                                # expected ROI when the bet was placed
    "sold_contracts": "INTEGER NOT NULL DEFAULT 0",
    "sell_proceeds": "REAL NOT NULL DEFAULT 0",
}


def init_db() -> None:
    with session() as conn:
        conn.executescript(SCHEMA)
        have = {r["name"] for r in conn.execute("PRAGMA table_info(bets)")}
        for col, ddl in BET_COLUMNS.items():
            if col not in have:
                conn.execute(f"ALTER TABLE bets ADD COLUMN {col} {ddl}")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS ix_bets_ext ON bets(ext_id) WHERE ext_id IS NOT NULL")


def get_meta(key: str, default=None):
    with session() as conn:
        row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return json.loads(row["value"]) if row else default


def set_meta(key: str, value) -> None:
    with session() as conn:
        conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, json.dumps(value)),
        )


def get_settings() -> dict:
    out = dict(DEFAULT_SETTINGS)
    with session() as conn:
        for row in conn.execute("SELECT key, value FROM settings"):
            if row["key"] in out:
                out[row["key"]] = json.loads(row["value"])
    return out


def save_settings(values: dict) -> dict:
    with session() as conn:
        for k, v in values.items():
            if k in DEFAULT_SETTINGS:
                conn.execute(
                    "INSERT INTO settings(key, value) VALUES(?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (k, json.dumps(float(v))),
                )
    return get_settings()
