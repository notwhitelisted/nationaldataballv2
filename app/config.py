import os
from datetime import datetime, date
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "ndb.sqlite3"
STATIC_DIR = ROOT / "static"

ET = ZoneInfo("America/New_York")


def _load_env_file() -> None:
    """Minimal .env support (KEY=VALUE lines) so API keys stay out of the code."""
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))


_load_env_file()
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ESPN_INJURIES = "https://site.api.espn.com/apis/site/v2/sports/{path}/injuries"
OPEN_METEO = "https://api.open-meteo.com/v1/forecast"

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
ESPN_NBA = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
NFLVERSE_GAMES = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"
NFLVERSE_PLAYER_WEEK = (
    "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{season}.csv"
)

# User-tunable settings (persisted in the settings table; these are defaults).
DEFAULT_SETTINGS = {
    "bankroll": 500.0,
    # Fraction of full Kelly to stake. Full Kelly is far too aggressive given model error.
    "kelly_fraction": 0.25,
    # Hard cap per position as a fraction of bankroll.
    "max_bet_pct": 0.03,
    # Minimum expected value (per $ risked, after fees) to show a market as an edge.
    "min_edge": 0.03,
    # Weight on our model vs the sportsbook line when both exist (0 = pure market, 1 = pure model).
    "model_weight": 0.35,
    # Kalshi taker fee coefficient: fee = ceil(coef * C * P * (1-P)) dollars.
    "kalshi_fee_coef": 0.07,
    # Paper trading: 1 = automatically paper-bet every qualifying edge (checked every 15 minutes).
    "auto_paper": 0.0,
    "paper_min_edge": 0.05,
}


def now_et() -> datetime:
    return datetime.now(ET)


def today_et() -> date:
    return now_et().date()


def current_nfl_season(d: date | None = None) -> int:
    d = d or today_et()
    return d.year if d.month >= 8 else d.year - 1


def current_nba_season(d: date | None = None) -> int:
    """ESPN labels NBA seasons by the year they end (2026-27 -> 2027)."""
    d = d or today_et()
    return d.year + 1 if d.month >= 9 else d.year
