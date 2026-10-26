"""NFL data from nflverse: schedule/results/closing lines and weekly player stats."""
import httpx
import pandas as pd

from ..config import DATA_DIR, ET, NFLVERSE_GAMES, NFLVERSE_PLAYER_WEEK, current_nfl_season

# nflverse abbreviations that differ from Kalshi's.
NFL_TO_KALSHI = {"JAX": "JAC", "LA": "LAR"}

GAMES_PATH = DATA_DIR / "nfl_games.csv"


def _player_path(season: int):
    return DATA_DIR / f"nfl_player_week_{season}.csv"


def to_kalshi(team: str) -> str:
    return NFL_TO_KALSHI.get(team, team)


def _download(client: httpx.Client, url: str, path) -> None:
    r = client.get(url)
    r.raise_for_status()
    path.write_bytes(r.content)


def refresh(log=print) -> None:
    season = current_nfl_season()
    with httpx.Client(follow_redirects=True, timeout=180) as client:
        log("NFL: downloading schedule, results and lines")
        _download(client, NFLVERSE_GAMES, GAMES_PATH)
        for s in (season - 1, season):
            path = _player_path(s)
            if s < season and path.exists():
                continue  # completed seasons don't change
            log(f"NFL: downloading {s} weekly player stats")
            _download(client, NFLVERSE_PLAYER_WEEK.format(season=s), path)
    log("NFL: done")


def has_data() -> bool:
    return GAMES_PATH.exists()


def load_games(seasons_back: int = 3) -> pd.DataFrame:
    if not GAMES_PATH.exists():
        return pd.DataFrame()
    g = pd.read_csv(GAMES_PATH, low_memory=False)
    g = g[g["season"] >= current_nfl_season() - seasons_back].copy()
    g["home_team"] = g["home_team"].map(to_kalshi)
    g["away_team"] = g["away_team"].map(to_kalshi)
    local = pd.to_datetime(g["gameday"] + " " + g["gametime"].fillna("13:00"))
    g["kickoff"] = local.dt.tz_localize(ET).dt.tz_convert("UTC")
    g["date"] = pd.to_datetime(g["gameday"])
    g["completed"] = g["result"].notna()
    return g.reset_index(drop=True)


def load_player_weeks() -> pd.DataFrame:
    season = current_nfl_season()
    frames = [pd.read_csv(_player_path(s), low_memory=False) for s in (season - 1, season) if _player_path(s).exists()]
    if not frames:
        return pd.DataFrame()
    p = pd.concat(frames, ignore_index=True)
    p = p[p["season_type"].isin(["REG", "POST"])].copy()
    p["team"] = p["team"].map(to_kalshi)
    p["opponent_team"] = p["opponent_team"].map(to_kalshi)
    num = [
        "completions", "attempts", "passing_yards", "passing_tds", "carries", "rushing_yards",
        "rushing_tds", "receptions", "targets", "receiving_yards", "receiving_tds",
    ]
    p[num] = p[num].fillna(0)
    p["tds"] = p["rushing_tds"] + p["receiving_tds"]
    # Global ordering key across seasons (weeks never exceed 22).
    p["t"] = p["season"] * 100 + p["week"]
    return p.reset_index(drop=True)
