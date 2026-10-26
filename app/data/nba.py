"""NBA data from ESPN's public site API: schedule, scores, DraftKings lines and box scores."""
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta

import httpx
import numpy as np
import pandas as pd

from ..config import ESPN_NBA, ET, current_nba_season, today_et
from ..db import session

# ESPN abbreviations that differ from Kalshi's (Kalshi uses standard NBA tricodes).
ESPN_TO_KALSHI = {"GS": "GSW", "NO": "NOP", "NY": "NYK", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS"}
# 2 = regular season, 3 = playoffs, 5 = play-in. Preseason (1) and All-Star (4) are skipped.
KEEP_SEASON_TYPES = {2, 3, 5}
WORKERS = 6


def norm_team(abbr: str) -> str:
    return ESPN_TO_KALSHI.get(abbr, abbr)


def _client() -> httpx.Client:
    return httpx.Client(timeout=30, headers={"User-Agent": "nationaldataball/0.1 (personal research)"})


def _to_int(v):
    try:
        return int(float(str(v).replace("+", "")))
    except (TypeError, ValueError):
        return 100 if str(v).upper() == "EVEN" else None


def _parse_odds(comp: dict, home: str, away: str) -> dict:
    out = {"book_home_margin": None, "book_total": None, "book_home_ml": None, "book_away_ml": None}
    odds = comp.get("odds") or []
    if not odds:
        return out
    o = odds[0]
    out["book_total"] = o.get("overUnder")
    details = (o.get("details") or "").strip()
    if details.upper() in ("EVEN", "PK", "PICK"):
        out["book_home_margin"] = 0.0
    elif details:
        abbr, _, line = details.rpartition(" ")
        try:
            line = abs(float(line))
            fav = norm_team(abbr)
            out["book_home_margin"] = line if fav == home else -line if fav == away else None
        except ValueError:
            pass
    ml = o.get("moneyline") or {}
    for side, key in (("home", "book_home_ml"), ("away", "book_away_ml")):
        s = ml.get(side) or {}
        for when in ("close", "current", "open"):
            v = _to_int((s.get(when) or {}).get("odds"))
            if v is not None:
                out[key] = v
                break
    return out


def parse_scoreboard(data: dict) -> list[dict]:
    games = []
    for e in data.get("events", []):
        stype = e.get("season", {}).get("type")
        if stype not in KEEP_SEASON_TYPES:
            continue
        comp = e["competitions"][0]
        teams = {c["homeAway"]: c for c in comp["competitors"]}
        home, away = norm_team(teams["home"]["team"]["abbreviation"]), norm_team(teams["away"]["team"]["abbreviation"])
        start = datetime.fromisoformat(e["date"].replace("Z", "+00:00"))
        completed = bool(comp["status"]["type"].get("completed"))
        g = {
            "game_id": e["id"],
            "season": e["season"]["year"],
            "season_type": stype,
            "date_et": start.astimezone(ET).date().isoformat(),
            "start_utc": start.isoformat(),
            "home": home,
            "away": away,
            "home_score": _to_int(teams["home"].get("score")) if completed else None,
            "away_score": _to_int(teams["away"].get("score")) if completed else None,
            "completed": int(completed),
        }
        g.update(_parse_odds(comp, home, away))
        games.append(g)
    return games


def _minutes(v) -> float:
    try:
        if isinstance(v, str) and ":" in v:
            m, s = v.split(":")
            return int(m) + int(s) / 60
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def parse_box(summary: dict, game: dict) -> list[dict]:
    rows = []
    for block in summary.get("boxscore", {}).get("players", []):
        team = norm_team(block["team"]["abbreviation"])
        is_home = team == game["home"]
        opp = game["away"] if is_home else game["home"]
        for st in block.get("statistics", [])[:1]:
            keys = st["keys"]
            for a in st.get("athletes", []):
                if a.get("didNotPlay") or not a.get("stats"):
                    continue
                v = dict(zip(keys, a["stats"]))
                mins = _minutes(v.get("minutes"))
                if mins <= 0:
                    continue
                fg3m = (v.get("threePointFieldGoalsMade-threePointFieldGoalsAttempted") or "0-0").split("-")[0]
                fga = (v.get("fieldGoalsMade-fieldGoalsAttempted") or "0-0").split("-")[-1]
                fta = (v.get("freeThrowsMade-freeThrowsAttempted") or "0-0").split("-")[-1]
                rows.append({
                    "game_id": game["game_id"],
                    "player_id": a["athlete"]["id"],
                    "name": a["athlete"]["displayName"],
                    "team": team,
                    "opp": opp,
                    "home": int(is_home),
                    "date_et": game["date_et"],
                    "starter": int(bool(a.get("starter"))),
                    "min": mins,
                    "pts": _to_int(v.get("points")) or 0,
                    "reb": _to_int(v.get("rebounds")) or 0,
                    "ast": _to_int(v.get("assists")) or 0,
                    "fg3m": _to_int(fg3m) or 0,
                    "fga": _to_int(fga) or 0,
                    "fta": _to_int(fta) or 0,
                    "tov": _to_int(v.get("turnovers")) or 0,
                    "stl": _to_int(v.get("steals")) or 0,
                    "blk": _to_int(v.get("blocks")) or 0,
                })
    return rows


UPSERT_GAME = """
INSERT INTO nba_games (game_id, season, season_type, date_et, start_utc, home, away, home_score, away_score,
                       completed, book_home_margin, book_total, book_home_ml, book_away_ml)
VALUES (:game_id, :season, :season_type, :date_et, :start_utc, :home, :away, :home_score, :away_score,
        :completed, :book_home_margin, :book_total, :book_home_ml, :book_away_ml)
ON CONFLICT(game_id) DO UPDATE SET
    date_et = excluded.date_et, start_utc = excluded.start_utc,
    home_score = excluded.home_score, away_score = excluded.away_score, completed = excluded.completed,
    book_home_margin = COALESCE(excluded.book_home_margin, book_home_margin),
    book_total = COALESCE(excluded.book_total, book_total),
    book_home_ml = COALESCE(excluded.book_home_ml, book_home_ml),
    book_away_ml = COALESCE(excluded.book_away_ml, book_away_ml)
"""

PLAYER_COLS = ["game_id", "player_id", "name", "team", "opp", "home", "date_et", "starter", "min",
               "pts", "reb", "ast", "fg3m", "fga", "fta", "tov", "stl", "blk"]
INSERT_PLAYER = (
    f"INSERT OR REPLACE INTO nba_player_games ({', '.join(PLAYER_COLS)}) "
    f"VALUES ({', '.join(':' + c for c in PLAYER_COLS)})"
)


def _season_days(season: int, until: date) -> list[date]:
    d, end = date(season - 1, 10, 1), min(date(season, 6, 30), until)
    out = []
    while d <= end:
        out.append(d)
        d += timedelta(days=1)
    return out


def refresh(log=print) -> None:
    season = current_nba_season()
    today = today_et()
    horizon = today + timedelta(days=7)
    days = _season_days(season - 1, horizon) + _season_days(season, horizon)
    with session() as conn:
        done = {r["day"] for r in conn.execute("SELECT day FROM fetched_days WHERE sport = 'nba'")}
    # Re-fetch recent/upcoming days every time: scores finalize and lines move.
    todo = [d for d in days if d.isoformat() not in done or d >= today - timedelta(days=2)]
    log(f"NBA: fetching {len(todo)} scoreboard days")

    def fetch_day(client, d):
        r = client.get(f"{ESPN_NBA}/scoreboard", params={"dates": d.strftime("%Y%m%d")})
        r.raise_for_status()
        return d, parse_scoreboard(r.json())

    with _client() as client, ThreadPoolExecutor(WORKERS) as pool:
        results = list(pool.map(lambda d: fetch_day(client, d), todo))
    with session() as conn:
        for d, games in results:
            for g in games:
                conn.execute(UPSERT_GAME, g)
            if d < today - timedelta(days=2) and all(g["completed"] for g in games):
                conn.execute("INSERT OR IGNORE INTO fetched_days VALUES ('nba', ?)", (d.isoformat(),))
        pending = [dict(r) for r in conn.execute("SELECT * FROM nba_games WHERE completed = 1 AND box_fetched = 0")]

    log(f"NBA: fetching {len(pending)} box scores")

    def fetch_box(client, g):
        try:
            r = client.get(f"{ESPN_NBA}/summary", params={"event": g["game_id"]})
            r.raise_for_status()
            return g, parse_box(r.json(), g)
        except httpx.HTTPError:
            return g, None

    done_count = 0
    with _client() as client, ThreadPoolExecutor(WORKERS) as pool:
        for g, rows in pool.map(lambda g: fetch_box(client, g), pending):
            if rows is None:
                continue
            with session() as conn:
                conn.executemany(INSERT_PLAYER, rows)
                conn.execute("UPDATE nba_games SET box_fetched = 1 WHERE game_id = ?", (g["game_id"],))
            done_count += 1
            if done_count % 100 == 0:
                log(f"NBA: {done_count}/{len(pending)} box scores")
    log("NBA: done")


def has_data() -> bool:
    with session() as conn:
        return conn.execute("SELECT COUNT(*) FROM nba_games WHERE completed = 1").fetchone()[0] > 0


def load_games() -> pd.DataFrame:
    with session() as conn:
        g = pd.read_sql_query("SELECT * FROM nba_games", conn)
    g["date"] = pd.to_datetime(g["date_et"])
    g["kickoff"] = pd.to_datetime(g["start_utc"], utc=True)
    g["completed"] = g["completed"].astype(bool)
    return g


def load_player_games() -> pd.DataFrame:
    with session() as conn:
        p = pd.read_sql_query("SELECT * FROM nba_player_games", conn)
    p["date"] = pd.to_datetime(p["date_et"])
    p["pra"] = p["pts"] + p["reb"] + p["ast"]
    p["season"] = np.where(p["date"].dt.month >= 9, p["date"].dt.year + 1, p["date"].dt.year)
    return p
