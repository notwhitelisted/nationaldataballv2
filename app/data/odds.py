"""Sportsbook lines from The Odds API, turned into a sharp-weighted no-vig consensus.

Credit usage (free tier = 500/month): cost = markets x regions, and a list of up to 10
bookmakers counts as one region. So game lines cost 3 credits per sport per pull, and
player props cost one credit per prop market per game. Nothing here runs automatically;
pulls only happen when the user asks, and the remaining quota is shown in the UI.
"""
from collections import defaultdict
from datetime import datetime, timezone

import httpx
from scipy import stats

from ..config import ET, ODDS_API_BASE, ODDS_API_KEY
from ..db import get_meta, session, set_meta
from ..models import pricing
from ..models.props import name_key
from . import teams

SPORT_KEYS = {"nfl": "americanfootball_nfl", "nba": "basketball_nba"}
BOOKS = "pinnacle,draftkings,fanduel,betmgm,williamhill_us,betonlineag,lowvig,bovada,betrivers,fanatics"
# Market-making books move first and carry less noise; weight them more in the consensus.
BOOK_WEIGHT = {"pinnacle": 3.0, "betonlineag": 1.5, "lowvig": 1.5}

PROP_MARKETS = {
    "nba": {"player_points": "pts", "player_rebounds": "reb", "player_assists": "ast",
            "player_threes": "fg3m", "player_points_rebounds_assists": "pra"},
    "nfl": {"player_pass_yds": "pass_yds", "player_pass_tds": "pass_tds", "player_rush_yds": "rush_yds",
            "player_reception_yds": "rec_yds", "player_receptions": "receptions", "player_anytime_td": "tds"},
}
# One-sided markets (anytime TD "Yes" only) can't be de-vigged; assume a typical ~7% hold.
ONE_SIDED_HOLD = 1.07


def enabled() -> bool:
    return bool(ODDS_API_KEY)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _get(path: str, params: dict):
    if not ODDS_API_KEY:
        raise RuntimeError("ODDS_API_KEY is not set. Add it to .env")
    with httpx.Client(timeout=30) as client:
        r = client.get(f"{ODDS_API_BASE}{path}", params={**params, "apiKey": ODDS_API_KEY})
    if "x-requests-remaining" in r.headers:
        set_meta("odds_quota", {
            "remaining": r.headers.get("x-requests-remaining"),
            "used": r.headers.get("x-requests-used"),
            "last_cost": r.headers.get("x-requests-last"),
            "at": _now(),
        })
    r.raise_for_status()
    return r.json()


def quota() -> dict | None:
    return get_meta("odds_quota")


# ---------------------------------------------------------------- game lines

def refresh_lines(sport: str) -> int:
    data = _get(f"/sports/{SPORT_KEYS[sport]}/odds",
                {"markets": "h2h,spreads,totals", "oddsFormat": "american", "bookmakers": BOOKS})
    now = _now()
    rows = []
    for e in data:
        home, away = teams.code(sport, e["home_team"]), teams.code(sport, e["away_team"])
        if not home or not away:
            continue
        for b in e.get("bookmakers", []):
            for mk in b.get("markets", []):
                for o in mk.get("outcomes", []):
                    side = "home" if o["name"] == e["home_team"] else "away" if o["name"] == e["away_team"] else o["name"].lower()
                    rows.append((sport, e["id"], e["commence_time"], home, away, b["key"], mk["key"], side,
                                 o.get("point"), int(o["price"]), now))
    with session() as conn:
        conn.executemany("INSERT INTO odds_lines VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    set_meta(f"odds_lines_at_{sport}", now)
    return len(data)


def _wavg(pairs):
    tw = sum(w for _, w in pairs)
    return sum(v * w for v, w in pairs) / tw if tw else None


def consensus(sport: str, sd: dict) -> dict:
    """Latest pull -> {(date_et, frozenset(teams)): consensus margin/total/home win prob}.

    Each book's spread (or total) and its two prices are converted to a fair expected margin:
    if a home -3.5 cover is 55% after removing the vig, the implied mean margin is
    3.5 + sd * Phi^-1(0.55). Books are then averaged with sharp books weighted up.
    """
    with session() as conn:
        last = conn.execute("SELECT MAX(fetched_at) FROM odds_lines WHERE sport = ?", (sport,)).fetchone()[0]
        if not last:
            return {}
        rows = conn.execute("SELECT * FROM odds_lines WHERE sport = ? AND fetched_at = ?", (sport, last)).fetchall()
    events = {}
    for r in rows:
        ev = events.setdefault(r["event_id"], {"r": r, "books": defaultdict(lambda: defaultdict(dict))})
        ev["books"][r["book"]][r["market"]][r["outcome"]] = (r["point"], r["price"])
    out = {}
    for eid, ev in events.items():
        margins, totals, wins = [], [], []
        for book, mk in ev["books"].items():
            w = BOOK_WEIGHT.get(book, 1.0)
            h2h = mk.get("h2h", {})
            if "home" in h2h and "away" in h2h:
                dv = pricing.devig_two_way(h2h["home"][1], h2h["away"][1])
                if dv:
                    wins.append((dv[0], w))
            sp = mk.get("spreads", {})
            if "home" in sp and "away" in sp and sp["home"][0] is not None:
                dv = pricing.devig_two_way(sp["home"][1], sp["away"][1])
                if dv:
                    margins.append((-sp["home"][0] + sd["margin"] * stats.norm.ppf(dv[0]), w))
            tot = mk.get("totals", {})
            if "over" in tot and "under" in tot and tot["over"][0] == tot["under"][0]:
                dv = pricing.devig_two_way(tot["over"][1], tot["under"][1])
                if dv:
                    totals.append((tot["over"][0] + sd["total"] * stats.norm.ppf(dv[0]), w))
        home_win = _wavg(wins)
        margin = _wavg(margins)
        if margin is None and home_win is not None:
            margin = pricing.margin_from_win_prob(home_win, sd["margin"])
        r = ev["r"]
        start = datetime.fromisoformat(r["commence_time"].replace("Z", "+00:00"))
        books = sorted(ev["books"])
        out[(start.astimezone(ET).date().isoformat(), frozenset((r["home"], r["away"])))] = {
            "event_id": eid, "margin": margin, "total": _wavg(totals), "home_win": home_win,
            "books": books, "fetched_at": last,
        }
    return out


# ---------------------------------------------------------------- player props

def prop_cost(sport: str) -> int:
    return len(PROP_MARKETS[sport])


def refresh_props(sport: str, event_id: str) -> int:
    data = _get(f"/sports/{SPORT_KEYS[sport]}/events/{event_id}/odds",
                {"markets": ",".join(PROP_MARKETS[sport]), "oddsFormat": "american", "bookmakers": BOOKS})
    now = _now()
    rows = []
    for b in data.get("bookmakers", []):
        for mk in b.get("markets", []):
            for o in mk.get("outcomes", []):
                if not o.get("description"):
                    continue
                rows.append((sport, event_id, b["key"], mk["key"], o["description"], o["name"].lower(),
                             o.get("point"), int(o["price"]), now))
    with session() as conn:
        conn.executemany("INSERT INTO odds_props VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    return len(rows)


def prop_consensus(sport: str, event_id: str) -> dict:
    """{(player name_key, stat): [{"book", "line", "p_over", "w"}]} from the latest pull for a game."""
    with session() as conn:
        last = conn.execute("SELECT MAX(fetched_at) FROM odds_props WHERE event_id = ?", (event_id,)).fetchone()[0]
        if not last:
            return {}
        rows = conn.execute("SELECT * FROM odds_props WHERE event_id = ? AND fetched_at = ?", (event_id, last)).fetchall()
    grouped = defaultdict(dict)
    for r in rows:
        stat = PROP_MARKETS[sport].get(r["market"])
        if stat:
            grouped[(r["player"], stat, r["book"], r["point"])][r["outcome"]] = r["price"]
    out = defaultdict(list)
    for (player, stat, book, point), sides in grouped.items():
        if "over" in sides and "under" in sides:
            dv = pricing.devig_two_way(sides["over"], sides["under"])
            p = dv[0] if dv else None
            line = point
        elif "yes" in sides:
            dv = pricing.devig_two_way(sides["yes"], sides["no"]) if "no" in sides else None
            p = dv[0] if dv else pricing.american_to_prob(sides["yes"]) / ONE_SIDED_HOLD
            line = 0.5
        else:
            continue
        if p is not None and line is not None:
            out[(name_key(player), stat)].append({"book": book, "line": line, "p_over": p, "w": BOOK_WEIGHT.get(book, 1.0)})
    return dict(out)


def book_prop_prob(quotes: list[dict], k: int, disp: float) -> float | None:
    """P(stat >= k) implied by sportsbook over/unders near Kalshi's threshold (None if none are near).

    For each quote, find the mean that reproduces the book's over probability under our count
    distribution, then read off P(X >= k). Quotes are averaged with sharp books weighted up.
    """
    vals = []
    for q in quotes:
        # Main lines only: extrapolating far into the tail is distribution guesswork, so skip it.
        if abs((k - 0.5) - q["line"]) > 0.15 * q["line"] + 1.5:
            continue
        mu = pricing.implied_mean(q["line"], q["p_over"], disp)
        if mu is not None:
            vals.append((pricing.count_prob_at_least(mu, disp, k), q["w"]))
    return _wavg(vals)
