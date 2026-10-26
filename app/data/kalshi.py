"""Read-only Kalshi market data via the public trade API (no account or key needed)."""
import re
import threading
import time
from datetime import datetime

import httpx

from ..config import KALSHI_BASE

SERIES = {
    "nfl": {
        "moneyline": "KXNFLGAME", "spread": "KXNFLSPREAD", "total": "KXNFLTOTAL", "team_total": "KXNFLTEAMTOTAL",
        "pass_yds": "KXNFLPASSYDS", "pass_tds": "KXNFLPASSTDS", "rush_yds": "KXNFLRSHYDS",
        "rec_yds": "KXNFLRECYDS", "receptions": "KXNFLREC", "tds": "KXNFLTD",
    },
    "nba": {
        "moneyline": "KXNBAGAME", "spread": "KXNBASPREAD", "total": "KXNBATOTAL", "team_total": "KXNBATEAMTOTAL",
        "pts": "KXNBAPTS", "reb": "KXNBAREB", "ast": "KXNBAAST", "fg3m": "KXNBA3PT", "pra": "KXNBAPRA",
    },
}
GAME_TYPES = {"moneyline", "spread", "total", "team_total"}

TEAM_CODES = {
    "nfl": {"ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAC",
            "KC", "LAC", "LAR", "LV", "MIA", "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB",
            "TEN", "WAS"},
    "nba": {"ATL", "BKN", "BOS", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW", "HOU", "IND", "LAC", "LAL",
            "MEM", "MIA", "MIL", "MIN", "NOP", "NYK", "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR",
            "UTA", "WAS"},
}

_cache: dict[str, tuple[float, list]] = {}
_lock = threading.Lock()
CACHE_SECONDS = 60


def _price(v) -> float | None:
    try:
        p = float(v)
    except (TypeError, ValueError):
        return None
    return p if 0 < p < 1 else None


def split_teams(sport: str, codes: str) -> tuple[str, str] | None:
    """Split a concatenated pair like 'CHISEA' or 'KCLV' into its two team codes."""
    known = TEAM_CODES[sport]
    for n in (2, 3, 4):
        a, b = codes[:n], codes[n:]
        if a in known and b in known:
            return a, b
    return None


def parse_market(m: dict, sport: str, mtype: str) -> dict | None:
    parts = m["ticker"].split("-")
    body = parts[1]
    try:
        game_date = datetime.strptime(body[:7].title(), "%y%b%d").date()
    except ValueError:
        return None
    teams = split_teams(sport, body[7:])
    if not teams:
        return None
    yes_bid, yes_ask = _price(m.get("yes_bid_dollars")), _price(m.get("yes_ask_dollars"))
    out = {
        "ticker": m["ticker"],
        "event_ticker": m["event_ticker"],
        "sport": sport,
        "type": mtype,
        "title": m.get("title") or m["ticker"],
        "date": game_date.isoformat(),
        "teams": list(teams),
        "strike": m.get("floor_strike"),
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "close_time": m.get("close_time"),
    }
    if mtype == "moneyline":
        out["team"] = parts[2]
    elif mtype in ("spread", "team_total"):
        out["team"] = re.match(r"[A-Z]+", parts[2]).group(0)
    elif mtype != "total":
        seg = parts[2]
        team = next((t for t in teams if seg.startswith(t)), None)
        if team is None or out["strike"] is None:
            return None
        # e.g. 'PHIMBROWN0' -> team PHI, key 'MBROWN', jersey 0
        mm = re.match(r"([A-Z]+?)(\d*)$", seg[len(team):])
        out["team"] = team
        out["player_key"] = mm.group(1) if mm else seg[len(team):]
        out["player_name"] = out["title"].split(":")[0].strip()
    return out


def _fetch_series(client: httpx.Client, series: str) -> list[dict]:
    markets, cursor = [], None
    while True:
        params = {"series_ticker": series, "status": "open", "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        r = client.get(f"{KALSHI_BASE}/markets", params=params)
        r.raise_for_status()
        d = r.json()
        markets += d.get("markets", [])
        cursor = d.get("cursor")
        if not cursor:
            return markets


def get_markets(sport: str, force: bool = False) -> list[dict]:
    with _lock:
        hit = _cache.get(sport)
        if hit and not force and time.time() - hit[0] < CACHE_SECONDS:
            return hit[1]
    out = []
    with httpx.Client(timeout=30) as client:
        for mtype, series in SERIES[sport].items():
            try:
                raw = _fetch_series(client, series)
            except httpx.HTTPError:
                continue
            for m in raw:
                p = parse_market(m, sport, mtype)
                if p:
                    out.append(p)
    with _lock:
        _cache[sport] = (time.time(), out)
    return out


def get_markets_by_ticker(tickers: list[str]) -> dict[str, dict]:
    """Raw market objects for many tickers (any status), 50 per request."""
    out = {}
    with httpx.Client(timeout=30) as client:
        for i in range(0, len(tickers), 50):
            chunk = tickers[i:i + 50]
            r = client.get(f"{KALSHI_BASE}/markets", params={"tickers": ",".join(chunk), "limit": 1000})
            r.raise_for_status()
            for m in r.json().get("markets", []):
                out[m["ticker"]] = m
    return out


def series_info(ticker: str) -> tuple[str, str | None]:
    """(sport, market type) for a ticker, e.g. KXNBAPTS-... -> ('nba', 'pts')."""
    series = ticker.split("-")[0]
    for sport, types in SERIES.items():
        for mtype, s in types.items():
            if s == series:
                return sport, mtype
    for prefix, sport in (("KXNFL", "nfl"), ("KXNBA", "nba"), ("KXWNBA", "wnba"), ("KXMLB", "mlb"), ("KXNHL", "nhl"),
                          ("KXNCAAF", "ncaaf"), ("KXNCAAMB", "ncaab"), ("KXMVE", "parlay")):
        if series.startswith(prefix):
            return sport, None
    return "other", None


def get_market(ticker: str) -> dict:
    with httpx.Client(timeout=30) as client:
        r = client.get(f"{KALSHI_BASE}/markets/{ticker}")
        r.raise_for_status()
        return r.json()["market"]
