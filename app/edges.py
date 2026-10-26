"""Join model probabilities with live Kalshi prices to surface +EV contracts."""
import math
import threading
import time
from datetime import date, datetime, timedelta, timezone

from .data import injuries, kalshi, nba, nfl, odds
from .models import games as gm
from .models import pricing, props

STATE_TTL = 300
_state: dict = {}
_lock = threading.Lock()

TYPE_LABELS = {
    "moneyline": "Moneyline", "spread": "Spread", "total": "Total", "team_total": "Team total",
    "pts": "Points", "reb": "Rebounds", "ast": "Assists", "fg3m": "Threes", "pra": "Pts+Reb+Ast",
    "pass_yds": "Pass yds", "pass_tds": "Pass TDs", "rush_yds": "Rush yds", "rec_yds": "Rec yds",
    "receptions": "Receptions", "tds": "Touchdowns",
}


def invalidate() -> None:
    with _lock:
        _state.clear()


def sport_state(sport: str, settings: dict) -> dict:
    """Upcoming game predictions + player projections, cached for a few minutes."""
    key = (sport, settings["model_weight"])
    with _lock:
        hit = _state.get(key)
        if hit and time.time() - hit["t"] < STATE_TTL:
            return hit
    games, model = gm.upcoming(sport, settings)
    projections, index, notes, prop_lines = {}, None, {}, {}
    if games:
        inj = injuries.lookup(sport)

        def injury_of(team, name, espn_id):
            return injuries.find(inj, team, name, espn_id)

        if sport == "nba":
            pg = nba.load_player_games()
            if not pg.empty:
                projections, index, notes = props.nba_projections(pg, games, injury_of)
        else:
            pw = nfl.load_player_weeks()
            if not pw.empty:
                projections, index, notes = props.nfl_projections(pw, games, injury_of)
        for g in games:
            g["flags"] = g["flags"] + notes.get(g["id"], [])
            if g.get("odds_event_id"):
                prop_lines[g["id"]] = odds.prop_consensus(sport, g["odds_event_id"])
    st = {"t": time.time(), "games": games, "model": model, "projections": projections, "index": index,
          "prop_lines": prop_lines}
    with _lock:
        _state[key] = st
    return st


def _game_lookup(games: list[dict]) -> dict:
    lk = {}
    for delta in (0, -1, 1):  # exact dates win over +/- 1 day (time zone edge cases)
        for g in games:
            d = date.fromisoformat(g["date"]) + timedelta(days=delta)
            lk.setdefault((d.isoformat(), frozenset((g["home"], g["away"]))), g)
    return lk


def _team_prob(m: dict, g: dict, view: dict) -> float | None:
    """P(yes) for a game-level market under a given view (model, book or fair)."""
    sd = view["sd"]
    t, team = m["type"], m.get("team")
    if t == "total":
        return pricing.prob_over(view["total"], sd["total"], m["strike"])
    if team not in (g["home"], g["away"]):
        return None
    sign = 1 if team == g["home"] else -1
    if t == "moneyline":
        return gm.prob_margin_over(m["sport"], sign * view["margin"], sd["margin"], 0)
    if t == "spread":
        return gm.prob_margin_over(m["sport"], sign * view["margin"], sd["margin"], m["strike"])
    if t == "team_total":
        pts = view["home_pts"] if team == g["home"] else view["away_pts"]
        return pricing.prob_over(pts, sd["team"], m["strike"])
    return None


def _mid(m: dict) -> float | None:
    if m["yes_bid"] is not None and m["yes_ask"] is not None:
        return (m["yes_bid"] + m["yes_ask"]) / 2
    return m["yes_ask"] or m["yes_bid"]


def evaluate_market(m: dict, g: dict, st: dict, settings: dict) -> dict | None:
    flags = list(g.get("flags", []))
    mid = _mid(m)
    book_prob = None
    player = None
    if m["type"] in kalshi.GAME_TYPES:
        model_p = _team_prob(m, g, g["model"])
        fair_p = _team_prob(m, g, g["fair"])
        has_book = g["book"]["margin"] is not None
        if has_book:
            book_p = {"margin": g["book"]["margin"], "total": g["book"]["total"] or g["model"]["total"],
                      "sd": g["book"]["sd"]}
            book_p["home_pts"] = (book_p["total"] + book_p["margin"]) / 2
            book_p["away_pts"] = (book_p["total"] - book_p["margin"]) / 2
            book_prob = _team_prob(m, g, book_p)
        elif mid is not None and m["yes_ask"] and m["yes_bid"] and m["yes_ask"] - m["yes_bid"] <= 0.10:
            # No sportsbook line: Kalshi's own price is the market prior.
            fair_p = settings["model_weight"] * model_p + (1 - settings["model_weight"]) * mid
            flags.append("No sportsbook line: blended with Kalshi price")
        else:
            flags.append("No sportsbook line and wide Kalshi book: model only")
    else:
        if not st["index"]:
            return None
        pid = st["index"].find(m["team"], m["player_key"], m["player_name"])
        proj = st["projections"].get((pid, g["id"]))
        if not proj:
            return None
        k = math.floor(m["strike"]) + 1
        model_p = props.prob_at_least(proj, m["type"], k)
        if model_p is None:
            return None
        player = {"id": pid, "name": proj["name"], "mean": proj[m["type"]]["mean"], "min": proj.get("min")}
        inj = proj.get("injury")
        if inj:
            flags.append(f"{inj['status']}: {inj['comment'][:90]}")
        elif proj.get("missed_last"):
            flags.append(f"{proj['name']} missed team's last game (check injury report)")
        if proj.get("min_boost", 0) >= 2:
            flags.append(f"+{proj['min_boost']:.0f} min projected (teammate out)")
        lines = st["prop_lines"].get(g["id"], {})
        quotes = lines.get((m["player_key"], m["type"])) or lines.get((props.name_key(proj["name"]), m["type"]))
        if quotes:
            book_prob = odds.book_prop_prob(quotes, k, proj[m["type"]]["disp"])
        w = settings["model_weight"]
        if book_prob is not None:
            fair_p = w * model_p + (1 - w) * book_prob
        elif mid is not None and m["yes_ask"] and m["yes_bid"] and m["yes_ask"] - m["yes_bid"] <= 0.10:
            # No sportsbook prop line pulled: use Kalshi's own mid as the market prior.
            fair_p = w * model_p + (1 - w) * mid
        else:
            fair_p = model_p
            flags.append("Wide/empty Kalshi book: model only")
    if fair_p is None:
        return None
    market_ref = book_prob if book_prob is not None else mid
    if market_ref is not None and abs(model_p - market_ref) > 0.15:
        flags.append("Model far from market: likely news we don't have (injury, lineup, QB). Verify first.")

    coef = settings["kalshi_fee_coef"]
    options = []
    if m["yes_ask"] is not None:
        options.append(("yes", fair_p, pricing.evaluate_side(fair_p, m["yes_ask"], coef)))
    if m["yes_bid"] is not None:
        options.append(("no", 1 - fair_p, pricing.evaluate_side(1 - fair_p, 1 - m["yes_bid"], coef)))
    if not options:
        return None
    side, prob, ev = max(options, key=lambda o: o[2]["roi"])
    stake = pricing.recommend_stake(prob, ev["cost"], settings)
    spread = m["yes_ask"] - m["yes_bid"] if m["yes_ask"] is not None and m["yes_bid"] is not None else None
    return {
        "ticker": m["ticker"],
        "sport": m["sport"],
        "type": m["type"],
        "type_label": TYPE_LABELS.get(m["type"], m["type"]),
        "title": m["title"],
        "game": f"{g['away']} @ {g['home']}",
        "game_id": g["id"],
        "kickoff": g["kickoff"],
        "strike": m["strike"],
        "yes_bid": m["yes_bid"],
        "yes_ask": m["yes_ask"],
        "kalshi_mid": mid,
        "model_prob": model_p,
        "book_prob": book_prob,
        "fair_prob": fair_p,
        "side": side,
        "side_prob": prob,
        "price": ev["price"],
        "cost": ev["cost"],
        "ev": ev["ev"],
        "roi": ev["roi"],
        "stake": stake,
        "player": player,
        "spread": spread,
        # Far-tail contracts are where distribution assumptions are weakest; the UI hides them by default.
        "tail": ev["price"] < 0.08 or ev["price"] > 0.92,
        "flags": flags,
    }


def edges(sport: str, settings: dict, force: bool = False) -> list[dict]:
    st = sport_state(sport, settings)
    if not st["games"]:
        return []
    lookup = _game_lookup(st["games"])
    now = datetime.now(timezone.utc)
    rows = []
    for m in kalshi.get_markets(sport, force=force):
        g = lookup.get((m["date"], frozenset(m["teams"])))
        if not g or datetime.fromisoformat(g["kickoff"]) <= now:
            continue  # unknown game, or already started (live prices need a live model)
        r = evaluate_market(m, g, st, settings)
        if r:
            rows.append(r)
    rows.sort(key=lambda r: -r["roi"])
    return rows


def games_view(sport: str, settings: dict) -> list[dict]:
    """Upcoming games with model/book/fair numbers plus Kalshi's implied win probability."""
    st = sport_state(sport, settings)
    lookup = _game_lookup(st["games"])
    kalshi_home = {}
    try:
        for m in kalshi.get_markets(sport):
            if m["type"] != "moneyline":
                continue
            g = lookup.get((m["date"], frozenset(m["teams"])))
            if g and m.get("team") == g["home"]:
                kalshi_home[g["id"]] = _mid(m)
    except Exception:
        pass
    return [{**g, "kalshi_home_win": kalshi_home.get(g["id"])} for g in st["games"]]


def projections_view(sport: str, settings: dict) -> list[dict]:
    st = sport_state(sport, settings)
    stats = props.NBA_STATS if sport == "nba" else list(props.NFL_STATS)
    out, seen = [], set()
    for p in st["projections"].values():  # ordered by kickoff, so the first entry is each player's next game
        if p["id"] in seen:
            continue
        seen.add(p["id"])
        row = {k: p[k] for k in ("id", "name", "team", "opp", "kickoff", "games", "missed_last", "last_played")}
        row["min"] = p.get("min")
        row["position"] = p.get("position")
        row["injury"] = p["injury"]["status"] if p.get("injury") else None
        for s in stats:
            row[s] = p[s]["mean"]
        out.append(row)
    return sorted(out, key=lambda r: (r["kickoff"], r["team"], -(r.get("min") or r.get("pass_yds", 0) + r.get("rec_yds", 0) + r.get("rush_yds", 0))))


def kickoff_for(ticker: str, settings: dict) -> str | None:
    """Kickoff time of the (upcoming) game a Kalshi ticker belongs to, if we know it."""
    sport, _ = kalshi.series_info(ticker)
    parts = ticker.split("-")
    if sport not in ("nfl", "nba") or len(parts) < 2:
        return None
    try:
        d = datetime.strptime(parts[1][:7].title(), "%y%b%d").date()
    except ValueError:
        return None
    teams = kalshi.split_teams(sport, parts[1][7:])
    if not teams:
        return None
    g = _game_lookup(sport_state(sport, settings)["games"]).get((d.isoformat(), frozenset(teams)))
    return g["kickoff"] if g else None
