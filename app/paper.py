"""Paper trading: record every qualifying edge as a no-money bet and grade it like a real one.

This is how to tell whether the model has an edge before risking money. Each market + side
is paper-bet once, at the price first seen, so closing line value (CLV) and results can be
measured per market type, edge size and price range.
"""
import threading
from datetime import datetime, timezone
import time
import traceback

from . import bets, edges
from .edges import ladder_key
from .db import get_settings, session

INTERVAL_SECONDS = 15 * 60
MAX_SPREAD = 0.06
# Long shots were the model's worst calibration: auto paper bets under 15c won 1 of 42
# against ~6 expected. Keep the all-edges book in the range where streaks stay survivable.
PRICE_RANGE = (0.25, 0.85)
# Shortlist picks lean on sportsbook prices; don't paper-bet off lines older than this.
MAX_LINE_AGE_HOURS = 12


def qualifies(r: dict, min_edge: float) -> bool:
    return (r["roi"] >= min_edge
            and PRICE_RANGE[0] <= r["price"] <= PRICE_RANGE[1]
            and r["spread"] is not None and r["spread"] <= MAX_SPREAD
            and not any(f.startswith("Model far from market") for f in r["flags"]))


def record(r: dict, settings: dict, source: str, prefix: str = "paper") -> dict | None:
    ext = f"{prefix}:{ladder_key(r)}"
    if bets.has_ext_id(ext):
        return None
    return bets.create({
        "ext_id": ext,
        "sport": r["sport"], "ticker": r["ticker"], "description": f"{r['side'].upper()} {r['title']} ({r['game']})",
        "side": r["side"], "price": r["price"], "contracts": max(1, r["stake"]["contracts"]),
        "fair_prob": r["side_prob"], "game_start": r["kickoff"], "mode": "paper", "source": source,
        "market_type": r["type"], "edge": r["roi"],
        "model_prob": r["model_prob"], "book_prob": r["book_prob"], "tags": edges.situation_tags(r),
    }, settings["kalshi_fee_coef"])


_last_snap: dict[str, tuple] = {}
SNAPSHOT_COLS = ["taken_at", "sport", "ticker", "market_type", "game_id", "kickoff", "strike", "yes_bid", "yes_ask",
                 "model_prob", "book_prob", "fair_prob", "player_id", "proj_mean", "min_boost", "usage_boost",
                 "carry_boost"]


def snapshot(sport: str, settings: dict) -> int:
    """Record every priced market whose price or probabilities moved since its last snapshot."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    for r in edges.edges(sport, settings):
        p = r["player"] or {}
        state = (r["yes_bid"], r["yes_ask"], round(r["model_prob"], 3),
                 None if r["book_prob"] is None else round(r["book_prob"], 3))
        if _last_snap.get(r["ticker"]) == state:
            continue
        _last_snap[r["ticker"]] = state
        rows.append((now, sport, r["ticker"], r["type"], r["game_id"], r["kickoff"], r["strike"],
                     r["yes_bid"], r["yes_ask"], r["model_prob"], r["book_prob"], r["fair_prob"],
                     p.get("id"), p.get("mean"), p.get("min_boost"), p.get("usage_boost"), p.get("carry_boost")))
    with session() as conn:
        conn.executemany(f"INSERT INTO market_snapshots ({', '.join(SNAPSHOT_COLS)}) "
                         f"VALUES ({', '.join('?' * len(SNAPSHOT_COLS))})", rows)
    return len(rows)


def run_once() -> dict:
    settings = get_settings()
    created = 0
    for sport in ("nfl", "nba"):
        best = {}
        for r in edges.edges(sport, settings, force=True):
            if qualifies(r, settings["paper_min_edge"]):
                k = ladder_key(r)
                if k not in best or r["roi"] > best[k]["roi"]:
                    best[k] = r
        for r in best.values():
            if record(r, settings, "auto"):
                created += 1
    return {"created": created, **bets.sync()}


def lines_fresh(lines_at: str | None) -> bool:
    if not lines_at:
        return False
    age = datetime.now(timezone.utc) - datetime.fromisoformat(lines_at)
    return age.total_seconds() <= MAX_LINE_AGE_HOURS * 3600


def run_shortlist() -> int:
    """Paper-bet each shortlist pick once (its own book, separate from the all-edges paper trader),
    so the shortlist's own CLV and results can be judged before real money follows it."""
    settings = get_settings()
    created = 0
    for sport in ("nfl", "nba"):
        res = edges.shortlist(sport, settings, force=True)
        if not lines_fresh(res["lines_at"]):
            continue
        for r in res["picks"]:
            if record(r, settings, "shortlist", prefix="paper-sl"):
                created += 1
    return created


def _loop() -> None:
    while True:
        try:
            s = get_settings()
            if s["paper_shortlist"] >= 1:
                run_shortlist()
            if s["auto_paper"] >= 1:
                run_once()
            else:
                bets.sync()  # keep closing prices and results current for logged bets
            for sport in ("nfl", "nba"):
                snapshot(sport, s)
        except Exception:
            traceback.print_exc()
        time.sleep(INTERVAL_SECONDS)


def start_background() -> None:
    threading.Thread(target=_loop, daemon=True, name="paper-trader").start()
