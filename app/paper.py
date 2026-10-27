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
from .db import get_settings

INTERVAL_SECONDS = 15 * 60
MAX_SPREAD = 0.06
# Shortlist picks lean on sportsbook prices; don't paper-bet off lines older than this.
MAX_LINE_AGE_HOURS = 12


def qualifies(r: dict, min_edge: float) -> bool:
    return (r["roi"] >= min_edge
            and 0.10 <= r["price"] <= 0.90
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
    }, settings["kalshi_fee_coef"])


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
        except Exception:
            traceback.print_exc()
        time.sleep(INTERVAL_SECONDS)


def start_background() -> None:
    threading.Thread(target=_loop, daemon=True, name="paper-trader").start()
