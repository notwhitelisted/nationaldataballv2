"""Paper trading: record every qualifying edge as a no-money bet and grade it like a real one.

This is how to tell whether the model has an edge before risking money. Each market + side
is paper-bet once, at the price first seen, so closing line value (CLV) and results can be
measured per market type, edge size and price range.
"""
import re
import threading
import time
import traceback

from . import bets, edges
from .db import get_settings

INTERVAL_SECONDS = 15 * 60
MAX_SPREAD = 0.06


def qualifies(r: dict, min_edge: float) -> bool:
    return (r["roi"] >= min_edge
            and 0.10 <= r["price"] <= 0.90
            and r["spread"] is not None and r["spread"] <= MAX_SPREAD
            and not any(f.startswith("Model far from market") for f in r["flags"]))


def ladder_key(r: dict) -> str:
    """Kalshi lists ladders (PHI by 3.5 / 4.5 / 7.5 ...). Rungs are highly correlated, so the
    paper book holds one position per ladder: per game for moneyline/total, per team for
    spreads/team totals, per player + stat for props."""
    parts = r["ticker"].split("-")
    event = "-".join(parts[:2])
    if r["type"] in ("moneyline", "total"):
        return event
    if r["type"] in ("spread", "team_total"):
        return f"{event}-{re.match(r'[A-Z]+', parts[2]).group(0)}"
    return "-".join(parts[:3])


def record(r: dict, settings: dict, source: str) -> dict | None:
    ext = f"paper:{ladder_key(r)}"
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


def _loop() -> None:
    while True:
        try:
            if get_settings()["auto_paper"] >= 1:
                run_once()
            else:
                bets.sync()  # keep closing prices and results current for logged bets
        except Exception:
            traceback.print_exc()
        time.sleep(INTERVAL_SECONDS)


def start_background() -> None:
    threading.Thread(target=_loop, daemon=True, name="paper-trader").start()
