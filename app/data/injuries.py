"""Current injury designations from ESPN's public injury feed (NFL and NBA)."""
import re
import threading
import time

import httpx

from ..config import ESPN_INJURIES
from ..models.props import name_key
from . import teams

PATHS = {"nfl": "football/nfl", "nba": "basketball/nba"}
# Designations that mean the player should be treated as not playing.
OUT = {"Out", "Doubtful", "Injured Reserve", "Suspension", "Physically Unable to Perform", "Not Active"}
CACHE_SECONDS = 600

_cache: dict[str, tuple[float, list]] = {}
_lock = threading.Lock()


def _parse(sport: str, data: dict) -> list[dict]:
    rows = []
    for block in data.get("injuries", []):
        team = teams.code(sport, block.get("displayName", ""))
        for i in block.get("injuries", []):
            status = i.get("status", "")
            if not status or status == "Active":
                continue
            a = i.get("athlete", {})
            espn_id = None
            for link in a.get("links", []):
                m = re.search(r"/id/(\d+)", link.get("href", ""))
                if m:
                    espn_id = m.group(1)
                    break
            rows.append({
                "team": team, "player": a.get("displayName", ""), "espn_id": espn_id, "status": status,
                "out": status in OUT, "comment": i.get("shortComment", ""), "updated": i.get("date"),
            })
    return rows


def get(sport: str, force: bool = False) -> list[dict]:
    with _lock:
        hit = _cache.get(sport)
        if hit and not force and time.time() - hit[0] < CACHE_SECONDS:
            return hit[1]
    try:
        with httpx.Client(timeout=30) as client:
            r = client.get(ESPN_INJURIES.format(path=PATHS[sport]))
            r.raise_for_status()
            rows = _parse(sport, r.json())
    except httpx.HTTPError:
        return hit[1] if hit else []
    with _lock:
        _cache[sport] = (time.time(), rows)
    return rows


def lookup(sport: str) -> dict:
    """Index for matching: ESPN athlete id (NBA box scores use it) and (team, name key)."""
    rows = get(sport)
    return {
        "by_id": {r["espn_id"]: r for r in rows if r["espn_id"]},
        "by_name": {(r["team"], name_key(r["player"])): r for r in rows},
    }


def find(idx: dict, team: str, name: str, espn_id: str | None = None) -> dict | None:
    if espn_id and espn_id in idx["by_id"]:
        return idx["by_id"][espn_id]
    return idx["by_name"].get((team, name_key(name)))
