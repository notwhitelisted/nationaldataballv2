import threading
import traceback
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import bets, edges, paper
from .config import STATIC_DIR
from .data import injuries, kalshi_account, nba, nfl, odds
from .db import get_meta, get_settings, init_db, save_settings, set_meta
from .models import games as gm
from .models import props_backtest

Sport = Literal["nfl", "nba"]

job = {"running": False, "log": [], "last": {}}
job_lock = threading.Lock()


def _log(msg: str) -> None:
    job["log"].append(f"{datetime.now():%H:%M:%S} {msg}")
    job["log"] = job["log"][-200:]


def _run_refresh(sports: list[str]) -> None:
    try:
        for s in sports:
            try:
                (nfl if s == "nfl" else nba).refresh(log=_log)
                injuries.get(s, force=True)
                job["last"][s] = datetime.now().isoformat(timespec="seconds")
            except Exception as e:  # keep going with the other sport
                _log(f"{s.upper()} refresh failed: {e}")
                traceback.print_exc()
        edges.invalidate()
    finally:
        job["running"] = False


def start_refresh(sports: list[str]) -> bool:
    with job_lock:
        if job["running"]:
            return False
        job["running"] = True
    threading.Thread(target=_run_refresh, args=(sports,), daemon=True).start()
    return True


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    missing = [s for s, mod in (("nfl", nfl), ("nba", nba)) if not mod.has_data()]
    if missing:
        start_refresh(missing)
    paper.start_background()
    yield


app = FastAPI(title="National Data Ball", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def revalidate_static(request, call_next):
    """Make browsers re-check the page and its JS/CSS so an update shows up without a hard refresh."""
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
def status():
    return {
        "refresh": job,
        "has_data": {"nfl": nfl.has_data(), "nba": nba.has_data()},
        "odds": {"enabled": odds.enabled(), "quota": odds.quota(), "prop_cost": {s: odds.prop_cost(s) for s in ("nfl", "nba")},
                 "lines_at": {s: get_meta(f"odds_lines_at_{s}") for s in ("nfl", "nba")}},
    }


@app.post("/api/odds/lines")
def odds_lines(sport: Sport):
    """Pull sportsbook game lines (costs ~3 Odds API credits)."""
    try:
        n = odds.refresh_lines(sport)
    except Exception as e:
        raise HTTPException(502, str(e))
    edges.invalidate()
    return {"events": n, "quota": odds.quota()}


@app.post("/api/odds/props")
def odds_props(sport: Sport, event_id: str):
    """Pull sportsbook player props for one game (costs one credit per prop market)."""
    try:
        n = odds.refresh_props(sport, event_id)
    except Exception as e:
        raise HTTPException(502, str(e))
    edges.invalidate()
    return {"outcomes": n, "quota": odds.quota()}


@app.get("/api/injuries")
def injury_list(sport: Sport):
    return sorted(injuries.get(sport), key=lambda r: (r["team"] or "", not r["out"], r["player"]))


@app.post("/api/refresh")
def refresh(sport: Literal["nfl", "nba", "all"] = "all"):
    sports = ["nfl", "nba"] if sport == "all" else [sport]
    return {"started": start_refresh(sports)}


@app.get("/api/settings")
def settings_get():
    return get_settings()


@app.post("/api/settings")
def settings_post(values: dict):
    s = save_settings(values)
    edges.invalidate()
    return s


@app.get("/api/games")
def games(sport: Sport):
    return edges.games_view(sport, get_settings())


@app.get("/api/edges")
def edge_list(sport: Sport, force: bool = False):
    return edges.edges(sport, get_settings(), force=force)


@app.get("/api/shortlist")
def shortlist(sport: Sport, force: bool = False):
    return edges.shortlist(sport, get_settings(), force=force)


@app.get("/api/projections")
def projections(sport: Sport):
    return edges.projections_view(sport, get_settings())


@app.get("/api/prop_lines")
def prop_lines(sport: Sport):
    return edges.prop_lines_view(sport, get_settings())


@app.get("/api/ratings")
def ratings(sport: Sport):
    st = edges.sport_state(sport, get_settings())
    m = st["model"]
    if not m:
        return {}
    return {"hfa": m.hfa, "mu": m.mu, "b2b": m.b2b, "sd_margin": m.sd_margin, "sd_total": m.sd_total,
            "teams": m.table()}


@app.get("/api/backtest")
def backtest(sport: Sport):
    return gm.backtest(sport)


props_bt = {"running": False}


def _run_props_backtest() -> None:
    try:
        res = props_backtest.backtest_nba(log=_log)
        res["run_at"] = datetime.now().isoformat(timespec="seconds")
        set_meta("props_backtest_nba", res)
    except Exception as e:
        _log(f"Props backtest failed: {e}")
        traceback.print_exc()
    finally:
        props_bt["running"] = False


@app.get("/api/backtest/props")
def props_backtest_get():
    """Last saved NBA props backtest (it takes minutes, so it runs on request, not per page load)."""
    return {"running": props_bt["running"], "result": get_meta("props_backtest_nba")}


@app.post("/api/backtest/props")
def props_backtest_run():
    with job_lock:
        if props_bt["running"]:
            return {"started": False}
        props_bt["running"] = True
    threading.Thread(target=_run_props_backtest, daemon=True).start()
    return {"started": True}


class BetIn(BaseModel):
    sport: Sport
    ticker: str
    description: str
    side: Literal["yes", "no"]
    price: float
    contracts: int
    fees: float | None = None
    fair_prob: float | None = None
    game_start: str | None = None
    notes: str | None = None
    mode: Literal["real", "paper"] = "real"
    market_type: str | None = None
    edge: float | None = None


@app.get("/api/bets")
def bet_list(mode: Literal["real", "paper", "all"] = "real"):
    return bets.list_bets(mode)


@app.post("/api/paper/run")
def paper_run():
    """Paper-bet every qualifying edge and Shortlist pick right now (also runs every 15 min)."""
    shortlist_created = paper.run_shortlist()
    return {**paper.run_once(), "shortlist_created": shortlist_created}


@app.get("/api/kalshi/account")
def kalshi_status():
    if not kalshi_account.configured():
        return {"configured": False}
    try:
        return {"configured": True, **kalshi_account.balance()}
    except Exception as e:
        return {"configured": True, "error": str(e)}


@app.post("/api/kalshi/import")
def kalshi_import():
    settings = get_settings()
    try:
        res = bets.import_from_kalshi(settings["kalshi_fee_coef"], lambda t: edges.kickoff_for(t, settings))
    except Exception as e:
        raise HTTPException(502, f"Kalshi import failed: {e}")
    # Keep bet sizing in step with the real account: bankroll = cash + open positions' value.
    try:
        b = kalshi_account.balance()
        res["bankroll"] = round(b["cash"] + b["portfolio_value"], 2)
        save_settings({"bankroll": res["bankroll"]})
    except Exception:
        pass
    return {**res, **bets.sync()}


@app.post("/api/bets")
def bet_create(b: BetIn):
    if not (0 < b.price < 1) or b.contracts <= 0:
        raise HTTPException(400, "price must be between 0 and 1 and contracts positive")
    return bets.create({**b.model_dump(), "source": "manual"}, get_settings()["kalshi_fee_coef"])


@app.post("/api/bets/sync")
def bet_sync():
    return bets.sync()


@app.post("/api/bets/{bet_id}/settle")
def bet_settle(bet_id: int, status: Literal["won", "lost", "void", "open"]):
    bets.settle(bet_id, status)
    return {"ok": True}


@app.delete("/api/bets/{bet_id}")
def bet_delete(bet_id: int):
    bets.delete(bet_id)
    return {"ok": True}
