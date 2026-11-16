"""Walk-forward backtest of NBA player prop probabilities.

Every game day is replayed with the live projection code (props.nba_projections) using only
box scores from before that day; the team model behind the scoring environment is refit
weekly. Each projection is scored on Kalshi-style ladders (10+/15+/20+ points, ...) against
what the player actually did.

Known differences from live:
  * No historical injury reports: a player who didn't play is treated as listed Out, so
    availability is known perfectly (minutes look a little better than they will live).
  * No historical sportsbook lines: the environment comes from our team model alone.

The baseline is the casual bettor's view: the player's hit rate at that threshold over their
last 20 games.
"""
from datetime import timedelta

import numpy as np
import pandas as pd

from ..data import nba
from . import games as gm
from . import pricing, props
from .ratings import fit

LADDERS = {
    "pts": range(5, 41, 5), "reb": range(2, 17, 2), "ast": range(2, 13, 2),
    "fg3m": range(1, 7), "pra": range(10, 56, 5),
}
WARMUP_DAYS = 30
BASELINE_GAMES = 20
P_RANGE = (0.03, 0.97)  # Kalshi rarely lists rungs priced outside this


def _played(day_games: pd.DataFrame) -> dict:
    return {gid: set(g.loc[g["min"] > 0, "player_id"]) for gid, g in day_games.groupby("game_id")}


def run_nba(log=print) -> pd.DataFrame:
    """One row per (player-game, stat, threshold) with the model and baseline P(stat >= k)."""
    pg = nba.load_player_games().sort_values("date", ascending=False)
    games = gm.load_games("nba")
    done = games[games["completed"]].sort_values("date")
    start = done["date"].min() + timedelta(days=WARMUP_DAYS)
    model, model_cut = None, None
    rows = []
    for d, day in done[done["date"] >= start].groupby("date"):
        if model is None or (d - model_cut).days >= 7:
            model, model_cut = fit(games, d, gm.PARAMS["nba"]), d
        hist = pg[pg["date"] < d]
        today = pg[pg["date"] == d]
        played = _played(today)
        glist = []
        for _, r in day.iterrows():
            pred = gm.game_prediction(model, r, 1.0, "nba")["model"]
            glist.append({"id": r["id"], "home": r["home_team"], "away": r["away_team"], "kickoff": d.isoformat(),
                          "fair": {"home_pts": pred["home_pts"], "away_pts": pred["away_pts"]}})
        team_game = {(g["home"], g["id"]) for g in glist} | {(g["away"], g["id"]) for g in glist}
        game_of_team = {t: gid for t, gid in team_game}

        def injury_of(team, name, pid):
            gid = game_of_team.get(team)
            if gid is not None and pid not in played.get(gid, set()):
                return {"out": True, "status": "Out", "comment": "did not play"}
            return None

        season = int(day["season"].iloc[0])
        proj, _, _ = props.nba_projections(hist, glist, injury_of, season=season)
        actual = today.set_index(["player_id", "game_id"])
        recent = dict(tuple(hist.groupby("player_id")))
        for (pid, gid), p in proj.items():
            if (pid, gid) not in actual.index:
                continue
            a = actual.loc[(pid, gid)]
            if a["team"] != p["team"] or a["min"] <= 0:
                continue  # traded since his last game
            last = recent[pid].head(BASELINE_GAMES)
            for stat, ks in LADDERS.items():
                mean, disp, y = p[stat]["mean"], p[stat]["disp"], float(a[stat])
                past = last[stat].to_numpy()
                for k in ks:
                    pm = pricing.count_prob_at_least(mean, disp, k)
                    if not P_RANGE[0] < pm < P_RANGE[1]:
                        continue
                    rows.append((d, season, pid, gid, stat, k, mean, y, pm,
                                 (np.sum(past >= k) + 1) / (len(past) + 2), y >= k, p["min"], float(a["min"]),
                                 p.get("min_boost", 0.0), p.get("usage_boost", 0.0), p["env"]))
        if d.day == 1:
            log(f"props backtest: {d.date()} ({len(rows)} rows)")
    return pd.DataFrame(rows, columns=["date", "season", "player_id", "game_id", "stat", "k", "mean", "actual",
                                       "p_model", "p_base", "hit", "proj_min", "min",
                                       "min_boost", "usage_boost", "env"])


# NFL rungs as Kalshi lists them (strike 9.5 -> 10+), scored where the model prices them in P_RANGE.
NFL_LADDERS = {
    "tds": range(1, 4), "pass_tds": range(1, 6), "receptions": range(1, 13),
    "rush_yds": range(10, 151, 10), "rec_yds": range(10, 151, 10), "pass_yds": range(125, 376, 25),
}
NFL_SEASONS = range(2022, 2027)  # history from the first; replayed from the second
NFL_WARMUP_WEEKS = 3
NFL_BASELINE_GAMES = 10


def run_nfl(log=print) -> pd.DataFrame:
    """NFL version of run_nba: every week replayed with props.nfl_projections on prior weeks only.

    Known differences from live:
      * No historical injury reports: a player with no stats that week is treated as Out (which
        also hands his carries to teammates), so availability is known perfectly.
      * The game environment uses nflverse closing lines, a little sharper than lines at bet time.
      * Wind is the recorded game-time wind, not a forecast.
    """
    from ..data import nfl
    from ..config import current_nfl_season
    seasons = [s for s in NFL_SEASONS if s <= current_nfl_season()]
    nfl.ensure_seasons(seasons, log)
    pw = nfl.load_player_weeks(seasons).dropna(subset=["player_id"]).sort_values("t", ascending=False)
    games = gm.load_games("nfl")
    done = games[games["completed"] & games["season"].isin(seasons[1:])]
    rows = []
    for (season, week), wk in done.groupby(["season", "week"]):
        if week <= NFL_WARMUP_WEEKS and season == seasons[1]:
            continue
        t = season * 100 + week
        hist = pw[(pw["t"] < t) & (pw["season"] >= season - 1)]
        today = pw[pw["t"] == t]
        played = set(zip(today["team"], today["player_display_name"]))
        glist = []
        for _, r in wk.iterrows():
            wind = r["wind"] if r["roof"] in ("outdoors", "open") and pd.notna(r["wind"]) else 0.0
            fair = None
            if pd.notna(r["spread_line"]) and pd.notna(r["total_line"]):
                fair = {"home_pts": (r["total_line"] + r["spread_line"]) / 2,
                        "away_pts": (r["total_line"] - r["spread_line"]) / 2, "margin": r["spread_line"]}
            glist.append({"id": r["id"], "home": r["home_team"], "away": r["away_team"], "kickoff": r["kickoff"].isoformat(),
                          "weather": {"roof": "outdoors", "wind_mph": float(wind)}, "fair": fair})
        teams = {g["home"] for g in glist} | {g["away"] for g in glist}

        def injury_of(team, name, _):
            if team in teams and (team, name) not in played:
                return {"out": True, "status": "Out", "comment": "did not play"}
            return None

        norms = props.nfl_team_points_norm(games, before=wk["date"].min())
        proj, _, _ = props.nfl_projections(hist, glist, injury_of, norms=norms, season=int(season))
        actual = today.drop_duplicates(["player_id", "game_id"]).set_index(["player_id", "game_id"])
        recent = dict(tuple(hist.groupby("player_id")))
        for (pid, gid), p in proj.items():
            if (pid, gid) not in actual.index:
                continue
            a = actual.loc[(pid, gid)]
            if a["team"] != p["team"]:
                continue
            last = recent[pid].head(NFL_BASELINE_GAMES)
            for stat, ks in NFL_LADDERS.items():
                if stat.startswith("pass") != (p["position"] == "QB"):
                    continue  # QBs are priced on passing, everyone else on rushing/receiving
                col = props.NFL_STATS[stat]
                mean, disp, y = p[stat]["mean"], p[stat]["disp"], float(a[col])
                past = last[col].to_numpy(float)
                for k in ks:
                    pm = pricing.count_prob_at_least(mean, disp, k)
                    if not P_RANGE[0] < pm < P_RANGE[1]:
                        continue
                    rows.append((wk["date"].min(), int(season), int(week), pid, gid, p["position"], stat, k,
                                 mean, disp, y, pm, (np.sum(past >= k) + 1) / (len(past) + 2), y >= k,
                                 p["env"], p.get("carry_boost", 0.0)))
        log(f"NFL props backtest: {season} week {week} ({len(rows)} rows)")
    return pd.DataFrame(rows, columns=["date", "season", "week", "player_id", "game_id", "position", "stat", "k",
                                       "mean", "disp", "actual", "p_model", "p_base", "hit", "env", "carry_boost"])


def _logloss(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _scores(df: pd.DataFrame) -> dict:
    y = df["hit"].to_numpy(float)
    bm, bb = float(np.mean((df["p_model"] - y) ** 2)), float(np.mean((df["p_base"] - y) ** 2))
    return {"n": len(df), "brier_model": bm, "brier_base": bb, "skill_vs_base": 1 - bm / bb,
            "logloss_model": _logloss(df["p_model"].to_numpy(), y), "logloss_base": _logloss(df["p_base"].to_numpy(), y)}


def _calibration(df: pd.DataFrame, bins: int = 10) -> list[dict]:
    b = pd.cut(df["p_model"], np.linspace(0, 1, bins + 1), include_lowest=True)
    cal = df.groupby(b, observed=True).agg(pred=("p_model", "mean"), actual=("hit", "mean"), n=("hit", "size"))
    return [{"pred": float(r.pred), "actual": float(r.actual), "n": int(r.n)} for r in cal.itertuples()]


def summarize(df: pd.DataFrame, sport: str = "nba") -> dict:
    if df.empty:
        return {"error": "not enough games"}
    pgames = df.drop_duplicates(["player_id", "game_id", "stat"])
    by_stat = []
    for stat, g in df.groupby("stat"):
        pgs = pgames[pgames["stat"] == stat]
        by_stat.append({"stat": stat, "player_games": len(pgs),
                        "proj": float(pgs["mean"].mean()), "actual": float(pgs["actual"].mean()),
                        "bias": float((pgs["actual"] - pgs["mean"]).mean()),
                        "mae": float((pgs["actual"] - pgs["mean"]).abs().mean()), **_scores(g),
                        "calibration": _calibration(g, 5)})
    # Rungs near 50/50 are the ones Kalshi prices as its main lines.
    main = df[df["p_model"].between(0.35, 0.65)]
    m = pgames.drop_duplicates(["player_id", "game_id"])
    out = {
        "sport": sport, "from": df["date"].min().date().isoformat(), "to": df["date"].max().date().isoformat(),
        "games": int(df["game_id"].nunique()), "player_games": int(len(m)), "predictions": len(df),
        "overall": _scores(df), "main_lines": _scores(main), "calibration": _calibration(df),
        "by_stat": by_stat,
        "by_season": [{"season": int(s), **_scores(g)} for s, g in df.groupby("season")],
    }
    if "proj_min" in m:
        out["minutes"] = {"mae": float((m["min"] - m["proj_min"]).abs().mean()), "bias": float((m["min"] - m["proj_min"]).mean())}
    return out


def backtest_nba(log=print) -> dict:
    res = summarize(run_nba(log))
    res["market"] = market_check("nba")
    return res


def backtest_nfl(log=print) -> dict:
    res = summarize(run_nfl(log), "nfl")
    res["market"] = market_check("nfl")
    return res


def market_check(sport: str, max_spread: float = 0.10, edge: float = 0.05) -> dict | None:
    """Score the model against Kalshi's own prices on settled player props.

    Uses the last market snapshot before kickoff for each ticker (the app records these while it
    runs), so it only covers games the app was open for. The outcome comes from the box scores.
    Also reports what buying every rung where model - ask >= edge would have returned.
    """
    from ..data import nfl
    from ..db import session
    with session() as conn:
        snaps = pd.read_sql_query(
            "SELECT ticker, market_type, game_id, kickoff, strike, yes_bid, yes_ask, model_prob, player_id, taken_at "
            "FROM market_snapshots WHERE sport = ? AND player_id IS NOT NULL AND taken_at < kickoff "
            "ORDER BY taken_at", conn, params=(sport,))
    if snaps.empty:
        return None
    last = snaps.groupby("ticker").tail(1)
    last = last[(last["yes_ask"] - last["yes_bid"] <= max_spread) & (last["yes_bid"] > 0) & (last["yes_ask"] < 1)]
    if sport == "nfl":
        pw = nfl.load_player_weeks().drop_duplicates(["player_id", "game_id"]).set_index(["player_id", "game_id"])
        stat_col = props.NFL_STATS
    else:
        pw = nba.load_player_games()
        pw = pw[pw["min"] > 0].set_index(["player_id", "game_id"])  # DNPs settle as voids
        stat_col = {s: s for s in LADDERS}
    rows = []
    for r in last.itertuples():
        if (r.player_id, r.game_id) not in pw.index or r.market_type not in stat_col:
            continue
        y = float(pw.loc[(r.player_id, r.game_id), stat_col[r.market_type]]) >= np.floor(r.strike) + 1
        rows.append((r.market_type, r.model_prob, (r.yes_bid + r.yes_ask) / 2, r.yes_ask, y, r.game_id))
    if not rows:
        return None
    d = pd.DataFrame(rows, columns=["stat", "p_model", "p_mkt", "ask", "hit", "game_id"])

    def score(g):
        y = g["hit"].to_numpy(float)
        bm, bk = float(np.mean((g["p_model"] - y) ** 2)), float(np.mean((g["p_mkt"] - y) ** 2))
        bets = g[g["p_model"] - g["ask"] >= edge]
        return {"n": len(g), "brier_model": bm, "brier_market": bk, "skill_vs_market": 1 - bm / bk,
                "bets": len(bets), "bet_hit_rate": float(bets["hit"].mean()) if len(bets) else None,
                "bet_roi": float((bets["hit"] - bets["ask"]).sum() / bets["ask"].sum()) if len(bets) else None}

    return {"games": int(d["game_id"].nunique()), "edge": edge, "overall": score(d),
            "by_stat": [{"stat": s, **score(g)} for s, g in d.groupby("stat")]}
