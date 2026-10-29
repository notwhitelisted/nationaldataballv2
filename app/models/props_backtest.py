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


def summarize(df: pd.DataFrame) -> dict:
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
    return {
        "sport": "nba", "from": df["date"].min().date().isoformat(), "to": df["date"].max().date().isoformat(),
        "games": int(df["game_id"].nunique()), "player_games": int(len(m)), "predictions": len(df),
        "overall": _scores(df), "main_lines": _scores(main), "calibration": _calibration(df),
        "minutes": {"mae": float((m["min"] - m["proj_min"]).abs().mean()), "bias": float((m["min"] - m["proj_min"]).mean())},
        "by_stat": by_stat,
        "by_season": [{"season": int(s), **_scores(g)} for s, g in df.groupby("season")],
    }


def backtest_nba(log=print) -> dict:
    return summarize(run_nba(log))
