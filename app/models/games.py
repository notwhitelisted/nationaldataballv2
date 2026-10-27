"""Game-level predictions (moneyline / spread / total / team total) for NFL and NBA."""
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import numpy as np
import pandas as pd

from ..config import today_et
from ..data import nba, nfl, odds, weather
from . import pricing
from .ratings import Params, TeamModel, add_back_to_back, fit

# Defaults chosen with the walk-forward backtest in this module (see /api/backtest).
PARAMS = {
    "nfl": Params(half_life=84, ridge=6.0, offseason_gap=120, use_b2b=False,
                  sd_bounds={"margin": (12.5, 14.5), "total": (12.0, 14.5), "team": (8.5, 10.5)}),
    "nba": Params(half_life=50, ridge=4.0, offseason_gap=60, use_b2b=True,
                  sd_bounds={"margin": (11.0, 14.0), "total": (16.0, 21.0), "team": (10.0, 13.5)}),
}


# Spread of actual results around the closing sportsbook line. NFL measured on 2023-26 nflverse
# closing lines; NBA from typical recent-season values (ESPN doesn't keep historical NBA lines).
MARKET_SD = {
    "nfl": {"margin": 12.6, "total": 12.9, "team": 9.0},
    "nba": {"margin": 12.3, "total": 17.8, "team": 11.0},
}


_nfl_hist: tuple[np.ndarray, np.ndarray, float] | None = None


def _nfl_history() -> tuple[np.ndarray, np.ndarray, float]:
    """(closing spreads, results, residual SD) since 1999, from both teams' perspectives."""
    global _nfl_hist
    if _nfl_hist is None:
        g = pd.read_csv(nfl.GAMES_PATH, usecols=["result", "spread_line"]).dropna()
        s, r = g["spread_line"].to_numpy(float), g["result"].to_numpy(float)
        _nfl_hist = (np.concatenate([s, -s]), np.concatenate([r, -r]), float((r - s).std()))
    return _nfl_hist


def prob_margin_over(sport: str, mean: float, sd: float, line: float) -> float:
    """P(margin > line).

    NFL margins pile up on 3 and 7, and where they pile up depends on the spread: a 7-point
    favorite wins by exactly 3 far more often than a 1-point favorite does. So NFL uses actual
    results of past games closed near `mean` (a triangular kernel, widened until there are
    enough games), stretched around `mean` when `sd` differs from the market's. Other sports
    use the normal approximation.
    """
    if sport != "nfl" or not nfl.GAMES_PATH.exists():
        return pricing.prob_over(mean, sd, line)
    spreads, results, resid_sd = _nfl_history()
    for width in (1.5, 2.5, 4.0):
        w = np.clip(1 - np.abs(spreads - mean) / width, 0, None)
        if w.sum() >= 400:
            outcome = mean + (results - mean) * (sd / resid_sd)
            return float(np.sum(w * (outcome > line)) / w.sum())
    # Very lopsided lines have too few past games; fall back to residuals around the spread.
    return float(np.mean((results - spreads) * (sd / resid_sd) > line - mean))


def load_games(sport: str) -> pd.DataFrame:
    """Normalized game table: date, kickoff, home_team, away_team, scores, completed, neutral, book lines."""
    if sport == "nfl":
        g = nfl.load_games()
        if g.empty:
            return g
        g = g[g["game_type"].isin(["REG", "WC", "DIV", "CON", "SB"])].copy()
        g["neutral"] = (g["location"] == "Neutral").astype(int)
        g["book_home_margin"] = g["spread_line"]
        g["book_total"] = g["total_line"]
        g["book_home_ml"] = g["home_moneyline"]
        g["book_away_ml"] = g["away_moneyline"]
        g["id"] = g["game_id"]
    else:
        g = nba.load_games()
        if g.empty:
            return g
        g = g.rename(columns={"home": "home_team", "away": "away_team"})
        g["neutral"] = 0
        g["id"] = g["game_id"]
        g = add_back_to_back(g)
    return g.reset_index(drop=True)


def fit_model(sport: str, games: pd.DataFrame, as_of: pd.Timestamp | None = None) -> TeamModel:
    as_of = as_of if as_of is not None else pd.Timestamp(today_et())
    return fit(games, as_of, PARAMS[sport])


def book_home_win_prob(row) -> float | None:
    dv = pricing.devig_two_way(row.get("book_home_ml"), row.get("book_away_ml"))
    return dv[0] if dv else None


def game_prediction(model: TeamModel, row, model_weight: float, sport: str, total_adj: float = 0.0) -> dict:
    hp, ap = model.predict(row["home_team"], row["away_team"], bool(row.get("neutral", 0)),
                           int(row.get("home_b2b", 0) or 0), int(row.get("away_b2b", 0) or 0))
    hp, ap = hp + total_adj / 2, ap + total_adj / 2  # weather adjustment (NFL wind)
    m_margin, m_total = hp - ap, hp + ap

    b_margin = row.get("book_home_margin")
    b_margin = None if b_margin is None or pd.isna(b_margin) else float(b_margin)
    b_total = row.get("book_total")
    b_total = None if b_total is None or pd.isna(b_total) else float(b_total)
    cons_hwp = row.get("cons_home_win")
    b_hwp = cons_hwp if cons_hwp is not None and not pd.isna(cons_hwp) else book_home_win_prob(row)
    if b_margin is None and b_hwp is not None:
        b_margin = pricing.margin_from_win_prob(b_hwp, model.sd_margin)

    w = model_weight
    f_margin = m_margin if b_margin is None else w * m_margin + (1 - w) * b_margin
    f_total = m_total if b_total is None else w * m_total + (1 - w) * b_total
    model_sd = {"margin": model.sd_margin, "total": model.sd_total, "team": model.sd_team}
    msd = MARKET_SD[sport]
    # The blended estimate is mostly market, so its error is closer to the market's than to our model's.
    fair_sd = model_sd if b_margin is None else {k: w * model_sd[k] + (1 - w) * msd[k] for k in msd}
    m_hwp = prob_margin_over(sport, m_margin, model.sd_margin, 0)
    # Win probability blends with the books' moneyline itself, not one derived from their spread.
    f_hwp = (w * m_hwp + (1 - w) * b_hwp if b_hwp is not None
             else prob_margin_over(sport, f_margin, fair_sd["margin"], 0))

    return {
        "model": {"margin": m_margin, "total": m_total, "home_pts": hp, "away_pts": ap,
                  "home_win": m_hwp, "sd": model_sd},
        "book": {"margin": b_margin, "total": b_total, "home_win": b_hwp, "sd": msd},
        "fair": {"margin": f_margin, "total": f_total,
                 "home_pts": (f_total + f_margin) / 2, "away_pts": (f_total - f_margin) / 2,
                 "home_win": f_hwp, "sd": fair_sd},
    }


def nfl_qb_flags(games: pd.DataFrame) -> dict:
    """Flag upcoming games whose listed starting QB differs from that team's last starter."""
    flags = {}
    done = games[games["completed"]].sort_values("date")
    last_qb = {}
    for _, r in done.iterrows():
        last_qb[r["home_team"]] = r.get("home_qb_name")
        last_qb[r["away_team"]] = r.get("away_qb_name")
    for _, r in games[~games["completed"]].iterrows():
        notes = []
        for side in ("home", "away"):
            team, qb = r[f"{side}_team"], r.get(f"{side}_qb_name")
            prev = last_qb.get(team)
            if isinstance(qb, str) and isinstance(prev, str) and qb != prev:
                notes.append(f"{team} QB change: {prev} -> {qb}")
        if notes:
            flags[r["id"]] = notes
    return flags


def upcoming(sport: str, settings: dict, days: int = 8) -> tuple[list[dict], TeamModel | None]:
    games = load_games(sport)
    if games.empty:
        return [], None
    model = fit_model(sport, games)
    today = pd.Timestamp(today_et())
    up = games[(~games["completed"]) & (games["date"] >= today) & (games["date"] <= today + timedelta(days=days))]
    qb_flags = nfl_qb_flags(games) if sport == "nfl" else {}
    cons = odds.consensus(sport, MARKET_SD[sport]) if odds.enabled() else {}
    if sport == "nfl":
        with ThreadPoolExecutor(8) as pool:  # warm the per-stadium forecast cache in parallel
            list(pool.map(weather.prefetch, up["stadium_id"].dropna().unique()))
    out = []
    for _, r in up.sort_values("kickoff").iterrows():
        r = r.copy()
        key_teams = frozenset((r["home_team"], r["away_team"]))
        c = next((cons[k] for d in (0, -1, 1)
                  if (k := ((r["date"] + timedelta(days=d)).date().isoformat(), key_teams)) in cons), None)
        if c and c["margin"] is not None:
            r["book_home_margin"], r["book_total"], r["cons_home_win"] = c["margin"], c["total"], c["home_win"]
            sharp = " incl. Pinnacle" if "pinnacle" in c["books"] else ""
            book_source = f"Consensus of {len(c['books'])} books{sharp}"
        elif pd.notna(r.get("book_home_margin")):
            book_source = "nflverse line" if sport == "nfl" else "DraftKings via ESPN"
        else:
            book_source = None
        flags = list(qb_flags.get(r["id"], []))
        wx = None
        if sport == "nfl":
            wx = weather.forecast(r.get("stadium_id"), r["kickoff"].to_pydatetime())
            wx["text"] = weather.describe(wx)
            if wx["total_adj"]:
                flags.append(f"Wind {wx['wind_mph']:.0f} mph: model total {wx['total_adj']:+.1f}")
        pred = game_prediction(model, r, settings["model_weight"], sport, wx["total_adj"] if wx else 0.0)
        out.append({
            "id": r["id"],
            "sport": sport,
            "date": r["date"].date().isoformat(),
            "kickoff": r["kickoff"].isoformat(),
            "home": r["home_team"],
            "away": r["away_team"],
            "neutral": bool(r.get("neutral", 0)),
            "home_b2b": int(r.get("home_b2b", 0) or 0),
            "away_b2b": int(r.get("away_b2b", 0) or 0),
            "flags": flags,
            "weather": wx,
            "book_source": book_source,
            "odds_event_id": c["event_id"] if c else None,
            "lines_at": c["fetched_at"] if c else None,
            **pred,
        })
    return out, model


def backtest(sport: str, edge_threshold: float | None = None) -> dict:
    """Walk-forward test: refit weekly using only prior data, predict the following week."""
    games = load_games(sport)
    if games.empty:
        return {"error": "no data"}
    edge_threshold = edge_threshold or (1.5 if sport == "nfl" else 3.0)
    done = games[games["completed"]].sort_values("date")
    start = done["date"].min() + timedelta(days=60 if sport == "nfl" else 30)
    if sport == "nfl":
        start = max(start, pd.Timestamp(f"{done['season'].max() - 1}-09-01"))
    test = done[done["date"] >= start]
    rows = []
    cut = start
    while cut <= test["date"].max():
        nxt = cut + timedelta(days=7)
        chunk = test[(test["date"] >= cut) & (test["date"] < nxt)]
        if len(chunk):
            m = fit(games, cut, PARAMS[sport])
            for _, r in chunk.iterrows():
                pred = game_prediction(m, r, 1.0, sport)
                rows.append({
                    "date": r["date"], "margin": r["home_score"] - r["away_score"],
                    "total": r["home_score"] + r["away_score"],
                    "m_margin": pred["model"]["margin"], "m_total": pred["model"]["total"],
                    "m_hwp": pred["model"]["home_win"],
                    "b_margin": pred["book"]["margin"], "b_total": pred["book"]["total"],
                    "b_hwp": pred["book"]["home_win"],
                })
        cut = nxt
    df = pd.DataFrame(rows)
    if df.empty:
        return {"error": "not enough games"}
    home_won = (df["margin"] > 0).astype(float)
    res = {
        "sport": sport,
        "games": len(df),
        "from": df["date"].min().date().isoformat(),
        "to": df["date"].max().date().isoformat(),
        "model": {
            "margin_mae": float((df["margin"] - df["m_margin"]).abs().mean()),
            "total_mae": float((df["total"] - df["m_total"]).abs().mean()),
            "win_brier": float(((df["m_hwp"] - home_won) ** 2).mean()),
            "straight_up_acc": float(((df["m_margin"] > 0) == (df["margin"] > 0)).mean()),
        },
    }
    lined = df.dropna(subset=["b_margin", "b_total"])
    if len(lined) >= 20:
        res["book"] = {
            "games": len(lined),
            "margin_mae": float((lined["margin"] - lined["b_margin"]).abs().mean()),
            "total_mae": float((lined["total"] - lined["b_total"]).abs().mean()),
            "model_margin_mae_same_games": float((lined["margin"] - lined["m_margin"]).abs().mean()),
            "model_total_mae_same_games": float((lined["total"] - lined["m_total"]).abs().mean()),
        }
        bw = lined.dropna(subset=["b_hwp"])
        if len(bw):
            hw = (bw["margin"] > 0).astype(float)
            res["book"]["win_brier"] = float(((bw["b_hwp"] - hw) ** 2).mean())
            res["book"]["model_win_brier_same_games"] = float(((bw["m_hwp"] - hw) ** 2).mean())
        # Betting against the closing line where the model disagrees by >= threshold.
        for key, actual, mcol, bcol in (("ats", "margin", "m_margin", "b_margin"), ("totals", "total", "m_total", "b_total")):
            diff = lined[mcol] - lined[bcol]
            pick = lined[diff.abs() >= edge_threshold]
            side = np.sign(diff[pick.index])
            outcome = np.sign(pick[actual] - pick[bcol]) * side
            wins, losses = int((outcome > 0).sum()), int((outcome < 0).sum())
            res[key] = {
                "threshold": edge_threshold, "bets": wins + losses, "wins": wins, "losses": losses,
                "win_pct": wins / (wins + losses) if wins + losses else None,
                "breakeven_at_-110": 0.5238,
            }
    # Calibration of model win probabilities.
    bins = np.linspace(0, 1, 11)
    df["bin"] = pd.cut(df["m_hwp"], bins, include_lowest=True)
    cal = df.groupby("bin", observed=True).agg(pred=("m_hwp", "mean"), n=("m_hwp", "size"))
    cal["actual"] = df.assign(hw=home_won).groupby("bin", observed=True)["hw"].mean()
    res["calibration"] = [{"pred": float(r.pred), "actual": float(r.actual), "n": int(r.n)} for r in cal.itertuples()]
    return res
