"""Player projections and over/under probabilities.

Approach (deliberately simple and transparent):
  * NBA: project minutes from recent games, multiply by recency-weighted per-minute rates,
    then scale by the game environment (our projected team points vs the team's norm) and,
    lightly, by what the opponent allows in rebounds/assists to the player's role and in threes.
  * NFL: recency-weighted per-game averages, scaled by how much the opponent allows
    relative to league average (shrunk halfway to 1.0).
  * Outcomes are modelled as negative binomial counts with the player's own dispersion
    (shrunk toward a stat-level default), which handles skew and the "X+" thresholds Kalshi lists.
"""
import re
import unicodedata

import numpy as np
import pandas as pd

from ..config import current_nba_season, current_nfl_season
from . import pricing

SUFFIXES = {"JR", "SR", "II", "III", "IV", "V"}


def name_key(name: str) -> str:
    """'Shai Gilgeous-Alexander' -> 'SGILGEOUSALEXANDER' (Kalshi ticker style)."""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    tokens = [re.sub(r"[^A-Za-z]", "", t).upper() for t in s.replace("-", " ").split()]
    tokens = [t for t in tokens if t]
    while len(tokens) > 2 and tokens[-1] in SUFFIXES:
        tokens.pop()
    if len(tokens) < 2:
        return "".join(tokens)
    return tokens[0][0] + "".join(tokens[1:])


def full_key(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Z]", "", s.upper())


class PlayerIndex:
    """Resolve Kalshi player references (team + 'MBROWN' key + display name) to our player ids."""

    def __init__(self, players: pd.DataFrame):
        # players: columns id, name, team (most recent team)
        self.by_key: dict[str, list] = {}
        self.by_full: dict[str, list] = {}
        for r in players.itertuples():
            self.by_key.setdefault(name_key(r.name), []).append(r)
            self.by_full.setdefault(full_key(r.name), []).append(r)

    def find(self, team: str, key: str, display_name: str):
        for cands in (self.by_key.get(key, []), self.by_full.get(full_key(display_name), [])):
            on_team = [c for c in cands if c.team == team]
            if len(on_team) == 1:
                return on_team[0].id
            if len(cands) == 1:
                return cands[0].id
        return None


def _decay_weights(n: int, decay: float) -> np.ndarray:
    return decay ** np.arange(n)


def _dispersion(values: np.ndarray, weights: np.ndarray, default: float, prior_n: float = 8) -> float:
    if len(values) < 3:
        return default
    m = np.average(values, weights=weights)
    if m <= 0:
        return default
    v = np.average((values - m) ** 2, weights=weights)
    emp = float(np.clip(v / m, 1.0, default * 2))
    n = len(values)
    return (n * emp + prior_n * default) / (n + prior_n)


# ---------------------------------------------------------------- NBA

NBA_STATS = ["pts", "reb", "ast", "fg3m", "pra"]
NBA_DISPERSION = {"pts": 2.6, "reb": 1.6, "ast": 1.7, "fg3m": 1.3, "pra": 3.2}
# How strongly each stat follows the game's scoring environment.
NBA_ENV_EXP = {"pts": 1.0, "fg3m": 1.0, "reb": 0.5, "ast": 0.5}


def nba_team_points_norm(pg: pd.DataFrame) -> dict:
    """Recency-weighted average points per game for each team over its last 15 games."""
    team_games = pg.groupby(["team", "game_id", "date"], as_index=False)["pts"].sum().sort_values("date", ascending=False)
    out = {}
    for team, g in team_games.groupby("team"):
        g = g.head(15)
        out[team] = float(np.average(g["pts"], weights=_decay_weights(len(g), 0.9)))
    return out


# Opponent defense beyond the scoring environment. Walk-forward test on 2025-26 (fit first
# half, scored second): rebounds and assists follow what the opponent allows to the player's
# role, threes what it allows relative to its points allowed. Each cut holdout deviance only
# ~0.3%; points gained nothing beyond the environment, and neither did pace.
NBA_DEF_WINDOW = 30
NBA_DEF_EXP = {"reb": 0.15, "ast": 0.20, "fg3m": 0.55}
NBA_DEF_CLIP = (0.85, 1.15)


def nba_role(reb_pm: float, ast_pm: float) -> str:
    if reb_pm >= 0.25 and ast_pm < 0.12:
        return "big"
    return "guard" if ast_pm >= 0.15 else "wing"


def _player_roles(pg: pd.DataFrame) -> pd.Series:
    """Role per player from their last NBA_DEF_WINDOW games (pg sorted most recent first)."""
    recent = pg.groupby("player_id").head(NBA_DEF_WINDOW)
    tot = recent.groupby("player_id")[["reb", "ast", "min"]].sum()
    tot = tot[tot["min"] > 0]
    return pd.Series([nba_role(r.reb / r.min, r.ast / r.min) for r in tot.itertuples()], index=tot.index)


def nba_defense_factors(pg: pd.DataFrame) -> dict:
    """{team: {"reb": {role: f}, "ast": {role: f}, "fg3m": f}} multipliers for opponents of
    `team` vs league average, over each defense's last NBA_DEF_WINDOW games (pg most recent first)."""
    pg = pg.assign(role=pg["player_id"].map(_player_roles(pg)))
    games = pg[["opp", "game_id", "date"]].drop_duplicates().sort_values("date", ascending=False)
    recent = games.groupby("opp").head(NBA_DEF_WINDOW)[["opp", "game_id"]]
    allowed = pg.merge(recent, on=["opp", "game_id"])
    by_role = allowed.groupby(["opp", "role"])[["reb", "ast", "min"]].sum()
    league_role = allowed.groupby("role")[["reb", "ast", "min"]].sum()
    per_game = allowed.groupby(["opp", "game_id"])[["pts", "fg3m"]].sum().groupby("opp").mean()
    lg_pts, lg_3 = per_game["pts"].mean(), per_game["fg3m"].mean()
    out = {}
    for team, g in per_game.iterrows():
        f = {"fg3m": (g["fg3m"] / lg_3) / (g["pts"] / lg_pts)}
        for s in ("reb", "ast"):
            f[s] = {}
            for role, lr in league_role.iterrows():
                if (team, role) in by_role.index:
                    r = by_role.loc[(team, role)]
                    if r["min"] > 0:
                        f[s][role] = (r[s] / r["min"]) / (lr[s] / lr["min"])
        out[team] = f
    return out


def nba_defense_mult(dfac: dict, stat: str, role: str) -> float:
    f = dfac.get(stat)
    if isinstance(f, dict):
        f = f.get(role)
    if not f:
        return 1.0
    return float(np.clip(f, *NBA_DEF_CLIP) ** NBA_DEF_EXP[stat])


# Minutes of current-season play needed before it outweighs the player's prior-season rate.
CURRENT_SEASON_K = 200
# Minutes of league-average play mixed into a prior-season rate (dominates only for rookies).
LEAGUE_PRIOR_MIN = 90
MIN_BUCKETS = (15, 24, 31)


def _min_bucket(minutes: float) -> int:
    return int(np.searchsorted(MIN_BUCKETS, minutes))


def nba_rate_priors(pg: pd.DataFrame) -> list[dict]:
    """League per-minute rates for players grouped by their average minutes."""
    avg_min = pg.groupby("player_id")["min"].transform("mean")
    bucket = np.searchsorted(MIN_BUCKETS, avg_min.to_numpy())
    out = []
    for b in range(len(MIN_BUCKETS) + 1):
        sub = pg[bucket == b]
        mins = sub["min"].sum() or 1.0
        out.append({s: float(sub[s].sum() / mins) for s in ("pts", "reb", "ast", "fg3m")})
    return out


def project_nba(hist: pd.DataFrame, env: float, priors: list[dict], season: int) -> dict | None:
    """hist: one player's games, most recent first. priors: league per-minute rates by minutes bucket.

    Per-minute rates are layered: league rate for similar-minute players -> the player's
    prior-season rate -> recency-weighted current-season rate. Each layer takes over as its
    sample grows, so stars keep their own rates and rookies start near league norms.
    """
    h = hist.head(60)
    if len(h) == 0:
        return None
    cur_mask = (h["season"] == season).to_numpy()
    cur, prev = h[cur_mask], h[~cur_mask].head(40)
    # Minutes: current season dominates once it exists (roles change between seasons).
    recent = h.head(12)
    mw = _decay_weights(len(recent), 0.8) * np.where(cur_mask[:len(recent)], 1.0, 0.15)
    proj_min = float(np.average(recent["min"], weights=mw))
    if proj_min <= 0:
        return None
    league = priors[_min_bucket(proj_min)]
    cw = _decay_weights(len(cur), 0.95)
    pw = _decay_weights(len(prev), 0.97)
    cur_min, prev_min = float(np.sum(cw * cur["min"])), float(np.sum(pw * prev["min"]))
    full = h[h["min"] >= 0.6 * proj_min].head(30)
    fw = _decay_weights(len(full), 0.93)
    usg = full["fga"] + 0.44 * full["fta"] + full["tov"]
    usg_min = float(np.sum(fw * full["min"]))
    out = {"min": proj_min, "games": len(h), "games_this_season": len(cur),
           # Possessions used per minute (shots, free-throw trips, turnovers).
           "usg_rate": float(np.sum(fw * usg) / usg_min) if usg_min > 0 else 0.0}
    for s in ("pts", "reb", "ast", "fg3m"):
        prev_rate = (np.sum(pw * prev[s]) + LEAGUE_PRIOR_MIN * league[s]) / (prev_min + LEAGUE_PRIOR_MIN)
        rate = float((np.sum(cw * cur[s]) + CURRENT_SEASON_K * prev_rate) / (cur_min + CURRENT_SEASON_K))
        mean = rate * proj_min * env ** NBA_ENV_EXP[s]
        out[s] = {"mean": mean, "disp": _dispersion(full[s].to_numpy(float), fw, NBA_DISPERSION[s])}
    out["pra"] = {
        "mean": out["pts"]["mean"] + out["reb"]["mean"] + out["ast"]["mean"],
        "disp": _dispersion(full["pra"].to_numpy(float), fw, NBA_DISPERSION["pra"]),
    }
    return out


def nba_projections(pg: pd.DataFrame, games: list[dict], injury_of=None) -> tuple[dict, PlayerIndex, dict]:
    """Project every player on a team in `games` (upcoming game predictions).

    injury_of(team, name, player_id) -> injury dict or None. Returns (projections keyed by
    (player_id, game_id), name index, {game_id: [injury notes]}).
    """
    pg = pg.sort_values("date", ascending=False)
    latest = pg.drop_duplicates("player_id")[["player_id", "name", "team"]].rename(columns={"player_id": "id"})
    index = PlayerIndex(latest)
    norms = nba_team_points_norm(pg)
    priors = nba_rate_priors(pg)
    dfacs = nba_defense_factors(pg)
    roles = _player_roles(pg)
    season = current_nba_season()
    team_last_game = pg.groupby("team")["date"].max()
    by_player = dict(tuple(pg.groupby("player_id")))
    out, notes = {}, {}
    for g in games:
        for side, opp in (("home", "away"), ("away", "home")):
            team = g[side]
            exp_pts = g["fair"][f"{side}_pts"]
            env = exp_pts / norms[team] if norms.get(team) else 1.0
            env = float(np.clip(env, 0.85, 1.15))
            roster = latest[latest["team"] == team]
            team_proj = {}
            for pid in roster["id"]:
                hist = by_player[pid]
                proj = project_nba(hist, env, priors, season)
                if not proj:
                    continue
                last = hist["date"].iloc[0]
                # Skip players who haven't played in a long while (injured / waived).
                if (team_last_game.get(team, last) - last).days > 21:
                    continue
                role = roles.get(pid, "wing")
                for s in NBA_DEF_EXP:
                    proj[s]["mean"] *= nba_defense_mult(dfacs.get(g[opp], {}), s, role)
                proj["pra"]["mean"] = proj["pts"]["mean"] + proj["reb"]["mean"] + proj["ast"]["mean"]
                name = hist["name"].iloc[0]
                proj.update({
                    "id": pid, "name": name, "team": team, "opp": g[opp], "game_id": g["id"],
                    "kickoff": g["kickoff"], "env": env, "role": role,
                    "missed_last": bool(last < team_last_game.get(team, last)),
                    "last_played": last.date().isoformat(),
                    "injury": injury_of(team, name, pid) if injury_of else None,
                })
                team_proj[pid] = proj
            notes.setdefault(g["id"], []).extend(_apply_nba_injuries(team_proj))
            for pid, proj in team_proj.items():
                out[(pid, g["id"])] = proj
    return out, index, notes


# Per-minute rate bump for teammates per unit of rotation usage share sitting out. Fit on
# 2025-26 box scores (14.6k teammate-games): with ~46% of rotation usage missing, teammates'
# points and assists per minute rose ~16%, rebounds and threes only ~4%.
NBA_USAGE_ABSORB = {"pts": 0.33, "ast": 0.30, "reb": 0.08, "fg3m": 0.08}
# Extra minutes per freed minute, by the teammate's minutes bucket (MIN_BUCKETS). Same data:
# coaches hand freed minutes to the bench; players already over 31 mpg barely gain any.
NBA_MIN_ABSORB = (0.11, 0.06, 0.025, 0.02)
ROTATION_MIN = 20
MAX_RATE_BOOST = 1.25


def _apply_nba_injuries(team_proj: dict) -> list[str]:
    """Drop players ruled out and hand their minutes and usage to healthy teammates.

    Only players who were in the rotation (played the team's last game) are redistributed;
    a player already missing games is already absent from teammates' recent numbers.
    Teammates gain minutes at the measured rate for their role (NBA_MIN_ABSORB, capped at 40;
    the rest goes to deep bench we don't project). Per-minute rates then rise with the share
    of rotation usage that is out (NBA_USAGE_ABSORB).
    """
    out_players = [p for p in team_proj.values() if p["injury"] and p["injury"]["out"]]
    notes = [f"{p['team']} {p['injury']['status']}: {p['name']} ({p['min']:.0f} min)"
             for p in out_players if p["min"] >= 15]
    fresh_out = [p for p in out_players if not p["missed_last"]]
    freed = sum(p["min"] for p in fresh_out)
    rotation_usg = sum(p["usg_rate"] * p["min"] for p in team_proj.values() if p["min"] >= ROTATION_MIN)
    out_usg = sum(p["usg_rate"] * p["min"] for p in fresh_out if p["min"] >= ROTATION_MIN)
    usg_share = out_usg / rotation_usg if rotation_usg > 0 else 0.0
    for p in out_players:
        del team_proj[p["id"]]
    healthy = list(team_proj.values())
    if freed > 0:
        for p in healthy:
            new_min = max(p["min"], min(40.0, p["min"] + freed * NBA_MIN_ABSORB[_min_bucket(p["min"])]))
            scale = new_min / p["min"]
            for s in NBA_STATS:
                p[s]["mean"] *= scale
            p["min_boost"] = new_min - p["min"]
            p["min"] = new_min
    if usg_share > 0:
        boost = {s: min(MAX_RATE_BOOST, 1 + a * usg_share) for s, a in NBA_USAGE_ABSORB.items()}
        for p in healthy:
            for s, b in boost.items():
                p[s]["mean"] *= b
            p["pra"]["mean"] = p["pts"]["mean"] + p["reb"]["mean"] + p["ast"]["mean"]
            p["usage_boost"] = boost["pts"] - 1
    return notes


# ---------------------------------------------------------------- NFL

NFL_STATS = {
    "pass_yds": "passing_yards", "pass_tds": "passing_tds", "rush_yds": "rushing_yards",
    "rec_yds": "receiving_yards", "receptions": "receptions", "tds": "tds",
}
NFL_DISPERSION = {"pass_yds": 22.0, "pass_tds": 1.0, "rush_yds": 16.0, "rec_yds": 20.0, "receptions": 1.3, "tds": 1.0}
# Which defensive "allowed" factor scales each stat.
NFL_DEF_FACTOR = {"pass_yds": "passing_yards", "rec_yds": "passing_yards", "receptions": "receptions",
                  "rush_yds": "rushing_yards", "pass_tds": "passing_tds", "tds": "tds"}
TD_PRIOR = {"RB": 0.45, "WR": 0.35, "TE": 0.25, "QB": 0.12}


def nfl_defense_factors(pw: pd.DataFrame) -> dict:
    """Per-defense multiplier for what it allows vs league average (recency weighted, shrunk 50%)."""
    allowed = pw.groupby(["opponent_team", "game_id", "t"], as_index=False)[
        ["passing_yards", "rushing_yards", "receptions", "passing_tds", "tds"]].sum()
    league = allowed[["passing_yards", "rushing_yards", "receptions", "passing_tds", "tds"]].mean()
    out = {}
    for team, g in allowed.sort_values("t", ascending=False).groupby("opponent_team"):
        g = g.head(10)
        w = _decay_weights(len(g), 0.88)
        out[team] = {c: float(1 + 0.5 * (np.average(g[c], weights=w) / league[c] - 1)) for c in league.index}
    return out


# Wind above 10 mph trims passing production (about 1.2% per mph).
WIND_PASS_PER_MPH = 0.012
WIND_STATS = {"pass_yds", "pass_tds", "rec_yds", "receptions"}


def project_nfl(hist: pd.DataFrame, dfac: dict, wind_mph: float = 0.0) -> dict | None:
    h = hist.head(16)
    if len(h) == 0:
        return None
    cur = (h["season"] == current_nfl_season()).to_numpy()
    w = _decay_weights(len(h), 0.85) * np.where(cur, 1.0, 0.5)
    pos = h["position"].iloc[0]
    out = {"games": len(h), "position": pos}
    for key, col in NFL_STATS.items():
        vals = h[col].to_numpy(float)
        if key == "tds":
            prior = TD_PRIOR.get(pos, 0.1)
            mean = (np.sum(w * vals) + 3 * prior) / (np.sum(w) + 3)
        else:
            mean = float(np.average(vals, weights=w))
        mean *= dfac.get(NFL_DEF_FACTOR[key], 1.0)
        if key in WIND_STATS:
            mean *= max(0.7, 1 - WIND_PASS_PER_MPH * max(0.0, wind_mph - 10))
        out[key] = {"mean": float(max(mean, 0.0)), "disp": _dispersion(np.clip(vals, 0, None), w, NFL_DISPERSION[key], 6)}
    return out


def nfl_projections(pw: pd.DataFrame, games: list[dict], injury_of=None) -> tuple[dict, PlayerIndex, dict]:
    """Same contract as nba_projections. Players ruled out are dropped (no target redistribution yet)."""
    pw = pw.dropna(subset=["player_id", "player_display_name"]).sort_values("t", ascending=False)
    latest = pw.drop_duplicates("player_id")[["player_id", "player_display_name", "team"]].rename(
        columns={"player_id": "id", "player_display_name": "name"})
    index = PlayerIndex(latest)
    dfacs = nfl_defense_factors(pw)
    team_last = pw.groupby("team")["t"].max()
    by_player = dict(tuple(pw.groupby("player_id")))
    out, notes = {}, {}
    for g in games:
        wx = g.get("weather") or {}
        wind = (wx.get("wind_mph") or 0.0) if wx.get("roof") == "outdoors" else 0.0
        for side, opp in (("home", "away"), ("away", "home")):
            team = g[side]
            roster = latest[latest["team"] == team]
            for pid in roster["id"]:
                hist = by_player[pid]
                if hist["position"].iloc[0] not in ("QB", "RB", "WR", "TE"):
                    continue
                last_t, tl = int(hist["t"].iloc[0]), int(team_last.get(team, hist["t"].iloc[0]))
                # Skip players absent for a month or more (injured reserve, cut, retired).
                if last_t // 100 != tl // 100 or tl - last_t > 3:
                    continue
                proj = project_nfl(hist, dfacs.get(g[opp], {}), wind)
                if not proj:
                    continue
                name = hist["player_display_name"].iloc[0]
                inj = injury_of(team, name, None) if injury_of else None
                if inj and inj["out"]:
                    volume = proj["pass_yds"]["mean"] / 4 + proj["rush_yds"]["mean"] + proj["rec_yds"]["mean"]
                    if volume >= 25:
                        notes.setdefault(g["id"], []).append(f"{team} {inj['status']}: {name} ({proj['position']})")
                    continue
                proj.update({
                    "id": pid, "name": name, "team": team, "opp": g[opp],
                    "game_id": g["id"], "kickoff": g["kickoff"],
                    "missed_last": bool(last_t < team_last.get(team, last_t)),
                    "last_played": f"{last_t // 100} wk {last_t % 100}",
                    "injury": inj,
                })
                out[(pid, g["id"])] = proj
    return out, index, notes


def prob_at_least(proj: dict, stat: str, k: int) -> float | None:
    s = proj.get(stat)
    if not s:
        return None
    return pricing.count_prob_at_least(s["mean"], s["disp"], k)
