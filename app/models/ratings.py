"""Team strength model: recency-weighted ridge regression on points scored.

    points(team, game) = mu + hfa*home + off[team] - def[opp] + b2b * (own_b2b - opp_b2b)

Each completed game contributes two rows (home and away). Older games are down-weighted
with an exponential half-life measured on an "effective" clock where the offseason gap is
compressed, so last season still informs early-season ratings without dominating them.
The ridge penalty shrinks every team toward league average, which matters a lot early on.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Params:
    half_life: float      # days on the effective clock
    ridge: float          # L2 penalty on off/def ratings
    offseason_gap: float  # effective days the offseason counts as
    use_b2b: bool
    sd_bounds: dict = field(default_factory=dict)  # name -> (lo, hi)


@dataclass
class TeamModel:
    params: Params
    mu: float
    hfa: float
    b2b: float
    off: dict
    dfn: dict
    sd_margin: float
    sd_total: float
    sd_team: float
    n_games: int

    def predict(self, home: str, away: str, neutral: bool = False, home_b2b: int = 0, away_b2b: int = 0):
        h = 0.0 if neutral else self.hfa
        rest = home_b2b - away_b2b
        hp = self.mu + h + self.off.get(home, 0.0) - self.dfn.get(away, 0.0) + self.b2b * rest
        ap = self.mu + self.off.get(away, 0.0) - self.dfn.get(home, 0.0) - self.b2b * rest
        return hp, ap

    def table(self) -> list[dict]:
        rows = [{"team": t, "off": self.off[t], "def": self.dfn[t], "net": self.off[t] + self.dfn[t]} for t in self.off]
        return sorted(rows, key=lambda r: -r["net"])


def effective_days(dates: pd.Series, offseason_gap: float) -> pd.Series:
    """Map calendar dates to a clock where any gap > 45 days counts as `offseason_gap` days."""
    uniq = np.sort(dates.unique())
    if len(uniq) == 0:
        return pd.Series(dtype=float)
    diffs = np.diff(uniq).astype("timedelta64[D]").astype(float)
    diffs = np.where(diffs > 45, offseason_gap, diffs)
    clock = pd.Series(np.concatenate([[0.0], np.cumsum(diffs)]), index=uniq)
    return dates.map(clock)


def add_back_to_back(games: pd.DataFrame) -> pd.DataFrame:
    """Flag teams playing on consecutive days (uses completed and scheduled games)."""
    g = games.copy()
    long = pd.concat([
        pd.DataFrame({"idx": g.index, "team": g["home_team"], "date": g["date"], "side": "home"}),
        pd.DataFrame({"idx": g.index, "team": g["away_team"], "date": g["date"], "side": "away"}),
    ]).sort_values(["team", "date"])
    long["b2b"] = (long.groupby("team")["date"].diff().dt.days == 1).astype(int)
    for side in ("home", "away"):
        part = long[long["side"] == side].set_index("idx")["b2b"]
        g[f"{side}_b2b"] = part.reindex(g.index).fillna(0).astype(int)
    return g


def fit(games: pd.DataFrame, as_of: pd.Timestamp, p: Params) -> TeamModel:
    """Fit on completed games strictly before `as_of`.

    `games` columns: date, home_team, away_team, home_score, away_score, completed,
    and optionally neutral / home_b2b / away_b2b.
    """
    g = games[games["completed"] & (games["date"] < as_of)]
    teams = sorted(set(g["home_team"]) | set(g["away_team"]))
    idx = {t: i for i, t in enumerate(teams)}
    n, T = len(g), len(teams)

    all_dates = pd.concat([g["date"], pd.Series([as_of])], ignore_index=True)
    clock = effective_days(all_dates, p.offseason_gap)
    age = clock.iloc[-1] - clock.iloc[:-1].to_numpy()
    w = 0.5 ** (age / p.half_life)
    ww = np.concatenate([w, w])

    base = 3  # mu, hfa, b2b
    X = np.zeros((2 * n, base + 2 * T))
    r = np.arange(n)
    hi = g["home_team"].map(idx).to_numpy()
    ai = g["away_team"].map(idx).to_numpy()
    neutral = g["neutral"].to_numpy().astype(float) if "neutral" in g else np.zeros(n)
    X[:, 0] = 1
    X[r, 1] = 1 - neutral
    X[r, base + hi] = 1
    X[r, base + T + ai] = -1
    X[n + r, base + ai] = 1
    X[n + r, base + T + hi] = -1
    if p.use_b2b and "home_b2b" in g:
        # Net rest effect on a team's scoring: its own fatigue minus its opponent's.
        rest = g["home_b2b"].to_numpy() - g["away_b2b"].to_numpy()
        X[r, 2] = rest
        X[n + r, 2] = -rest
    y = np.concatenate([g["home_score"].to_numpy(float), g["away_score"].to_numpy(float)])

    penalty = np.zeros(X.shape[1])
    penalty[base:] = p.ridge
    penalty[2] = 1e-6 if p.use_b2b else 1e9  # effectively drop the b2b column when unused
    A = X.T @ (X * ww[:, None]) + np.diag(penalty)
    beta = np.linalg.solve(A, X.T @ (ww * y))

    pred = X @ beta
    hp, ap = pred[:n], pred[n:]
    hs, as_ = y[:n], y[n:]

    def wsd(resid, key):
        wr = w if len(resid) == n else ww
        sd = float(np.sqrt(np.sum(wr * resid ** 2) / np.sum(wr))) * 1.08  # in-sample residuals understate error
        lo, hi_ = p.sd_bounds.get(key, (0, np.inf))
        return float(np.clip(sd, lo, hi_))

    return TeamModel(
        params=p,
        mu=float(beta[0]),
        hfa=float(beta[1]),
        b2b=float(beta[2]) if p.use_b2b else 0.0,
        off={t: float(beta[base + i]) for t, i in idx.items()},
        dfn={t: float(beta[base + T + i]) for t, i in idx.items()},
        sd_margin=wsd((hs - as_) - (hp - ap), "margin"),
        sd_total=wsd((hs + as_) - (hp + ap), "total"),
        sd_team=wsd(np.concatenate([hs - hp, as_ - ap]), "team"),
        n_games=n,
    )
