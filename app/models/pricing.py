"""Probability, odds and bet-sizing math shared by every model."""
import math

from scipy import optimize, stats


def american_to_prob(odds: float | None) -> float | None:
    if odds is None or (isinstance(odds, float) and math.isnan(odds)) or odds == 0:
        return None
    return 100 / (odds + 100) if odds > 0 else -odds / (-odds + 100)


def prob_to_american(p: float) -> int:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return round(-100 * p / (1 - p)) if p >= 0.5 else round(100 * (1 - p) / p)


def devig_two_way(odds_a: float | None, odds_b: float | None) -> tuple[float, float] | None:
    """Remove the bookmaker margin from a two-way market (multiplicative method)."""
    pa, pb = american_to_prob(odds_a), american_to_prob(odds_b)
    if pa is None or pb is None:
        return None
    s = pa + pb
    return pa / s, pb / s


def prob_over(mean: float, sd: float, line: float) -> float:
    """P(X > line) for a normally distributed quantity (margins, totals)."""
    return float(stats.norm.sf(line, loc=mean, scale=sd))


def margin_from_win_prob(p: float, sd: float) -> float:
    """Expected margin implied by a win probability under the normal margin model."""
    p = min(max(p, 1e-4), 1 - 1e-4)
    return float(stats.norm.ppf(p) * sd)


def count_prob_at_least(mean: float, var_ratio: float, k: int) -> float:
    """P(X >= k) for an over-dispersed count: negative binomial with var = var_ratio * mean.

    Falls back to Poisson when the data is not over-dispersed.
    """
    if k <= 0:
        return 1.0
    mean = max(mean, 1e-6)
    if var_ratio <= 1.02:
        return float(stats.poisson.sf(k - 1, mean))
    n = mean / (var_ratio - 1)
    p = 1 / var_ratio
    return float(stats.nbinom.sf(k - 1, n, p))


def implied_mean(line: float, p_over: float, var_ratio: float) -> float | None:
    """Mean of the count distribution under which P(X > line) equals p_over."""
    k = math.floor(line) + 1
    p_over = min(max(p_over, 0.01), 0.99)
    try:
        return float(optimize.brentq(lambda mu: count_prob_at_least(mu, var_ratio, k) - p_over,
                                     1e-3, max(5 * line + 20, 60)))
    except ValueError:
        return None


def kalshi_fee(price: float, coef: float, contracts: int = 1) -> float:
    """Kalshi trading fee in dollars: ceil to the cent of coef * C * P * (1 - P)."""
    raw = coef * contracts * price * (1 - price)
    return math.ceil(round(raw * 100, 6)) / 100


def evaluate_side(fair_prob: float, price: float, fee_coef: float) -> dict:
    """Expected value of buying one contract at `price` that pays $1 with prob `fair_prob`.

    The fee is amortised per contract (as if buying ~100 contracts) so tiny orders
    don't look artificially bad because of cent rounding.
    """
    fee = fee_coef * price * (1 - price)
    cost = price + fee
    ev = fair_prob - cost
    return {"price": price, "fee": fee, "cost": cost, "ev": ev, "roi": ev / cost if cost > 0 else 0.0}


def max_price(prob: float, min_roi: float, coef: float) -> float | None:
    """Highest whole-cent price at which a contract winning with `prob` still returns `min_roi`
    after the taker fee: the most to pay when placing a limit order."""
    for c in range(99, 0, -1):
        if evaluate_side(prob, c / 100, coef)["roi"] >= min_roi:
            return c / 100
    return None


def kelly_fraction(fair_prob: float, cost: float) -> float:
    """Full-Kelly bankroll fraction for a binary contract costing `cost` that pays 1."""
    if cost <= 0 or cost >= 1:
        return 0.0
    return max(0.0, (fair_prob - cost) / (1 - cost))


def recommend_stake(fair_prob: float, cost: float, settings: dict) -> dict:
    f = kelly_fraction(fair_prob, cost) * settings["kelly_fraction"]
    f = min(f, settings["max_bet_pct"])
    dollars = f * settings["bankroll"]
    return {"stake_pct": f, "stake": round(dollars, 2), "contracts": int(dollars // cost) if cost > 0 else 0}
