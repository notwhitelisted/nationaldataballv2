import pytest

from app.data import kalshi
from app.models import pricing
from app.models.props import name_key


def test_american_odds_roundtrip():
    assert pricing.american_to_prob(-110) == pytest.approx(0.5238, abs=1e-4)
    assert pricing.american_to_prob(150) == pytest.approx(0.4)
    assert pricing.prob_to_american(0.4) == 150
    assert pricing.prob_to_american(0.6) == -150


def test_devig_sums_to_one():
    a, b = pricing.devig_two_way(-150, 130)
    assert a + b == pytest.approx(1.0)
    assert a > b


def test_kalshi_fee_rounds_up_to_cent():
    # 0.07 * 10 * 0.55 * 0.45 = 0.17325 -> $0.18
    assert pricing.kalshi_fee(0.55, 0.07, 10) == 0.18
    assert pricing.kalshi_fee(0.50, 0.07, 100) == 1.75


def test_evaluate_side_and_kelly():
    ev = pricing.evaluate_side(0.60, 0.50, 0.07)
    assert ev["fee"] == pytest.approx(0.0175)
    assert ev["ev"] == pytest.approx(0.60 - 0.5175)
    assert pricing.kelly_fraction(0.60, ev["cost"]) == pytest.approx((0.60 - 0.5175) / (1 - 0.5175))
    assert pricing.kelly_fraction(0.40, 0.50) == 0.0


def test_stake_respects_cap():
    settings = {"kelly_fraction": 1.0, "max_bet_pct": 0.02, "bankroll": 1000}
    s = pricing.recommend_stake(0.9, 0.5, settings)
    assert s["stake"] == 20.0
    assert s["contracts"] == 40


def test_count_probabilities():
    # Poisson fallback and negative binomial both behave like probabilities.
    p_pois = pricing.count_prob_at_least(2.0, 1.0, 6)
    p_nb = pricing.count_prob_at_least(2.0, 2.0, 6)
    assert 0 < p_pois < 1 and 0 < p_nb < 1
    assert p_nb > p_pois  # over-dispersion fattens the far upper tail
    assert pricing.count_prob_at_least(5.0, 1.5, 0) == 1.0


def test_normal_margin_symmetry():
    assert pricing.prob_over(0, 12, 0) == pytest.approx(0.5)
    assert pricing.prob_over(3, 12, 0) > 0.5
    assert pricing.margin_from_win_prob(0.5, 12) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("name,key", [
    ("Shai Gilgeous-Alexander", "SGILGEOUSALEXANDER"),
    ("Kel'el Ware", "KWARE"),
    ("Jaren Jackson Jr.", "JJACKSON"),
    ("Nikola Jokić", "NJOKIC"),
    ("Marquise Brown", "MBROWN"),
])
def test_name_key(name, key):
    assert name_key(name) == key


def test_split_teams():
    assert kalshi.split_teams("nfl", "CHISEA") == ("CHI", "SEA")
    assert kalshi.split_teams("nfl", "KCLV") == ("KC", "LV")
    assert kalshi.split_teams("nfl", "LARSF") == ("LAR", "SF")
    assert kalshi.split_teams("nba", "GSWDEN") == ("GSW", "DEN")


def test_parse_markets():
    base = {"yes_bid_dollars": "0.4000", "yes_ask_dollars": "0.4400", "close_time": None}
    spread = kalshi.parse_market({**base, "ticker": "KXNFLSPREAD-26NOV01PHIWAS-WAS8",
                                  "event_ticker": "KXNFLSPREAD-26NOV01PHIWAS", "floor_strike": 7.5,
                                  "title": "Washington wins by over 7.5 points?"}, "nfl", "spread")
    assert spread["team"] == "WAS" and spread["date"] == "2026-11-01" and spread["teams"] == ["PHI", "WAS"]
    assert spread["yes_bid"] == 0.40 and spread["yes_ask"] == 0.44

    prop = kalshi.parse_market({**base, "ticker": "KXNFLRECYDS-26OCT26DALPHI-PHIMBROWN0-50",
                                "event_ticker": "KXNFLRECYDS-26OCT26DALPHI", "floor_strike": 49.5,
                                "title": "Hollywood Brown: 50+ receiving yards"}, "nfl", "rec_yds")
    assert prop["team"] == "PHI" and prop["player_key"] == "MBROWN" and prop["player_name"] == "Hollywood Brown"

    empty = kalshi.parse_market({**base, "yes_bid_dollars": "0.0000", "ticker": "KXNBAGAME-26OCT29MEMPOR-POR",
                                 "event_ticker": "KXNBAGAME-26OCT29MEMPOR", "title": "Portland wins"}, "nba", "moneyline")
    assert empty["yes_bid"] is None and empty["team"] == "POR"


def test_implied_mean_reproduces_book_probability():
    mu = pricing.implied_mean(24.5, 0.55, 2.5)
    assert pricing.count_prob_at_least(mu, 2.5, 25) == pytest.approx(0.55, abs=1e-6)
    assert pricing.implied_mean(0.5, 0.40, 1.0) == pytest.approx(-__import__("math").log(0.60), abs=1e-6)


def test_book_prop_prob_ignores_far_lines():
    from app.data import odds
    quotes = [{"book": "pinnacle", "line": 13.5, "p_over": 0.5, "w": 3.0}]
    assert odds.book_prop_prob(quotes, 25, 2.6) is None       # 25+ is far from a 13.5 line
    assert 0.4 < odds.book_prop_prob(quotes, 14, 2.6) < 0.6   # 14+ is the line itself


def test_injury_parse():
    from app.data import injuries
    data = {"injuries": [{"displayName": "Baltimore Ravens", "injuries": [
        {"status": "Out", "shortComment": "ankle", "date": "2026-10-26T18:00Z",
         "athlete": {"displayName": "Lamar Jackson", "links": [{"href": "https://www.espn.com/nfl/player/_/id/3916387/lamar-jackson"}]}},
        {"status": "Active", "athlete": {"displayName": "Someone Else", "links": []}},
    ]}]}
    rows = injuries._parse("nfl", data)
    assert len(rows) == 1
    assert rows[0]["team"] == "BAL" and rows[0]["espn_id"] == "3916387" and rows[0]["out"]
    idx = {"by_id": {}, "by_name": {("BAL", "LJACKSON"): rows[0]}}
    assert injuries.find(idx, "BAL", "Lamar Jackson") is rows[0]


def test_weather_wind_adjustment(monkeypatch):
    from datetime import datetime, timezone
    from app.data import weather
    hourly = {"time": ["2026-11-01T18:00"], "temperature_2m": [40.0], "wind_speed_10m": [20.0],
              "wind_gusts_10m": [30.0], "precipitation_probability": [10], "precipitation": [0.0], "snowfall": [0.0]}
    monkeypatch.setattr(weather, "_hourly", lambda lat, lon: hourly)
    wx = weather.forecast("BUF00", datetime(2026, 11, 1, 18, tzinfo=timezone.utc))
    assert wx["total_adj"] == pytest.approx(-3.0)
    assert weather.forecast("DET00", datetime(2026, 11, 1, 18, tzinfo=timezone.utc))["roof"] == "dome"
    assert weather.forecast("DAL00", datetime(2026, 11, 1, 18, tzinfo=timezone.utc))["total_adj"] == 0.0


def test_settlement_pnl_with_partial_sell():
    from app import bets

    class Conn:
        def execute(self, sql, params):
            self.params = params

    b = {"id": 1, "contracts": 10, "price": 0.40, "fees": 0.17, "sold_contracts": 4, "sell_proceeds": 2.40}
    c = Conn()
    bets._apply_settlement(c, b, "won")
    assert c.params[1] == pytest.approx(2.40 + 6 - 4.00 - 0.17)
    bets._apply_settlement(c, b, "lost")
    assert c.params[1] == pytest.approx(2.40 - 4.00 - 0.17)


def test_fill_price_formats():
    from app import bets
    assert bets._fill_price({"yes_price": 42}, "yes") == pytest.approx(0.42)
    assert bets._fill_price({"no_price_dollars": "0.5800"}, "no") == pytest.approx(0.58)


def test_kalshi_signature_verifies(tmp_path, monkeypatch):
    import base64
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding, rsa
    from app.data import kalshi_account

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = tmp_path / "k.pem"
    pem.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                      serialization.NoEncryption()))
    monkeypatch.setenv("KALSHI_API_KEY_ID", "test-id")
    monkeypatch.setenv("KALSHI_PRIVATE_KEY_PATH", str(pem))
    monkeypatch.setattr(kalshi_account, "_private_key", None)
    assert kalshi_account.configured()
    h = kalshi_account._sign("GET", "/portfolio/fills")
    msg = (h["KALSHI-ACCESS-TIMESTAMP"] + "GET/trade-api/v2/portfolio/fills").encode()
    key.public_key().verify(base64.b64decode(h["KALSHI-ACCESS-SIGNATURE"]), msg,
                            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
                            hashes.SHA256())
    assert h["KALSHI-ACCESS-KEY"] == "test-id"


def test_ladder_keys_group_rungs():
    from app.edges import ladder_key
    a = ladder_key({"ticker": "KXNFLSPREAD-26NOV01PHIWAS-WAS8", "type": "spread"})
    b = ladder_key({"ticker": "KXNFLSPREAD-26NOV01PHIWAS-WAS4", "type": "spread"})
    c = ladder_key({"ticker": "KXNFLSPREAD-26NOV01PHIWAS-PHI4", "type": "spread"})
    assert a == b != c
    assert ladder_key({"ticker": "KXNBAPTS-26OCT26MINCLE-MINLBALL1-30", "type": "pts"}) == \
        ladder_key({"ticker": "KXNBAPTS-26OCT26MINCLE-MINLBALL1-25", "type": "pts"})


def _edge_row(**kw):
    r = {"side": "yes", "book_prob": 0.60, "cost": 0.52, "price": 0.50, "roi": 0.08, "spread": 0.01,
         "flags": [], "player": None}
    return {**r, **kw}


def test_shortlist_uses_book_prob_for_the_side_bet():
    from app.edges import book_side_ev, shortlist_reject
    assert shortlist_reject(_edge_row()) is None
    # Hawks +2.5 on Oct 26: NO on "HOU by over 2.5" at ~44.6c all-in, books had HOU covering 57%.
    hawks = _edge_row(side="no", book_prob=0.57, cost=0.446, price=0.43, roi=0.063)
    assert book_side_ev(hawks) < 0
    assert shortlist_reject(hawks) == "sportsbooks don't see value"


def test_shortlist_rejections():
    from app.edges import shortlist_reject
    assert shortlist_reject(_edge_row(book_prob=None)) == "no sportsbook price"
    assert shortlist_reject(_edge_row(price=0.12, cost=0.13, book_prob=0.2)) == "long shot / heavy favorite"
    assert shortlist_reject(_edge_row(spread=0.06)) == "wide Kalshi book"
    assert shortlist_reject(_edge_row(player={"name": "X"}, flags=["Questionable: ankle"])) == "player injury status"
    # Team-level injury notes on a game don't block that game's bets.
    assert shortlist_reject(_edge_row(flags=["HOU Out: Marcus Smart (25 min)"])) is None


def test_max_price_is_highest_cent_clearing_the_bar():
    p = pricing.max_price(0.55, 0.04, 0.07)
    assert pricing.evaluate_side(0.55, p, 0.07)["roi"] >= 0.04
    assert pricing.evaluate_side(0.55, p + 0.01, 0.07)["roi"] < 0.04
    assert pricing.max_price(0.01, 0.04, 0.07) is None


def test_paper_lines_freshness():
    from datetime import datetime, timedelta, timezone
    from app.paper import lines_fresh
    now = datetime.now(timezone.utc)
    assert lines_fresh((now - timedelta(hours=2)).isoformat())
    assert not lines_fresh((now - timedelta(hours=13)).isoformat())
    assert not lines_fresh(None)


def _nba_proj(pid, minutes, usg_rate, out=False, missed_last=False):
    stats = {s: {"mean": 10.0, "disp": 1.5} for s in ("pts", "reb", "ast", "fg3m")}
    return {"id": pid, "name": pid, "team": "IND", "min": minutes, "usg_rate": usg_rate, "missed_last": missed_last,
            "injury": {"out": True, "status": "Out"} if out else None, **stats, "pra": {"mean": 30.0, "disp": 3.2}}


def test_nba_injury_hands_out_minutes_and_usage():
    from app.models.props import NBA_USAGE_ABSORB, _apply_nba_injuries
    # Star (40% of rotation usage) out: teammates get some of his minutes and a scoring-rate bump.
    team = {"star": _nba_proj("star", 30, 0.6, out=True), "a": _nba_proj("a", 30, 0.3),
            "bench": _nba_proj("bench", 10, 0.3), "b": _nba_proj("b", 30, 0.3)}
    notes = _apply_nba_injuries(team)
    assert "star" not in team and notes == ["IND Out: star (30 min)"]
    a = team["a"]
    assert a["min"] == pytest.approx(30.75)  # 24-31 mpg players gain 0.025 per freed minute
    assert team["bench"]["min"] == pytest.approx(13.3)  # the bench gains most
    share = 0.6 * 30 / (0.6 * 30 + 0.3 * 30 * 2)
    assert a["usage_boost"] == pytest.approx(NBA_USAGE_ABSORB["pts"] * share)
    assert a["pts"]["mean"] == pytest.approx(10 * 30.75 / 30 * (1 + NBA_USAGE_ABSORB["pts"] * share))
    assert a["reb"]["mean"] < a["pts"]["mean"]  # rebounds barely follow usage
    assert a["pra"]["mean"] == pytest.approx(a["pts"]["mean"] + a["reb"]["mean"] + a["ast"]["mean"])


def test_nba_injury_already_missing_changes_nothing():
    from app.models.props import _apply_nba_injuries
    # Out for weeks: teammates' recent games already reflect it.
    team = {"x": _nba_proj("x", 30, 0.6, out=True, missed_last=True), "a": _nba_proj("a", 30, 0.3)}
    _apply_nba_injuries(team)
    assert team["a"]["pts"]["mean"] == 10.0 and "usage_boost" not in team["a"]
