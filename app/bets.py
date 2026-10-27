"""Bet tracker: real and paper positions, Kalshi import, auto-settlement and CLV analytics."""
from collections import defaultdict
from datetime import datetime, timezone

import httpx

from .data import kalshi, kalshi_account
from .db import session
from .models import pricing


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


INSERT_COLS = ["created_at", "sport", "ticker", "description", "side", "price", "contracts", "fees", "fair_prob",
               "game_start", "notes", "mode", "source", "ext_id", "market_type", "edge"]


def create(b: dict, fee_coef: float) -> dict:
    row = {c: b.get(c) for c in INSERT_COLS}
    row["created_at"] = b.get("created_at") or _now()
    row["mode"] = b.get("mode") or "real"
    row["source"] = b.get("source") or "manual"
    if row["fees"] is None:
        row["fees"] = pricing.kalshi_fee(b["price"], fee_coef, b["contracts"])
    with session() as conn:
        cur = conn.execute(
            f"INSERT INTO bets ({', '.join(INSERT_COLS)}) VALUES ({', '.join('?' * len(INSERT_COLS))})",
            [row[c] for c in INSERT_COLS],
        )
        return dict(conn.execute("SELECT * FROM bets WHERE id = ?", (cur.lastrowid,)).fetchone())


def has_ext_id(ext_id: str) -> bool:
    with session() as conn:
        return conn.execute("SELECT 1 FROM bets WHERE ext_id = ?", (ext_id,)).fetchone() is not None


def delete(bet_id: int) -> None:
    with session() as conn:
        conn.execute("DELETE FROM bets WHERE id = ?", (bet_id,))


def settle(bet_id: int, status: str) -> None:
    with session() as conn:
        b = conn.execute("SELECT * FROM bets WHERE id = ?", (bet_id,)).fetchone()
        if b:
            _apply_settlement(conn, dict(b), status)


def _apply_settlement(conn, b: dict, status: str) -> None:
    held = b["contracts"] - (b["sold_contracts"] or 0)
    cost = b["contracts"] * b["price"] + b["fees"]
    proceeds = b["sell_proceeds"] or 0.0
    if status == "won":
        pnl = proceeds + held - cost
    elif status == "lost":
        pnl = proceeds - cost
    elif status == "void":
        pnl = 0.0
    else:
        pnl, status = None, "open"
    conn.execute("UPDATE bets SET status = ?, pnl = ?, settled_at = ? WHERE id = ?",
                 (status, pnl, _now() if status != "open" else None, b["id"]))


def sync() -> dict:
    """Settle finished markets and record pre-game prices (for CLV) for every open bet."""
    with session() as conn:
        open_bets = [dict(r) for r in conn.execute("SELECT * FROM bets WHERE status = 'open'")]
    if not open_bets:
        return {"settled": 0, "open": 0}
    try:
        markets = kalshi.get_markets_by_ticker(sorted({b["ticker"] for b in open_bets}))
    except httpx.HTTPError as e:
        return {"error": str(e)}
    now = datetime.now(timezone.utc)
    settled = 0
    with session() as conn:
        for b in open_bets:
            m = markets.get(b["ticker"])
            if not m:
                continue
            result = (m.get("result") or "").lower()
            if result in ("yes", "no"):
                _apply_settlement(conn, b, "won" if result == b["side"] else "lost")
                settled += 1
                continue
            if result == "void":
                _apply_settlement(conn, b, "void")
                settled += 1
                continue
            # Before kickoff, keep recording the price so the last value is our closing line.
            start = datetime.fromisoformat(b["game_start"]) if b.get("game_start") else None
            if start is None or now < start:
                try:
                    bid, ask = float(m.get("yes_bid_dollars") or 0), float(m.get("yes_ask_dollars") or 0)
                except ValueError:
                    continue
                if 0 < bid < 1 and 0 < ask < 1:
                    mid = (bid + ask) / 2
                    conn.execute("UPDATE bets SET close_price = ? WHERE id = ?",
                                 (mid if b["side"] == "yes" else 1 - mid, b["id"]))
    return {"settled": settled, "open": len(open_bets) - settled}


# ---------------------------------------------------------------- Kalshi import

def _num(d: dict, *keys):
    for k in keys:
        v = d.get(k)
        if v not in (None, ""):
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return None


def _fill_price(f: dict, side: str) -> float | None:
    p = _num(f, f"{side}_price_dollars", f"{side}_price_fixed")
    if p is None:
        p = _num(f, f"{side}_price")
        if p is not None and p > 1:
            p /= 100  # cents
    return p


def import_from_kalshi(fee_coef: float, kickoff_for) -> dict:
    """Pull your Kalshi fills and mirror them as real bets (one row per market + side).

    Buys set the average price and contract count; sells before settlement are tracked so
    P&L stays right. A bet you logged by hand for the same market and side is matched and
    updated with the real fills (keeping its fair probability). kickoff_for(ticker) -> ISO time.
    """
    agg = {}
    for f in sorted(kalshi_account.fills(), key=lambda f: f.get("created_time", "")):
        side, action = f.get("side"), f.get("action")
        if side not in ("yes", "no") or action not in ("buy", "sell"):
            continue
        # Kalshi reports closing a YES position as "sell no" (and vice versa), priced on the
        # position's own side. So a sell reduces the opposite side's holding.
        pos_side = side if action == "buy" else ("yes" if side == "no" else "no")
        n, price = _num(f, "count_fp", "count"), _fill_price(f, pos_side)
        if not n or price is None:
            continue
        a = agg.setdefault((f["ticker"], pos_side), {"bought": 0.0, "cost": 0.0, "sold": 0.0, "proceeds": 0.0,
                                                     "fees": 0.0, "first": f.get("created_time")})
        fee = _num(f, "fee_cost_dollars", "fee_cost")
        if fee is None:  # older payloads have no fee field: estimate (makers usually pay less or nothing)
            fee = pricing.kalshi_fee(price, fee_coef, max(1, round(n))) if f.get("is_taker", True) else 0.0
        a["fees"] += fee
        if action == "buy":
            a["bought"] += n
            a["cost"] += n * price
        else:
            a["sold"] += n
            a["proceeds"] += n * price

    agg = {k: a for k, a in agg.items() if a["bought"] > 0}
    titles = {}
    try:
        titles = {t: m.get("title", t) for t, m in kalshi.get_markets_by_ticker(sorted({t for t, _ in agg})).items()}
    except httpx.HTTPError:
        pass
    created = updated = 0
    with session() as conn:
        for (ticker, side), a in agg.items():
            ext = f"kalshi:{ticker}:{side}"
            # Kalshi allows fractional contracts, so counts are kept exact.
            vals = {"price": a["cost"] / a["bought"], "contracts": round(a["bought"], 2), "fees": round(a["fees"], 4),
                    "sold_contracts": round(a["sold"], 2), "sell_proceeds": a["proceeds"]}
            row = conn.execute("SELECT id FROM bets WHERE ext_id = ?", (ext,)).fetchone()
            if row is None:
                row = conn.execute("SELECT id FROM bets WHERE mode = 'real' AND ext_id IS NULL AND ticker = ? AND side = ? "
                                   "ORDER BY created_at LIMIT 1", (ticker, side)).fetchone()
            if row:
                conn.execute("UPDATE bets SET price = :price, contracts = :contracts, fees = :fees, "
                             "sold_contracts = :sold_contracts, sell_proceeds = :sell_proceeds, ext_id = :ext, "
                             "source = 'kalshi' WHERE id = :id", {**vals, "ext": ext, "id": row["id"]})
                updated += 1
                continue
            sport, mtype = kalshi.series_info(ticker)
            conn.execute(
                "INSERT INTO bets (created_at, sport, ticker, description, side, price, contracts, fees, game_start, "
                "mode, source, ext_id, market_type, sold_contracts, sell_proceeds) VALUES "
                "(:created_at, :sport, :ticker, :description, :side, :price, :contracts, :fees, :game_start, "
                "'real', 'kalshi', :ext, :mtype, :sold_contracts, :sell_proceeds)",
                {**vals, "created_at": a["first"] or _now(), "sport": sport, "ticker": ticker, "side": side,
                 "description": f"{side.upper()} {titles.get(ticker, ticker)}", "game_start": kickoff_for(ticker),
                 "ext": ext, "mtype": mtype},
            )
            created += 1
    return {"created": created, "updated": updated}


# ---------------------------------------------------------------- reporting

GAME_TYPES = {"moneyline", "spread", "total", "team_total"}


def _edge_bucket(e):
    if e is None:
        return "unknown"
    for hi, label in ((0.05, "< 5%"), (0.10, "5-10%"), (0.20, "10-20%")):
        if e < hi:
            return label
    return "20%+"


def _price_bucket(p):
    return "< 25¢" if p < 0.25 else "25-50¢" if p < 0.5 else "50-75¢" if p < 0.75 else "75¢+"


def _stats(rows: list[dict]) -> dict:
    settled = [r for r in rows if r["status"] in ("won", "lost")]
    risked = sum(r["price"] * r["contracts"] + r["fees"] for r in settled)
    pnl = sum(r["pnl"] or 0 for r in settled)
    clv = [r for r in rows if r["close_price"] is not None]
    return {
        "bets": len(rows),
        "open": sum(r["status"] == "open" for r in rows),
        "won": sum(r["status"] == "won" for r in rows),
        "lost": sum(r["status"] == "lost" for r in rows),
        "risked": risked,
        "pnl": pnl,
        "roi": pnl / risked if risked else None,
        # Positive CLV (price moved toward us before kickoff) is the best early signal of real edge.
        "avg_clv": sum(r["close_price"] - r["price"] for r in clv) / len(clv) if clv else None,
        "clv_beat_pct": sum(r["close_price"] > r["price"] for r in clv) / len(clv) if clv else None,
        "clv_n": len(clv),
        "expected_pnl": sum((r["fair_prob"] - r["price"]) * r["contracts"] - r["fees"]
                            for r in rows if r["fair_prob"] is not None),
    }


SOURCE_LABELS = {"shortlist": "Shortlist", "auto": "All qualifying edges", "manual": "Logged by hand",
                 "kalshi": "Imported from Kalshi"}


def list_bets(mode: str = "real") -> dict:
    with session() as conn:
        q = "SELECT * FROM bets" + ("" if mode == "all" else " WHERE mode = ?") + " ORDER BY created_at DESC"
        rows = [dict(r) for r in conn.execute(q, () if mode == "all" else (mode,))]
    groupings = {
        "Market": lambda r: ("Game: " if r["market_type"] in GAME_TYPES else "Prop: ") + (r["market_type"] or "?")
                            if r["market_type"] else "unknown",
        "Sport": lambda r: r["sport"],
        "Strategy": lambda r: SOURCE_LABELS.get(r["source"] or "manual", r["source"]),
        "Edge at bet": lambda r: _edge_bucket(r["edge"]),
        "Price": lambda r: _price_bucket(r["price"]),
    }
    breakdown = {}
    for name, keyf in groupings.items():
        groups = defaultdict(list)
        for r in rows:
            groups[keyf(r)].append(r)
        breakdown[name] = sorted(({"group": k, **_stats(v)} for k, v in groups.items()), key=lambda g: -g["bets"])
    return {"summary": _stats(rows), "breakdown": breakdown, "bets": rows}
