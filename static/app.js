const state = { shortlist: null, betMode: "real", sport: "nfl", tab: "edges", edges: [], games: [], props: [], propLines: [], injuries: [], sort: {}, pending: null, odds: null };
const $ = (s) => document.querySelector(s);

async function api(path, opts = {}) {
  const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
  return r.json();
}

const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const pct = (p, d = 1) => (p == null ? "–" : (p * 100).toFixed(d) + "%");
const cents = (p) => (p == null ? "–" : Math.round(p * 100) + "¢");
const num = (x, d = 1) => (x == null ? "–" : Number(x).toFixed(d));
const money = (x) => (x == null ? "–" : (x < 0 ? "-$" : "$") + Math.abs(x).toFixed(2));
const when = (iso) => new Date(iso).toLocaleString([], { weekday: "short", month: "numeric", day: "numeric", hour: "numeric", minute: "2-digit" });
const signed = (x, d = 1) => (x == null ? "–" : (x > 0 ? "+" : "") + Number(x).toFixed(d));
// Express a home-team margin as a conventional spread, e.g. "PHI -3.5".
const spreadText = (home, away, m) => (m == null ? "–" : Math.abs(m) < 0.05 ? "PK" : `${m > 0 ? home : away} -${Math.abs(m).toFixed(1)}`);
const flags = (fs) => (fs || []).map((f) => `<span class="flag">${esc(f)}</span>`).join("");

/** Render a sortable table. cols: [{label, get, html?, num?}] */
function table(el, cols, rows, key) {
  const s = state.sort[key];
  const sorted = s
    ? [...rows].sort((a, b) => {
        const va = cols[s.i].get(a), vb = cols[s.i].get(b);
        if (va == null) return 1;
        if (vb == null) return -1;
        return (va > vb ? 1 : va < vb ? -1 : 0) * s.dir;
      })
    : rows;
  el._rows = sorted;
  if (!sorted.length) {
    el.innerHTML = `<div class="empty">Nothing to show. Try Refresh data or loosen the filters.</div>`;
    return;
  }
  const head = cols.map((c, i) => `<th data-i="${i}" class="${c.num ? "num" : ""}">${c.label}${s && s.i === i ? (s.dir > 0 ? " ▲" : " ▼") : ""}</th>`).join("");
  const body = sorted
    .map((r, ri) => `<tr data-ri="${ri}">${cols.map((c) => `<td class="${c.num ? "num" : ""}">${c.html ? c.html(r) : esc(c.get(r))}</td>`).join("")}</tr>`)
    .join("");
  el.innerHTML = `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
  el.querySelectorAll("th").forEach((th) => {
    th.onclick = () => {
      const i = +th.dataset.i, cur = state.sort[key];
      state.sort[key] = { i, dir: cur && cur.i === i ? -cur.dir : -1 };
      table(el, cols, rows, key);
    };
  });
}

// ---------------------------------------------------------------- Edges

const GAME_TYPES = new Set(["moneyline", "spread", "total", "team_total"]);
const forSide = (r, p) => (p == null ? null : r.side === "yes" ? p : 1 - p);

function filteredEdges() {
  const group = $("#f-group").value;
  const minEv = parseFloat($("#f-minev").value || "0") / 100;
  const maxSpread = parseFloat($("#f-spread").value || "100") / 100;
  const hideTails = $("#f-tails").checked;
  const hideFlagged = $("#f-flagged").checked;
  const q = $("#f-search").value.trim().toLowerCase();
  return state.edges.filter((r) => {
    if (group === "game" && !GAME_TYPES.has(r.type)) return false;
    if (group === "props" && GAME_TYPES.has(r.type)) return false;
    if (r.roi < minEv) return false;
    if (r.spread == null || r.spread > maxSpread + 1e-9) return false;
    if (hideTails && (r.price < 0.1 || r.price > 0.9)) return false;
    if (hideFlagged && r.flags.some((f) => f.startsWith("Model far from market"))) return false;
    if (q && !`${r.title} ${r.game}`.toLowerCase().includes(q)) return false;
    return true;
  });
}

const isShortlist = () => $("#f-view").value === "shortlist";

function shortlistRows() {
  const group = $("#f-group").value;
  const q = $("#f-search").value.trim().toLowerCase();
  return (state.shortlist?.picks || []).filter((r) => {
    if (group === "game" && !GAME_TYPES.has(r.type)) return false;
    if (group === "props" && GAME_TYPES.has(r.type)) return false;
    return !q || `${r.title} ${r.game}`.toLowerCase().includes(q);
  });
}

function renderEdges() {
  const el = $("#edges-table");
  const sl = isShortlist();
  $("#f-advanced").hidden = sl;
  $("#shortlist-note").hidden = !sl;
  const missing = sl ? state.shortlist?.games_without_props || [] : [];
  const linesAt = sl ? state.shortlist?.lines_at : null;
  const ageH = linesAt ? (Date.now() - new Date(linesAt)) / 3.6e6 : null;
  $("#shortlist-missing").innerHTML = !sl ? "" : [
    ageH == null ? `<span class="flag">No sportsbook lines yet: click Update lines.</span>`
      : `<span class="${ageH > 12 ? "flag" : "muted"}">Sportsbook lines are ${ageH < 1 ? "under an hour" : Math.round(ageH) + " hours"} old${ageH > 12 ? ": click Update lines before betting" : ""}.</span>`,
    missing.length ? `<span class="flag">No sportsbook props pulled for: ${esc(missing.join(", "))}</span>
       <span class="muted">Prop bets from these games can't make the shortlist until you click Props on the Games tab.</span>` : "",
  ].join(" ");
  table(el, [
    { label: "Kickoff", get: (r) => r.kickoff, html: (r) => when(r.kickoff) },
    { label: "Game", get: (r) => r.game },
    { label: "Market", get: (r) => r.type_label },
    { label: "Bet", get: (r) => r.title, html: (r) => `<span class="side-${r.side}">${r.side.toUpperCase()}</span> ${esc(r.title)}${r.player ? ` <span class="muted small">(proj ${num(r.player.mean)})</span>` : ""}` },
    { label: "Price", num: true, get: (r) => r.price, html: (r) => cents(r.price) },
    { label: "Fair", num: true, get: (r) => r.side_prob, html: (r) => pct(r.side_prob) },
    { label: "Model", num: true, get: (r) => forSide(r, r.model_prob), html: (r) => pct(forSide(r, r.model_prob)) },
    { label: "Book", num: true, get: (r) => forSide(r, r.book_prob), html: (r) => pct(forSide(r, r.book_prob)) },
    { label: "Bid/Ask (yes)", num: true, get: (r) => r.spread, html: (r) => `${cents(r.yes_bid)} / ${cents(r.yes_ask)}` },
    { label: "EV", num: true, get: (r) => r.roi, html: (r) => `<span class="${r.roi > 0 ? "pos" : "neg"}">${pct(r.roi)}</span>` },
    ...(sl ? [
      { label: "Book-only EV", num: true, get: (r) => r.book_roi, html: (r) => pct(r.book_roi) },
      { label: "Max price", num: true, get: (r) => r.max_price, html: (r) => `<b>${cents(r.max_price)}</b>` },
    ] : []),
    { label: "Stake", num: true, get: (r) => r.stake.stake, html: (r) => `${money(r.stake.stake)} <span class="muted small">(${r.stake.contracts})</span>` },
    { label: "Notes", get: (r) => r.flags.length, html: (r) => flags(r.flags) },
    { label: "", get: () => 0, html: (r) => `<button data-log="${esc(r.ticker)}">Log</button> <button data-paper="${esc(r.ticker)}" title="Record a no-money test bet at this price">Paper</button>` },
  ], sl ? shortlistRows() : filteredEdges(), sl ? "shortlist" : "edges");
}

async function loadEdges(force = false) {
  $("#edges-table").innerHTML = `<div class="empty">Loading Kalshi markets and model…</div>`;
  try {
    if (isShortlist()) state.shortlist = await api(`/api/shortlist?sport=${state.sport}&force=${force}`);
    else state.edges = await api(`/api/edges?sport=${state.sport}&force=${force}`);
    renderEdges();
  } catch (e) {
    $("#edges-table").innerHTML = `<div class="empty">Error: ${esc(e.message)}</div>`;
  }
}

// ---------------------------------------------------------------- Games

async function loadGames() {
  const el = $("#games-table");
  el.innerHTML = `<div class="empty">Loading…</div>`;
  const rows = await api(`/api/games?sport=${state.sport}`);
  const b2b = (g, side) => (g[`${side}_b2b`] ? ` <span class="flag">B2B</span>` : "");
  table(el, [
    { label: "Kickoff", get: (g) => g.kickoff, html: (g) => when(g.kickoff) },
    { label: "Matchup", get: (g) => `${g.away} @ ${g.home}`, html: (g) => `${g.away}${b2b(g, "away")} @ ${g.home}${b2b(g, "home")}${g.neutral ? ' <span class="flag">neutral</span>' : ""}` },
    { label: "Model line", get: (g) => g.model.margin, html: (g) => spreadText(g.home, g.away, g.model.margin) },
    { label: "Book line", get: (g) => g.book.margin,
      html: (g) => `<span title="${esc(g.book_source || "no line")}">${spreadText(g.home, g.away, g.book.margin)}</span>` },
    { label: "Fair line", get: (g) => g.fair.margin, html: (g) => `<b>${spreadText(g.home, g.away, g.fair.margin)}</b>` },
    { label: "Model total", num: true, get: (g) => g.model.total, html: (g) => num(g.model.total) },
    { label: "Book total", num: true, get: (g) => g.book.total, html: (g) => num(g.book.total) },
    { label: "Fair total", num: true, get: (g) => g.fair.total, html: (g) => `<b>${num(g.fair.total)}</b>` },
    { label: "Home win: model", num: true, get: (g) => g.model.home_win, html: (g) => pct(g.model.home_win) },
    { label: "book", num: true, get: (g) => g.book.home_win, html: (g) => pct(g.book.home_win) },
    { label: "Kalshi", num: true, get: (g) => g.kalshi_home_win, html: (g) => pct(g.kalshi_home_win) },
    ...(state.sport === "nfl" ? [{ label: "Weather", get: (g) => g.weather?.wind_mph, html: (g) => esc(g.weather?.text || "–") }] : []),
    { label: "Notes", get: (g) => g.flags.length, html: (g) => flags(g.flags) },
    { label: "", get: () => 0, html: (g) => (g.odds_event_id && state.odds?.enabled
        ? `<button data-props="${esc(g.odds_event_id)}" title="Pull sportsbook player props for this game">Props (${state.odds.prop_cost[state.sport]} cr)</button>` : "") },
  ], rows, "games");
}

// ---------------------------------------------------------------- Projections

const linesView = () => $("#p-view").value === "lines";

function renderPropLines() {
  const q = $("#p-search").value.trim().toLowerCase();
  const mainOnly = $("#p-main").checked, evOnly = $("#p-ev").checked;
  const rows = state.propLines.filter((r) => (!mainOnly || r.main) && (!evOnly || r.roi > 0)
    && (!q || `${r.name} ${r.game}`.toLowerCase().includes(q)));
  const side = (r) => (r.side === "yes" ? "Over" : "Under");
  table($("#props-table"), [
    { label: "Tip", get: (r) => r.kickoff, html: (r) => when(r.kickoff) },
    { label: "Game", get: (r) => r.game },
    { label: "Player", get: (r) => r.name },
    { label: "Stat", get: (r) => r.type_label },
    { label: "Line", num: true, get: (r) => r.line, html: (r) => num(r.line) },
    { label: "Proj", num: true, get: (r) => r.proj, html: (r) => num(r.proj) },
    { label: "Proj vs line", num: true, get: (r) => r.diff,
      html: (r) => `<span class="${r.diff >= 0 ? "pos" : "neg"}">${r.diff >= 0 ? "Over" : "Under"} ${num(Math.abs(r.diff))}</span>` },
    { label: "Over % model", num: true, get: (r) => r.model_over, html: (r) => pct(r.model_over) },
    { label: "Over % books", num: true, get: (r) => r.book_over, html: (r) => pct(r.book_over) },
    { label: "Over % Kalshi", num: true, get: (r) => r.kalshi_over, html: (r) => pct(r.kalshi_over) },
    { label: "Pick", get: (r) => side(r), html: (r) => `<span class="side-${r.side}">${side(r)}</span> ${cents(r.price)} <span class="muted small">(${pct(r.side_prob, 0)} to hit)</span>` },
    { label: "EV", num: true, get: (r) => r.roi, html: (r) => `<span class="${r.roi > 0 ? "pos" : "neg"}">${pct(r.roi)}</span>` },
    { label: "Notes", get: (r) => r.flags.length, html: (r) => flags(r.flags) },
  ], rows, `lines-${state.sport}`);
}

function renderProps() {
  $("#p-line-filters").hidden = !linesView();
  $("#p-lines-note").hidden = !linesView();
  if (linesView()) return renderPropLines();
  const q = $("#p-search").value.trim().toLowerCase();
  const rows = state.props.filter((p) => !q || `${p.name} ${p.team} ${p.opp}`.toLowerCase().includes(q));
  const boost = (p) => [p.min_boost >= 2 ? `+${p.min_boost.toFixed(0)} min` : "", p.usage_boost >= 0.03 ? `+${pct(p.usage_boost, 0)} rate` : ""]
    .filter(Boolean).map((b) => ` <span class="flag" title="Teammate out">${b}</span>`).join("");
  const common = [
    { label: "Player", get: (p) => p.name,
      html: (p) => `${esc(p.name)}${p.injury ? ` <span class="flag">${esc(p.injury)}</span>` : p.missed_last ? ' <span class="flag">missed last game</span>' : ""}${boost(p)}` },
    { label: "Team", get: (p) => p.team },
    { label: "Opp", get: (p) => p.opp },
    { label: "Kickoff", get: (p) => p.kickoff, html: (p) => when(p.kickoff) },
  ];
  const stat = (label, key, d = 1) => ({ label, num: true, get: (p) => p[key], html: (p) => num(p[key], d) });
  const cols = state.sport === "nba"
    ? [...common, stat("Min", "min"), stat("PTS", "pts"), stat("REB", "reb"), stat("AST", "ast"), stat("3PM", "fg3m"), stat("PRA", "pra")]
    : [...common, { label: "Pos", get: (p) => p.position }, stat("Pass yds", "pass_yds"), stat("Pass TD", "pass_tds", 2),
       stat("Rush yds", "rush_yds"), stat("Rec", "receptions"), stat("Rec yds", "rec_yds"),
       { label: "Anytime TD", num: true, get: (p) => p.tds, html: (p) => pct(1 - Math.exp(-p.tds)) }];
  cols.push({ label: "Sample", num: true, get: (p) => p.games, html: (p) => `${p.games} g` });
  table($("#props-table"), cols, rows, `props-${state.sport}`);
}

async function loadProps() {
  $("#props-table").innerHTML = `<div class="empty">Loading${linesView() ? " Kalshi lines and model" : ""}…</div>`;
  try {
    if (linesView()) state.propLines = await api(`/api/prop_lines?sport=${state.sport}`);
    else state.props = await api(`/api/projections?sport=${state.sport}`);
    renderProps();
  } catch (e) {
    $("#props-table").innerHTML = `<div class="empty">Error: ${esc(e.message)}</div>`;
  }
}

// ---------------------------------------------------------------- Injuries

function renderInjuries() {
  const q = $("#i-search").value.trim().toLowerCase();
  const onlyOut = $("#i-out").checked;
  const rows = state.injuries.filter((r) => (!onlyOut || r.out) && (!q || `${r.team} ${r.player}`.toLowerCase().includes(q)));
  table($("#injuries-table"), [
    { label: "Team", get: (r) => r.team },
    { label: "Player", get: (r) => r.player },
    { label: "Status", get: (r) => r.status, html: (r) => `<span class="${r.out ? "neg" : ""}">${esc(r.status)}</span>` },
    { label: "Updated", get: (r) => r.updated, html: (r) => (r.updated ? when(r.updated) : "–") },
    { label: "Details", get: (r) => r.comment, html: (r) => `<span class="small" style="white-space:normal">${esc(r.comment)}</span>` },
  ], rows, `inj-${state.sport}`);
}

async function loadInjuries() {
  $("#injuries-table").innerHTML = `<div class="empty">Loading…</div>`;
  state.injuries = await api(`/api/injuries?sport=${state.sport}`);
  renderInjuries();
}

// ---------------------------------------------------------------- Bets

async function loadBets() {
  const { summary: s, bets, breakdown } = await api(`/api/bets?mode=${state.betMode}`);
  const card = (k, v, cls = "") => `<div class="card"><div class="k">${k}</div><div class="v ${cls}">${v}</div></div>`;
  $("#bets-summary").innerHTML = [
    card("Record (W-L)", `${s.won}-${s.lost}`),
    card("Open", s.open),
    card("Profit / loss", money(s.pnl), s.pnl >= 0 ? "pos" : "neg"),
    card("ROI", pct(s.roi), (s.roi || 0) >= 0 ? "pos" : "neg"),
    card("Avg CLV", s.avg_clv == null ? "–" : signed(s.avg_clv * 100) + "¢", (s.avg_clv || 0) >= 0 ? "pos" : "neg"),
    card("Beat closing price", pct(s.clv_beat_pct, 0)),
    card("Expected profit", money(s.expected_pnl)),
  ].join("");
  const statCols = [
    { label: "Bets", num: true, get: (g) => g.bets },
    { label: "W-L", get: (g) => g.won - g.lost, html: (g) => `${g.won}-${g.lost}` },
    { label: "P&L", num: true, get: (g) => g.pnl, html: (g) => `<span class="${g.pnl >= 0 ? "pos" : "neg"}">${money(g.pnl)}</span>` },
    { label: "ROI", num: true, get: (g) => g.roi, html: (g) => pct(g.roi) },
    { label: "Avg CLV", num: true, get: (g) => g.avg_clv,
      html: (g) => (g.avg_clv == null ? "–" : `<span class="${g.avg_clv >= 0 ? "pos" : "neg"}">${signed(g.avg_clv * 100)}¢</span>`) },
    { label: "Beat close", num: true, get: (g) => g.clv_beat_pct, html: (g) => pct(g.clv_beat_pct, 0) },
  ];
  $("#bets-breakdown").innerHTML = Object.keys(breakdown)
    .map((k, i) => `<h3>By ${k.toLowerCase()}</h3><div class="table-wrap" id="bd-${i}"></div>`).join("");
  Object.entries(breakdown).forEach(([k, rows], i) =>
    table($(`#bd-${i}`), [{ label: k, get: (g) => g.group }, ...statCols], rows, `bd-${k}`));
  table($("#bets-table"), [
    { label: "Placed", get: (b) => b.created_at, html: (b) => when(b.created_at) },
    { label: "Bet", get: (b) => b.description,
      html: (b) => `${esc(b.description)}${b.mode === "paper" ? ' <span class="flag">paper</span>' : ""}${b.source === "kalshi" ? ' <span class="flag">from Kalshi</span>' : ""}` },
    { label: "Edge", num: true, get: (b) => b.edge, html: (b) => pct(b.edge) },
    { label: "Price", num: true, get: (b) => b.price, html: (b) => cents(b.price) },
    { label: "Qty", num: true, get: (b) => b.contracts },
    { label: "Fair", num: true, get: (b) => b.fair_prob, html: (b) => pct(b.fair_prob) },
    { label: "Close", num: true, get: (b) => b.close_price, html: (b) => cents(b.close_price) },
    { label: "CLV", num: true, get: (b) => (b.close_price == null ? null : b.close_price - b.price),
      html: (b) => (b.close_price == null ? "–" : `<span class="${b.close_price >= b.price ? "pos" : "neg"}">${signed((b.close_price - b.price) * 100, 0)}¢</span>`) },
    { label: "Status", get: (b) => b.status },
    { label: "P&L", num: true, get: (b) => b.pnl, html: (b) => (b.pnl == null ? "–" : `<span class="${b.pnl >= 0 ? "pos" : "neg"}">${money(b.pnl)}</span>`) },
    { label: "", get: () => 0, html: (b) => `<select data-settle="${b.id}"><option value="">set…</option><option>won</option><option>lost</option><option>void</option><option>open</option></select> <button data-del="${b.id}">✕</button>` },
  ], bets, "bets");
  const acct = await api("/api/kalshi/account");
  $("#kalshi-import").style.display = acct.configured ? "" : "none";
  $("#kalshi-balance").textContent = !acct.configured ? "Kalshi account not linked (see README)"
    : acct.error ? `Kalshi: ${acct.error}` : `Kalshi cash ${money(acct.cash)} · positions ${money(acct.portfolio_value)}`;
}

// ---------------------------------------------------------------- Model

async function loadModel() {
  const el = $("#model-backtest");
  el.innerHTML = `<div class="empty">Running walk-forward backtest…</div>`;
  const [bt, rt] = await Promise.all([api(`/api/backtest?sport=${state.sport}`), api(`/api/ratings?sport=${state.sport}`)]);
  if (bt.error) {
    el.innerHTML = `<div class="empty">${esc(bt.error)}</div>`;
  } else {
    const card = (k, v) => `<div class="card"><div class="k">${k}</div><div class="v">${v}</div></div>`;
    const b = bt.book;
    let html = `<h3>Walk-forward backtest (${bt.games} games, ${bt.from} → ${bt.to})</h3>
      <p class="muted small">Each week the model is refit using only earlier games, then scored on that week. Lower error is better.</p>
      <div class="cards">
        ${card("Model margin error (avg pts)", num(bt.model.margin_mae, 2))}
        ${b ? card("Closing line margin error", num(b.margin_mae, 2)) : ""}
        ${card("Model total error", num(bt.model.total_mae, 2))}
        ${b ? card("Closing total error", num(b.total_mae, 2)) : ""}
        ${card("Model win Brier", num(bt.model.win_brier, 4))}
        ${b && b.win_brier ? card("Market win Brier", num(b.win_brier, 4)) : ""}
        ${card("Picks winner", pct(bt.model.straight_up_acc))}
      </div>`;
    if (bt.ats) {
      html += `<div class="cards">
        ${card(`ATS when model differs ≥${bt.ats.threshold} pts`, `${bt.ats.wins}-${bt.ats.losses} (${pct(bt.ats.win_pct)})`)}
        ${card(`Totals when model differs ≥${bt.totals.threshold}`, `${bt.totals.wins}-${bt.totals.losses} (${pct(bt.totals.win_pct)})`)}
        ${card("Break-even at -110", "52.4%")}
      </div>
      <p class="muted small">If the model's raw error is higher than the closing line's, it shouldn't be bet on its own. That's why
      Fair blends it with the market (Settings → model weight).</p>`;
    } else {
      html += `<p class="muted small">No historical sportsbook lines are stored for this sport yet. Lines are saved on each refresh
      from now on, so this comparison will fill in over the season.</p>`;
    }
    html += `<h3>Calibration</h3><div class="table-wrap"><table><thead><tr><th class="num">Predicted home win</th><th class="num">Actual</th><th class="num">Games</th></tr></thead><tbody>
      ${bt.calibration.map((c) => `<tr><td class="num">${pct(c.pred)}</td><td class="num">${pct(c.actual)}</td><td class="num">${c.n}</td></tr>`).join("")}
      </tbody></table></div>`;
    el.innerHTML = html;
  }
  loadPropsBacktest();
  if (rt.teams) {
    table($("#model-ratings"), [
      { label: "Team", get: (t) => t.team },
      { label: "Net", num: true, get: (t) => t.net, html: (t) => signed(t.net) },
      { label: "Offense", num: true, get: (t) => t.off, html: (t) => signed(t.off) },
      { label: "Defense", num: true, get: (t) => t.def, html: (t) => signed(t.def) },
    ], rt.teams, "ratings");
  }
}

const STAT_NAMES = { pts: "Points", reb: "Rebounds", ast: "Assists", fg3m: "Threes", pra: "Pts+Reb+Ast" };

async function loadPropsBacktest() {
  const el = $("#model-props-bt");
  if (state.sport !== "nba") {
    el.innerHTML = "";
    return;
  }
  const { running, result: r } = await api("/api/backtest/props");
  const button = `<button id="props-bt-run" ${running ? "disabled" : ""}>${running ? "Running… (a few minutes)" : r ? "Re-run" : "Run props backtest"}</button>`;
  let html = `<h3>Player props backtest</h3>
    <p class="muted small">Every past game day is replayed with the live projection code, using only earlier box scores, and each
    projection is scored on Kalshi-style ladders (10+/15+/20+ points, …). Baseline = the player's hit rate over their last 20 games.
    Players who didn't play count as ruled out, so availability is known perfectly here. ${button}</p>`;
  if (r && !r.error) {
    const card = (k, v) => `<div class="card"><div class="k">${k}</div><div class="v">${v}</div></div>`;
    const o = r.overall, m = r.main_lines;
    html += `<p class="muted small">${r.games} games, ${r.player_games.toLocaleString()} player-games, ${r.predictions.toLocaleString()} predictions (${r.from} → ${r.to}). Run ${when(r.run_at)}.</p>
      <div class="cards">
        ${card("Brier: model / baseline", `${num(o.brier_model, 4)} / ${num(o.brier_base, 4)}`)}
        ${card("Better than baseline by", pct(o.skill_vs_base))}
        ${card("Main lines (35–65%): model / baseline", `${num(m.brier_model, 4)} / ${num(m.brier_base, 4)}`)}
        ${card("Minutes error (avg)", num(r.minutes.mae, 1))}
        ${card("Minutes bias (actual − proj)", signed(r.minutes.bias))}
      </div>
      <h3>By stat</h3><div class="table-wrap" id="props-bt-stats"></div>
      <h3>Calibration (all rungs)</h3><div class="table-wrap"><table><thead><tr><th class="num">Model said</th><th class="num">Actually hit</th><th class="num">Predictions</th></tr></thead><tbody>
      ${r.calibration.map((c) => `<tr><td class="num">${pct(c.pred)}</td><td class="num">${pct(c.actual)}</td><td class="num">${c.n.toLocaleString()}</td></tr>`).join("")}
      </tbody></table></div>`;
  } else if (r && r.error) {
    html += `<div class="empty">${esc(r.error)}</div>`;
  }
  el.innerHTML = html;
  if (r && !r.error) {
    table($("#props-bt-stats"), [
      { label: "Stat", get: (s) => STAT_NAMES[s.stat] || s.stat },
      { label: "Player-games", num: true, get: (s) => s.player_games, html: (s) => s.player_games.toLocaleString() },
      { label: "Avg proj", num: true, get: (s) => s.proj, html: (s) => num(s.proj) },
      { label: "Avg actual", num: true, get: (s) => s.actual, html: (s) => num(s.actual) },
      { label: "Bias (actual − proj)", num: true, get: (s) => s.bias, html: (s) => signed(s.bias, 2) },
      { label: "Avg error", num: true, get: (s) => s.mae, html: (s) => num(s.mae, 2) },
      { label: "Brier model", num: true, get: (s) => s.brier_model, html: (s) => num(s.brier_model, 4) },
      { label: "Brier baseline", num: true, get: (s) => s.brier_base, html: (s) => num(s.brier_base, 4) },
      { label: "Better by", num: true, get: (s) => s.skill_vs_base, html: (s) => pct(s.skill_vs_base) },
    ], r.by_stat, "props-bt-stats");
  }
  $("#props-bt-run").onclick = async () => {
    await api("/api/backtest/props", { method: "POST" });
    loadPropsBacktest();
  };
  if (running) setTimeout(() => state.tab === "model" && state.sport === "nba" && loadPropsBacktest(), 15000);
}

// ---------------------------------------------------------------- Settings

const SETTINGS_HELP = {
  bankroll: ["Bankroll ($)", "Money set aside for betting. Stakes are sized from this."],
  kelly_fraction: ["Kelly fraction", "0.25 = quarter Kelly. Full Kelly assumes a perfect model and swings wildly."],
  max_bet_pct: ["Max bet (% of bankroll, as decimal)", "Hard cap per position, e.g. 0.03 = 3%."],
  min_edge: ["Min edge (decimal)", "Reference value for what counts as an edge. The Edges tab filter is what applies."],
  model_weight: ["Model weight (0–1)", "How much Fair trusts our model vs the market. Raise it only once your CLV shows the model adds value."],
  kalshi_fee_coef: ["Kalshi fee coefficient", "Taker fee = coef × price × (1 − price) per contract. Check Kalshi's fee schedule."],
  auto_paper: ["Auto paper trading (0 = off, 1 = on)", "When on, every 15 minutes the app paper-bets each qualifying edge (one per ladder) while it's running."],
  paper_shortlist: ["Paper-track the Shortlist (0 = off, 1 = on)", "When on, every 15 minutes each new Shortlist pick is recorded once as a paper bet (Strategy: Shortlist), as long as sportsbook lines are under 12 hours old."],
  paper_min_edge: ["Paper min edge (decimal)", "Minimum EV for the paper trader, e.g. 0.05 = 5%. Long shots, wide books and far-from-market flags are always skipped."],
};

async function loadSettings() {
  const s = await api("/api/settings");
  $("#settings-form").innerHTML = Object.entries(SETTINGS_HELP)
    .map(([k, [label, help]]) => `<label>${label}<input name="${k}" type="number" step="any" value="${s[k]}"><span class="help">${help}</span></label>`)
    .join("") + `<div><button class="primary" type="submit">Save</button> <span id="settings-saved" class="muted"></span></div>`;
}

$("#settings-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const values = Object.fromEntries(new FormData(e.target).entries());
  await api("/api/settings", { method: "POST", body: JSON.stringify(values) });
  $("#settings-saved").textContent = "Saved.";
});

// ---------------------------------------------------------------- Wiring

const loaders = { edges: () => loadEdges(), games: loadGames, props: loadProps, injuries: loadInjuries, bets: loadBets, model: loadModel, settings: loadSettings };

function show(tab) {
  state.tab = tab;
  document.querySelectorAll("nav button").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.id === `tab-${tab}`));
  loaders[tab]().catch((e) => console.error(e));
}

document.querySelectorAll("nav button").forEach((b) => (b.onclick = () => show(b.dataset.tab)));
document.querySelectorAll(".sport-toggle button").forEach((b) => {
  b.onclick = () => {
    state.sport = b.dataset.sport;
    pollStatus();
    document.querySelectorAll(".sport-toggle button").forEach((x) => x.classList.toggle("active", x === b));
    show(state.tab);
  };
});
["#f-group", "#f-minev", "#f-spread", "#f-tails", "#f-flagged", "#f-search"].forEach((s) => $(s).addEventListener("input", renderEdges));
$("#edges-reload").onclick = () => loadEdges(true);
$("#f-view").addEventListener("change", () => loadEdges());
$("#p-search").addEventListener("input", renderProps);
$("#p-view").addEventListener("change", loadProps);
$("#p-main").addEventListener("change", renderProps);
$("#p-ev").addEventListener("change", renderProps);
$("#i-search").addEventListener("input", renderInjuries);
$("#i-out").addEventListener("input", renderInjuries);

$("#games-table").addEventListener("click", async (e) => {
  const b = e.target.closest("[data-props]");
  if (!b) return;
  const cost = state.odds.prop_cost[state.sport];
  if (!confirm(`Pull sportsbook player props for this game? Uses ${cost} Odds API credits.`)) return;
  b.disabled = true;
  try {
    await api(`/api/odds/props?sport=${state.sport}&event_id=${encodeURIComponent(b.dataset.props)}`, { method: "POST" });
    b.textContent = "Props pulled";
    pollStatus();
  } catch (err) {
    alert(err.message);
    b.disabled = false;
  }
});

$("#lines-btn").onclick = async () => {
  if (!confirm(`Pull ${state.sport.toUpperCase()} sportsbook lines? Uses about 3 Odds API credits.`)) return;
  $("#lines-btn").disabled = true;
  try {
    await api(`/api/odds/lines?sport=${state.sport}`, { method: "POST" });
    await pollStatus();
    show(state.tab);
  } catch (err) {
    alert(err.message);
  } finally {
    $("#lines-btn").disabled = false;
  }
};

$("#edges-table").addEventListener("click", async (e) => {
  const pb = e.target.closest("[data-paper]");
  if (pb) {
    const r = $("#edges-table")._rows[+pb.closest("tr").dataset.ri];
    await api("/api/bets", {
      method: "POST",
      body: JSON.stringify({
        sport: r.sport, ticker: r.ticker, description: `${r.side.toUpperCase()} ${r.title} (${r.game})`, side: r.side,
        price: r.price, contracts: Math.max(1, r.stake.contracts), fair_prob: r.side_prob, game_start: r.kickoff,
        mode: "paper", market_type: r.type, edge: r.roi,
      }),
    });
    pb.textContent = "Saved";
    pb.disabled = true;
    return;
  }
  const t = e.target.closest("[data-log]");
  if (!t) return;
  const r = $("#edges-table")._rows[+t.closest("tr").dataset.ri];
  state.pending = r;
  const f = $("#bet-form");
  $("#bet-desc").textContent = `${r.game} · ${r.title}`;
  f.side.value = r.side;
  f.price.value = r.price.toFixed(2);
  f.contracts.value = Math.max(1, r.stake.contracts);
  f.notes.value = "";
  $("#bet-dialog").showModal();
});

$("#bet-dialog").addEventListener("close", async () => {
  if ($("#bet-dialog").returnValue !== "save" || !state.pending) return;
  const r = state.pending, f = $("#bet-form");
  const side = f.side.value;
  await api("/api/bets", {
    method: "POST",
    body: JSON.stringify({
      sport: r.sport, ticker: r.ticker, description: `${side.toUpperCase()} ${r.title} (${r.game})`, side,
      price: parseFloat(f.price.value), contracts: parseInt(f.contracts.value, 10),
      fair_prob: side === r.side ? r.side_prob : 1 - r.side_prob, game_start: r.kickoff, notes: f.notes.value || null,
      market_type: r.type, edge: side === r.side ? r.roi : null,
    }),
  });
  state.pending = null;
});

$("#bets-table").addEventListener("change", async (e) => {
  const sel = e.target.closest("[data-settle]");
  if (!sel || !sel.value) return;
  await api(`/api/bets/${sel.dataset.settle}/settle?status=${sel.value}`, { method: "POST" });
  loadBets();
});
$("#bets-table").addEventListener("click", async (e) => {
  const del = e.target.closest("[data-del]");
  if (!del || !confirm("Delete this bet?")) return;
  await api(`/api/bets/${del.dataset.del}`, { method: "DELETE" });
  loadBets();
});
document.querySelectorAll("#bet-mode button").forEach((b) => {
  b.onclick = () => {
    state.betMode = b.dataset.mode;
    document.querySelectorAll("#bet-mode button").forEach((x) => x.classList.toggle("active", x === b));
    loadBets();
  };
});
$("#kalshi-import").onclick = async () => {
  $("#kalshi-import").disabled = true;
  try {
    const r = await api("/api/kalshi/import", { method: "POST" });
    const br = r.bankroll != null ? `
Bankroll set to $${r.bankroll.toFixed(2)} from your Kalshi balance.` : "";
    alert(`Imported ${r.created} new and updated ${r.updated} existing positions from Kalshi.${br}`);
    await loadBets();
  } catch (err) {
    alert(err.message);
  } finally {
    $("#kalshi-import").disabled = false;
  }
};
$("#paper-run").onclick = async () => {
  $("#paper-run").disabled = true;
  try {
    const r = await api("/api/paper/run", { method: "POST" });
    alert(`Recorded ${r.created} new paper bets, plus ${r.shortlist_created} Shortlist picks.`);
    state.betMode = "paper";
    document.querySelectorAll("#bet-mode button").forEach((x) => x.classList.toggle("active", x.dataset.mode === "paper"));
    await loadBets();
  } finally {
    $("#paper-run").disabled = false;
  }
};
$("#bets-sync").onclick = async () => {
  $("#bets-sync").disabled = true;
  try { await api("/api/bets/sync", { method: "POST" }); await loadBets(); } finally { $("#bets-sync").disabled = false; }
};

let polling = null;
async function pollStatus() {
  const s = await api("/api/status");
  state.odds = s.odds;
  const qt = s.odds.quota, at = s.odds.lines_at[state.sport];
  $("#lines-btn").style.display = s.odds.enabled ? "" : "none";
  $("#odds-quota").textContent = !s.odds.enabled ? "No Odds API key (.env)"
    : `${qt ? qt.remaining + " credits left" : ""}${at ? ` · lines ${new Date(at).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}` : " · no lines yet"}`;
  const r = s.refresh;
  const last = r.log[r.log.length - 1] || "";
  $("#refresh-status").textContent = r.running ? last : r.last[state.sport] ? `Updated ${new Date(r.last[state.sport]).toLocaleTimeString()}` : "";
  $("#refresh-btn").disabled = r.running;
  if (r.running && !polling) polling = setInterval(pollStatus, 2000);
  if (!r.running && polling) {
    clearInterval(polling);
    polling = null;
    show(state.tab);
  }
}
$("#refresh-btn").onclick = async () => {
  await api("/api/refresh?sport=all", { method: "POST" });
  pollStatus();
};

pollStatus();
show("edges");
