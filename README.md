# National Data Ball

A local web app that prices NFL and NBA markets on Kalshi: moneylines, spreads, totals, team totals
and player props. It compares a fair probability against live Kalshi bid/ask after fees,
sizes positions with fractional Kelly, and tracks your bets with closing line value (CLV).

## Run it

```powershell
.\run.ps1
```

This opens http://127.0.0.1:8000. The first launch downloads about two seasons of data
(roughly 2 to 4 minutes). After that, **Refresh data** only pulls what is new.

Tests: `.venv\Scripts\python.exe -m pytest`

## Data sources (all free, no keys)

| What | Source |
| --- | --- |
| NFL schedule, results, closing spread/total/moneyline, starting QBs | [nflverse `games.csv`](https://github.com/nflverse/nfldata) |
| NFL weekly player stats | nflverse `stats_player_week_{season}.csv` |
| NBA schedule, scores, DraftKings lines, box scores | ESPN public site API |
| Kalshi markets and prices, settlement results | Kalshi public trade API v2 (read-only, no login) |
| Injury designations (both leagues) | ESPN public injury feed, refreshed every 10 min |
| NFL kickoff weather (temp, wind, precipitation) | [Open-Meteo](https://open-meteo.com) forecast, no key needed |
| Sportsbook consensus lines and player props | [The Odds API](https://the-odds-api.com), needs a key |

Everything is cached in `data/` (SQLite and CSV files).

### The Odds API key and credits

Put your key in `.env` (git ignores this file):

```
ODDS_API_KEY=your_key_here
```

The free tier gives 500 credits a month. Nothing is pulled automatically. You decide when to spend:

| Action | Cost |
| --- | --- |
| **Update lines** (spreads, totals, moneylines for one sport) | 3 credits |
| **Props** button on a game (NBA / NFL) | 5 / 6 credits |

Lines are averaged across 8 to 10 books after removing the vig, with Pinnacle weighted 3x.
Every pull is stored, which builds a line-movement history.

### Injuries and weather in the model

- Players listed Out, Doubtful, IR or suspended are dropped. In the NBA, when a rotation player
  sits, teammates gain minutes at the rate measured from past absences (bench players gain the
  most, players over 31 mpg barely any; flagged "+X min"), and their per-minute points and
  assists rise with the share of team usage that is out (about +16% when a star with ~45% of
  rotation usage sits; flagged "+X% scoring rate").
- Questionable and day-to-day players stay in, with their status shown on every edge.
- NFL outdoor games: each mph of wind above 10 lowers the model total by 0.3 points and
  passing projections by about 1.2%. Domes are ignored. Retractable roofs are assumed to close
  in bad weather.

## How the numbers are made

**Team model** ([app/models/ratings.py](app/models/ratings.py)). A ridge regression of points scored on
offense/defense ratings, home field and NBA back-to-backs. Recent games are weighted more, and the
offseason is compressed so last season informs early-season ratings without dominating them.
Outputs are a projected margin and total, with a normal error distribution.

**Fair price** ([app/models/games.py](app/models/games.py)). This is the model blended with the sportsbook line
(`model_weight` in Settings, default 0.35). Without a sportsbook line, the blend uses Kalshi's own mid
price instead. The market is very hard to beat on its own: in the walk-forward backtest
(Model tab), the raw NFL model has a higher error than the closing line. So the model is treated as
one input, not the truth.

**Player props** ([app/models/props.py](app/models/props.py)).
- NBA: projected minutes × layered per-minute rates (league average for similar-minute players,
  then last season, then this season), scaled by the projected game environment.
- NFL: recency-weighted per-game averages × an opponent-allowed factor.
- Probabilities come from negative binomial distributions with each player's own variance.

**Edges** ([app/edges.py](app/edges.py)). For each open Kalshi market whose game hasn't started, the app
computes the EV of buying YES at the ask or NO at (1 − bid), after the taker fee. Notes flag known
blind spots: QB changes, a player missing the last game, and big gaps between model and market,
which are usually injuries or news the model doesn't know about.

## Linking your Kalshi account (read-only)

1. On Kalshi, go to Account & security → API keys → Create key. Save the private key file
   (`.pem`) it gives you. Kalshi shows it only once.
2. Put the file in this folder (git ignores `*.pem`) and add the following to `.env`:
   ```
   KALSHI_API_KEY_ID=your-key-id
   KALSHI_PRIVATE_KEY_PATH=kalshi_private_key.pem
   ```
3. Restart the app. The Bets tab shows your balance, and **Import from my Kalshi account**
   pulls every fill. Each market + side becomes one bet with your real average price and fees,
   and sells before settlement are accounted for. If you already logged that bet by hand, it is
   updated rather than duplicated, and keeps the fair probability recorded when you logged it.

The app only calls Kalshi's GET `/portfolio` endpoints. There is no order code in it.

## Paper trading

Paper bets are graded exactly like real ones (results, P&L, CLV) but cost nothing.

- **Paper** button on any edge: record that one bet.
- **Paper-bet current edges** (Bets tab): record every qualifying edge right now.
- **Auto paper trading** (Settings → 1): the app does that every 15 minutes while it's running.

A paper bet qualifies at the paper min edge (default 5%) when it is priced between 10¢ and 90¢,
the Kalshi spread is at most 6¢, and it has no far-from-market flag. Only the best rung of each
ladder is taken (one per game line, one per player + stat), so correlated rungs don't inflate
the sample. The Bets tab breaks results down by market, sport, edge size and price. Use that to
decide what to bet with real money and which market types to ignore.

## Using it responsibly

- **Your scoreboard is CLV, not win rate.** The Bets tab records your price against Kalshi's last
  pre-game price. If you consistently beat the close, you have an edge. If you don't, more volume
  won't help. A few hundred bets are needed before win/loss tells you much.
- Most "edges" in the far tails (prices under 10¢ or over 90¢) come from the model's distribution
  assumptions, not real mispricing. They are hidden by default.
- Check injury reports before acting on any prop. The model only sees box scores.
- Bet amounts you are comfortable losing. Quarter Kelly with a 3% cap is the default for a reason.

## Roadmap (highest value first)

1. **Faster injury news.** Parse the official NBA injury report PDF, which updates through the
   day, and NFL inactives about 90 minutes before kickoff. Redistribute NFL targets.
2. **Alternate prop lines.** The Odds API's `*_alternate` markets would price Kalshi's ladders
   directly instead of only near the main line. They cost extra credits.
3. **Kalshi price snapshots.** Store Kalshi prices over time to measure CLV for every market the
   app flagged, not just the ones you bet.
4. **Kalshi authenticated API.** Import fills and positions automatically. Order placement should
   stay manual until CLV proves an edge.
5. **Better NFL inputs.** EPA/success rate from nflverse play-by-play, plus rest.
