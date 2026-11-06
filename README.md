# Market Monitor — live 5-minute FX setups, with their evidence attached

An entry / stop / target alert system for an intraday mean-reversion strategy on 5-minute
bars, running on GitHub Actions every 5 minutes and pushing to a phone through ntfy.

**Start with this sentence: the 77% win rate in the alerts is a correctly measured number
and an unproven edge.** Every alert carries its own counter-evidence, because a message that
says "CHFJPY LONG, 77% win rate" without also saying "the same trade with no filters wins
69.7%" is a message that would be doing something dishonest at the exact moment it is hardest
to argue with.

---

## What was actually found

26 pairs, 81 days of Yahoo 5-minute bars (437,498 bars), 3 entry settings x 13 exit configs =
1,014 looks. Reproduce with `python -m lab.intraday`; full output in
[results/log_intraday.txt](results/log_intraday.txt).

| pair | win rate | trades | expectancy | PF | same trade, filters removed | z vs own control | win rate at 2x spread |
|---|---|---|---|---|---|---|---|
| **CHFJPY** | **77.0%** | 61 | +0.031R | 1.54 | 69.7% | +1.16 | 63.9% |
| EURAUD | 74.4% | 39 | +0.032R | 1.71 | 66.0% | +1.03 | 69.2% |
| CADJPY | 73.6% | 72 | +0.008R | 1.11 | 66.4% | +1.19 | 56.9% |
| GBPJPY | 72.9% | 85 | -0.013R | 0.89 | 71.2% | +0.31 | 64.7% |
| USDSGD | 72.7% | 55 | -0.029R | 0.66 | 65.2% | +1.08 | 29.1% |
| USDJPY | 72.2% | 72 | -0.006R | 0.93 | 68.7% | +0.59 | 63.9% |
| GBPCAD | 71.4% | 70 | -0.016R | 0.85 | 64.1% | +1.17 | 65.7% |
| GBPUSD | 70.4% | 159 | +0.011R | 1.15 | 62.4% | **+1.93** | 52.2% |

The four with positive expectancy are CHFJPY, EURAUD, CADJPY and GBPUSD. Only CHFJPY and
EURAUD were positive in both out-of-sample halves *and* above the search bar on nothing at
all, which is the correct way to read that: neither cleared it.

### Three things that are true about that 77%

1. **It needs no skill to reproduce most of it.** The winning config has no profit target.
   The exit is "price is back at the mean it left", and the stop is 12 x ATR(5m) away — about
   95 pips on CHFJPY. A position like that wins whenever price does not first travel 95 pips
   against it. Strip out the trend filter, the RSI gate and the session window and the same
   geometry still wins 69.7% of 357 trades. The filters are worth roughly 7 points, which at
   n=61 is z = 1.16. With 1,014 looks, the bar for "this is not the search finding a fluke"
   is about z = 3.7.
2. **Ask for a payoff and 70% disappears.** At a 0.5R target, where the barrier null already
   pays a random entry 66.7%, **zero of 265** results cleared 70% — where 74 should have come
   from worthless signals. At 1R the best was 60.7%. The high hit rate and a real target are
   the same dial turned in opposite directions.
3. **Costs are the whole story.** Median 5m bar range is 0.7 to 10.5 pips while retail spread
   is 1.1 to 3.0, so the strategy was built to enter on the 5m flush and place barriers an
   hour wide, keeping cost near 2 to 4% of R. Even so, doubling the assumed spread puts every
   single one of these pairs under 70% and turns the two best expectancies flat or negative.

So: use it to see what the system is doing in real time. Do not size up on the strength of
the number in the title.

## The strategy

`lab/strategies.py:intraday_mr` — on a closed 5m bar, when price has flushed `dev_k` ATRs
below its own `dev_n`-bar mean, RSI(2) is under 20, and price is still above the completed
hour's 24-span EMA, go long. Exit long when close is back at that mean. Stop 12 x ATR(14) of
the 5m frame. Session 07:00-20:00 UTC. Time stop 72 bars.

The trend filter reads the hour **after** it closes (`resample('1h').last().shift(1)`), so a
decision at 14:35 never sees the 14:00 bar's final value. The monitor signals only on closed
bars and executes at the next open, exactly like `lab/backtest.py`.

## What an alert looks like

```
CHFJPY LONG 191.42200                                     p4

LONG  entry 191.42200
STOP    189.99711  (142 pips = 12x ATR(14))
TARGET  191.60927  (19 pips away: the 96-bar mean, which keeps moving)
SIZE    0.07 lot on a $10,000 account = $50 at risk (0.5%)
        spread 2.6 pips = 3.6% of R
TIME    close out after 72 bars (~6 h) if neither barrier is hit

Evidence: 77.0% of 61 trades over 81 days, +0.031R per trade, profit factor 1.54.
NOT PROVEN: the same trade with the trend/RSI/session filters removed wins 69.7% of 357 (z=+1.16, needs +3.7).
It only reaches 70% because this config sets no profit target - the exit is the mean.
At 2x the assumed spread it falls under 70% and stops paying.
```

Notes on reading it:

- **TARGET is not a level you can leave a limit order at.** The exit is the mean, and the
  mean moves every bar. The number printed is where it stood at entry. The monitor tells you
  when it is reached.
- **The stop is wide on purpose.** 95-140 pips is what makes the cost a fraction of R, and
  what makes this a wide-stop mean reversion that happens to be timed on 5m bars rather than
  the scalp the question implied.
- **Closures arrive as separate alerts** at priority 3 (`STOPPED`, `TARGET`, `TIME STOP`) with
  the R multiple.

## Receiving it on your phone

1. Install **ntfy** from your app store (Android and iOS both have an official client; the
   project page is https://ntfy.sh), or just open https://ntfy.sh in a browser and subscribe
   there.
2. Subscribe to the topic that is stored as the `NTFY_TOPIC` secret in this repo's
   Settings -> Secrets and variables -> Actions. Treat that string as a password: anyone who
   knows it can both read your alerts and send fake ones to your phone. If it leaks, set a
   new one and restart nothing — the next run picks it up.
3. Priority 4 and 5 alerts can bypass silent mode once you enable it per topic in the app.
   Priority 5 is reserved for "the monitor is blind", so it should.

## Running it

```bash
pip install -r requirements.txt
python -m lab.monitor --dry --limit 2      # print instead of push, no credentials needed
NTFY_TOPIC=<topic> python -m lab.monitor   # one pass, then exit
NTFY_TOPIC=<topic> python -m lab.monitor --daemon --interval 300   # keep looping
```

`--account`, `--risk` and `--limit` control the ticket; the environment variables do the same
for Actions. The watchlist itself is not configurable, and that is deliberate:
`lab/monitor.py:evidence()` reads `results/intraday_5m.csv` and its filter-free control twin,
takes each pair's own best config, and keeps those at or above 70%. Re-run the sweep and the
monitor follows the new leaderboard by itself. Edit the CSVs and the alerts change, which is
why they are checked in.

## Deployment facts worth knowing

- **Cadence.** `.github/workflows/live.yml` fires at `*/5 * * * *`. GitHub documents
  scheduled runs as best-effort: under load a run can start minutes late, and on a 5-minute
  system that is a whole bar. The alert prints the signal bar's timestamp, so check it before
  acting on a stale entry.
- **Why this repo is public.** A private repo gets 2,000 free Actions minutes a month and a
  5-minute scan of 3 pairs burns 8,800 to 13,000 — the job would stop mid-month and go quiet.
  Public repos are not capped, so the schedule is honest. The cost is that the strategy, the
  watchlist and the position book are readable by anyone.
- **The position book is committed back** to `state/positions.json`, because Actions has no
  memory between runs. It commits only when the book actually changed rather than every pass
  on purpose: 288 "no change" commits a day would make the history unreadable. The commit
  carries `[skip ci]` and the `push` trigger ignores `state/**`, so it cannot loop.
- **A schedule can be switched off and no alert will tell you.** GitHub disables scheduled
  workflows on a repo with no activity for 60 days and emails the account owner. Because a
  quiet position book means no commits, that timer can genuinely run down here. This is the
  one silence this design cannot break: `MONITOR BLIND` fires when a run executes and cannot
  reach Yahoo, whereas a disabled schedule never executes at all. If alerts go quiet for a
  whole weekday session, check Settings -> Actions before concluding the market had nothing
  to say.
- **Yahoo blocks datacenter IPs** often enough that a single retry is not a hedge, so the
  fetcher does four with backoff and jitter. There is no key-free second source for 5m FX
  bars to fall back on — stooq sits behind a JavaScript proof-of-work wall and everything
  else needs an account. When the feed still fails, the monitor sends a priority-5
  **MONITOR BLIND** alert naming the pairs it could not evaluate, rather than saying nothing.
- **Stale frames never signal.** Yahoo keeps serving the last Friday bar through the weekend.
  If the newest closed bar is more than 45 minutes old the pass reports the gap and declines
  to open, so no ticket is ever written on a Saturday.

## Repository

```
lab/
  monitor.py      live loop: fetch, signal, size, alert, resolve, persist
  intraday.py     the 5m sweep, the null/control/OOS tests and the tradability table
  strategies.py   the 9 strategy families, including intraday_mr
  backtest.py     event-driven engine: barriers resolved inside the bar, gaps at the open
  data.py         Yahoo download + cache, the 26-pair universe, pip/spread/slippage table
  live.py         daily scanner, lot sizing, closed-bar rules
  search.py       walk-forward grid and exit configs
  tradable.py     what a real account can actually place, and what that does to expectancy
  dirt.py         which bars are corrupt, and why none are auto-corrected
results/          every log and sweep, including the control run
state/            the position book carried between runs
```
