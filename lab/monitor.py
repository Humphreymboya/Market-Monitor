"""
Live 5-minute monitor: watches the pairs whose measured win rate reached the target, and
pushes an entry / stop / target ticket to a phone through ntfy.

Read this before you trust it, because the honest framing is the point of the module.

The numbers that chose this watchlist come from lab/intraday.py over 81 days of Yahoo 5m
bars. On that frame CHFJPY resolves 77.0% of its trades. That figure is measured correctly
and it is NOT an established edge, for three reasons which are printed into every alert:

  1. the same trade shape with the trend, RSI and session filters removed wins 69.7% on the
     same pair, so the filters - the only part that could be skill - are worth about 7
     points, which is z ~ 1.2 against its own control where 1,014 looks demand z ~ 3.7;
  2. it only reaches 70% because the config has no profit target. The exit is "price is back
     at the mean it left", which wins whenever price does not first travel the whole stop -
     a 12 x ATR(5m) excursion of about 95 pips. Ask for a 0.5R target instead and zero of
     265 pair/config results cleared 70% where 74 should have from worthless signals;
  3. at double the assumed spread CHFJPY falls to 63.9% with negative expectancy. The edge
     is well under a pip per trade, so it does not survive an uncompetitive fill.

So this is a monitor, not a signal service. It reports what the strategy would be doing
right now, at the frequency it historically did it, with its own evidence attached.

Two deliberate behaviours.

Silence is never allowed to mean "nothing to report". A failed feed is announced at ntfy
urgent priority, because a monitor that quietly stops monitoring is worse than no monitor.
There is no key-free second source for 5m FX bars - stooq sits behind a JavaScript
proof-of-work wall and every real alternative needs an account and a key - so the only
available hedge is to shout when the one feed we have goes down.

Staleness is checked before anything is signalled. Yahoo keeps serving the last Friday bar
all weekend, and a monitor that cannot tell the difference will happily "find" the same
setup on Saturday morning and again on Sunday. So a pass only produces tickets when the
freshest closed bar is recent enough to be the market actually trading, and the gap that
made it refuse is reported.

Signals are taken on a CLOSED bar and executed at the next open, exactly the way
lab/backtest.py does it. The last bar of any Yahoo intraday frame is still forming and its
"close" is a live price; signalling off it would be lookahead in production.
"""
import argparse
import json
import os
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
import yfinance as yf

from . import data as D
from . import indicators as ta
from . import live as L
from . import strategies as S

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RESULTS = os.path.join(ROOT, "results")
STATE = os.path.join(ROOT, "state")
TF = "5m"
PERIOD = "7d"                 # the longest 5m window Yahoo serves is ~60d; the filters need ~4d
QUALIFY_WR = 70.0            # the target the user set, and the only thing that picks the list
MIN_TRADES = 30              # below this the win rate is not a measurement, it is a coin toss
STALE_AFTER = pd.Timedelta("45min")   # a weekend or a dead feed, either way: do not signal

# live.sizing needs a USD price for the QUOTE currency of every pair it sizes, and the
# monitor may well watch EURAUD and CHFJPY without ever loading AUDUSD or USDJPY. So the
# currencies are listed here and only the ones actually missing get fetched.
CONVERSION = {"JPY": "USDJPY", "CHF": "USDCHF", "CAD": "USDCAD", "SGD": "USDSGD",
              "EUR": "EURUSD", "GBP": "GBPUSD", "AUD": "AUDUSD", "NZD": "NZDUSD"}


def meta(name):
    return D.PAIRS[[s for s, m in D.PAIRS.items() if m["name"] == name][0]]


def usd_has(usable):
    """Quote currencies the frames already in hand can price. USD is always known."""
    return {c for c, sym in CONVERSION.items() if sym in usable} | {"USD"}


# ------------------------------------------------------------------ data
def fetch_live(name, period=PERIOD, tries=4):
    """
    Fresh 5m bars for one pair, no cache accepted.

    data.fetch() returns a cached file up to three days old, which is right for research and
    fatal for a live scanner - it would be deciding on a stale frame. This path always asks
    Yahoo and retries with backoff and jitter, because a datacenter runner gets rate limited
    often enough that one retry is not a hedge.
    """
    last_err = "no attempt"
    for k in range(tries):
        try:
            raw = yf.download(meta(name)["name"] + "=X", interval="5m", period=period,
                              progress=False, auto_adjust=False)
            if raw is None or raw.empty:
                raise RuntimeError("empty response")
            if len(raw) < 150:
                raise RuntimeError(f"only {len(raw)} bars back")
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [c[0] for c in raw.columns]
            df = raw.dropna(subset=["Open", "High", "Low", "Close"])
            df = df[~df.index.duplicated(keep="last")]
            df.index = pd.to_datetime(df.index, utc=True, format="mixed")
            return df.sort_index()[["Open", "High", "Low", "Close"]].astype(float)
        except Exception as e:                                    # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(min(2.0 ** k, 8.0) + float(np.random.uniform(0, 1.5)))
    raise RuntimeError(f"{name}: live fetch failed after {tries} tries - {last_err}")


# ------------------------------------------------------------------ evidence
def two_prop_z(w1, n1, w2, n2):
    p = (n1 * w1 + n2 * w2) / (n1 + n2)
    se = (p * (1 - p) * (1 / n1 + 1 / n2)) ** 0.5
    return (w1 - w2) / se if se > 0 else float("nan")


def evidence(qualify=QUALIFY_WR, min_trades=MIN_TRADES):
    """
    The watchlist, derived from the sweep instead of typed in here.

    Reads results/intraday_5m.csv plus its filter-free control twin, takes each pair's own
    best config, keeps those at or above the target win rate, and carries through the
    numbers an alert should state. If a future sweep moves the leaderboard, this monitor
    follows it without anyone editing a list.
    """
    res = pd.read_csv(os.path.join(RESULTS, "intraday_5m.csv"))
    ctl = pd.read_csv(os.path.join(RESULTS, "intraday_5m_control.csv"))
    ok = res[(res.n >= min_trades) & res.wr.notna()]
    best = ok.loc[ok.groupby("pair")["wr"].idxmax()]
    hit = best[best.wr >= qualify].sort_values("wr", ascending=False)
    out = []
    for _, r in hit.iterrows():
        c = ctl[(ctl.pair == r.pair) & (ctl.kw == r.kw) & (ctl.cfg == r.cfg)
                & (ctl.n >= min_trades)]
        cw = float(c.wr.iloc[0]) if len(c) else float("nan")
        cn = int(c.n.iloc[0]) if len(c) else 0
        out.append(dict(pair=r.pair, kw=json.loads(r.kw), cfg=json.loads(r.cfg),
                        wr=float(r.wr), n=int(r.n), exp=float(r.exp), pf=float(r.pf),
                        ctrl_wr=cw, ctrl_n=cn,
                        z_ctrl=two_prop_z(r.wr / 100.0, int(r.n), cw / 100.0, cn)
                        if cn else float("nan")))
    return out


# ------------------------------------------------------------------ levels
def levels(df, kw, cfg):
    """Signal state and the levels a ticket needs, read off the last CLOSED bar."""
    c = df["Close"]
    n = L.closed_through(df, TF)
    sig = S.REGISTRY["intraday_mr"](df, **kw)
    a = ta.atr(df, 14)
    mean = ta.sma(c, kw["dev_n"])        # the same expression the strategy's exit is defined on
    i = n - 1
    return dict(n_closed=n, bar=df.index[i], close=float(c.iloc[i]),
                atr=float(a.iloc[i]), mean=float(mean.iloc[i]),
                dirn=int(sig["dir"][i]), last_px=float(c.iloc[-1]))


def ticket(name, lv, cfg, usd_table, account, risk_pct):
    """One order ticket, in the fill convention the backtest actually used."""
    m = meta(name)
    d, pip = lv["dirn"], m["pip"]
    sd = cfg["atr_mult"] * lv["atr"]                 # stop distance in price units
    px = lv["last_px"]                               # filled at the next open, i.e. right now
    stop_pips = sd / pip
    return dict(dir="LONG" if d > 0 else "SHORT", entry=px, stop=px - d * sd,
                target=lv["mean"], sd=sd, stop_pips=stop_pips,
                mean_pips=abs(lv["mean"] - px) / pip, bars=cfg["max_bars"],
                # sizing() returns None outright when it cannot price the quote currency;
                # an empty dict lets every caller use .get() instead of crashing on a ticket
                sizing=L.sizing(pip, px, name[3:], usd_table,
                                account * risk_pct / 100.0, stop_pips) or {},
                spread=m["spread"], cost_r=2 * m["spread"] / stop_pips)


# ------------------------------------------------------------------ alerts
# set once by run(); notify reads them so every call site does not re-thread credentials
TOPIC = None
TOKEN = None
DRY = False


def notify(title, body, priority=3, tags=None):
    """POST to ntfy. True only if it was actually delivered (or deliberately printed)."""
    if DRY or not TOPIC:
        print(f"[{'dry' if DRY else 'no topic'}] {title} p{priority} "
              f"| {body.replace(chr(10), ' / ')}")
        return bool(DRY)
    req = urllib.request.Request(f"https://ntfy.sh/{TOPIC}", data=body.encode(), method="POST")
    req.add_header("Title", title)
    req.add_header("Priority", str(priority))
    if tags:
        req.add_header("Tags", ",".join(tags))
    if TOKEN:
        req.add_header("Authorization", f"Bearer {TOKEN}")
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return 200 <= r.status < 300
    except (urllib.error.URLError, OSError) as e:
        print(f"  ! ntfy delivery failed: {e}")
        return False


def body_for(ev, tk, account, risk_pct):
    s = tk["sizing"]
    size = ("size unknown - no USD price for the quote currency on this pass"
            if s.get("lots") is None else
            f"{s['lots']:.2f} lot on a ${account:,.0f} account "
            f"= ${account * risk_pct / 100:,.0f} at risk ({risk_pct}%)")
    lines = [
        f"{tk['dir']}  entry {tk['entry']:.5f}",
        f"STOP    {tk['stop']:.5f}  ({tk['stop_pips']:.0f} pips = "
        f"{ev['cfg']['atr_mult']:.0f}x ATR(14))",
        f"TARGET  {tk['target']:.5f}  ({tk['mean_pips']:.0f} pips away: the "
        f"{ev['kw']['dev_n']}-bar mean, which keeps moving)",
        f"SIZE    {size}",
        f"        spread {tk['spread']:.1f} pips = {100 * tk['cost_r']:.1f}% of R",
    ]
    if ev["cfg"]["max_bars"]:
        lines.append(f"TIME    close out after {ev['cfg']['max_bars']} bars "
                     f"(~{ev['cfg']['max_bars'] * 5 / 60:.0f} h) if neither barrier is hit")
    lines += [
        "",
        f"Evidence: {ev['wr']:.1f}% of {ev['n']} trades over 81 days, "
        f"{ev['exp']:+.3f}R per trade, profit factor {ev['pf']:.2f}.",
        f"NOT PROVEN: the same trade with the trend/RSI/session filters removed wins "
        f"{ev['ctrl_wr']:.1f}% of {ev['ctrl_n']} (z={ev['z_ctrl']:+.2f}, needs +3.7).",
        "It only reaches 70% because this config sets no profit target - the exit is the "
        "mean.",
        "At 2x the assumed spread it falls under 70% and stops paying.",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------ state
def state_path():
    os.makedirs(STATE, exist_ok=True)
    return os.path.join(STATE, "positions.json")


def load_state():
    try:
        return json.load(open(state_path()))
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(st):
    p = state_path()
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, indent=1, sort_keys=True, default=str)
    os.replace(tmp, p)


# ------------------------------------------------------------------ exits
def check_exit(pos, df, kw):
    """
    Did the recorded position resolve? The three ways the backtest closes one, in the order
    it checks them: the stop, then the mean being reclaimed, then the bar count.

    R is always distance / the distance that defined the risk, in price units. Mixing pips
    into that ratio is how a monitor starts reporting -0.003R for what was a full stop-out.
    """
    d = pos["dir"]
    after = df[df.index > pd.Timestamp(pos["since"], tz="UTC")]
    if not len(after):
        return None
    stop_hit = (after["Low"] <= pos["stop"]) if d > 0 else (after["High"] >= pos["stop"])
    if stop_hit.any():
        t = after.index[stop_hit.argmax()]
        side = "low" if d > 0 else "high"
        return ("STOPPED", f"{side} pierced {pos['stop']:.5f} at {t}", -1.0 - pos["cost_r"])
    mean = ta.sma(df["Close"], kw["dev_n"]).reindex(after.index)
    touched = after["Close"] >= mean if d > 0 else after["Close"] <= mean
    if touched.any():
        j = touched.argmax()
        r = ((mean.iloc[j] - pos["entry"]) * d) / pos["sd"] - pos["cost_r"]
        return ("TARGET", f"price back at the mean at {after.index[j]}", r)
    if pos["max_bars"] and len(after) > pos["max_bars"]:
        r = ((after["Close"].iloc[-1] - pos["entry"]) * d) / pos["sd"] - pos["cost_r"]
        return ("TIME STOP", f"{pos['max_bars']} bars elapsed", r)
    return None


# ------------------------------------------------------------------ run
def one_pass(watch, data, st, usd_table, account, risk_pct, quiet=False):
    """One sweep of the watchlist. Returns (tickets opened, closures, notes)."""
    opened, closed, notes = [], [], []
    now = pd.Timestamp.now(tz="UTC")
    for ev in watch:
        name = ev["pair"]
        df = data.get(name)
        if df is None:
            continue
        lv = levels(df, ev["kw"], ev["cfg"])
        age = now - lv["bar"]
        if age > STALE_AFTER:
            # the market is shut, or the feed is not delivering. Either way there is no live
            # bar to act on, and pretending otherwise is how a weekend ticket gets written
            notes.append(f"{name} stale: newest closed bar is {age} old - not signalling")
            continue
        pos = st.get(name)
        if pos:
            done = check_exit(pos, df, ev["kw"])
            if done:
                why, detail, r = done
                st.pop(name)
                closed.append((name, why, r))
                notify(f"{name} {why} {r:+.2f}R", detail, priority=3,
                       tags=["heavy_minus_sign" if r < 0 else "white_check_mark"])
        if lv["dirn"] and name not in st:
            if lv["n_closed"] < ev["kw"]["dev_n"] + 20:
                notes.append(f"{name}: {lv['n_closed']} closed bars, need "
                             f"{ev['kw']['dev_n'] + 20} for the {ev['kw']['dev_n']}-bar mean")
                continue
            tk = ticket(name, lv, ev["cfg"], usd_table, account, risk_pct)
            if not tk["sizing"] or tk["sizing"].get("below_min_lot"):
                notes.append(f"{name}: cannot size this trade" if not tk["sizing"] else
                             f"{name}: {tk['sizing']['lots']:.4f} lot is under the "
                             f"{tk['sizing']['min_lot']}-lot floor at this account size")
            st[name] = dict(dir=lv["dirn"], entry=tk["entry"], stop=tk["stop"],
                            target=tk["target"], sd=tk["sd"], since=str(lv["bar"]),
                            max_bars=ev["cfg"]["max_bars"], cost_r=tk["cost_r"],
                            signal=lv["close"], lots=tk["sizing"].get("lots"))
            opened.append((name, tk))
            notify(f"{name} {tk['dir']} {tk['entry']:.5f}",
                   body_for(ev, tk, account, risk_pct), priority=4,
                   tags=["arrows_counterclockwise"])
        elif not quiet and not pos and lv["dirn"] == 0:
            notes.append(f"{name} flat at {lv['bar']}")
    return opened, closed, notes


def run(account=10_000.0, risk_pct=0.5, topic=None, token=None, limit=3, dry=False,
        once=True, interval=300, notify_failed=True):
    watch = evidence()
    if not watch:
        print("no pair currently qualifies - rerun: python -m lab.intraday")
        return
    watch = watch[:limit]
    print(f"watching {[e['pair'] for e in watch]} | ${account:,.0f} @ {risk_pct}% | "
          f"ntfy {'dry-run' if dry else (topic[:8] + '...' if topic else 'NO TOPIC')}")
    global TOPIC, TOKEN, DRY
    TOPIC, TOKEN, DRY = topic, token, dry
    st = load_state()
    fails = []
    while True:
        data = {}
        for ev in watch:
            try:
                data[ev["pair"]] = fetch_live(ev["pair"])
            except RuntimeError as e:
                fails.append(str(e))
                data[ev["pair"]] = None
        usable = {k: v for k, v in data.items() if v is not None}
        # a ticket cannot be sized without knowing what a pip of the quote currency is worth
        for cur in {ev["pair"][3:] for ev in watch} - {"USD"} - set(usd_has(usable)):
            sym = CONVERSION.get(cur)
            if sym and sym not in usable:
                try:
                    usable[sym] = fetch_live(sym)
                except RuntimeError as e:
                    fails.append(str(e))
        usd_table, _ = L.build_usd_table(usable) if usable else ({}, {})
        opened, closed, notes = one_pass(watch, data, st, usd_table, account,
                                          risk_pct)
        for n in notes:
            print("  " + n)
        for name, tk in opened:
            print(f"  OPEN {name} {tk['dir']} @ {tk['entry']:.5f} stop {tk['stop']:.5f} "
                  f"tgt {tk['target']:.5f} "
                  f"{(tk['sizing'].get('lots') or float('nan')):.2f} lot")
        for name, why, r in closed:
            print(f"  CLOSED {name}: {why} {r:+.3f}R")
        save_state(st)
        if fails:
            msg = "\n".join(fails)
            print("  DATA FAILURE:\n" + msg)
            if notify_failed:
                notify("MONITOR BLIND - data feed failed",
                       msg + "\n\nThose pairs were NOT evaluated on this pass, so anything "
                       "open in them is unwatched. This is not 'no signal'.",
                       priority=5, tags=["warning"])
            fails = []
        if once:
            break
        time.sleep(max(interval - 20, 60))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--account", type=float, default=float(os.environ.get("ACCOUNT", 10_000)))
    ap.add_argument("--risk", type=float, default=float(os.environ.get("RISK_PCT", 0.5)))
    ap.add_argument("--limit", type=int, default=int(os.environ.get("WATCH_TOP", 3)))
    ap.add_argument("--topic", default=os.environ.get("NTFY_TOPIC"))
    ap.add_argument("--token", default=os.environ.get("NTFY_TOKEN"))
    ap.add_argument("--daemon", action="store_true", help="loop instead of one pass")
    ap.add_argument("--interval", type=int, default=int(os.environ.get("POLL_SEC", 300)))
    ap.add_argument("--dry", action="store_true", help="print tickets instead of pushing")
    a = ap.parse_args()
    run(a.account, a.risk, a.topic, a.token, a.limit, a.dry, once=not a.daemon,
        interval=a.interval)


if __name__ == "__main__":
    main()
