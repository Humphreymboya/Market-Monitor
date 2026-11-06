"""
Live setup scanner.

Pulls the most recent bars, replays the validated strategy through the SAME engine that
produced the backtest, over the bars that are actually finished, and prints the state each
pair is in: an open position with its live barriers, a setup that just fired and has not been
filled yet, the trades that recently resolved, or nothing.

That is the only honest output for a barrier-only exit. A signal here is not a suggestion that
expires - it becomes a position that lives until price reaches the stop or the target, which
for a 4-ATR stop against a 0.5R target averages about a month per pair. Listing the last
thirty signals, the way this file used to, made a system that fires roughly twice a year per
pair look like it was screaming at you every day.

Quote caveat printed in the output: Yahoo FX is near-real-time but not an execution feed.
Treat the numbers as a plan to place with your broker, not as a fill you can take.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from . import backtest as B
from . import data as D
from . import strategies as S
from . import search as SR

HERE = os.path.dirname(os.path.abspath(__file__))
CFG_FILE = os.path.join(HERE, "..", "results", "chosen_strategy.json")
RECENT_BARS = 15


def quote_to_usd(quote_ccy, rates):
    """USD value of 1 unit of `quote_ccy`, from the pairs we already have."""
    if quote_ccy == "USD":
        return 1.0
    r = rates.get(quote_ccy)
    return r if r else None


def build_usd_table(datasets):
    """Last close of each pair, used to convert pip value into USD."""
    last = {name: float(df["Close"].iloc[-1]) for name, df in datasets.items()}
    # quote-ccy -> USD
    t = {"USD": 1.0}
    if "USDJPY" in last:
        t["JPY"] = 1.0 / last["USDJPY"]
    if "USDCHF" in last:
        t["CHF"] = 1.0 / last["USDCHF"]
    if "USDCAD" in last:
        t["CAD"] = 1.0 / last["USDCAD"]
    for ccy, nm in (("EUR", "EURUSD"), ("GBP", "GBPUSD"), ("AUD", "AUDUSD"), ("NZD", "NZDUSD"),
                    ("SGD", "USDSGD")):
        if nm in last:
            t[ccy] = last[nm] if nm.endswith("USD") else 1.0 / last[nm]
    return t, last


def pip_usd(pip, quote_ccy, usd_table):
    """USD value of one pip per standard lot, for the quote currency of a pair."""
    q_usd = quote_to_usd(quote_ccy, usd_table)
    return None if not q_usd else pip * 100_000.0 * q_usd


def sizing(pip, price, quote_ccy, usd_table, risk_usd, stop_pips):
    """Lot count such that hitting the stop loses ~risk_usd.

    A wide stop on a cheap pip can push the honest size under the 0.01-lot floor most retail
    accounts cannot trade below. Rounding up is not a cosmetic fix - it silently multiplies
    the risk, so the number is reported with the multiple attached rather than printed as 0.00.
    """
    pip_usd_ = pip_usd(pip, quote_ccy, usd_table)
    if not pip_usd_ or stop_pips <= 0:
        return None
    lots = risk_usd / (stop_pips * pip_usd_)
    MIN_LOT = 0.01
    risk_at_min = MIN_LOT * stop_pips * pip_usd_
    return {"lots": lots, "pip_usd_per_lot": pip_usd_,
            "notional_base": lots * 100_000.0,
            "below_min_lot": lots < MIN_LOT, "min_lot": MIN_LOT,
            "risk_at_min_usd": risk_at_min, "risk_at_min_x": risk_at_min / risk_usd}


def closed_through(df, tf):
    """Index of the last bar that is actually finished.

    A daily bar whose date is today, or an hourly bar in the current hour, is still
    forming - its 'close' is a live price, not a settled one. Signalling off it is the
    intraday equivalent of lookahead, so the scanner refuses to.

    Intraday frames are judged by their own spacing, measured from the index, not assumed
    to be hourly. The old hourly test silently threw away the freshest bar for up to 55
    minutes on the 5m frame - eleven bars, most of a bar's worth of decision time - which
    is the difference between a live signal and a stale one.
    """
    now = pd.Timestamp.now(tz="UTC")
    last = df.index[-1]
    if tf in ("1d", "1d_clean", "1d_rep"):
        forming = last.normalize() == now.normalize()
    else:
        if len(df) > 5:
            step = pd.Timedelta(np.median(np.diff(df.index[-60:])))
        else:
            step = pd.Timedelta("1h")
        forming = (now - last) < step
    return len(df) - (1 if forming else 0)


ABSUSD_STOP_FRAC = 0.05


def _ticket(meta, usd_table, account, risk_pct, name, px, direction, sd, rr):
    """One printable order ticket: the two barrier levels, the size, the cost of entry.

    The absurd_stop flag is not decoration. ATR(14) is built from the frame's own High/Low, and
    a vendor bar with a typo'd extreme inflates it for the next fourteen bars, so a stop can come
    out at 29% of price - a level the pair has never traded near in twenty-two years. The engine
    will happily hold that position forever and the ticket will look like a real trade. It is not:
    the barrier was never set by the market, only by the corrupt bar.
    """
    stop = px - direction * sd
    stop_pips = sd / meta["pip"]
    tgt = px + direction * sd * rr if rr else None
    frac = sd / px if px else 0.0
    return {"dir": "LONG" if direction == 1 else "SHORT", "signal_close": px,
            "stop": stop, "stop_pips": stop_pips, "target": tgt, "rr": rr,
            "stop_pct_of_price": frac, "absurd_stop": frac > ABSUSD_STOP_FRAC,
            "sizing": sizing(meta["pip"], px, name[3:], usd_table,
                             account * risk_pct / 100.0, stop_pips),
            "spread_pips": meta["spread"],
            "spread_pct_of_r": meta["spread"] / stop_pips}


def scan(cfg_entry, datasets, usd_table, last_px, tf="1d", account=10_000.0,
         risk_pct=0.5, recent_bars=RECENT_BARS, verbose=True):
    """Per-pair position state, produced by replaying the backtest engine on closed bars.

    One code path for both the history and the present is the point: a setup printed here is
    a setup the statistics were computed on, and the barriers a pair is sitting under right now
    are the ones the engine has been resolving for twenty-two years.
    """
    fam, kw, cfg = cfg_entry["family"], cfg_entry["params"], cfg_entry["cfg"]
    rr = float(cfg.get("rr") or 0.0)
    out = []
    for name, df in datasets.items():
        meta = D.PAIRS[name + "=X"]
        try:
            sig = S.REGISTRY[fam](df, **kw)
        except Exception:  # noqa: BLE001
            continue
        sub, s2 = SR.window(df, sig, None, None)
        n_closed = closed_through(sub, tf)
        if n_closed < SR.MIN_BARS:
            continue
        sub_c = sub.iloc[:n_closed]
        s2_c = {k: (None if v is None else np.asarray(v)[:n_closed])
                for k, v in s2.items()}
        res = B.run(sub_c, s2_c, cfg, meta["pip"], meta["spread"], meta["slip"], SR.RISK)
        now_px = float(sub_c["Close"].iloc[-1])
        rec = {"pair": name, "pip": meta["pip"], "spread_pips": meta["spread"],
               "last_closed_bar": str(sub_c.index[-1]), "last_close": now_px,
               "forming_bar": str(sub.index[-1]) if n_closed < len(sub) else None,
               "open": None, "pending": None, "recent_closed": [],
               "history_trades": res["n_trades"]}

        op = res.get("open")
        if op:
            sd = abs(op["entry"] - op["stop"])
            p = _ticket(meta, usd_table, account, risk_pct, name, op["entry"],
                        op["dir"], sd, rr)
            p.update(entry=op["entry"], bars_held=op["bars_held"],
                     entry_bar=str(sub_c.index[-1 - op["bars_held"]]),
                     unrealised_r=op["unrealised_r"],
                     open_pips=(now_px - op["entry"]) * op["dir"] / meta["pip"],
                     pips_to_stop=abs(p["stop"] - now_px) / meta["pip"],
                     pips_to_target=(abs(p["target"] - now_px) / meta["pip"]
                                     if p["target"] else None))
            rec["open"] = p
        elif int(s2_c["dir"][-1]) != 0:
            a = s2_c["atr"][-1]
            sd = cfg["atr_mult"] * a if a and not np.isnan(a) else 0.0
            if sd > 0:
                rec["pending"] = _ticket(meta, usd_table, account, risk_pct, name, now_px,
                                         int(s2_c["dir"][-1]), sd, rr)
        # the engine marks an unfinished position closed at the last bar to keep the equity
        # curve complete. That is bookkeeping, not an exit: the barriers are still live, and
        # the same position is already reported under `open`. Printing it here would show one
        # trade as both held and resolved.
        real = [t for t in res["trades"] if t[5] not in ("eod", "eod_ps")]
        for t in real[-5:]:
            if t[0] >= n_closed - recent_bars:
                rec["recent_closed"].append(
                    {"bar": str(sub_c.index[t[0]]), "dir": "LONG" if t[1] == 1 else "SHORT",
                     "r": round(t[2], 3), "pips": round(t[3], 1), "bars": int(t[4]),
                     "reason": t[5]})
        out.append(rec)
    return out


def _print_ticket(p):
    tgt = "-" if not p["target"] else f"{p['target']:.5f}"
    print(f"     entry {p['signal_close']:.5f}   stop {p['stop']:.5f} "
          f"({p['stop_pips']:.0f} pips)   target {tgt}  [{p['rr']}R]")
    if p["absurd_stop"]:
        print(f"     !!  DATA ARTEFACT: this stop is {p['stop_pct_of_price']:.1%} of the price. "
              f"No real barrier - ATR was inflated by a corrupt bar, so this position can never "
              f"resolve. Do NOT trade it; and note the backtest counts it as still open too.")
    if p["sizing"]:
        s = p["sizing"]
        if s["below_min_lot"]:
            print(f"     size needs {s['lots']:.4f} lots - BELOW the {s['min_lot']:.2f}-lot floor. "
                  f"Skip this one, or place {s['min_lot']:.2f} and risk "
                  f"${s['risk_at_min_usd']:,.0f} = {s['risk_at_min_x']:.1f}x what you planned.")
        else:
            print(f"     size {s['lots']:.2f} lots ({s['notional_base']/1000:.0f}k base) | "
                  f"pip value ${s['pip_usd_per_lot']:.2f}/lot | "
                  f"spread {p['spread_pips']:.1f} pips = {p['spread_pct_of_r']:.1%} of your R")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="1d_rep")
    ap.add_argument("--refresh", action="store_true", help="re-download the latest bars")
    ap.add_argument("--config", default=CFG_FILE)
    ap.add_argument("--account", type=float, default=10_000.0)
    ap.add_argument("--risk", type=float, default=0.5, help="percent of account per trade")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    if not os.path.exists(a.config):
        print("no chosen_strategy.json - pick a validated config first")
        return
    entry = json.load(open(a.config))
    datasets = D.load_clean(a.tf, refresh=a.refresh)
    usd_table, last_px = build_usd_table(datasets)
    res = scan(entry, datasets, usd_table, last_px, a.tf, account=a.account,
               risk_pct=a.risk)

    if a.json:
        print(json.dumps(res, indent=2, default=str))
        return

    print(f"STRATEGY  {entry['family']}  {json.dumps(entry['params'], sort_keys=True)}")
    print(f"EXITS     {json.dumps(entry['cfg'], sort_keys=True)}")
    print(f"TF {a.tf} | {len(res)} pairs | account {a.account:,.0f} USD | risk {a.risk}% "
          f"per trade ({a.account * a.risk / 100:,.0f} USD)")
    fb = [r for r in res if r["forming_bar"]]
    if fb:
        print(f"Replay stops at the last CLOSED bar. Still forming, ignored: "
              f"{fb[0]['forming_bar'][:16]}")
    print(f"Last closed bar: {res[0]['last_closed_bar'][:16]}\n")

    held = [r for r in res if r["open"]]
    new = [r for r in res if r["pending"]]
    closed = [r for r in res if r["recent_closed"]]

    print("=" * 96)
    print(f"1. HOLDING - {len(held)} position(s) whose barriers are still outstanding")
    print("=" * 96)
    for r in sorted(held, key=lambda x: x["open"]["bars_held"], reverse=True):
        p = r["open"]
        tt = "-" if p["pips_to_target"] is None else f"{p['pips_to_target']:.0f} pips away"
        print(f"  {r['pair']:7s} {p['dir']:5s}  filled {p['entry_bar'][:16]} "
              f"({p['bars_held']} bars ago) at {p['entry']:.5f}")
        print(f"     now {r['last_close']:.5f} -> {p['open_pips']:+.0f} pips = "
              f"{p['unrealised_r']:+.2f}R  |  stop {p['stop']:.5f} is "
              f"{p['pips_to_stop']:.0f} pips away  |  target is {tt}")
        _print_ticket(p)
        print()

    print("=" * 96)
    print(f"2. NEW SIGNAL on the most recent closed bar, not yet filled - {len(new)}")
    print("=" * 96)
    for r in new:
        print(f"  {r['pair']:7s} {r['pending']['dir']:5s}  signal bar "
              f"{r['last_closed_bar'][:16]}")
        _print_ticket(r["pending"])
        print()
    if not new:
        print("  nothing fired on the last closed bar, and that is the normal state. This\n"
              "  system takes a position roughly twice a year per pair and holds it until a\n"
              "  barrier is reached. Do not manufacture a trade to feel busy.\n")

    n_tr = sum(len(r["recent_closed"]) for r in closed)
    print("=" * 96)
    print(f"3. RESOLVED within the last {RECENT_BARS} bars - {n_tr} trade(s) on "
          f"{len(closed)} pair(s)")
    print("=" * 96)
    for r in closed:
        for t in r["recent_closed"]:
            print(f"  {r['pair']:7s} {t['dir']:5s} closed {t['bar'][:16]} after "
                  f"{t['bars']:3d} bars -> {t['r']:+.2f}R ({t['pips']:+.0f} pips) "
                  f"[{t['reason']}]")
    if not closed:
        print("  none")

    print("\nNOTE: quotes are delayed consumer data, not a broker feed. The entry shown for a\n"
          "      NEW signal is the last closed bar's price; the real fill is the next bar's\n"
          "      open. Place a limit at the level the rules specify and let the broker tell\n"
          "      you the truth.")
    print("NOTE: measured expectancy is +0.015R a trade. At 0.5% risk that is 0.008% of\n"
          "      equity a trade, roughly +1%/year over ~130 trades a year across the book.\n"
          "      This is a thin edge, not a paycheck, and it did not clear the significance\n"
          "      bar for the amount of searching it took to find.")
    print("NOTE: the backtest sizes in fractional lots with no floor, and 816 of its 4,358\n"
    "      trades (18.7%) need less than the 0.01 lot a retail account starts with. Those are\n"
    "      the widest-stop trades - median 751 pips - and they are NOT where the edge isn't:\n"
    "      they carry +0.0452R while the 3,542 trades a $10k account can actually place carry\n"
    "      +0.0088R. Skip the sub-lot tail and you keep 81% of the trades but only 56% of the\n"
    "      expectancy, leaving a profit factor of 1.03 - inside the noise of the costs. Every\n"
    "      pair stays tradeable, so this is not a coverage problem; it is a size problem. To\n"
    "      place the whole book at 0.5% risk you need roughly $12k (90th-pct stop) to $69k\n"
    "      (the widest stop the market ever produced), or a tighter ATR multiple that gives up\n"
    "      the exact trades doing the earning.")


if __name__ == "__main__":
    main()
