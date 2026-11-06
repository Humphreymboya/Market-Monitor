"""
Automated walk-forward strategy search.

The point is NOT "find the highest win rate" - a 90% win rate is trivially manufactured
with a tiny target and a colossal stop. The gate below requires win rate AND expectancy
AND out-of-sample persistence AND cross-pair consistency together.

  python -m lab.search --tf 1d --smoke
  python -m lab.search --tf 1d --families rsi2_mr,pullback
"""
import argparse
import itertools
import json
import os
import time

import numpy as np
import pandas as pd

from . import data as D
from . import strategies as S
from . import backtest as B

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
RISK = 0.005
# a window shorter than this cannot support a trade-level verdict on a pair
MIN_BARS = 110

# ---- exit grids: ATR stop multiple, target in R, time stop, ATR trailing stop
CFG_GRID = [
    dict(atr_mult=1.5, rr=0.0, max_bars=0, trail_atr=0.0),
    dict(atr_mult=2.0, rr=0.0, max_bars=0, trail_atr=0.0),
    dict(atr_mult=3.0, rr=0.0, max_bars=0, trail_atr=0.0),
    dict(atr_mult=2.0, rr=1.0, max_bars=0, trail_atr=0.0),
    dict(atr_mult=2.0, rr=1.5, max_bars=0, trail_atr=0.0),
    dict(atr_mult=2.0, rr=2.0, max_bars=0, trail_atr=0.0),
    dict(atr_mult=3.0, rr=1.0, max_bars=0, trail_atr=0.0),
    dict(atr_mult=3.0, rr=2.0, max_bars=0, trail_atr=0.0),
    dict(atr_mult=2.0, rr=0.6, max_bars=0, trail_atr=0.0),
    dict(atr_mult=2.0, rr=0.5, max_bars=10, trail_atr=0.0),
    dict(atr_mult=2.0, rr=0.0, max_bars=5, trail_atr=0.0),
    dict(atr_mult=2.0, rr=0.0, max_bars=10, trail_atr=0.0),
    dict(atr_mult=2.0, rr=1.0, max_bars=15, trail_atr=3.0),
    dict(atr_mult=3.0, rr=0.0, max_bars=25, trail_atr=3.0),
    dict(atr_mult=2.0, rr=0.8, max_bars=12, trail_atr=2.5),
]

# Scale-out: bank part of the size at a near target, then run the rest off breakeven. This
# is the one exit change that raises hit rate without shortening the payoff by construction,
# because the runner keeps its full R multiple instead of being capped.
CFG_PARTIAL = [
    dict(atr_mult=2.0, rr=0.0, max_bars=0, trail_atr=0.0, rr1=0.5, scale=0.5),
    dict(atr_mult=2.0, rr=0.0, max_bars=10, trail_atr=0.0, rr1=0.5, scale=0.5),
    dict(atr_mult=3.0, rr=0.0, max_bars=0, trail_atr=0.0, rr1=0.6, scale=0.5),
    dict(atr_mult=2.0, rr=2.0, max_bars=0, trail_atr=0.0, rr1=0.75, scale=0.5),
    dict(atr_mult=2.0, rr=1.5, max_bars=0, trail_atr=0.0, rr1=0.5, scale=0.6),
    dict(atr_mult=3.0, rr=2.0, max_bars=25, trail_atr=3.0, rr1=0.6, scale=0.5),
]
CFG_GRID = CFG_GRID + CFG_PARTIAL

GRIDS = {
    "rsi2_mr": dict(rsi_n=[2, 3, 4], buy_th=[5, 10, 15, 20], trend=[100, 150, 200],
                    exit_sma=[5, 10], adx_max=[0, 25]),
    "bb_fade": dict(bb_n=[15, 20, 30], k=[1.5, 2.0, 2.5], rsi_buy=[30, 35, 40],
                    trend=[0, 200], adx_max=[0, 20, 25]),
    "donchian_bo": dict(n=[15, 20, 30, 55], exit_n=[5, 10, 20], adx_min=[0, 20, 25],
                        regime_lo=[0.1], regime_hi=[0.95]),
    "pullback": dict(fast=[20, 50], slow=[100, 200], down_n=[2, 3, 4],
                     use_rsi=[False, True], rsi_max=[40, 45, 50], adx_min=[0, 20],
                     exit_sma=[0, 10]),
    "session_bo": dict(sess_start=[7, 8, 9], sess_len=[1, 2, 3], rr=[1.0], adx_min=[0, 20]),
    "tiny_tp_mr": dict(rsi_n=[2, 3], buy_th=[10, 15, 20], trend=[200]),
    "extreme_fade": dict(k=[0.8, 1.0, 1.2, 1.5, 2.0], trend=[100, 200],
                         use_trend=[True, False], tgt_prev=[True, False], adx_max=[0, 20]),
    "week_gap_fade": dict(thresh=[10.0, 15.0, 20.0, 30.0], sl_mult=[1.5, 2.0, 3.0],
                          max_gap_mult=[0.0, 3.0, 5.0]),
    # atr_mult has to reach into the 8-24 range here: on 5m bars a 1-2 ATR stop is 2-4 pips,
    # which is less than the spread, so the only survivable part of this grid sizes the
    # barrier off roughly an hourly range while still entering on the 5m flush.
    "intraday_mr": dict(dev_n=[24, 48, 96], dev_k=[0.75, 1.0, 1.5], rsi_buy=[10, 20, 30],
                        htf_n=[12, 24, 48], use_rsi=[True, False], use_sess=[True, False]),
}

# Families that supply their own stop level and/or an explicit price target ignore the
# ATR/rr part of the grid, so sweeping 15 exit variants on them is wasted work.
CFG_BY_FAMILY = {
    "week_gap_fade": [
        dict(atr_mult=2.0, rr=0.0, max_bars=0, trail_atr=0.0),
        dict(atr_mult=2.0, rr=0.0, max_bars=24, trail_atr=0.0),
        dict(atr_mult=2.0, rr=0.0, max_bars=48, trail_atr=0.0),
        dict(atr_mult=2.0, rr=0.0, max_bars=0, trail_atr=2.0),
        dict(atr_mult=2.0, rr=0.0, max_bars=36, trail_atr=2.5),
        dict(atr_mult=2.0, rr=0.0, max_bars=0, trail_atr=0.0, rr1=0.7, scale=0.5),
        dict(atr_mult=2.0, rr=0.0, max_bars=48, trail_atr=0.0, rr1=0.7, scale=0.5),
    ],
    # 5m exits. atr_mult is multiplied by the 5m ATR, so 12-24 is where a stop becomes wide
    # enough that the round-trip cost is a real fraction of R instead of the whole of it.
    # rr=0.5 is included knowing what it does: a 0.5R barrier wins 2/3 of the time on a
    # driftless walk, so it is listed to be compared against its null, not to be admired.
    "intraday_mr": [
        dict(atr_mult=8.0, rr=0.5, max_bars=72, trail_atr=0.0, use_exit_sig=False),
        dict(atr_mult=12.0, rr=0.5, max_bars=72, trail_atr=0.0, use_exit_sig=False),
        dict(atr_mult=12.0, rr=0.5, max_bars=144, trail_atr=0.0, use_exit_sig=False),
        dict(atr_mult=18.0, rr=0.5, max_bars=144, trail_atr=0.0, use_exit_sig=False),
        dict(atr_mult=12.0, rr=0.75, max_bars=144, trail_atr=0.0, use_exit_sig=False),
        dict(atr_mult=12.0, rr=1.0, max_bars=144, trail_atr=0.0, use_exit_sig=False),
        dict(atr_mult=18.0, rr=1.0, max_bars=288, trail_atr=0.0, use_exit_sig=False),
        dict(atr_mult=24.0, rr=1.0, max_bars=288, trail_atr=0.0, use_exit_sig=False),
        dict(atr_mult=12.0, rr=1.5, max_bars=288, trail_atr=0.0, use_exit_sig=False),
        dict(atr_mult=12.0, rr=2.0, max_bars=288, trail_atr=0.0, use_exit_sig=False),
        dict(atr_mult=12.0, rr=0.0, max_bars=72, trail_atr=0.0, use_exit_sig=True),
        dict(atr_mult=18.0, rr=0.0, max_bars=144, trail_atr=6.0, use_exit_sig=True),
        dict(atr_mult=12.0, rr=0.0, max_bars=0, trail_atr=12.0, use_exit_sig=True),
    ],
}


def cfg_grid_for(fam):
    return CFG_BY_FAMILY.get(fam, CFG_GRID)

GATES = dict(
    is_min_trades=60, is_min_wr=55.0, is_min_pf=1.10,
    oos_min_trades=30, oos_min_wr=65.0, oos_min_pf=1.30, oos_min_exp=0.08,
    oos_max_dd=0.30, oos_min_pair_wr=60.0, oos_min_pair_frac=0.55, oos_min_pairs=4,
)


def _ts(x):
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t


def window(df, sig, start, end):
    """Slice bars to a date window. Indicators were computed on the full history, which a
    live trader would also have, so warm-up uses pre-window data but nothing leaks forward."""
    m = np.ones(len(df), dtype=bool)
    if start:
        m &= np.asarray(df.index >= _ts(start))
    if end:
        m &= np.asarray(df.index < _ts(end))
    sub = df.iloc[m]
    out = {}
    for k, v in sig.items():
        out[k] = None if v is None else np.asarray(v)[m]
    return sub, out


class Engine:
    def __init__(self, tf):
        self.data = D.load_clean(tf)
        # copies, not aliases: a cost-stress run that edits self.meta must not silently
        # rewrite the global PAIRS table for every later engine in the process
        self.meta = {D.PAIRS[s]["name"]: dict(D.PAIRS[s]) for s in D.PAIRS}
        self._sig = {}
        self.tf = tf
        bad = max(D.integrity(d) for d in self.data.values())
        if bad > 0.001:
            print(f"WARNING: worst pair has {bad:.2%} bars failing the OHLC envelope check")
        if tf == "1d_rep":
            # repaired bars pass the envelope check, but cross-comparison against the exact
            # hourly-derived bars shows their High/Low still sit 1-5 pips inside the true
            # daily range. Second-order against 60+ pip daily stop distances, yet it means
            # 1d_rep is for statistical power and 1d_clean stays the fidelity check.
            print("note: 1d_rep = 23y vendor daily, envelope-repaired; ranges still ~1-5 pips "
                  "narrower than hourly-derived truth (power frame, not the fidelity frame)")

    def signals(self, fam, kw, name, df):
        key = (fam, tuple(sorted(kw.items())), name)
        if key not in self._sig:
            try:
                self._sig[key] = S.REGISTRY[fam](df, **kw)
            except Exception:  # noqa: BLE001
                self._sig[key] = None
        return self._sig[key]

    def clear_signals(self):
        """
        Drop the signal cache.

        Signals are only ever reused inside one parameter set (every exit config and fold of
        that set), so on a 23-year frame holding them across sets costs ~7 MB per set and
        buys nothing.
        """
        self._sig.clear()

    def evaluate(self, fam, kw, cfg, start, end):
        trades, eqs, per_pair = [], {}, []
        for name, df in self.data.items():
            sig = self.signals(fam, kw, name, df)
            if sig is None:
                continue
            sub, s2 = window(df, sig, start, end)
            if len(sub) < MIN_BARS:
                continue
            m = self.meta[name]
            res = B.run(sub, s2, cfg, m["pip"], m["spread"], m["slip"], RISK)
            res["pair"] = name
            eqs[name] = res.pop("eq")
            res.pop("open", None)
            trades.extend([(t, name) for t in res.pop("trades")])
            per_pair.append(res)
        if not per_pair:
            return None
        return summarise(trades, eqs, per_pair)


def summarise(trades, eqs, per_pair):
    r = np.array([t[0][2] for t in trades], dtype=float)
    pips = np.array([t[0][3] for t in trades], dtype=float)
    if len(r) == 0:
        return None
    wins, losses = r[r > 0], r[r <= 0]
    gl = abs(losses.sum())
    idx = None
    for s in eqs.values():
        idx = s.index if idx is None else idx.union(s.index)
    mat = pd.DataFrame({k: v.reindex(idx).ffill() for k, v in eqs.items()})
    port = mat.mean(axis=1)
    port = port / port.iloc[0]
    dd = float(-((port - port.cummax()) / port.cummax()).min())
    yrs = max((idx[-1] - idx[0]).days / 365.25, 1e-6)
    act = [p for p in per_pair if p["n_trades"] >= 5]
    good = [p for p in act if p["win_rate"] >= GATES["oos_min_pair_wr"]]
    daily = port.pct_change().dropna()
    sharpe = float(daily.mean() / daily.std() * np.sqrt(252)) if len(daily) > 20 and daily.std() > 0 else 0.0
    return {
        "n_trades": len(r),
        "win_rate": 100.0 * len(wins) / len(r),
        "pf": float(wins.sum() / gl) if gl > 1e-9 else 99.0,
        "exp_r": float(r.mean()),
        "avg_win_r": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss_r": float(losses.mean()) if len(losses) else 0.0,
        "trades_per_yr": len(r) / yrs,
        "max_dd": dd,
        "cagr": 100.0 * (float(port.iloc[-1]) ** (1 / yrs) - 1),
        "sharpe": sharpe,
        "avg_pips": float(pips.mean()),
        "active_pairs": len(act),
        "pair_wr_frac": len(good) / max(len(act), 1),
        "per_pair": per_pair,
        "trades": trades,
        "port": port,
    }


def pooled(trades):
    """
    Trade-level statistics over a set of folds treated as one sample.

    A single 5-month fold holding ~40 trades has a standard error on the win rate of about
    7 points, so demanding 65% inside every fold rejects genuine 65% processes. Pooling the
    disjoint folds gives a precise estimate; the folds are then used for the question they
    can actually answer - is the sign of the edge consistent across regimes.
    """
    if not trades:
        return None
    r = np.array([t[0][2] for t in trades], dtype=float)
    pips = np.array([t[0][3] for t in trades], dtype=float)
    wins, losses = r[r > 0], r[r <= 0]
    by_pair = {}
    for t, name in trades:
        by_pair.setdefault(name, []).append(t[2])
    act = [(k, v) for k, v in by_pair.items() if len(v) >= 5]
    good = [1 for k, v in act if 100.0 * sum(1 for x in v if x > 0) / len(v) >= GATES["oos_min_pair_wr"]]
    gl = abs(losses.sum())
    return {
        "n": len(r),
        "win_rate": 100.0 * len(wins) / len(r),
        "pf": float(wins.sum() / gl) if gl > 1e-9 else 99.0,
        "exp_r": float(r.mean()),
        "se_wr": 100.0 * float(np.sqrt((len(wins) / len(r)) * (1 - len(wins) / len(r)) / len(r))),
        "avg_pips": float(pips.mean()),
        "active_pairs": len(act),
        "pair_wr_frac": len(good) / max(len(act), 1),
    }


def passes_is(s):
    return (s and s["n_trades"] >= GATES["is_min_trades"] and s["win_rate"] >= GATES["is_min_wr"]
            and s["pf"] >= GATES["is_min_pf"] and s["exp_r"] > 0.03)


def passes_oos(s):
    if not s:
        return False
    return (s["n_trades"] >= GATES["oos_min_trades"]
            and s["win_rate"] >= GATES["oos_min_wr"]
            and s["pf"] >= GATES["oos_min_pf"]
            and s["exp_r"] >= GATES["oos_min_exp"]
            and s["max_dd"] <= GATES["oos_max_dd"]
            and s["active_pairs"] >= GATES["oos_min_pairs"]
            and s["pair_wr_frac"] >= GATES["oos_min_pair_frac"])


def combos(fam):
    keys = list(GRIDS[fam].keys())
    for vals in itertools.product(*[GRIDS[fam][k] for k in keys]):
        yield dict(zip(keys, vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="1d")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--families", default="")
    ap.add_argument("--split", default="2017-01-01")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    eng = Engine(a.tf)
    print(f"{a.tf}: {len(eng.data)} pairs loaded")
    fams = a.families.split(",") if a.families else list(GRIDS)
    if a.smoke:
        fams = fams[:2]

    t0 = time.time()
    rows = []
    for fam in fams:
        gs = list(combos(fam))
        if a.smoke:
            gs = gs[:6]
        elif a.limit:
            gs = gs[: a.limit]
        for kw in gs:
            for ci, cfg in enumerate(cfg_grid_for(fam)):
                iss = eng.evaluate(fam, kw, cfg, None, a.split)
                if not passes_is(iss):
                    continue
                oos = eng.evaluate(fam, kw, cfg, a.split, None)
                if not oos:
                    continue
                rows.append(dict(
                    family=fam, params=json.dumps(kw, sort_keys=True), cfg_i=ci,
                    cfg=json.dumps(cfg, sort_keys=True),
                    is_n=iss["n_trades"], is_wr=round(iss["win_rate"], 2),
                    is_pf=round(iss["pf"], 3), is_exp=round(iss["exp_r"], 4),
                    oos_n=oos["n_trades"], oos_wr=round(oos["win_rate"], 2),
                    oos_pf=round(oos["pf"], 3), oos_exp=round(oos["exp_r"], 4),
                    oos_dd=round(oos["max_dd"], 4), oos_pairs=oos["active_pairs"],
                    oos_pairwrfrac=round(oos["pair_wr_frac"], 2),
                    oos_tpy=round(oos["trades_per_yr"], 1),
                    oos_avgpips=round(oos["avg_pips"], 1), PASS=passes_oos(oos)))
        print(f"[{fam}] {time.time()-t0:.0f}s  gated rows={len(rows)}  "
              f"OOS-pass={sum(1 for r in rows if r['PASS'])}", flush=True)
        if a.smoke and rows:
            break

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, f"search_{a.tf}_{a.families.replace(',', '_') or 'all'}.csv"),
              index=False)
    cols = ["family", "params", "cfg_i", "oos_n", "oos_wr", "oos_pf", "oos_exp", "oos_dd"]
    if len(df):
        print(f"\nIS-viable={len(df)}  OOS gate PASS={int(df['PASS'].sum())}")
        print(df[df.PASS][cols].head(15).to_string(index=False))
        print("\nBest 12 by OOS expectancy:")
        print(df.sort_values("oos_exp", ascending=False)[cols].head(12).to_string(index=False))
    else:
        print("nothing cleared the in-sample screen")


if __name__ == "__main__":
    main()
