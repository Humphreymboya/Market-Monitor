"""
The last question, asked once and answered with full statistical power.

Everything upstream (lab.search, lab.edge, lab.tradeoff) shows the same shape: raw win rate
is set by the payoff, not by the signal. With a 1R stop and an rr R target a driftless walk
wins 1/(1+rr) of the time before costs, so 65% is free at rr<=0.54 and costs nothing to
claim. The only number that can therefore mean "working strategy" is the win-rate EXCESS
over that barrier line, measured with enough trades and enough disjoint eras that a lucky
decade cannot carry it.

This module sweeps the low-target region - the only region where >65% and positive net
expectancy can coexist at retail cost - across the surviving entries and grades each on:

  pool_n      trades behind the estimate (SE of a win rate is sqrt(p(1-p)/n))
  excess      points of hit rate above 1/(1+rr)
  z           the excess divided by its own standard error
  eras+       disjoint eras (2004-11, 11-18, 18-26) with positive expectancy
  worst_z     the smallest single-era z - one bad era is a failed era
  holdout     2025-07 onward, never used by any search or gate in this project
  x2/x3       expectancy when the spread and slippage are doubled and tripled
  loo         the worst expectancy after deleting one pair, so no single pair carries it

The accept rule is deliberately brutal: z >= 3.9 pooled (the Bonferroni bar for the number
of looks this project has taken), positive in 3 of 3 eras, positive at x2 cost, and a
holdout that is not negative.
"""
import argparse
import copy
import json
import os

import numpy as np
import pandas as pd

from . import edge as ED
from . import search as SR
from . import validate as V

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
HOLDOUT = "2025-07-01"

# entries that reached a profitable point on the barrier frontier, plus the controls that
# never did, so the table cannot be read as a cherry-pick of winners
CANDIDATES = [
    ("bb_fade", dict(bb_n=15, k=2.0, rsi_buy=35, trend=200, adx_max=0)),
    ("bb_fade", dict(bb_n=30, k=1.5, rsi_buy=40, trend=0, adx_max=0)),
    ("bb_fade", dict(bb_n=20, k=1.5, rsi_buy=35, trend=0, adx_max=0)),
    ("extreme_fade", dict(k=1.0, trend=200, use_trend=True, tgt_prev=False, adx_max=0)),
    ("rsi2_mr", dict(rsi_n=2, buy_th=10, trend=200, exit_sma=5, adx_max=0)),
    ("donchian_bo", dict(n=20, exit_n=10, adx_min=20, regime_lo=0.1, regime_hi=0.95)),
]
RRS = [0.30, 0.40, 0.50, 0.60, 0.75, 1.00]
ATRS = [2.0, 3.0, 4.0]


def score(eng, fam, kw, cfg, gauntlet=False):
    """One candidate at one payoff: pooled excess plus the robustness columns."""
    m = ED.measure(eng, fam, kw, cfg)
    if m is None:
        return None
    rr = cfg["rr"]
    p0 = ED.null_wr(rr)
    pool = eng.evaluate(fam, kw, cfg, None, None)
    if not pool:
        return None
    ho = eng.evaluate(fam, kw, cfg, HOLDOUT, None)

    # the expensive checks only run on a system that has already earned them
    earns = gauntlet and pool["win_rate"] >= 65.0 and pool["exp_r"] > 0
    stressed, loo = {}, None
    if earns:
        meta0 = {k: dict(v) for k, v in eng.meta.items()}
        try:
            for f_mult in (2.0, 3.0):
                for name in eng.meta:
                    eng.meta[name]["spread"] = meta0[name]["spread"] * f_mult
                    eng.meta[name]["slip"] = meta0[name]["slip"] * f_mult
                ss = eng.evaluate(fam, kw, cfg, None, None)
                stressed[f_mult] = None if not ss else ss["exp_r"]
        finally:
            eng.meta = {k: dict(v) for k, v in meta0.items()}
        loo = V.leave_one_out(eng, fam, kw, cfg, None, None)
    elif gauntlet:
        stressed = {2.0: None, 3.0: None}
    eras_pos = sum(1 for r in m["rows"] if r["exp_R"] > 0)

    return dict(
        family=fam, params=json.dumps(kw, sort_keys=True),
        atr=cfg["atr_mult"], rr=rr, null_wr=round(100 * p0, 1),
        n=m["pool_n"], wr=m["pool_wr"], excess=m["pool_excess"], z=m["pool_z"],
        worst_z=m["worst_z"], eras_pos=eras_pos,
        exp=round(pool["exp_r"], 4), pf=round(pool["pf"], 2),
        dd=round(pool["max_dd"], 3), tpy=round(pool["trades_per_yr"], 1),
        ann_pct=round(100 * SR.RISK * pool["exp_r"] * pool["trades_per_yr"], 2),
        ho_n=0 if not ho else ho["n_trades"],
        ho_wr=None if not ho else round(ho["win_rate"], 1),
        ho_exp=None if not ho else round(ho["exp_r"], 4),
        exp_x2=None if stressed[2.0] is None else round(stressed[2.0], 4),
        exp_x3=None if stressed[3.0] is None else round(stressed[3.0], 4),
        loo_exp_min=None if not loo else loo["exp_min"],
        verdict=accept(dict(z=m["pool_z"], worst_z=m["worst_z"], eras_pos=eras_pos,
                             exp=pool["exp_r"],
                             ho_exp=None if not ho else ho["exp_r"],
                             exp_x2=stressed[2.0],
                             loo_exp_min=None if not loo else loo["exp_min"],
                             wr=m["pool_wr"])),
    )


def accept(s):
    """>65% hit rate that is also real, robust, and still true after paying more."""
    return bool(s["wr"] >= 65.0 and s["exp"] > 0 and s["z"] >= 3.9
                and s["eras_pos"] == 3 and s["worst_z"] > 0
                and s["exp_x2"] is not None and s["exp_x2"] > 0
                and (s["ho_exp"] is None or s["ho_exp"] > 0)
                and (s["loo_exp_min"] is None or s["loo_exp_min"] > 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="1d_rep")
    ap.add_argument("--rr", default="", help="comma list, defaults to the low-target sweep")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    eng = SR.Engine(a.tf)
    rrs = [float(x) for x in a.rr.split(",")] if a.rr else RRS
    print(f"{a.tf}: {len(eng.data)} pairs, {len(CANDIDATES)} entries x {len(rrs)} targets "
          f"x {len(ATRS)} stops = {len(CANDIDATES)*len(rrs)*len(ATRS)} barrier-only systems\n")

    rows = []
    for fam, kw in CANDIDATES:
        for atr in ATRS:
            for rr in rrs:
                cfg = dict(atr_mult=atr, rr=rr, max_bars=0, trail_atr=0.0,
                           use_exit_sig=False)
                s = score(eng, fam, kw, cfg, gauntlet=True)
                if s is None:
                    continue
                rows.append(s)
                print(f"{fam:12s} {atr:.0f}ATR {rr:.2f}R  null {s['null_wr']:5.1f}%  "
                      f"WR {s['wr']:5.1f}%  excess {s['excess']:+5.1f}  z {s['z']:+5.2f}  "
                      f"worst-z {s['worst_z']:+5.2f}  eras {s['eras_pos']}/3  "
                      f"exp {s['exp']:+.4f}R  x2 {s['exp_x2']}  holdout {s['ho_exp']}  "
                      f"{'ACCEPT' if s['verdict'] else ''}", flush=True)
        eng.clear_signals()

    df = pd.DataFrame(rows)
    out = a.out or os.path.join(OUT, f"final_{a.tf}.csv")
    df.to_csv(out, index=False)

    print("\n" + "=" * 100)
    print("THE 65% FRONTIER - every system with a measured win rate at or above 65%")
    print("=" * 100)
    hi = df[df.wr >= 65.0].sort_values(["wr"], ascending=False)
    cols = ["family", "atr", "rr", "null_wr", "wr", "excess", "z", "eras_pos", "exp",
            "pf", "ann_pct", "ho_exp", "exp_x2", "verdict"]
    print(hi[cols].to_string(index=False) if len(hi) else "none")

    print("\n" + "=" * 100)
    print("BEST EVIDENCE ANYWHERE - highest pooled z on this table, whatever the win rate")
    print("=" * 100)
    print(df.sort_values("z", ascending=False)[cols].head(10).to_string(index=False))

    n_acc = int(df.verdict.sum())
    print("\n" + "-" * 100)
    print(f"{len(df)} systems graded. Passing every test (>=65% WR, z>=3.9, 3/3 eras, "
          f"profitable at 2x cost, clean holdout, no pair dependency): {n_acc}")
    best_z = df.loc[df.z.idxmax()]
    print(f"  highest z reached: {best_z.z:+.2f} ({best_z.family} {best_z.atr:.0f}ATR "
          f"{best_z.rr:.2f}R, WR {best_z.wr}%, exp {best_z.exp:+.4f}R)")
    hi_pos = hi[hi.exp > 0]
    if len(hi_pos):
        b = hi_pos.iloc[0]
        print(f"  best WR with positive net expectancy: {b.wr}% at {b.rr:.2f}R "
              f"(null {b.null_wr}%, excess {b.excess:+.1f} points, z {b.z:+.2f}, "
              f"{b.exp:+.4f}R = {b.ann_pct:+.2f}%/yr at 0.5% risk)")
    print(f"  written: {out}")


if __name__ == "__main__":
    main()
