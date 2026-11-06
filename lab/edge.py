"""
Rank signals by the only win-rate number that means anything: the excess over the
gambler's-ruin null, measured independently in three disjoint eras.

Why this exists. The tradeoff sweep shows observed win rate sits at ~57% whether the target
is 0.25R or 3R. That single fact kills raw win rate as a search target - it is set by the
payoff, not the signal. With a stop worth 1R and a target worth rr R, a driftless walk hits
the target first with probability 1/(1+rr). So the question is never "is the win rate above
65%" but "does this signal clear that barrier line by enough, in every era, to pay the
spread".

Protocol, deliberately stricter than the parameter search:
  * pure barrier exits (no signal exit, no time stop, no trail) so every trade ends at the
    stop or the target and WR vs 1/(1+rr) is an exact comparison
  * three disjoint eras, no overlap; an edge has to show up in all three, not once
  * z is computed on the pooled era count, so a 2000-trade era gives ~1 point of precision
  * rank by the WORST era z, so one lucky decade cannot carry a parameter set
"""
import argparse
import itertools
import json

import numpy as np
import pandas as pd

from . import search as SR
from .data import PAIRS

ERAS = [("2004-2011", "2004-01-01", "2011-07-01"),
        ("2011-2018", "2011-07-01", "2018-01-01"),
        ("2018-2026", "2018-01-01", None)]

BARRIER_CFG = dict(atr_mult=2.0, rr=1.0, max_bars=0, trail_atr=0.0, use_exit_sig=False)


def null_wr(rr):
    """Coin-flip barrier probability. Only meaningful for an actual two-sided target."""
    return 1.0 / (1.0 + rr) if rr and rr > 0 else float("nan")


def measure(eng, fam, kw, cfg, eras=ERAS):
    """Per-era win-rate excess over the barrier null, plus the pooled z of the worst era."""
    rr = cfg["rr"]
    p0 = null_wr(rr)
    # an exit with no price target (signal or time exit) has no barrier null to compare to
    has_null = bool(p0 == p0) and 0.0 < p0 < 1.0
    out, pooled = [], [0, 0]
    for label, s, e in eras:
        st = eng.evaluate(fam, kw, cfg, s, e)
        if not st or st["n_trades"] < SR.GATES["oos_min_trades"]:
            return None
        n, wr = st["n_trades"], st["win_rate"] / 100.0
        z = ((wr - p0) / np.sqrt(p0 * (1 - p0) / n)) if has_null else np.nan
        out.append(dict(era=label, n=n, wr=round(100 * wr, 1),
                        excess=round(100 * (wr - p0), 1) if has_null else np.nan,
                        z=round(z, 2),
                        exp_R=round(st["exp_r"], 4), pf=round(st["pf"], 2),
                        dd=round(st["max_dd"], 3)))
        pooled[0] += wr * n
        pooled[1] += n
    pe = pooled[0] / pooled[1]
    wz = min(o["z"] for o in out) if has_null else np.nan
    pz = ((pe - p0) / np.sqrt(p0 * (1 - p0) / pooled[1])) if has_null else np.nan
    return dict(rows=out, worst_z=wz, pool_n=pooled[1],
                pool_wr=round(100 * pe, 1),
                pool_excess=round(100 * (pe - p0), 1) if has_null else np.nan,
                pool_z=round(pz, 2))


def combos_limited(fam, cap):
    grid = SR.GRIDS[fam]
    keys = list(grid)
    all_c = list(itertools.product(*[grid[k] for k in keys]))
    if cap and len(all_c) > cap:
        step = max(1, len(all_c) // cap)
        all_c = all_c[::step]
    return [dict(zip(keys, c)) for c in all_c]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="1d_rep")
    ap.add_argument("--families", default="rsi2_mr,bb_fade,donchian_bo,pullback")
    ap.add_argument("--atr-mult", type=float, default=2.0)
    ap.add_argument("--rr", type=float, default=1.0)
    ap.add_argument("--cap", type=int, default=60, help="max parameter sets per family")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg = dict(BARRIER_CFG, atr_mult=a.atr_mult, rr=a.rr)
    eng = SR.Engine(a.tf)
    print(f"{a.tf}: {len(eng.data)} pairs, barrier-only exits, "
          f"stop={a.atr_mult} ATR, target={a.rr}R, null WR {100*null_wr(a.rr):.1f}%")
    print("eras: " + " | ".join(f"{l} {s[4:7]}-{(e or 'now')[:4]}" for l, s, e in ERAS), flush=True)

    rows = []
    for fam in a.families.split(","):
        kws = combos_limited(fam, a.cap)
        for i, kw in enumerate(kws):
            m = measure(eng, fam, kw, cfg)
            eng.clear_signals()
            if m is None:
                continue
            rows.append(dict(family=fam, params=json.dumps(kw, sort_keys=True),
                             worst_z=m["worst_z"], pool_wr=m["pool_wr"],
                             pool_excess=m["pool_excess"], pool_n=m["pool_n"],
                             pool_z=m["pool_z"],
                             **{f"{r['era']}_excess": r["excess"] for r in m["rows"]},
                             **{f"{r['era']}_exp": r["exp_R"] for r in m["rows"]},
                             detail=json.dumps(m["rows"])))
            if (i + 1) % 20 == 0:
                print(f"   [{fam}] {i+1}/{len(kws)} done, best worst-z "
                      f"{max((r['worst_z'] for r in rows), default=-9)}", flush=True)

    df = pd.DataFrame(rows)
    if not len(df):
        print("nothing measurable")
        return df
    df = df.sort_values("worst_z", ascending=False)
    out = a.out or f"results/edge_{a.tf}_{a.rr}R.csv"
    df.to_csv(out, index=False)

    print(f"\n{len(df)} parameter sets measured. Ranked by the WORST of the three eras.\n")
    cols = ["family", "params", "pool_n", "pool_wr", "pool_excess", "pool_z", "worst_z"]
    print(df[cols].head(15).to_string(index=False))

    n_trials = len(df)
    # Bonferroni-style bar: with ~n_trials independent-ish looks, a z of 3.9 is p ~ 1e-4
    bar = 3.9
    real = df[(df.worst_z >= bar) & (df.pool_excess > 0)]
    print(f"\nClearing the multiple-comparison bar (worst-era z >= {bar}): {len(real)}")
    if len(real):
        print(real[cols].head(10).to_string(index=False))
        for _, r in real.head(3).iterrows():
            print(f"\n{r.family} {r.params}\n  " + json.dumps(json.loads(r.detail)))
    else:
        best = df.iloc[0]
        print(f"  best worst-era z was {best.worst_z} ({best.family} {best.params}) - "
              f"a real but sub-threshold hit-rate excess")
    return df


if __name__ == "__main__":
    main()
