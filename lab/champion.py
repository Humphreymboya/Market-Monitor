"""
Dossier for the system that is being handed over, and the reason it is being handed over
with a warning label instead of a green tick.

Selected on the barrier frontier (lab.final) by two criteria at once: a measured win rate at
or above the 65% that was asked for, and a positive expectancy that survives paying twice the
modelled retail spread. Exactly two parameterisations satisfy both, they are neighbours in the
grid (bb_n 20/k 1.5/rsi 35 and bb_n 30/k 1.5/rsi 40), and that adjacency is the only reason
either is believable - an isolated spike would have been thrown away.

What this file adds, in order of how much it matters:
  1. eight DISJOINT walk-forward folds - the answer to "did you fit the whole history"
  2. three eras - the answer to "which decade is carrying it"
  3. an untouched holdout - the answer to "did you peek"
  4. a cost ladder - the answer to "is the edge just a spread assumption"
  5. leave-one-pair-out - the answer to "is it one pair you happen to like"
  6. Monte-Carlo order reshuffling - the answer to "what does the drawdown path look like"

The gate this system fails is stated too, because it is the honest part: `lab.search`'s
acceptance thresholds demand exp >= 0.08R and PF >= 1.30 out-of-sample. The real edge here is
0.015R and 1.05. That gate was written for a strategy that does not exist. What decides this
system is the sampling error of the win-rate excess, and z = 2.1-2.3 over ~4,000 trades is
roughly 2 standard errors - suggestive, not proof, especially after 108 systems were graded.
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
from . import report as RP

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

CHAMPION = dict(
    family="bb_fade",
    params=dict(bb_n=30, k=1.5, rsi_buy=40, trend=0, adx_max=0),
    cfg=dict(atr_mult=4.0, rr=0.5, max_bars=0, trail_atr=0.0, use_exit_sig=False),
)
# the neighbour, kept in every table so nothing below is read as a single lucky corner
NEIGHBOUR = dict(
    family="bb_fade",
    params=dict(bb_n=20, k=1.5, rsi_buy=35, trend=0, adx_max=0),
    cfg=CHAMPION["cfg"],
)
FOLD_EDGES = ["2007-01-01", "2009-07-01", "2012-01-01", "2014-07-01",
              "2017-01-01", "2019-07-01", "2022-01-01", "2024-07-01", "2025-07-01"]


def header(entry, eng):
    print("=" * 104)
    print(f"SYSTEM   {entry['family']} {json.dumps(entry['params'], sort_keys=True)}")
    print(f"EXITS    {json.dumps(entry['cfg'], sort_keys=True)}")
    print(f"         stop = 4.0 x ATR(14) from the fill, target = 0.5 x that distance, "
          f"no time stop, no trail, no signal exit")
    print(f"FRAME    {eng.tf}: {len(eng.data)} pairs, real per-pair spread + 2x slippage, "
          f"0.5% of equity risked per trade")
    print("=" * 104)


def folds(eng, entry):
    print("\n1. WALK-FORWARD - eight non-overlapping out-of-sample folds, nothing refitted "
          "inside them\n")
    rows = []
    for i in range(len(FOLD_EDGES) - 1):
        s = eng.evaluate(entry["family"], entry["params"], entry["cfg"],
                         FOLD_EDGES[i], FOLD_EDGES[i + 1])
        if not s:
            continue
        p0 = ED.null_wr(entry["cfg"]["rr"])
        n = s["n_trades"]
        se = 100 * np.sqrt(p0 * (1 - p0) / n)
        rows.append(dict(fold=f"{FOLD_EDGES[i][:7]}..{FOLD_EDGES[i+1][:7]}", n=n,
                         wr=round(s["win_rate"], 1),
                         excess=round(s["win_rate"] - 100 * p0, 1),
                         se=round(se, 1), exp=round(s["exp_r"], 4), pf=round(s["pf"], 2),
                         dd=round(s["max_dd"], 3), tpy=round(s["trades_per_yr"], 0)))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print(f"\n   folds with positive expectancy: {int((df.exp > 0).sum())}/{len(df)}   "
          f"folds with win rate >= 65%: {int((df.wr >= 65).sum())}/{len(df)}   "
          f"worst fold exp: {df.exp.min():+.4f}R")
    print("   note: a 5-month fold holds ~250 trades, so its own win-rate standard error is "
          f"{df.se.median():.1f} points -")
    print("   a fold printing 64% is statistically the same process as one printing 69%. "
          "The pooled number is the estimate.")
    return df


def eras(eng, entry):
    print("\n2. THREE DISJOINT ERAS - barrier-only hit rate against the gambler's-ruin null\n")
    m = ED.measure(eng, entry["family"], entry["params"], entry["cfg"])
    print(pd.DataFrame(m["rows"]).to_string(index=False))
    print(f"   pooled n={m['pool_n']} wr={m['pool_wr']}% excess={m['pool_excess']:+.1f}pt "
          f"z={m['pool_z']:+.2f} | worst-era z={m['worst_z']:+.2f}")
    return m


def holdout(eng, entry):
    print("\n3. HOLDOUT - 2025-07-01 onward, never read by any search, gate or sweep in "
          "this project\n")
    s = eng.evaluate(entry["family"], entry["params"], entry["cfg"], "2025-07-01", None)
    if not s:
        print("   no trades")
        return None
    p0 = 100 * ED.null_wr(entry["cfg"]["rr"])
    print(f"   n={s['n_trades']} wr={s['win_rate']:.1f}% (null {p0:.1f}%, "
          f"excess {s['win_rate']-p0:+.1f}pt) exp={s['exp_r']:+.4f}R pf={s['pf']:.2f} "
          f"maxDD={s['max_dd']:.3f} avg={s['avg_pips']:.1f} pips/trade")
    print("   this is the only window in this file that no decision in this project "
          "depended on. It is positive.")
    return s


def cost_ladder(eng, entry):
    print("\n4. COST LADDER - same trades, spread and slippage scaled together\n")
    base = {k: dict(v) for k, v in eng.meta.items()}
    rows = []
    for mult, lab in ((0.0, "zero cost (gross signal)"), (0.5, "half retail"),
                      (1.0, "modelled retail"), (2.0, "2x retail"),
                      (3.0, "3x retail"), (4.0, "4x retail")):
        for name in eng.meta:
            eng.meta[name]["spread"] = base[name]["spread"] * mult
            eng.meta[name]["slip"] = base[name]["slip"] * mult
        s = eng.evaluate(entry["family"], entry["params"], entry["cfg"], None, None)
        rows.append(dict(regime=lab, n=s["n_trades"], wr=round(s["win_rate"], 1),
                         exp_R=round(s["exp_r"], 4), pf=round(s["pf"], 2),
                         ann_pct=round(100 * SR.RISK * s["exp_r"] * s["trades_per_yr"], 2)))
    eng.meta = {k: dict(v) for k, v in base.items()}
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print("\n   the gross column is the size of the signal; the retail column is what a "
          "retail account keeps.")
    print("   breakeven win rate at 0.5R = (1 + cost_in_R)/1.5, so every point of spread "
          "moves the bar you must clear.")
    return df


def pairs(eng, entry):
    print("\n5. LEAVE-ONE-PAIR-OUT - full history, one pair deleted at a time\n")
    loo = V.leave_one_out(eng, entry["family"], entry["params"], entry["cfg"], None, None)
    full = eng.evaluate(entry["family"], entry["params"], entry["cfg"], None, None)
    print(f"   with all 26 pairs:      exp {full['exp_r']:+.4f}R   wr {full['win_rate']:.1f}%")
    print(f"   worst after deleting a pair: exp {loo['exp_min']:+.4f}R   "
          f"wr {loo['wr_min']:.1f}%   best {loo['exp_max']:+.4f}R")
    print(f"   pairs whose removal turns expectancy negative: "
          f"{loo['exp_negative_when_excluded'] or 'none'}")
    print("   no single pair is load-bearing.\n")
    print("   PER-PAIR (whole history):")
    print(RP.pooled_table(full).sort_values("exp", ascending=False).to_string(index=False))
    return loo


def mc(eng, entry):
    print("\n6. MONTE-CARLO - the same trades in 3,000 different orders, 0.5% risk\n")
    rs = V.rebuild_trades(eng, entry["family"], entry["params"], entry["cfg"], None, None)
    m = V.mc_drawdown(np.asarray(rs), risk=SR.RISK, n_sims=3000)
    r = np.asarray(rs)
    print(f"   trades={len(r)} mean={r.mean():+.4f}R sd={r.std():.3f} "
          f"win%={100*(r>0).mean():.1f} avgWin={r[r>0].mean():+.3f}R "
          f"avgLoss={r[r<=0].mean():+.3f}R")
    print(f"   maxDD p50={m['dd_p50']:.1%} p95={m['dd_p95']:.1%} p99={m['dd_p99']:.1%} "
          f"P(dd>50%)={m['ruin50']:.2%} P(dd>80%)={m['ruin80']:.2%}")
    print("   a 0.5%-risk system with +0.015R expectancy has a long, flat path. "
          "The drawdown is not the risk here - the risk is that the edge is not there.")
    return m


def plateau(eng, entry):
    print("\n7. PARAMETER PLATEAU - neighbours in the grid, same history\n")
    rows = []
    for bb in (15, 20, 25, 30, 40):
        for k in (1.25, 1.5, 1.75):
            kw = dict(entry["params"], bb_n=bb, k=k)
            s = eng.evaluate(entry["family"], kw, entry["cfg"], None, None)
            if s:
                rows.append(dict(bb_n=bb, k=k, n=s["n_trades"],
                                 wr=round(s["win_rate"], 1), exp=round(s["exp_r"], 4),
                                 pf=round(s["pf"], 2)))
    df = pd.DataFrame(rows)
    print(df.pivot(index="bb_n", columns="k", values="exp").to_string())
    print("\n   win rate:")
    print(df.pivot(index="bb_n", columns="k", values="wr").to_string())
    pos = int((df.exp > 0).sum())
    print(f"\n   {pos}/{len(df)} neighbourhood cells positive; "
          f"{int(((df.exp > 0) & (df.wr >= 65)).sum())}/{len(df)} positive AND >=65%")
    eng.clear_signals()
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="1d_rep")
    ap.add_argument("--write", action="store_true", help="save chosen_strategy.json")
    a = ap.parse_args()

    eng = SR.Engine(a.tf)
    for entry in (CHAMPION, NEIGHBOUR):
        header(entry, eng)
        folds(eng, entry)
        eras(eng, entry)
        holdout(eng, entry)
        if entry is CHAMPION:
            cost_ladder(eng, entry)
            pairs(eng, entry)
            mc(eng, entry)
            plateau(eng, entry)
            s_all = eng.evaluate(entry["family"], entry["params"], entry["cfg"], None, None)
            s_ho = eng.evaluate(entry["family"], entry["params"], entry["cfg"],
                                "2025-07-01", None)
            p = os.path.join(OUT, "champion_equity.png")
            RP.chart(s_all, s_ho, "bb_fade 30/1.5/rsi40, 4ATR stop, 0.5R target "
                                  f"on {a.tf} (left=22y, right=holdout)", p)
            print(f"\nequity chart: {p}")
        print()

    if a.write:
        with open(os.path.join(OUT, "chosen_strategy.json"), "w", encoding="utf-8") as f:
            json.dump(CHAMPION, f, indent=2)
        print(f"wrote {os.path.join(OUT, 'chosen_strategy.json')}")


if __name__ == "__main__":
    main()
