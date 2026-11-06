"""
Automated multi-round search driver.

One train/test split is not validation - it is one bet you happened to take. This driver
rotates several out-of-sample windows over the same history and only promotes a candidate
that clears the gate in a majority of rounds, so a winner has to work in more than one
market regime. It stops as soon as it has survivors, which is the "repeat until it works"
loop, but pointed at a stricter acceptance test rather than at a looser one.
"""
import argparse
import json
import os
import time

import pandas as pd

from . import search as SR
from . import validate as V

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def fold_gate(oos_folds, min_folds_positive=2):
    """
    Judge a candidate on the POOLED disjoint out-of-sample folds, then ask whether the
    sign survives regime changes. Pooled win rate is the precise estimate; folds of ~40
    trades are far too small to carry a 65% threshold individually.
    """
    g = SR.GATES
    tr = [t for s in oos_folds for t in (s["trades"] if s else [])]
    stats = SR.pooled(tr)
    if not stats:
        return False, None
    have = [s for s in oos_folds if s and s["n_trades"] >= 5]
    pos = sum(1 for s in have if s["exp_r"] > 0)
    stats["folds_ok"] = pos
    ok = (stats["n"] >= 60
          and stats["win_rate"] >= g["oos_min_wr"]
          and stats["pf"] >= g["oos_min_pf"]
          and stats["exp_r"] >= g["oos_min_exp"]
          and stats["active_pairs"] >= g["oos_min_pairs"]
          and stats["pair_wr_frac"] >= g["oos_min_pair_frac"]
          and pos >= min(min_folds_positive, len(have))
          and len(have) >= min_folds_positive
          and all(s["win_rate"] >= 50.0 for s in have)
          and all(s["exp_r"] > -0.20 for s in have)
          and max(s["max_dd"] for s in have) <= g["oos_max_dd"])
    return ok, stats


def folds_for(edges):
    """
    Expanding-window walk-forward over NON-OVERLAPPING out-of-sample periods.

    Measuring the same long tail three times looks like three tests but is one sample
    sampled three ways. Folds here are disjoint; everything after the last edge is never
    read by the search and is kept as a holdout.
    """
    return [(None, edges[i], edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def run(tf, families, edges, min_pass=2, verbose=True, cfgs=None, progress_every=25, cap=0):
    eng = SR.Engine(tf)
    n = len(eng.data)
    span = (max(d.index.max() for d in eng.data.values())
            - min(d.index.min() for d in eng.data.values()))
    folds = folds_for(edges)
    print(f"{tf}: {n} pairs, {span.days/365.25:.1f}y, {len(folds)} disjoint OOS folds "
          f"{[(f[2][:7], f[3][:7]) for f in folds]} need >={min_pass}; "
          f"holdout {edges[-1][:7]}+ untouched", flush=True)
    hits, tried, t0 = [], 0, time.time()

    for fam in families:
        kws = list(SR.combos(fam))
        if cap and len(kws) > cap:
            # even spacing, not a head cut: the grid is ordered by parameter, so taking the
            # first N would silently test only one corner of it
            kws = kws[::max(1, len(kws) // cap)]
        fam_cfgs = cfgs or SR.cfg_grid_for(fam)
        for ki, kw in enumerate(kws):
            for name in eng.data:            # cache signals once, reused by every fold/cfg
                eng.signals(fam, kw, name, eng.data[name])
            if eng.signals(fam, kw, list(eng.data)[0], list(eng.data.values())[0]) is None:
                continue
            for ci, cfg in enumerate(fam_cfgs):
                tried += 1
                rounds = []
                for (is0, is1, o0, o1) in folds:
                    iss = eng.evaluate(fam, kw, cfg, is0, is1)
                    if not SR.passes_is(iss):
                        if not rounds:
                            rounds = None
                            break
                        continue
                    oos = eng.evaluate(fam, kw, cfg, o0, o1)
                    rounds.append((o0, iss, oos))
                if not rounds:
                    continue
                rounds = [r for r in rounds if r[2]]
                if len(rounds) < min_pass:
                    continue
                ok, stats = fold_gate([r[2] for r in rounds], min_folds_positive=min_pass)
                if ok:
                    rec = {"family": fam, "params": json.dumps(kw, sort_keys=True),
                           "cfg_i": ci, "cfg": json.dumps(cfg, sort_keys=True),
                           "rounds_passed": stats["folds_ok"], "rounds_total": len(folds),
                           "pool_n": stats["n"], "pool_wr": round(stats["win_rate"], 1),
                           "pool_se": round(stats["se_wr"], 1), "pool_pf": round(stats["pf"], 2),
                           "pool_exp": round(stats["exp_r"], 4),
                           "fold_wr": [round(o["win_rate"], 1) for (_, _, o) in rounds],
                           "fold_exp": [round(o["exp_r"], 3) for (_, _, o) in rounds],
                           "fold_n": [o["n_trades"] for (_, _, o) in rounds]}
                    hits.append(rec)
                    if verbose:
                        print(f"  HIT {fam} {rec['params'][:52]} cfg{ci} "
                              f"WR {rec['pool_wr']}+/-{rec['pool_se']} pf {rec['pool_pf']} "
                              f"exp {rec['pool_exp']} n{rec['pool_n']} "
                              f"folds_wr={rec['fold_wr']}", flush=True)
            eng.clear_signals()
            if verbose and (ki + 1) % progress_every == 0:
                print(f"   [{fam}] {ki+1}/{len(kws)} sets, {tried} cfgs, {len(hits)} hits, "
                      f"{time.time()-t0:.0f}s", flush=True)
    return pd.DataFrame(hits), tried, time.time() - t0


def gauntlet(eng_tf, df, top=6, split=None):
    """Run the full validation gauntlet on multi-round survivors."""
    eng = SR.Engine(eng_tf)
    last = max(d.index.max() for d in eng.data.values())
    subs = V.sub_segments(split, last)
    rows = []
    for _, r in df.head(top).iterrows():
        fam, kw, cfg = r["family"], json.loads(r["params"]), json.loads(r["cfg"])
        seg = [eng.evaluate(fam, kw, cfg, s0, s1) for (s0, s1, _) in subs]
        oos = eng.evaluate(fam, kw, cfg, split, None)
        if not oos:
            continue
        mc = V.mc_drawdown(V.rebuild_trades(eng, fam, kw, cfg, split, None))
        pl = V.plateau(eng, fam, kw, cfg, split)
        loo = V.leave_one_out(eng, fam, kw, cfg, split, None)
        base = {k: dict(v) for k, v in eng.meta.items()}
        stress = {}
        for f_mult in (2.0, 3.0):
            for k2 in eng.meta:
                eng.meta[k2]["spread"] = base[k2]["spread"] * f_mult
                eng.meta[k2]["slip"] = base[k2]["slip"] * f_mult
            s = eng.evaluate(fam, kw, cfg, split, None)
            stress[f"x{int(f_mult)}"] = None if not s else round(s["exp_r"], 4)
            for k2 in eng.meta:
                eng.meta[k2]["spread"] = base[k2]["spread"]
                eng.meta[k2]["slip"] = base[k2]["slip"]
        okh = [V.gate(s) for s in seg]
        row = dict(family=fam, params=r["params"], cfg=r["cfg"], cfg_i=r["cfg_i"],
                   rounds_passed=r["rounds_passed"],
                   oos_wr=round(oos["win_rate"], 1), oos_n=oos["n_trades"],
                   oos_pf=round(oos["pf"], 2), oos_exp=round(oos["exp_r"], 4),
                   oos_dd=round(oos["max_dd"], 4), oos_sharpe=round(oos["sharpe"], 2),
                   oos_avgpips=round(oos["avg_pips"], 1), oos_tpy=round(oos["trades_per_yr"], 1),
                   halves="/".join(str(int(x)) for x in okh),
                   exp_x2=stress["x2"], exp_x3=stress["x3"],
                   dd_p95=round(mc.get("dd_p95", 0), 3), plateau=pl,
                   loo_wrmin=(loo or {}).get("wr_min"), loo_expmin=(loo or {}).get("exp_min"))
        row["SURVIVE"] = bool(all(okh) and row["oos_wr"] >= 65 and row["oos_exp"] >= 0.05
                              and stress["x3"] > 0 and row["dd_p95"] < 0.45
                              and pl >= 50 and (loo or {}).get("wr_min", 0) >= 55)
        rows.append(row)
        print(f"  {fam:12s} {r['params'][:52]:52s} WR {row['oos_wr']:5.1f} n{row['oos_n']:4d} "
              f"pf{row['oos_pf']:5.2f} exp{row['oos_exp']:+.3f} x3{stress['x3']} "
              f"dd95 {row['dd_p95']} pla{pl}% halves={row['halves']} -> "
              f"{'SURVIVES' if row['SURVIVE'] else 'rejected'}", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="1d_clean")
    ap.add_argument("--families", default="rsi2_mr,bb_fade,pullback,donchian_bo")
    ap.add_argument("--edges", default="2024-10-01,2025-05-01,2025-11-01,2026-04-01",
                    help="fold boundaries; the last one starts the untouched holdout")
    ap.add_argument("--min-pass", type=int, default=2)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--cap", type=int, default=0,
                    help="max parameter sets per family (evenly spaced); 0 = whole grid")
    ap.add_argument("--final-split", default=None, help="default = last edge (holdout start)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    edges = a.edges.split(",")
    df, tried, secs = run(a.tf, a.families.split(","), edges, a.min_pass, cap=a.cap)
    print(f"\nsearch: {tried} configs in {secs:.0f}s -> {len(df)} multi-round survivors")
    if len(df):
        df.to_csv(os.path.join(OUT, f"auto_{a.tf}.csv"), index=False)
        fs = a.final_split or edges[-1]
        print(f"\ngauntlet on untouched holdout from {fs}:\n")
        g = gauntlet(a.tf, df, top=a.top, split=fs)
        g.to_csv(os.path.join(OUT, f"gauntlet_{a.tf}.csv"), index=False)
        s = g[g.SURVIVE] if "SURVIVE" in g else g.iloc[:0]
        print(f"\nSURVIVORS: {len(s)}")
        chosen = []
        for _, x in s.iterrows():
            chosen.append({"family": x["family"], "params": json.loads(x["params"]),
                           "cfg": json.loads(x["cfg"]), "cfg_i": int(x["cfg_i"]),
                           "oos_wr": float(x["oos_wr"]), "oos_exp": float(x["oos_exp"]),
                           "oos_n": int(x["oos_n"]), "tf": a.tf})
        if chosen:
            p = os.path.join(OUT, "chosen_strategy.json")
            json.dump(chosen[0], open(p, "w"), indent=2)
            print(json.dumps(chosen, indent=2))
            print(f"primary written to {p}")
    else:
        print("no candidate survived the multi-round gate")


if __name__ == "__main__":
    main()
