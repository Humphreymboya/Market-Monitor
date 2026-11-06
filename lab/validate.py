"""
Validation gauntlet for search finalists.

A backtest that only passes one train/test split is a coin flip you happened to win.
Every finalist here has to survive:
  1. two non-overlapping out-of-sample periods, independently gated
  2. a cost-stress pass (spread x2 and x3) - FX spreads widen and retail pays more
  3. Monte-Carlo trade-order reshuffling for the real drawdown distribution
  4. parameter plateau: neighbours in the grid must still work (no razor-edge spikes)
  5. leave-one-pair-out: no single pair may be carrying the whole result
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from . import search as SR
from . import backtest as B
from . import data as D

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def sub_segments(oos_start, idx_max):
    """Split the out-of-sample span into two independent halves to test for gate."""
    t0 = pd.Timestamp(oos_start, tz="UTC")
    t1 = pd.Timestamp(idx_max)
    if t1.tzinfo is None:
        t1 = t1.tz_localize("UTC")
    mid = t0 + (t1 - t0) / 2
    lab = lambda a, b: f"{a.year}-{b.year}"  # noqa: E731
    return [(t0.isoformat(), mid.isoformat(), lab(t0, mid)),
            (mid.isoformat(), "2099-01-01", lab(mid, t1))]


def mc_drawdown(rs, risk=0.005, n_sims=2000, seed=7):
    """Distribution of max drawdown when the same trades occur in a different order."""
    rng = np.random.default_rng(seed)
    rs = np.asarray(rs, dtype=float)
    if len(rs) < 20:
        return {}
    dds = np.empty(n_sims)
    for k in range(n_sims):
        eq = np.cumprod(1.0 + risk * rng.permutation(rs))
        peak = np.maximum.accumulate(eq)
        dds[k] = ((peak - eq) / peak).max()
    return {"dd_p50": float(np.percentile(dds, 50)), "dd_p95": float(np.percentile(dds, 95)),
            "dd_p99": float(np.percentile(dds, 99)),
            "ruin50": float((dds > 0.5).mean()), "ruin80": float((dds > 0.8).mean())}


def plateau(eng, fam, kw, cfg, is_end):
    """Re-score the config with each exit/stop parameter nudged one step."""
    variants = []
    for mult in (0.75, 1.0, 1.25):
        c = dict(cfg)
        if c.get("atr_mult"):
            c["atr_mult"] = round(c["atr_mult"] * mult, 3)
        variants.append(c)
    for rr in (0.5, 1.0, 1.5, 2.0):
        if "rr" in cfg:
            c = dict(cfg)
            c["rr"] = rr
            variants.append(c)
    ok = tot = 0
    for c in variants:
        s = eng.evaluate(fam, kw, c, None, is_end)
        if not s:
            continue
        tot += 1
        if s["exp_r"] > 0.02 and s["pf"] >= 1.05:
            ok += 1
    return round(100.0 * ok / tot, 1) if tot else 0.0


def leave_one_out(eng, fam, kw, cfg, start, end):
    full_data = eng.data
    base = eng.evaluate(fam, kw, cfg, start, end)
    if not base:
        return None
    worst = {"exp": base["exp_r"], "wr": base["win_rate"]}
    drops = []
    for name in list(full_data):
        eng.data = {k: v for k, v in full_data.items() if k != name}
        s = eng.evaluate(fam, kw, cfg, start, end)
        if s and s["n_trades"] >= 20:
            drops.append((name, round(s["exp_r"], 4), round(s["win_rate"], 1)))
    eng.data = full_data
    if not drops:
        return None
    e = [d[1] for d in drops]
    w = [d[2] for d in drops]
    return {"exp_min": min(e), "exp_max": max(e), "wr_min": min(w), "wr_max": max(w),
            "exp_negative_when_excluded": [d[0] for d in drops if d[1] <= 0]}


def gate(s, min_trades=12):
    if not s:
        return False
    return (s["n_trades"] >= min_trades and s["win_rate"] >= SR.GATES["oos_min_wr"]
            and s["pf"] >= 1.10 and s["exp_r"] > 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="1d")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--split", default="2017-01-01")
    ap.add_argument("--only-pass", action="store_true")
    a = ap.parse_args()

    path = None
    for f in os.listdir(OUT):
        if f.startswith(f"search_{a.tf}") and f.endswith(".csv"):
            path = os.path.join(OUT, f)
    if not path:
        print("no search results yet - run lab.search first")
        return
    df = pd.read_csv(path)
    if a.only_pass and df["PASS"].sum():
        df = df[df.PASS]
    cand = df.sort_values("oos_exp", ascending=False).head(a.top)
    print(f"validating {len(cand)} candidates from {os.path.basename(path)}\n")

    eng = SR.Engine(a.tf)
    last_idx = max(d.index.max() for d in eng.data.values())
    SUBS = sub_segments(a.split, last_idx)
    print(f"OOS halves: {[s[2] for s in SUBS]}\n")
    rows = []
    for _, r in cand.iterrows():
        fam = r["family"]
        kw = json.loads(r["params"])
        cfg = json.loads(r["cfg"])
        seg = [eng.evaluate(fam, kw, cfg, s0, s1) for (s0, s1, _) in SUBS]
        sub1, sub2 = seg[0], seg[1]
        # cost stress
        base_meta = {k: dict(v) for k, v in eng.meta.items()}
        stress = {}
        for f_mult in (2.0, 3.0):
            for name in eng.meta:
                eng.meta[name]["spread"] = base_meta[name]["spread"] * f_mult
                eng.meta[name]["slip"] = base_meta[name]["slip"] * f_mult
            ss = eng.evaluate(fam, kw, cfg, a.split, None)
            stress[f"x{int(f_mult)}"] = None if not ss else round(ss["exp_r"], 4)
            for name in eng.meta:
                eng.meta[name]["spread"] = base_meta[name]["spread"]
                eng.meta[name]["slip"] = base_meta[name]["slip"]

        oos = eng.evaluate(fam, kw, cfg, a.split, None)
        mc = mc_drawdown(rebuild_trades(eng, fam, kw, cfg, a.split, None))
        pl = plateau(eng, fam, kw, cfg, a.split)
        loo = leave_one_out(eng, fam, kw, cfg, a.split, None)

        ok_both = bool(gate(sub1) and gate(sub2))
        rows.append(dict(
            family=fam, params=r["params"][:70],
            oos_wr=round(oos["win_rate"], 1) if oos else 0,
            oos_pf=round(oos["pf"], 2) if oos else 0,
            oos_exp=round(oos["exp_r"], 4) if oos else 0,
            oos_n=oos["n_trades"] if oos else 0,
            wr_a=round(sub1["win_rate"], 1) if sub1 else 0,
            n_a=sub1["n_trades"] if sub1 else 0,
            wr_b=round(sub2["win_rate"], 1) if sub2 else 0,
            n_b=sub2["n_trades"] if sub2 else 0,
            both_oos_pass=ok_both,
            exp_x2=stress.get("x2"), exp_x3=stress.get("x3"),
            mc_dd_p95=round(mc.get("dd_p95", 0), 3), mc_dd_p99=round(mc.get("dd_p99", 0), 3),
            plateau_pct=pl,
            loo_exp_min=(loo or {}).get("exp_min"), loo_wr_min=(loo or {}).get("wr_min"),
            loo_killers=",".join((loo or {}).get("exp_negative_when_excluded", []) or [])[:40],
        ))
        la, lb = SUBS[0][2], SUBS[1][2]
        print(f"{fam:12s} {r['params'][:60]:60s} "
              f"WR {rows[-1]['oos_wr']:5.1f} n{rows[-1]['oos_n']:4d} | "
              f"{la} {rows[-1]['wr_a']:5.1f}(n{rows[-1]['n_a']})  "
              f"{lb} {rows[-1]['wr_b']:5.1f}(n{rows[-1]['n_b']}) both={int(ok_both)} | "
              f"x2 {stress.get('x2')} x3 {stress.get('x3')} | dd95 {rows[-1]['mc_dd_p95']} | "
              f"pla {pl}% | LOOwrmin {rows[-1]['loo_wr_min']}", flush=True)

    out = pd.DataFrame(rows)
    out["VERDICT"] = (out["both_oos_pass"] & (out["exp_x2"] > 0) & (out["exp_x3"] > 0)
                      & (out["mc_dd_p95"] < 0.45)
                      & (out["plateau_pct"] >= 50) & (out["loo_wr_min"] >= 55)
                      & (out["oos_exp"] >= 0.05))
    out.to_csv(os.path.join(OUT, f"validated_{a.tf}.csv"), index=False)
    print("\n=== SURVIVORS ===")
    surv = out[out.VERDICT]
    print(surv.to_string(index=False) if len(surv) else "none")


def rebuild_trades(eng, fam, kw, cfg, start, end):
    """Raw pooled R-multiples, re-sliced the same way the engine does."""
    rs = []
    for name, df in eng.data.items():
        sig = eng.signals(fam, kw, name, df)
        if sig is None:
            continue
        sub, s2 = SR.window(df, sig, start, end)
        if len(sub) < SR.MIN_BARS:
            continue
        m = eng.meta[name]
        res = B.run(sub, s2, cfg, m["pip"], m["spread"], m["slip"], SR.RISK)
        rs.extend([t[2] for t in res["trades"]])
    return rs


if __name__ == "__main__":
    main()
