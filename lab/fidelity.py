"""
Fidelity check for the two 68% systems found on the power frame.

lab.data repairs the vendor daily bars by widening each High/Low only enough to contain its
own stray Open/Close, so a repaired bar's range is a LOWER bound on the true intraday range.
A barrier system's win rate depends entirely on which barrier the true path touched first,
so a frame that systematically understates range is exactly the kind of frame that would
flatter a 0.5R-target system: the target is ~65 pips away and the residual shortfall is
1-5 pips. If the 68% is an artefact of the repair, the hourly-derived frame - where every
bar is the exact min/max of 24 real hourly bars - will not reproduce it.

This compares, on the same entry and the same exit, three things:
  * the repaired daily frame vs the exact hourly frame over the window both cover
  * barrier-only exits vs the exits a live trader would actually place (signal exit, time
    stop, breakeven) - the frontier result is only tradable if a real exit rule holds it
  * the win rate against the gambler's-ruin null at each payoff, so no frame is read as
    skill on its own
"""
import argparse
import json

import numpy as np
import pandas as pd

from . import edge as ED
from . import search as SR

SYSTEMS = [
    ("bb_fade", dict(bb_n=30, k=1.5, rsi_buy=40, trend=0, adx_max=0)),
    ("bb_fade", dict(bb_n=20, k=1.5, rsi_buy=35, trend=0, adx_max=0)),
]
EXITS = [
    ("barrier 4.0ATR 0.5R", dict(atr_mult=4.0, rr=0.5, max_bars=0, trail_atr=0.0,
                                use_exit_sig=False)),
    ("barrier 4.0ATR 0.4R", dict(atr_mult=4.0, rr=0.4, max_bars=0, trail_atr=0.0,
                                use_exit_sig=False)),
    ("barrier 4.0ATR 0.75R", dict(atr_mult=4.0, rr=0.75, max_bars=0, trail_atr=0.0,
                                 use_exit_sig=False)),
    ("barrier 4.0ATR 1.0R", dict(atr_mult=4.0, rr=1.0, max_bars=0, trail_atr=0.0,
                                use_exit_sig=False)),
    ("sigexit 4.0ATR 0.5R", dict(atr_mult=4.0, rr=0.5, max_bars=0, trail_atr=0.0,
                                use_exit_sig=True)),
    ("timecap 4.0ATR 0.5R", dict(atr_mult=4.0, rr=0.5, max_bars=10, trail_atr=0.0,
                                use_exit_sig=False)),
    ("scale-out .5R/50% +1.5R", dict(atr_mult=4.0, rr=1.5, max_bars=0, trail_atr=0.0,
                                     use_exit_sig=False, rr1=0.5, scale=0.5)),
]


def row(eng, fam, kw, cfg, start=None, end=None):
    s = eng.evaluate(fam, kw, cfg, start, end)
    if not s or s["n_trades"] < 15:
        return None
    p0 = ED.null_wr(cfg["rr"]) if not cfg.get("scale") else np.nan
    ex = 100 * (s["win_rate"] / 100 - p0) if p0 == p0 else np.nan
    z = (ex / 100) / np.sqrt(p0 * (1 - p0) / s["n_trades"]) if p0 == p0 else np.nan
    return dict(label=f"{eng.tf}", n=s["n_trades"], wr=round(s["win_rate"], 1),
                null=None if not (p0 == p0) else round(100 * p0, 1),
                excess=None if not (ex == ex) else round(ex, 1),
                z=None if not (z == z) else round(z, 2),
                exp=round(s["exp_r"], 4), pf=round(s["pf"], 2),
                dd=round(s["max_dd"], 3), tpy=round(s["trades_per_yr"], 1),
                ann=round(100 * SR.RISK * s["exp_r"] * s["trades_per_yr"], 2),
                pairs=s["active_pairs"])


def line(lab, tf, r):
    def num(v, fmt):
        return "   -  " if v is None else format(v, fmt)
    print(f"{lab:22s} {tf:10s} {r['n']:5d} {r['wr']:6.1f} "
          f"{num(r['null'], '6.1f')} {num(r['excess'], '+6.1f')} "
          f"{num(r['z'], '+6.2f')} {r['exp']:+8.4f} {r['pf']:5.2f} {r['dd']:6.3f} "
          f"{r['tpy']:6.1f} {r['ann']:+6.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="2024-11-01", help="window both daily frames cover")
    a = ap.parse_args()

    frames = {}
    for tf in ("1d_clean", "1d_rep"):
        frames[tf] = SR.Engine(tf)
    common = set.intersection(*[set(e.data) for e in frames.values()])
    print(f"exit-architecture and frame comparison on {len(common)} pairs shared by both "
          f"daily frames\n")

    hdr = (f"{'exit':22s} {'frame':10s} {'n':>5s} {'wr%':>6s} {'null%':>6s} "
           f"{'exc':>6s} {'z':>6s} {'exp_R':>8s} {'pf':>5s} {'dd':>6s} "
           f"{'tr/yr':>6s} {'ann%':>6s}")
    for fam, kw in SYSTEMS:
        print("=" * 108)
        print(f"{fam} {json.dumps(kw, sort_keys=True)}")
        print("=" * 108)
        print(hdr)
        for lab, cfg in EXITS:
            for tf, eng in frames.items():
                r = row(eng, fam, kw, cfg, a.split, None)
                if r is not None:
                    line(lab, tf, r)
            r = row(frames["1d_rep"], fam, kw, cfg, None, None)
            if r:
                line(lab, "1d_rep 22y", r)
            print()

    print("=" * 108)
    print("ERA STABILITY of the tradable exit (barrier 4ATR / 0.5R), repaired daily frame")
    print("=" * 108)
    eng = frames["1d_rep"]
    cfg = EXITS[0][1]
    for fam, kw in SYSTEMS:
        m = ED.measure(eng, fam, kw, cfg)
        print(f"\n{fam} {json.dumps(kw, sort_keys=True)}")
        print(pd.DataFrame(m["rows"]).to_string(index=False))
        print(f"  pooled n={m['pool_n']} wr={m['pool_wr']}% excess={m['pool_excess']:+.1f} "
              f"z={m['pool_z']:+.2f} | worst-era z={m['worst_z']:+.2f}")
        ho = eng.evaluate(fam, kw, cfg, "2025-07-01", None)
        if ho:
            print(f"  HOLDOUT 2025-07+ (never searched): n={ho['n_trades']} "
                  f"wr={ho['win_rate']:.1f}% exp={ho['exp_r']:+.4f}R pf={ho['pf']:.2f}")


if __name__ == "__main__":
    main()
