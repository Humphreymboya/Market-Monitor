"""
Final report: given a chosen strategy config, produce the numbers and the charts.

  python -m lab.report --tf 1d_clean --split 2026-01-01
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import search as SR
from . import tradeoff as TO
from . import edge as ED

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def pooled_table(s):
    rows = []
    for p in s["per_pair"]:
        rows.append(dict(pair=p["pair"], n=p["n_trades"], wr=round(p["win_rate"], 1),
                         pf=round(p["pf"], 2), exp=round(p["exp_r"], 3),
                         avg_pips=round(p["avg_pips"], 1), tot_pips=round(p["tot_pips"], 0),
                         avg_bars=round(p.get("avg_bars", 0), 1)))
    return pd.DataFrame(rows).sort_values("exp", ascending=False)


def chart(s_is, s_oos, title, path):
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    for ax_, s, lab in ((ax[0][0], s_is, "in-sample"), (ax[0][1], s_oos, "OUT-OF-SAMPLE")):
        if s is None:
            continue
        port = s["port"]
        ax_.plot(port.index, port.values, lw=1.1, color="navy")
        dd = (port - port.cummax()) / port.cummax()
        ax_.fill_between(port.index, dd.values, 0, color="firebrick", alpha=0.25)
        ax_.set_title(f"{lab}: equity + drawdown\nWR {s['win_rate']:.1f}%  "
                      f"PF {s['pf']:.2f}  exp {s['exp_r']:+.3f}R  n={s['n_trades']}  "
                      f"maxDD {100*s['max_dd']:.1f}%")
        ax_.grid(alpha=0.3)
    # per-pair expectancy
    if s_is:
        pp = pooled_table(s_is)
        ax[1][0].barh(pp["pair"], pp["exp"], color=np.where(pp["exp"] > 0, "seagreen", "indianred"))
        ax[1][0].set_title("per-pair expectancy (R) - in-sample")
        ax[1][0].grid(alpha=0.3)
    if s_oos:
        pp = pooled_table(s_oos)
        ax[1][1].barh(pp["pair"], pp["wr"], color=np.where(pp["wr"] >= 65, "seagreen",
                                                            np.where(pp["wr"] >= 55, "goldenrod", "indianred")))
        ax[1][1].axvline(65, color="black", ls="--", lw=1)
        ax[1][1].set_title("per-pair win rate % - out-of-sample (dashed = 65% target)")
        ax[1][1].grid(alpha=0.3)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="1d_clean")
    ap.add_argument("--config", default=os.path.join(OUT, "chosen_strategy.json"))
    ap.add_argument("--split", default="2026-01-01")
    a = ap.parse_args()

    cfgp = json.load(open(a.config))
    fam, kw, cfg = cfgp["family"], cfgp["params"], cfgp["cfg"]
    eng = SR.Engine(a.tf)
    s_is = eng.evaluate(fam, kw, cfg, None, a.split)
    s_oos = eng.evaluate(fam, kw, cfg, a.split, None)
    s_all = eng.evaluate(fam, kw, cfg, None, None)

    lines = []
    def say(x=""):
        print(x)
        lines.append(str(x))

    say("=" * 78)
    say(f"STRATEGY  {fam}")
    say(f"PARAMS    {json.dumps(kw, sort_keys=True)}")
    say(f"EXITS     {json.dumps(cfg, sort_keys=True)}")
    say(f"TIMEFRAME {a.tf} | {len(eng.data)} pairs | costs: real spread + slippage per pair")
    say("=" * 78)
    hdr = f"{'window':16s} {'trades':>7s} {'win%':>7s} {'PF':>6s} {'E[R]':>8s} {'maxDD':>7s} {'Sharpe':>7s} {'pips/trade':>11s} {'tr/yr':>7s}"
    say(hdr)
    say("-" * 78)
    for lab, s in (("in-sample", s_is), ("OUT-OF-SAMPLE", s_oos), ("everything", s_all)):
        if not s:
            say(f"{lab:16s}  (no trades in window)")
            continue
        say(f"{lab:16s} {s['n_trades']:7d} {s['win_rate']:7.1f} {s['pf']:6.2f} "
            f"{s['exp_r']:+8.4f} {100*s['max_dd']:6.1f}% {s['sharpe']:7.2f} "
            f"{s['avg_pips']:11.1f} {s['trades_per_yr']:7.1f}")
    say()
    if s_all:
        say(f"avg win {s_all['avg_win_r']:+.2f}R | avg loss {s_all['avg_loss_r']:+.2f}R | "
            f"payoff {abs(s_all['avg_win_r']/s_all['avg_loss_r']) if s_all['avg_loss_r'] else 0:.2f}")
        say("")
        say("PER-PAIR (out-of-sample):")
        if s_oos:
            say(pooled_table(s_oos).to_string(index=False))
        say("")
        # the payoff/win-rate curve: what a 65% win rate actually costs you
        say("WIN-RATE vs PAYOFF (same entries, target swept; null = what a coin flip earns):")
        rows = []
        for rr in (0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
            c2 = dict(cfg)
            c2["rr"] = rr
            s2 = eng.evaluate(fam, kw, c2, None, None)
            if not s2 or s2["n_trades"] < 10:
                continue
            p0 = TO.null_wr(rr)
            rows.append(dict(rr=rr, n=s2["n_trades"], wr=round(s2["win_rate"], 1),
                             wr_null=round(100 * p0, 1),
                             excess=round(s2["win_rate"] - 100 * p0, 1),
                             exp_R=round(s2["exp_r"], 4), pf=round(s2["pf"], 2)))
        say(pd.DataFrame(rows).to_string(index=False))
    say("")
    say("How to read this: win rate alone is not an edge. At target = 0.5R a random walk")
    say(f"already wins ~{100*TO.null_wr(0.5):.0f}% of the time. 'excess' above the null column")
    say("is the only part that is skill; expectancy in R is what pays the bills.")

    # The same entries, but with every trade forced to end at a barrier, so win rate is
    # directly comparable to the coin-flip probability. Early signal exits move the win rate
    # without proving direction; this section is the one that does or does not.
    say("")
    say("BARRIER TEST - same entries, exits restricted to stop/target, three disjoint eras")
    say("(answers: does this entry pick DIRECTION better than a driftless walk, repeatedly?)")
    br = dict(atr_mult=cfg.get("atr_mult", 2.0), rr=cfg.get("rr") or 1.0,
              max_bars=0, trail_atr=0.0, use_exit_sig=False)
    m = ED.measure(eng, fam, kw, br)
    if m:
        say(pd.DataFrame(m["rows"]).to_string(index=False))
        say(f"pooled: n={m['pool_n']} wr={m['pool_wr']}% excess={m['pool_excess']:+.1f}pt "
            f"z={m['pool_z']:+.2f} | worst-era z={m['worst_z']:+.2f} "
            f"(multiple-comparison bar for this much searching: z >= 3.9)")
        say("null at this payoff = "
            f"{100*ED.null_wr(br['rr']):.1f}%; excess <= 0 means the win rate is the payoff, not skill")
    else:
        say("(too few trades in one or more eras to measure)")

    p = chart(s_is, s_oos, f"{fam} {json.dumps(kw, sort_keys=True)} on {a.tf}",
              os.path.join(OUT, f"report_{a.tf}.png"))
    say(f"\nchart: {p}")
    with open(os.path.join(OUT, f"report_{a.tf}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
