"""
Win-rate vs payoff decomposition.

A win rate cannot be read on its own. With a stop worth 1R and a target worth rr R, a
driftless random walk hits the target first with probability 1/(1+rr) - so 65% is
*automatic* if you set the target at ~0.54R, and it says nothing about skill. The profit
line is WR > (1 + cost_R) / (1 + rr).

This module sweeps the target multiple and reports, for every point:
  WR observed, WR expected from payoff alone (null), the excess (real edge), a binomial z
  for that excess, expectancy in R, profit factor and drawdown.
"""
import argparse
import json

import numpy as np

from . import search as SR
from . import backtest as B


def null_wr(rr):
    """Gambler's-ruin hit probability for a driftless walk with barriers at +rr and -1."""
    return 1.0 / (1.0 + rr) if rr > 0 else np.nan


def required_wr(rr, cost_r):
    return (1.0 + cost_r) / (1.0 + rr) if rr > 0 else np.nan


def sweep(tf, fam, params, base_cfg, rrs, split=None):
    eng = SR.Engine(tf)
    kw = params if isinstance(params, dict) else json.loads(params)
    rows = []
    for rr in rrs:
        cfg = dict(base_cfg)
        cfg["rr"] = rr
        s_is = eng.evaluate(fam, kw, cfg, None, split)
        s_oos = eng.evaluate(fam, kw, cfg, split, None) if split else s_is
        for label, s in (("IS", s_is), ("OOS", s_oos)):
            if not s or s["n_trades"] < 8:
                continue
            n = s["n_trades"]
            wr = s["win_rate"] / 100.0
            p0 = null_wr(rr)
            z = (wr - p0) / np.sqrt(p0 * (1 - p0) / n) if p0 and 0 < p0 < 1 else np.nan
            rows.append(dict(tf=tf, period=label, rr=rr, n=n, wr_obs=round(100 * wr, 1),
                             wr_null=round(100 * p0, 1), excess=round(100 * (wr - p0), 1),
                             z=round(z, 2), req_wr=round(100 * required_wr(rr, 0.03), 1),
                             exp_R=round(s["exp_r"], 4), pf=round(s["pf"], 2),
                             dd=round(s["max_dd"], 3), avgwin=round(s["avg_win_r"], 2),
                             avgloss=round(s["avg_loss_r"], 2),
                             tpy=round(s["trades_per_yr"], 1)))
    return rows, eng


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="1d_clean")
    ap.add_argument("--family", default="rsi2_mr")
    ap.add_argument("--params", default='{"rsi_n":2,"buy_th":10,"trend":200,"exit_sma":5,"adx_max":0}')
    ap.add_argument("--atr-mult", type=float, default=2.0)
    ap.add_argument("--split", default=None)
    ap.add_argument("--rrs", default="0.25,0.4,0.5,0.6,0.8,1.0,1.5,2.0,3.0")
    a = ap.parse_args()
    rrs = [float(x) for x in a.rrs.split(",")]
    cfg = dict(atr_mult=a.atr_mult, rr=0.0, max_bars=0, trail_atr=0.0)
    rows, eng = sweep(a.tf, a.family, a.params, cfg, rrs, a.split)
    import pandas as pd
    df = pd.DataFrame(rows)
    cols = ["period", "rr", "n", "wr_obs", "wr_null", "excess", "z", "req_wr", "exp_R", "pf", "dd", "avgwin", "avgloss", "tpy"]
    print(f"\n{a.tf} {a.family} {a.params}  stop={a.atr_mult}ATR")
    print("  req_wr = win rate needed for positive expectancy after costs at that payoff")
    print(df[cols].to_string(index=False))
    print("\nReading: 'excess' is the win rate you get beyond what the payoff ratio alone "
          "buys you. Positive expectancy needs wr_obs > req_wr, and the edge should be "
          "persistent (z > 2) rather than a small sample fluke.")


if __name__ == "__main__":
    main()
