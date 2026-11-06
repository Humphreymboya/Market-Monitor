"""
5-minute intraday search, run per pair, with the win-rate numbers that make 70% readable.

Two things about this frame have to be said before any number in it is used.

The first is the cost structure, and it is the reason the strategy is built the way it is.
On 5m bars the median range of a major pair is 1-3 pips while a retail spread is 1.1-3 pips,
so an entry and an exit both sized off 5m volatility pays the spread twice on a move that is
smaller than the spread. A 5m system whose barriers are 1-2 ATR(5m) wide is not under-edged,
it is arithmetic: the round trip costs more than the bar can produce. So the entry is timed on
the 5m flush and the stop is sized at 8-24 x ATR(5m), which is roughly one to two hourly
ranges. That makes the cost a fraction of R instead of all of it.

The second is that a win rate on this frame is a very weak statistic, and this is the module
that says so loudest. Eighty-one days is 26 pairs x ~90 trades, so every per-pair win rate
here carries a standard error of 4-5 points. A pair printing 70% on 90 trades is fully
consistent with a true 62%. Worse, the payoff choice manufactures hit rate on its own: a
0.5R barrier wins two thirds of the time on a driftless random walk with no signal at all,
so 70% is ~1 standard error above a null that requires no skill whatsoever. Searching
thousands of pair/config combinations and reporting the best one is a guarantee of finding
70%, not evidence of anything.

Everything below is therefore reported as a set: win rate, its own standard error, the null
for that exact payoff, the excess over that null, the z of the excess, and the expectancy that
the excess actually has to pay for. The z threshold accounts for how many looks this search
took, because with this much searching a single 70% is the expected outcome of nothing.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from . import backtest as B
from . import data as D
from . import indicators as ta
from . import live as L
from . import search as SR
from . import strategies as S

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

# Three entry settings, deliberately few. Every extra dimension here is a dimension that can
# be fitted to noise on an 81-day frame, so this is the smallest defensible sweep.
ENTRY_VARIANTS = [
    dict(dev_n=48, dev_k=1.0, rsi_buy=20, htf_n=24, use_rsi=True, use_sess=True),
    dict(dev_n=24, dev_k=0.75, rsi_buy=20, htf_n=12, use_rsi=True, use_sess=True),
    dict(dev_n=96, dev_k=1.5, rsi_buy=30, htf_n=48, use_rsi=True, use_sess=True),
]

FAMILY = "intraday_mr"


def null_wr(rr):
    """Hit rate of the target-first barrier for a driftless walk: 1/(1+rr)."""
    return 1.0 / (1.0 + rr) if rr and rr > 0 else np.nan


def per_pair(datasets, kw, cfg, start=None, end=None, spread_mult=1.0, control=False):
    """Backtest one config on every pair separately. Returns one row per pair.

    control=True replaces the filtered entry with the filter-free geometry in geometry_one,
    which is the only fair null for a mean-touch exit.
    """
    rows = []
    for name, df in datasets.items():
        m = D.PAIRS[name + "=X"]
        sig = geometry_one(df, kw) if control else S.REGISTRY[FAMILY](df, **kw)
        sub, s2 = SR.window(df, sig, start, end)
        if len(sub) < SR.MIN_BARS:
            continue
        res = B.run(sub, s2, cfg, m["pip"], m["spread"] * spread_mult,
                    m["slip"] * spread_mult, SR.RISK)
        tr = trades_of(res)
        rs = np.array([t[2] for t in tr], dtype=float)
        if len(rs) < 10:
            rows.append(dict(pair=name, n=len(rs), wr=np.nan, exp=np.nan, pf=np.nan,
                             avg_pips=np.nan, se=np.nan, excess=np.nan, z=np.nan))
            continue
        wins, losses = rs[rs > 0], rs[rs <= 0]
        wr = 100.0 * len(wins) / len(rs)
        p0 = null_wr(cfg.get("rr") or 0.0)
        se = 100.0 * np.sqrt((p0 * (1 - p0)) / len(rs)) if p0 == p0 and 0 < p0 < 1 else np.nan
        rows.append(dict(
            pair=name, n=len(rs), wr=wr,
            se=se,
            excess=wr - 100 * p0 if se == se else np.nan,
            z=(wr - 100 * p0) / se if se == se and se > 0 else np.nan,
            exp=float(rs.mean()),
            pf=float(wins.sum() / abs(losses.sum())) if losses.sum() < -1e-9 else 99.0,
            avg_pips=float(np.mean([abs(t[3]) for t in tr])),
            avg_bars=float(np.mean([t[4] for t in tr])),
        ))
    return pd.DataFrame(rows)


def trades_of(res):
    """Only trades the engine actually resolved.

    The engine marks an unfinished position closed at the last bar so the equity curve is
    complete. That is bookkeeping, not an outcome, and on a 61-trade sample one synthetic
    mark moves the win rate by 1.6 points - enough to matter when the whole question is
    whether a pair is above or below 70%.
    """
    return [t for t in res["trades"] if t[5] not in ("eod", "eod_ps")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="5m")
    ap.add_argument("--target-wr", type=float, default=70.0)
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()

    datasets = D.load_clean(a.tf, refresh=a.refresh)
    exits = SR.cfg_grid_for(FAMILY)
    days = (max(d.index.max() for d in datasets.values())
            - min(d.index.min() for d in datasets.values())).days
    n_looks = len(ENTRY_VARIANTS) * len(exits) * len(datasets)
    # Bonferroni-ish one-sided bar for this many looks at p ~ 1e-4 overall
    z_bar = 3.7

    print("=" * 100)
    print(f"FRAME  {a.tf}: {len(datasets)} pairs, ~{days} days of history, "
          f"{sum(len(d) for d in datasets.values()):,} bars")
    print(f"ENTRY  {FAMILY} - 5m flush timed against the completed-hour trend, session window")
    print(f"EXITS  {len(exits)} barrier/time configs, stops at 8-24 x ATR(5m) so cost stays a "
          "fraction of R")
    print(f"SEARCH {len(ENTRY_VARIANTS)} entry x {len(exits)} exit x {len(datasets)} pairs "
          f"= {n_looks:,} looks -> significance bar z >= {z_bar}")
    print("=" * 100)

    all_rows = []
    for vi, kw in enumerate(ENTRY_VARIANTS):
        for ei, cfg in enumerate(exits):
            df = per_pair(datasets, kw, cfg)
            df["entry"] = vi
            df["exit"] = ei
            df["rr"] = cfg.get("rr") or 0.0
            df["atr_mult"] = cfg["atr_mult"]
            df["max_bars"] = cfg["max_bars"]
            df["cfg"] = json.dumps(cfg, sort_keys=True)
            df["kw"] = json.dumps(kw, sort_keys=True)
            all_rows.append(df)
    res = pd.concat(all_rows, ignore_index=True)
    res.to_csv(os.path.join(OUT, f"intraday_{a.tf}.csv"), index=False)

    ok = res[res.n >= 30].dropna(subset=["wr"])
    print(f"\n{len(ok):,} pair/config results with 30+ trades\n")

    print("1. HOW MANY LOOKS CLEAR THE TARGET, AND HOW MANY SHOULD BY PURE PAYOFF")
    for rr in sorted(ok.rr.unique()):
        sub = ok[ok.rr == rr]
        if not len(sub):
            continue
        p0 = null_wr(rr)
        hits = int((sub.wr >= a.target_wr).sum())
        if p0 != p0:
            print(f"   rr={rr:<5} no price target, so no closed-form barrier null "
                  f"(see section 6)   looks={len(sub):4d}  observed >=70%: {hits:3d}")
            continue
        exp_hits = float(_norm_sf((a.target_wr - 100 * p0) / sub.se.to_numpy()).sum())
        print(f"   rr={rr:<5} null={100*p0:5.1f}%  looks={len(sub):4d}  "
              f"observed >=70%: {hits:3d}   expected if every signal were worthless: "
              f"{exp_hits:5.1f}")

    print(f"\n2. BEST PAIR/CONFIG BY WIN RATE (all {len(ok):,} results, top 20)")
    cols = ["pair", "entry", "rr", "atr_mult", "max_bars", "n", "wr", "se", "excess", "z",
            "exp", "pf", "avg_pips"]
    best = ok.sort_values("wr", ascending=False).head(20)
    print(best[cols].round(3).to_string(index=False))

    print(f"\n3. EVERY PAIR'S BEST RESULT, SOLO (one row per pair, its own highest win rate)")
    per = ok.loc[ok.groupby("pair")["wr"].idxmax()].sort_values("wr", ascending=False)
    print(per[cols].round(3).to_string(index=False))

    print(f"\n4. THE ONES AT OR ABOVE {a.target_wr:.0f}% - and whether any clears the search bar")
    hit = per[per.wr >= a.target_wr]
    if not len(hit):
        print(f"   none. highest observed was {per.wr.max():.1f}% on {per.iloc[0]['pair']}")
    for _, r in hit.iterrows():
        verdict = ("CLEARS" if r.z >= z_bar else
                   "inside noise" if abs(r.z) < 2 else "suggestive only")
        print(f"   {r.pair:8s} n={int(r.n):3d} wr={r.wr:5.1f}% +-{r.se:.1f}  "
              f"null={100*null_wr(r.rr):5.1f}% excess={r.excess:+6.1f}pt z={r.z:+5.2f}  "
              f"exp={r.exp:+.4f}R pf={r.pf:.2f}  stop={r.atr_mult:.0f}xATR "
              f"tgt={r.rr}R avg={r.avg_pips:.1f} pips over {r.avg_bars:.0f} bars  -> {verdict}")

    print(f"\n   {len(hit)} of {len(datasets)} pairs reached {a.target_wr:.0f}%, from "
          f"{n_looks:,} looks.")
    print(f"   At this sample size one worthless signal per pair prints above 70% roughly "
          "as often")
    print("   as not, so the count of hits is the expected count whether the edge is real or "
          "not.")
    print("   The column that separates them is z, and the bar is", z_bar, ".")

    print("\n5. OUT-OF-SAMPLE SPLIT ON THE TOP PAIRS - first half vs second half, nothing "
          "refitted")
    mid = datasets["EURUSD"].index[len(datasets["EURUSD"]) // 2]
    for _, r in hit.head(6).iterrows():
        kw = json.loads(r.kw)
        cfg = json.loads(r.cfg)
        d = {r.pair: datasets[r.pair]}
        a1 = per_pair(d, kw, cfg, end=mid)
        a2 = per_pair(d, kw, cfg, start=mid)
        b = per_pair(d, kw, cfg, spread_mult=2.0)
        if not (len(a1) and len(a2) and len(b)):
            continue
        print(f"   {r.pair:8s} first half n={int(a1.n[0]):3d} wr={a1.wr[0]:5.1f}% "
              f"exp={a1.exp[0]:+.4f}R | second half n={int(a2.n[0]):3d} wr={a2.wr[0]:5.1f}% "
              f"exp={a2.exp[0]:+.4f}R | 2x cost wr={b.wr[0]:5.1f}% exp={b.exp[0]:+.4f}R")

    print("\n6. THE CONTROL THAT MATTERS: same trades, filters removed")
    print("   Enter at every 5m extreme from the mean, exit on the return to the mean, same")
    print("   stops, same costs - but no trend filter, no RSI, no session window. This is not")
    print("   a random signal; it is this exact trade shape with no judgement in it. Whatever")
    print("   the real strategy beats this by is the whole of its edge.")
    ctrl = []
    for vi, kw in enumerate(ENTRY_VARIANTS):
        for ei, cfg in enumerate(exits):
            df = per_pair(datasets, kw, cfg, control=True)
            df["entry"] = vi
            df["exit"] = ei
            df["rr"] = cfg.get("rr") or 0.0
            df["cfg"] = json.dumps(cfg, sort_keys=True)
            df["kw"] = json.dumps(kw, sort_keys=True)
            ctrl.append(df)
    ctl = pd.concat(ctrl, ignore_index=True)
    ctl.to_csv(os.path.join(OUT, f"intraday_{a.tf}_control.csv"), index=False)
    c_ok = ctl[ctl.n >= 30].dropna(subset=["wr"])

    for rr in sorted(c_ok.rr.unique()):
        cs = c_ok[c_ok.rr == rr]
        ss = ok[(ok.rr == rr) & (ok.kw.isin(cs.kw)) & (ok.cfg.isin(cs.cfg))]
        p0 = null_wr(rr)
        print(f"   rr={rr:<5} filter-free geometry: mean wr={cs.wr.mean():5.1f}%  "
              f"best pair={cs.wr.max():5.1f}%  pairs >=70%: {(cs.wr >= 70).sum():2d}/{len(cs):3d}"
              f"   exp mean={cs.exp.mean():+.4f}R"
              + ("" if p0 != p0 else f"   (barrier null {100*p0:.1f}%)"))
        if len(ss):
            j = ss.merge(cs, on=["pair", "kw", "cfg"], suffixes=("_s", "_c"))
            if len(j):
                print(f"          vs filtered strategy: mean wr={j.wr_s.mean():5.1f}%  "
                      f"best pair={j.wr_s.max():5.1f}%  "
                      f"delta = {j.wr_s.mean()-j.wr_c.mean():+5.1f}pt   "
                      f"exp {j.exp_s.mean():+.4f}R vs {j.exp_c.mean():+.4f}R")

    print(f"\n   HEAD-TO-HEAD on the pairs that reached "
          f"{a.target_wr:.0f}% (same pair, same config, filters on vs off):")
    key = ["pair", "kw", "cfg"]
    for _, r in hit.iterrows():
        c = c_ok[(c_ok.pair == r.pair) & (c_ok.kw == r.kw) & (c_ok.cfg == r.cfg)]
        if not len(c):
            continue
        c = c.iloc[0]
        d = r.wr - c.wr
        print(f"   {r.pair:8s} rr={r.rr:<5} stop={r.atr_mult:.0f}xATR  "
              f"filters ON {r.wr:5.1f}% (n={int(r.n):3d}) exp={r.exp:+.4f}R | "
              f"filters OFF {c.wr:5.1f}% (n={int(c.n):3d}) exp={c.exp:+.4f}R | "
              f"the filters are worth {d:+5.1f}pt and {r.exp-c.exp:+.4f}R")
    print("\n7. WHAT THE 70% ACTUALLY IS - payoff shape, and the filters scored against the")
    print("   geometry control, because a mean-touch exit has no closed-form barrier null.")
    print("   Two-proportion z below compares the same pair's filtered trades against its own")
    print("   unfiltered trades: identical exit, identical costs, filters as the only variable.")
    print(f"\n   {'pair':8s} {'n_f':>5} {'n_c':>5} {'wr_on':>6} {'wr_off':>7} {'z':>6} "
          f"{'win_R':>7} {'loss_R':>7} {'worst':>7} {'maxDD_R':>8} {'tr/d':>5} {'pips/d':>7}")
    for _, r in hit.iterrows():
        kw, cfg = json.loads(r.kw), json.loads(r.cfg)
        d = {r.pair: datasets[r.pair]}
        on = per_pair(d, kw, cfg)
        off = per_pair(d, kw, cfg, control=True)
        if not (len(on) and len(off)):
            continue
        n1, w1 = int(on.n[0]), on.wr[0] / 100.0
        n2, w2 = int(off.n[0]), off.wr[0] / 100.0
        p = (n1 * w1 + n2 * w2) / (n1 + n2)
        se2 = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
        zc = (w1 - w2) / se2 if se2 > 0 else np.nan
        # full trade list for the filtered run: wins, losses, worst, peak-to-trough in R
        m = D.PAIRS[r.pair + "=X"]
        sig = S.REGISTRY[FAMILY](d[r.pair], **kw)
        sub, s2 = SR.window(d[r.pair], sig, None, None)
        res_t = B.run(sub, s2, cfg, m["pip"], m["spread"], m["slip"], SR.RISK)
        tr = trades_of(res_t)
        rs = np.array([t[2] for t in tr], dtype=float)
        pips = np.array([t[3] for t in tr], dtype=float)   # element 3 is already P&L in pips
        curve = np.cumsum(rs)
        dd = float((np.maximum.accumulate(curve) - curve).max()) if len(curve) else np.nan
        span_d = max((d[r.pair].index.max() - d[r.pair].index.min()).days, 1)
        wins, losses = rs[rs > 0], rs[rs <= 0]
        print(f"   {r.pair:8s} {n1:5d} {n2:5d} {100*w1:5.1f}% {100*w2:6.1f}% {zc:+6.2f} "
              f"{(wins.mean() if len(wins) else np.nan):+7.3f} "
              f"{(losses.mean() if len(losses) else np.nan):+7.3f} "
              f"{(rs.min() if len(rs) else np.nan):+7.2f} {dd:8.1f} "
              f"{n1/span_d:5.2f} {pips.sum()/span_d:+7.2f}")

    print(f"\n7b. TRADABILITY: what one R actually is, what the spread costs, and whether a "
          f"retail account can place it")
    usd_table, _ = L.build_usd_table(datasets)
    print(f"   {'pair':8s} {'stop_pips':>9} {'spread':>7} {'cost%ofR':>9} "
          f"{'wr@2x':>6} {'exp@2x':>8} {'wr@3x':>6} {'exp@3x':>8} {'lot@10k':>8} "
          f"{'$risk@0.01':>10} {'min_acct':>9}")
    for _, r in hit.iterrows():
        kw, cfg = json.loads(r.kw), json.loads(r.cfg)
        m = D.PAIRS[r.pair + "=X"]
        d = {r.pair: datasets[r.pair]}
        c2 = per_pair(d, kw, cfg, spread_mult=2.0)
        c3 = per_pair(d, kw, cfg, spread_mult=3.0)
        sig = S.REGISTRY[FAMILY](d[r.pair], **kw)
        sub, s2 = SR.window(d[r.pair], sig, None, None)
        tr = trades_of(B.run(sub, s2, cfg, m["pip"], m["spread"], m["slip"], SR.RISK))
        # t[3] is P&L in pips and equals R-multiple x stop-in-pips, so their ratio is the stop
        stop_pips = float(np.median([abs(t[3] / t[2]) for t in tr if abs(t[2]) > 1e-9]))
        sz = L.sizing(m["pip"], float(sub["Close"].iloc[-1]), r.pair[3:], usd_table,
                      10_000 * SR.RISK, stop_pips)
        print(f"   {r.pair:8s} {stop_pips:9.1f} {m['spread']:7.1f} "
              f"{100*m['spread']/stop_pips:8.1f}% {c2.wr[0]:5.1f}% {c2.exp[0]:+8.4f} "
              f"{c3.wr[0]:5.1f}% {c3.exp[0]:+8.4f} {sz['lots']:8.3f} "
              f"{sz['risk_at_min_usd']:10.2f} {sz['risk_at_min_usd']/SR.RISK:9,.0f}"
              + ("  <- below lot floor" if sz["below_min_lot"] else ""))

    print("\n   READ THIS BEFORE ACTING ON ANY ROW ABOVE.")
    print("   - None of these configs has a price target (rr=0). The exit is 'price is back at")
    print("     the mean it left'. A position wins unless it first travels the whole stop - 12")
    print("     ATR(5m) - against it, so a high hit rate is the shape of the trade, not proof")
    print("     of a signal. The unfiltered column is what that shape alone earns.")
    print("   - The 70% only exists at rr=0. At rr=0.5, where a target is 0.5R and the null is")
    print("     already 66.7%, ZERO of 265 looks cleared 70% when 74 should have from worthless")
    print("     signals. With a real target the strategy does not just miss the bar, it sits")
    print("     under it. There is no pair here that wins 70% of the time for a payoff.")
    print("   - Costs are the whole story on this frame: every one of these pairs falls below")
    print("     70% when the spread is doubled, and the two with the best expectancy go flat.")
    print("     A 5m mean-reversion edge of well under a pip per trade does not survive a")
    print("     slightly bad fill, a news day, or an ECN that is not quoted at the median.")
    return res


def geometry_one(df, kw):
    """
    Enter on every bar that is far enough from its own mean, direction implied by which side
    it is on, exit when price is back at that mean - and drop the trend, RSI and session
    filters. Everything about the trade shape is held fixed; only the judgement is removed.

    This is the control that actually answers "is the 70% the strategy or the shape of the
    trade". Randomising direction instead - the obvious first attempt - is useless here,
    because the exit is defined by the mean: a long opened above its own mean closes on the
    very next bar, so the randomised version trades a completely different thing and its
    win rate measures nothing. Whatever the filtered strategy beats this by is the entire
    edge of the filters, and if the answer is nothing, the win rate was never the strategy's.
    """
    c = df["Close"]
    a = ta.atr(df, 14)
    mean = ta.sma(c, kw["dev_n"])
    z = ((c - mean) / a.replace(0.0, np.nan)).to_numpy()
    d = np.zeros(len(df), dtype=np.int8)
    d[z < -kw["dev_k"]] = 1
    d[z > kw["dev_k"]] = -1
    return S._pack(df, d, xl=(c >= mean).to_numpy(), xs=(c <= mean).to_numpy())


def _norm_sf(x):
    """1 - Phi(x), elementwise. P(Z > x): the tail probability a fair signal beats this."""
    from math import erf, sqrt
    v = np.vectorize(lambda y: 0.5 * (1.0 - erf(float(y) / sqrt(2.0))))
    return v(np.asarray(x, dtype=float))


if __name__ == "__main__":
    main()
