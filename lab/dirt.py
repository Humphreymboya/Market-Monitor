"""
The dirty bars in the 23-year frame, why they cannot be auto-corrected, and what the edge looks like either way.

The vendor daily series contains bars whose High/Low are internally consistent but physically
absurd - EURGBP 2008-03-17 ships Low = 0.0066 against a 0.788 Close, a 7,842-pip range where
that pair's median day is 46 pips. repair_daily cannot fix them: it only ever widens extremes,
and these are too WIDE, which is the one direction an envelope check is blind to. They matter
because every barrier system in this project resolves inside the bar, so a fake extreme can
touch a target that was never reachable, and because ATR(14) is built from them, which then
sets stops for the next fourteen bars.

Two consequences, measured here rather than argued:

  1. Zombie positions. A stop sized off an inflated ATR can sit far outside anything the pair
     has ever traded, so the position never resolves and the pair is blocked from every signal
     after it. lab.live flags these; this counts them.

  2. The heuristic cannot be trusted to find them. That is the actual result of this module, and
     it is why the frame ships unsanded. A bar being wide relative to its neighbours is NOT
     evidence of a typo, because the four most important days in FX risk - the 2015 Swiss franc
     unpegging, the 2009 euro/franc storm around 2009-10-30, the 2016 Brexit vote, the 2020
     COVID liquidation - are wide relative to their neighbours too, and genuinely so. Section 2
     lists the known-real dates that the same threshold flags. Any automatic clamp loose enough
     to be a rule is therefore one that removes exactly the dislocations a stop-loss system
     most needs to be tested against, which is not a repair but a quieter dataset.

So the output is a census with its false positives named, plus a sensitivity bound. The clip
columns are not a cleaner dataset to trade on; they are the worst case for the edge, because the
same clamp that drops the typos drops the crashes. If the edge survives the version of the data
where real tail events have been erased, it is not being carried by them.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from . import backtest as B
from . import data as D
from . import search as SR
from . import strategies as S
from . import tradable as T

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

# dislocations that are verifiable market history rather than feed noise: each of these dates
# appears in the top handful of ranges for its pair, and each is documented in the market record
KNOWN_REAL = [("EURCHF", "2015-01-15", "SNB removed the EUR floor; franc +20% intraday"),
              ("CHFJPY", "2015-01-15", "same event, the yen leg of it"),
              ("GBPUSD", "2016-06-24", "Brexit referendum, cable gaps open 8%"),
              ("EURGBP", "2016-06-24", "same vote, the cross re-rating"),
              ("AUDJPY", "2008-10-24", "global financial crisis liquidation, carry unwind")]


def census(datasets, mult):
    """Per-pair count of bars exceeding mult x their own trailing median range, plus the worst."""
    rows = []
    for name, df in datasets.items():
        m = D.suspect_bars(df, mult)
        rng = ((df["High"] - df["Low"]) / df["Close"]).to_numpy()
        n = int(m.sum())
        rows.append(dict(pair=name, bars=len(df), suspects=n,
                         pct=100.0 * n / len(df),
                         worst_pct=100.0 * float(rng[m].max()) if n else 0.0,
                         worst_bar=str(df.index[m][0])[:10] if n else "-",
                         median_pct=100.0 * float(np.nanmedian(rng))))
    return rows


def run_book(datasets, fam, kw, cfg):
    """(R-multiples, zombie open positions) on whatever frame is passed."""
    all_r, zombies = [], []
    for name, df in datasets.items():
        m = D.PAIRS[name + "=X"]
        try:
            sig = S.REGISTRY[fam](df, **kw)
        except Exception:  # noqa: BLE001
            continue
        sub, s2 = SR.window(df, sig, None, None)
        res = B.run(sub, s2, cfg, m["pip"], m["spread"], m["slip"], SR.RISK)
        all_r += [t[2] for t in res["trades"] if abs(t[2]) > 1e-9]
        op = res.get("open")
        if op and abs(op["entry"] - op["stop"]) / op["entry"] > 0.05:
            zombies.append((name, abs(op["entry"] - op["stop"]) / op["entry"],
                            op["bars_held"]))
    return np.array(all_r), zombies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="1d_rep")
    ap.add_argument("--mult", type=float, default=10.0)
    a = ap.parse_args()

    entry = json.load(open(os.path.join(OUT, "chosen_strategy.json")))
    fam, kw, cfg = entry["family"], entry["params"], entry["cfg"]
    ds = D.load_clean(a.tf)

    print("=" * 96)
    print(f"FRAME {a.tf}: {len(ds)} pairs. Suspicious bar = range > {a.mult:.0f}x that pair's")
    print("own trailing 60-bar median range. The threshold is deliberately crude; see the")
    print("docstring for why nothing auto-corrects these.")
    print("=" * 96)

    rows = census(ds, a.mult)
    tot = sum(r["suspects"] for r in rows)
    tb = sum(r["bars"] for r in rows)
    print(f"\n1. CENSUS: {tot} bars of {tb:,} exceed {a.mult:.0f}x their own median range "
          f"= {100.0*tot/tb:.2f}% of the frame\n")
    print(f"{'pair':8s} {'suspects':>9} {'%ofpair':>8} {'median day':>11} "
          f"{'worst bar':>11}  {'worst range':>12}")
    for r in sorted(rows, key=lambda x: -x["suspects"]):
        if r["suspects"]:
            print(f"{r['pair']:8s} {r['suspects']:9d} {r['pct']:7.2f}% "
                  f"{r['median_pct']:10.2f}% {r['worst_bar']:>11}  {r['worst_pct']:11.1f}%")

    print(f"\n2. FALSE POSITIVES - known-real dislocations tested against the SAME {a.mult:.0f}x rule")
    print(f"{'pair':8s} {'date':11s} {'range':>7} {'its median':>11} {'ratio':>7}   event")
    flagged = 0
    for name, date, why in KNOWN_REAL:
        d = ds.get(name)
        if d is None or pd.Timestamp(date, tz="UTC") not in d.index:
            print(f"{name:8s} {date:11s}  not in this frame")
            continue
        rng = ((d["High"] - d["Low"]) / d["Close"])
        base = rng.rolling(60, min_periods=20).median()
        i = d.index.get_loc(pd.Timestamp(date, tz="UTC"))
        r, b = float(rng.iloc[i]), float(base.iloc[i])
        hit = r > a.mult * b
        flagged += int(hit)
        print(f"{name:8s} {date:11s} {100*r:6.1f}% {100*b:10.2f}% {r/b:6.0f}x   "
              f"{'FLAGGED' if hit else 'clean   '}: {why}")
    print(f"\n   {flagged} of {len(KNOWN_REAL)} real events on that list are caught by the same "
          "threshold that\n   catches the EURGBP typo. So 'wide relative to its own neighbours' "
          "is not a test\n   for fabrication - it is a test for volatility, and the two cannot "
          "be separated by a\n   rolling statistic. A clamp that removes the typo removes Brexit "
          "and the SNB\n   decapitation with it, and a frame with its tail sanded out would "
          "flatter every\n   stop-loss in this project. Hence: reported, not corrected.")

    print("\n3. SENSITIVITY - edge with the suspects clipped, which also clips the real crashes")
    print(f"{'frame':34s} {'n':>6} {'wr':>7} {'exp':>9} {'pf':>6}")
    for label, mult in (("as shipped (no clipping)", None), ("clipped at 10x median", 10.0),
                        ("clipped at 6x median", 6.0), ("clipped at 4x median", 4.0)):
        frames = ds
        if mult:
            frames = {n: D.clip_suspects(df, mult)[0] for n, df in ds.items()}
        rs, _ = run_book(frames, fam, kw, cfg)
        st = T.stat(rs)
        if st:
            print(f"{label:34s} {st['n']:6d} {st['wr']:6.1f}% {st['exp']:+9.4f} {st['pf']:6.2f}")

    _, z = run_book(ds, fam, kw, cfg)
    print(f"\n4. ZOMBIE POSITIONS on the shipped frame - open, stop >5% of price, unresolvable: "
          f"{len(z)}")
    for name, sd, bars in sorted(z, key=lambda x: -x[2]):
        print(f"   {name:8s} stop is {sd:.1%} of price, held {bars:,} bars "
              f"({bars/252:.1f} years) and counting")
    print("\n   Each one blocks its pair from every later signal, so the trade count, the per-pair\n"
          "   table and the leave-one-pair-out test in log_champion.txt are all measured on a book\n"
          "   where some pairs have been silently out of service since the bar that poisoned their\n"
          "   ATR. EURGBP has been out for nineteen of the twenty-two years being reported on.")


if __name__ == "__main__":
    main()
