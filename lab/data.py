"""FX data layer: download + cache real OHLC from Yahoo Finance."""
import os
import numpy as np
import pandas as pd
import yfinance as yf

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

# pip size + realistic retail all-in spread (pips) and per-side slippage (pips)
PAIRS = {
    "EURUSD=X": dict(name="EURUSD", pip=0.0001, spread=1.1, slip=0.3),
    "GBPUSD=X": dict(name="GBPUSD", pip=0.0001, spread=1.4, slip=0.3),
    "USDJPY=X": dict(name="USDJPY", pip=0.01, spread=1.2, slip=0.3),
    "USDCHF=X": dict(name="USDCHF", pip=0.0001, spread=1.6, slip=0.3),
    "AUDUSD=X": dict(name="AUDUSD", pip=0.0001, spread=1.4, slip=0.3),
    "USDCAD=X": dict(name="USDCAD", pip=0.0001, spread=1.8, slip=0.3),
    "NZDUSD=X": dict(name="NZDUSD", pip=0.0001, spread=1.9, slip=0.3),
    "EURJPY=X": dict(name="EURJPY", pip=0.01, spread=2.0, slip=0.4),
    "GBPJPY=X": dict(name="GBPJPY", pip=0.01, spread=2.8, slip=0.4),
    "EURGBP=X": dict(name="EURGBP", pip=0.0001, spread=1.8, slip=0.4),
    "AUDJPY=X": dict(name="AUDJPY", pip=0.01, spread=2.4, slip=0.4),
    "CADJPY=X": dict(name="CADJPY", pip=0.01, spread=2.6, slip=0.4),
    "NZDJPY=X": dict(name="NZDJPY", pip=0.01, spread=3.0, slip=0.4),
    "EURCHF=X": dict(name="EURCHF", pip=0.0001, spread=2.2, slip=0.4),
    # --- second universe: more instruments, so the win-rate estimate has fewer standard
    # errors around it rather than more hope
    "AUDCAD=X": dict(name="AUDCAD", pip=0.0001, spread=1.9, slip=0.4),
    "AUDNZD=X": dict(name="AUDNZD", pip=0.0001, spread=2.2, slip=0.4),
    "AUDCHF=X": dict(name="AUDCHF", pip=0.0001, spread=2.4, slip=0.4),
    "CADCHF=X": dict(name="CADCHF", pip=0.0001, spread=2.3, slip=0.4),
    "EURAUD=X": dict(name="EURAUD", pip=0.0001, spread=2.3, slip=0.4),
    "EURCAD=X": dict(name="EURCAD", pip=0.0001, spread=2.4, slip=0.4),
    "GBPCAD=X": dict(name="GBPCAD", pip=0.0001, spread=2.5, slip=0.4),
    "GBPNZD=X": dict(name="GBPNZD", pip=0.0001, spread=2.9, slip=0.5),
    "GBPCHF=X": dict(name="GBPCHF", pip=0.0001, spread=2.7, slip=0.5),
    "CHFJPY=X": dict(name="CHFJPY", pip=0.01, spread=2.6, slip=0.4),
    "NZDCAD=X": dict(name="NZDCAD", pip=0.0001, spread=2.6, slip=0.5),
    "USDSGD=X": dict(name="USDSGD", pip=0.0001, spread=2.4, slip=0.5),
}

# timeframe -> (yahoo interval, lookback)
TFS = {"1d": ("1d", "1990-01-01"), "1h": ("1h", "730d"), "15m": ("15m", "60d"),
       "5m": ("5m", "60d")}


def _flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df


def _as_utc(df):
    """Normalise the index to a UTC DatetimeIndex (CSV round-trips can lose the dtype)."""
    idx = pd.to_datetime(df.index, utc=True, format="mixed")
    df = df.copy()
    df.index = idx
    return df.sort_index()


def fetch(symbol, tf="1d", force=False):
    """Return OHLC DataFrame for symbol/tf, using local cache when possible."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{PAIRS[symbol]['name']}_{tf}.csv")
    interval, look = TFS[tf]
    if os.path.exists(path) and not force:
        df = _as_utc(pd.read_csv(path, index_col=0))
        if df.index.max().strftime("%Y-%m-%d") >= (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=3)).strftime("%Y-%m-%d"):
            return df
    kw = dict(period=look) if look.endswith("d") else dict(start=look)
    raw = yf.download(symbol, interval=interval, progress=False, **kw)
    if raw is None or raw.empty:
        if os.path.exists(path):
            return _as_utc(pd.read_csv(path, index_col=0))
        raise RuntimeError(f"no data for {symbol} {tf}")
    df = _flatten(raw).dropna(subset=["Open", "High", "Low", "Close"])
    df = df[~df.index.duplicated(keep="last")]
    df = _as_utc(df[["Open", "High", "Low", "Close"]].astype(float))
    df.to_csv(path)
    return df


def integrity(df):
    """Fraction of bars whose Close falls outside their own High/Low envelope."""
    return float(((df["Close"] > df["High"]) | (df["Close"] < df["Low"])
                  | (df["High"] < df["Low"])).mean())


def suspect_bars(df, mult=10.0, window=60):
    """Mask of bars whose High-Low range is implausibly wide for that pair's own history.

    These survive repair_daily by construction: the envelope check only catches extremes that
    are too NARROW, and a typo that is too wide is internally consistent. See repair_daily for
    why nothing is allowed to auto-correct them here.
    """
    rng = (df["High"] - df["Low"]) / df["Close"]
    base = rng.rolling(window, min_periods=20).median()
    return (rng > mult * base) & base.notna()


def clip_suspects(df, mult=10.0):
    """Sensitivity-test transform: clamp suspect extremes toward the Close, re-enveloping after.

    NOT the shipped frame. This exists so the question 'does the edge depend on the dirty
    bars' can be measured rather than argued; lab.dirt reports both sides of that comparison.
    Because the threshold is a multiple of a rolling median it also catches genuine dislocations
    (EURCHF 2015-01-15 is one), so results here are a lower bound on robustness, not a cleaner
    dataset.
    """
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    hi = np.maximum.reduce([h, c, o])
    lo = np.minimum.reduce([l, c, o])
    rng = pd.Series(np.asarray(hi) - np.asarray(lo), index=df.index)
    cap = (mult * rng.rolling(60, min_periods=20).median()).to_numpy()
    bad = (rng.to_numpy() > cap) & ~np.isnan(cap)
    half = cap / 2.0
    out = df[["Open", "High", "Low", "Close"]].copy()
    out["High"] = np.maximum.reduce([np.where(bad, np.minimum(hi, c + half), hi), c, o])
    out["Low"] = np.minimum.reduce([np.where(bad, np.maximum(lo, c - half), lo), c, o])
    return out, int(bad.sum())


def load_all(tf="1d", refresh=False):
    out = {}
    for sym in PAIRS:
        try:
            d = fetch(sym, tf, force=refresh)
            if len(d) > 500:
                out[PAIRS[sym]["name"]] = d
        except Exception as e:  # noqa: BLE001
            print(f"  ! {sym} {tf}: {type(e).__name__} {e}")
    return out


def repair_daily(df):
    """
    Make a vendor daily bar internally consistent, without inventing a favourable extreme.

    A bar whose Close (or Open) sits outside its own High/Low is a reporting failure: the
    true range was at least as wide as the price we do know. So the minimal, one-sided
    repair is to widen High up to the stray Close/Open and Low down. Widening can only
    ever make a stop *more* likely to be hit and a target more likely to be reached, so any
    edge that survives the repaired frame survived a harder test than the raw one.

    This deliberately does NOT narrow anything, and it is worth being explicit about why
    not, because the frame still contains roughly 214 bars whose High/Low are internally
    consistent but physically absurd - EURGBP 2008-03-17 ships Low = 0.0066 against a 0.788
    Close, a 7,842-pip range where the pair's median day is 46. Such a bar is poison for a
    barrier backtest: the fake extreme can touch a target that was never reachable, and it
    inflates ATR(14), which then sets every subsequent stop. One of them put EURGBP's stop
    2,300 pips (29% of price) above its entry and left the position unresolvable for 4,852
    bars, which is what lab.live is now careful about printing.

    The obvious fix - clamp any bar exceeding N times its own rolling median range - was
    tried and is wrong. EURCHF 2015-01-15, the Swiss franc unpegging, has a range 272x its
    own 60-bar median: by that statistic the largest genuine risk event in twenty-two years
    of the sample is indistinguishable from a data-entry error, so a rule that removes one
    removes the other. Sanding the crash out of the frame is not a repair, it is exactly the
    favourable extreme this function exists to avoid inventing. lab.dirt enumerates the
    suspects instead; the sensitivity question - whether any edge survives with them clipped
    anyway - is measured, not assumed.
    Returns (repaired_frame, fraction_of_bars_changed, median_widening_in_pips).
    """
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    hi = np.maximum.reduce([h, c, o])
    lo = np.minimum.reduce([l, c, o])
    changed = ((hi > h) | (lo < l)).to_numpy()
    pip = 0.01 if float(df["Close"].median()) > 20 else 0.0001
    widen = ((hi - h) + (l - lo)).to_numpy() / pip
    out = df[["Open", "High", "Low", "Close"]].copy()
    out["High"], out["Low"] = hi, lo
    return out, float(changed.mean()), float(np.median(widen[changed])) if changed.any() else 0.0


def load_clean(tf="1h", min_bars=18, refresh=False):
    """
    Bars safe for pip-level stop/target logic.

    The vendor 1d series is internally inconsistent (a few percent of daily bars have a
    Close outside their own High/Low, and day-to-day gaps average ~35 pips on EURUSD, which
    is an aggregation artefact rather than a real move). Anything that resolves a stop
    inside a bar will manufacture edge from that. So the daily frame used for real testing
    is rebuilt from the hourly feed, where every bar passes the envelope check and the only
    gaps are the genuine weekend ones.

    '1d_clean' is the truth we can see (2 years, from hourly). '1d_rep' applies the minimal
    one-sided envelope repair to the 23-year vendor series, which is the only way to get an
    out-of-sample pool large enough for the win-rate estimate to mean anything.
    """
    if tf == "1d":
        return load_all("1d", refresh)  # vendor daily: long history, envelope-imperfect
    if tf == "1d_rep":
        rep = {}
        for name, df in load_all("1d", refresh).items():
            r, frac, med = repair_daily(df)
            r.attrs["repaired_frac"], r.attrs["median_widen_pips"] = frac, med
            rep[name] = r
        return rep
    if tf == "1h":
        return load_all("1h", refresh)
    if tf == "1d_clean":
        hrs = load_all("1h", refresh)
        out = {}
        for name, df in hrs.items():
            sh = df.copy()
            sh.index = sh.index + pd.Timedelta(hours=2)   # 22:00->22:00 UTC sessions
            g = sh.resample("1D")
            d = pd.DataFrame({
                "Open": g["Open"].first(), "High": g["High"].max(),
                "Low": g["Low"].min(), "Close": g["Close"].last(), "nbars": g.size(),
            }).dropna()
            d = d[d["nbars"] >= min_bars]
            out[name] = d[["Open", "High", "Low", "Close"]].astype(float)
        return out
    if tf == "15m":
        return load_all("15m")
    if tf == "5m":
        # ~81 days of history, not 23 years: Yahoo caps 5m at 60d. The frame is short in
        # TIME even though it is dense in bars, which is the binding limit on any intraday
        # verdict here - see lab.intraday for how that is handled.
        return load_all("5m", refresh)
    raise ValueError(tf)


if __name__ == "__main__":
    for tf in ("1d", "1h"):
        d = load_all(tf, refresh=True)
        print(f"\n{tf}: {len(d)} pairs")
        for k, v in d.items():
            print(f"  {k:8s} {len(v):6d} bars  {v.index.min()} -> {v.index.max()}")
