"""Strategy families. Each returns a signal pack computed only from bar-close data."""
import numpy as np
import pandas as pd
from . import indicators as ta


def _pack(df, dirn, stop=None, xl=None, xs=None, atr_n=14, tgt=None):
    a = ta.atr(df, atr_n)
    return {
        "dir": np.asarray(dirn, dtype=np.int8),
        "stop": None if stop is None else np.asarray(stop, dtype=float),
        "tgt": None if tgt is None else np.asarray(tgt, dtype=float),
        "xl": None if xl is None else np.asarray(xl, dtype=bool),
        "xs": None if xs is None else np.asarray(xs, dtype=bool),
        "atr": a.to_numpy(float),
    }


# ------------------------------------------------------------------ 1. RSI-2 mean reversion (Connors style)
def rsi2_mr(df, rsi_n=2, buy_th=10, sell_th=90, trend=200, exit_sma=5, atr_mult=6.0,
            atr_frac=0.0, adx_max=0, day_filter=None):
    c = df["Close"]
    r = ta.rsi(c, rsi_n)
    tr = ta.sma(c, trend)
    ex = ta.sma(c, exit_sma)
    long_sig = (c > tr) & (r < buy_th)
    short_sig = (c < tr) & (r > sell_th)
    if adx_max:
        adx, _, _ = ta.adx(df)
        f = adx < adx_max
        long_sig &= f
        short_sig &= f
    d = np.where(long_sig.to_numpy(), 1, np.where(short_sig.to_numpy(), -1, 0)).astype(np.int8)
    xl = (c > ex).to_numpy()
    xs = (c < ex).to_numpy()
    return _pack(df, d, xl=xl, xs=xs)


# ------------------------------------------------------------------ 2. Bollinger band fade
def bb_fade(df, bb_n=20, k=2.0, rsi_n=14, rsi_buy=35, rsi_sell=65, trend=0,
             adx_max=25):
    c = df["Close"]
    mid, up, lo = ta.bbands(c, bb_n, k)
    r = ta.rsi(c, rsi_n)
    lsg = (c < lo) & (r < rsi_buy)
    ssg = (c > up) & (r > rsi_sell)
    if trend:
        tr = ta.sma(c, trend)
        lsg &= c > tr
        ssg &= c < tr
    if adx_max:
        adx, _, _ = ta.adx(df)
        f = adx < adx_max
        lsg &= f
        ssg &= f
    d = np.where(lsg.to_numpy(), 1, np.where(ssg.to_numpy(), -1, 0)).astype(np.int8)
    xl = (c >= mid).to_numpy()      # exit into the mean
    xs = (c <= mid).to_numpy()
    return _pack(df, d, xl=xl, xs=xs)


# ------------------------------------------------------------------ 3. Donchian breakout (trend)
def donchian_bo(df, n=20, exit_n=10, adx_min=20, atr_mult=3.0, trail=True, regime_lo=0.2, regime_hi=0.9):
    h, l, mid = ta.donchian(df, n)
    xh, xl_, _ = ta.donchian(df, exit_n)
    adx, pdi, mdi = ta.adx(df)
    a = ta.atr(df, 14)
    # volatility regime filter: only when ATR percentile is mid-to-high
    pct = a.rolling(250, min_periods=100).rank(pct=True)
    f = (pct > regime_lo) & (pct < regime_hi)
    lsg = (df["Close"] > h.shift(1)) & (adx > adx_min) & (pdi > mdi) & f
    ssg = (df["Close"] < l.shift(1)) & (adx > adx_min) & (mdi > pdi) & f
    d = np.where(lsg.to_numpy(), 1, np.where(ssg.to_numpy(), -1, 0)).astype(np.int8)
    exl = (df["Close"] < xl_.shift(1)).to_numpy()
    exs = (df["Close"] > xh.shift(1)).to_numpy()
    return _pack(df, d, xl=exl, xs=exs)


# ------------------------------------------------------------------ 4. Pullback-in-uptrend
def pullback(df, fast=50, slow=200, down_n=3, rsi_max=45, rsi_n=14, use_rsi=True,
             adx_min=0, exit_sma=0, break_low=True):
    c, l = df["Close"], df["Low"]
    f1, f2 = ta.sma(c, fast), ta.sma(c, slow)
    up = (c > f1) & (f1 > f2)
    dn = (c < f1) & (f1 < f2)
    downcloses = (c.diff() < 0).rolling(down_n).sum() >= down_n
    d = np.zeros(len(df), dtype=np.int8)
    lsg = up & downcloses
    ssg = dn & (c.diff() > 0).rolling(down_n).sum().ge(down_n)
    if use_rsi:
        r = ta.rsi(c, rsi_n)
        lsg &= r < rsi_max
        ssg &= r > 100 - rsi_max
    if adx_min:
        adx, pdi, mdi = ta.adx(df)
        lsg &= adx > adx_min
        ssg &= adx > adx_min
    d[lsg.to_numpy()] = 1
    d[ssg.to_numpy()] = -1
    stop = np.full(len(df), np.nan)
    if break_low:
        stop[lsg.to_numpy()] = l[lsg].to_numpy() - 0.3 * ta.atr(df, 14)[lsg].to_numpy()
        stop[ssg.to_numpy()] = df["High"][ssg].to_numpy() + 0.3 * ta.atr(df, 14)[ssg].to_numpy()
    if exit_sma:
        e = ta.sma(c, exit_sma)
        xl = (c < e).to_numpy()
        xs = (c > e).to_numpy()
    else:
        xl = (c.diff() > 0).rolling(2).sum().ge(2).to_numpy()
        xs = (c.diff() < 0).rolling(2).sum().ge(2).to_numpy()
    return _pack(df, d, stop=stop, xl=xl, xs=xs)


# ------------------------------------------------------------------ 5. Session breakout (hourly)
def session_bo(df, sess_start=8, sess_len=2, lookback_bars=None, rr=1.0, adx_min=0):
    idx = df.index
    if idx.tz is not None:
        hh = idx.tz_convert("UTC").hour.to_numpy()
    else:
        hh = idx.hour.to_numpy()
    win = (hh >= sess_start) & (hh < sess_start + sess_len)
    if not win.any():
        return _pack(df, np.zeros(len(df), dtype=np.int8))
    # build per-day opening range, then trade breakouts later that same day
    dates = idx.normalize()
    hi = df["High"].where(win).groupby(dates).transform("max")
    lo = df["Low"].where(win).groupby(dates).transform("min")
    after = (hh >= sess_start + sess_len) & (hh < 21)
    adx, pdi, mdi = ta.adx(df)
    lsg = after & (df["Close"] > hi).to_numpy() & hi.notna().to_numpy()
    ssg = after & (df["Close"] < lo).to_numpy() & lo.notna().to_numpy()
    if adx_min:
        lsg &= adx > adx_min
        ssg &= adx > adx_min
    d = np.where(lsg.to_numpy(), 1, np.where(ssg.to_numpy(), -1, 0)).astype(np.int8)
    stop = np.full(len(df), np.nan)
    stop[lsg.to_numpy()] = (hi - 0.25 * (hi - lo))[lsg].to_numpy()
    stop[ssg.to_numpy()] = (lo + 0.25 * (hi - lo))[ssg].to_numpy()
    # flat handled by the engine's time stop (max_bars) in the parameter grid
    return _pack(df, d, stop=stop)


# ------------------------------------------------------------------ 6. Deliberate win-rate inflated mean reversion
def tiny_tp_mr(df, rsi_n=2, buy_th=15, trend=200, atr_mult=8.0, rr=0.4):
    """Very small target vs very wide stop: manufactures a high win rate.
    Included on purpose so the search can be judged on expectancy, not win rate."""
    c = df["Close"]
    r = ta.rsi(c, rsi_n)
    tr = ta.sma(c, trend)
    lsg = (c > tr) & (r < buy_th)
    ssg = (c < tr) & (r > 100 - buy_th)
    d = np.where(lsg.to_numpy(), 1, np.where(ssg.to_numpy(), -1, 0)).astype(np.int8)
    return _pack(df, d, xl=lsg.to_numpy(), xs=ssg.to_numpy())


# ------------------------------------------------------------------ 7. Fade an outsized single-bar extension
def extreme_fade(df, k=1.2, atr_n=14, trend=200, use_trend=True, tgt_prev=True, adx_max=0):
    """
    A one-bar move several ATRs against the prevailing trend is the FX version of a
    panicked flush. Buy the down-extension inside an uptrend (and mirror), aiming the
    price back at yesterday's close rather than at a multiple of the risk.
    """
    c = df["Close"]
    pc = c.shift(1)
    a = ta.atr(df, atr_n)
    move = c - pc
    lsg = move < -k * a
    ssg = move > k * a
    if use_trend:
        tr = ta.sma(c, trend)
        lsg = lsg & (c > tr)
        ssg = ssg & (c < tr)
    if adx_max:
        adx, _, _ = ta.adx(df)
        lsg = lsg & (adx < adx_max)
        ssg = ssg & (adx < adx_max)
    d = np.where(lsg.to_numpy(), 1, np.where(ssg.to_numpy(), -1, 0)).astype(np.int8)
    tgt = pc.to_numpy() if tgt_prev else None
    return _pack(df, d, tgt=tgt)


# ------------------------------------------------------------------ 8. Weekend gap fill (hourly)
def _pip_hint(df):
    return 0.01 if float(df["Close"].median()) > 20.0 else 0.0001


def week_gap_fade(df, thresh=15.0, sl_mult=2.0, atr_n=14, max_gap_mult=0.0):
    """
    The market reopens Sunday ~22:00-23:00 UTC with a gap against Friday's close. If the
    gap is still unfilled by the end of that first bar, the historical tendency is for it
    to fill. Target is literally Friday's close, so the trade's R depends on how far the
    gap is relative to the stop - that asymmetry is the whole question, not a formality.
    """
    idx = df.index
    hrs = idx.to_series().diff().dt.total_seconds().div(3600)
    week_start = hrs > 36
    pip = _pip_hint(df)
    c, pc, op = df["Close"], df["Close"].shift(1), df["Open"]
    a = ta.atr(df, atr_n)
    gap = op - pc
    lsg = week_start & (gap < -thresh * pip) & (c < pc)
    ssg = week_start & (gap > thresh * pip) & (c > pc)
    if max_gap_mult:                       # skip news-driven gaps too large to fade
        big = gap.abs() > max_gap_mult * a
        lsg = lsg & ~big
        ssg = ssg & ~big
    d = np.where(lsg.to_numpy(), 1, np.where(ssg.to_numpy(), -1, 0)).astype(np.int8)
    stop = np.full(len(df), np.nan)
    av = a.to_numpy()
    for mask, sgn in ((lsg.to_numpy(), 1), (ssg.to_numpy(), -1)):
        stop[mask] = c[mask].to_numpy() - sgn * sl_mult * av[mask]
    return _pack(df, d, stop=stop, tgt=pc.to_numpy())


# ------------------------------------------------------------------ 9. Intraday dip-buy against the hourly trend
def intraday_mr(df, dev_n=48, dev_k=1.0, rsi_n=2, rsi_buy=20, htf_n=24, htf_tf="1h",
                sess_lo=7, sess_hi=20, atr_n=14, use_htf=True, use_rsi=True, use_sess=True):
    """
    Buy a 5-minute flush inside an hourly uptrend (and mirror), aiming back at the mean.

    Built for what the 5m frame actually allows. The median 5-minute range on the majors is
    1-3 pips while a retail spread is 1.1-3 pips, so an entry and an exit both taken off 5m
    volatility pays the spread twice on a move smaller than the spread - the cost is the whole
    trade. The fix is asymmetric: use the 5m bar only to TIME the entry, and size the barrier
    off the higher timeframe so the round-trip cost is a fraction of R rather than all of it.
    The engine's atr_mult is applied to this pack's atr, which is deliberately the 5m ATR, so
    the grid has to reach multiples of 8-20 to produce hourly-scale stops. That is not padding,
    it is the only part of the grid where a 5m system can survive its own costs.

    Three filters, each of which is the whole argument:
      * htf trend from a resampled hourly close, shifted by one bar so only a COMPLETED hour
        is ever visible to an intraday decision - the usual way to leak a future close into
        an intraday backtest is to resample without the shift
      * RSI(2) on 5m closes, so the entry waits for the flush rather than catching it early
      * session window, because the Asia dead zone and the 22:00 rollover are where the
        quoted spread is widest relative to the movement available to pay for it
    """
    c = df["Close"]
    a = ta.atr(df, atr_n)
    mean = ta.sma(c, dev_n)
    z = (c - mean) / a.replace(0.0, np.nan)

    lsg = z < -dev_k
    ssg = z > dev_k
    if use_rsi:
        r = ta.rsi(c, rsi_n)
        lsg &= r < rsi_buy
        ssg &= r > 100 - rsi_buy
    if use_htf:
        h = c.resample(htf_tf).last().shift(1)
        hma = h.ewm(span=htf_n, adjust=False).mean()
        up = hma.reindex(c.index, method="ffill")
        lsg &= c > up
        ssg &= c < up
    if use_sess:
        hh = df.index.hour.to_numpy()
        on = (hh >= sess_lo) & (hh < sess_hi)
        lsg &= on
        ssg &= on

    d = np.where(lsg.to_numpy(), 1, np.where(ssg.to_numpy(), -1, 0)).astype(np.int8)
    xl = (c >= mean).to_numpy()          # exit when price is back at the mean it left
    xs = (c <= mean).to_numpy()
    return _pack(df, d, xl=xl, xs=xs)


REGISTRY = {
    "rsi2_mr": rsi2_mr,
    "bb_fade": bb_fade,
    "donchian_bo": donchian_bo,
    "pullback": pullback,
    "session_bo": session_bo,
    "tiny_tp_mr": tiny_tp_mr,
    "extreme_fade": extreme_fade,
    "week_gap_fade": week_gap_fade,
    "intraday_mr": intraday_mr,
}
