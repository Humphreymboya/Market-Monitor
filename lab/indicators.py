"""Vectorised indicators (numpy/pandas only, no TA-lib dependency)."""
import numpy as np
import pandas as pd


def sma(s, n):
    return s.rolling(n, min_periods=n).mean()


def ema(s, n):
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    # Wilder smoothing
    ru = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rd = dn.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = ru / rd.replace(0.0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.fillna(100.0)


def true_range(df):
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    return pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)


def atr(df, n=14):
    return true_range(df).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def bbands(close, n=20, k=2.0):
    m = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    return m, m + k * sd, m - k * sd


def donchian(df, n=20):
    hh = df["High"].rolling(n, min_periods=n).max()
    ll = df["Low"].rolling(n, min_periods=n).min()
    mid = (hh + ll) / 2
    return hh, ll, mid


def adx(df, n=14):
    h, l = df["High"], df["Low"]
    up = h.diff()
    dn = -l.diff()
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = true_range(df)
    atr_n = tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    pdi = 100 * pd.Series(plus, index=df.index).ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / atr_n
    mdi = 100 * pd.Series(minus, index=df.index).ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / atr_n
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean().fillna(0.0), pdi, mdi


def zscore(close, n=20):
    m = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    return (close - m) / sd.replace(0.0, np.nan)
