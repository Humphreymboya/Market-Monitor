"""
Event-driven FX backtester - the single execution engine used by the search and the report.

Honesty rules baked in:
  * Signals are computed on the CLOSE of bar i; fills happen at the OPEN of bar i+1.
    (No lookahead, and never a fill inside the bar that produced the decision.)
  * Round-trip cost = broker spread + 2x slippage, per pair, deducted from every trade.
  * If both stop and target are touched inside the same bar, the STOP is assumed to fill
    (worst-case resolution).
  * A gap through a level fills at the open, not at the level.
  * Equity is marked to market on every bar, so drawdown reflects actually being in the
    market, not just realised trade outcomes.
"""
import numpy as np
import pandas as pd


def run(df, sig, cfg, pip, spread_pips, slip_pips, risk=0.005):
    """Backtest one pair over `df`.

    sig:  dict of arrays aligned to df:
          dir   int8   +1 long / -1 short, fired at that bar's close (0 = nothing)
          stop  float  protective stop price at that bar (NaN -> ATR-derived)
          xl/xs bool   close-based exit signals
          atr   float  ATR used for stop sizing
    cfg:  atr_mult, rr (0 = no target), max_bars (0 = none), trail_atr, trail_lag,
          use_exit_sig
    """
    o = df["Open"].to_numpy(float)
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    c = df["Close"].to_numpy(float)
    n = len(df)
    d = sig["dir"]
    atrv = sig["atr"]
    stopp = sig.get("stop")
    tgtarr = sig.get("tgt")
    xl = sig.get("xl")
    xs = sig.get("xs")
    dates = df.index

    atr_mult = cfg.get("atr_mult", 2.0)
    rr = cfg.get("rr", 0.0)
    max_bars = cfg.get("max_bars", 0)
    trail_atr = cfg.get("trail_atr", 0.0)
    use_exit = cfg.get("use_exit_sig", True)
    rr1 = cfg.get("rr1", 0.0)
    scale = cfg.get("scale", 0.0)
    to_be = cfg.get("to_be", True)

    cost = (spread_pips + 2.0 * slip_pips) * pip  # round trip, price units

    pos = 0
    entry = stop = tgt = 0.0
    risk_dist = 1.0
    bars_held = 0
    eq = 1.0
    eqv = np.full(n, np.nan)
    trades = []          # (exit_index, direction, R multiple, net pips, bars, reason)
    pending = 0
    pending_stop = np.nan
    pending_tgt = np.nan
    close_pos = 0
    # scale-out: bank `scale` of the size at rr1, then run the remainder off a breakeven floor.
    partial = float(rr1) > 0 and float(scale) > 0
    rem = 1.0
    booked_r = 0.0
    cost_r = 0.0
    tgt1 = 0.0

    for i in range(n):
        # ---- execute what was decided at the previous close ----
        if pos == 0 and pending != 0:
            px = o[i]
            sd = (px - pending_stop) if pending == 1 else (pending_stop - px)
            if np.isnan(pending_stop) or sd <= 0:
                a = atrv[i]
                sd = atr_mult * a if (a and not np.isnan(a) and a > 0) else 0.0
            if sd > 0:
                pos = pending
                entry = px
                risk_dist = sd
                stop = px - pending * sd
                tgt = px + pending * sd * rr if rr > 0 else 0.0
                if not np.isnan(pending_tgt):          # explicit price level beats the R multiple
                    tgt = pending_tgt
                # the whole size is still open until tgt1 actually fills, so `rem` only
                # drops when the partial is taken - setting it at entry would book a full
                # stop-out as (1-scale) of a loss and inflate both win rate and expectancy
                tgt1 = (px + pending * rr1 * sd
                        if partial and (not tgt or rr1 < abs(tgt - px) / sd) else 0.0)
                rem = 1.0
                booked_r = 0.0
                cost_r = cost / sd
                bars_held = 0
        elif pos != 0 and close_pos:
            px = o[i]
            r = booked_r + rem * (((px - entry) * pos - cost) / risk_dist)
            eq *= 1.0 + risk * (r - booked_r)
            trades.append((i, pos, r, r * risk_dist / pip, bars_held,
                           "signal_ps" if booked_r else "signal"))
            pos = 0
            booked_r = 0.0
            rem = 1.0
        pending = 0
        pending_stop = np.nan
        pending_tgt = np.nan
        close_pos = 0

        # ---- manage the open position through this bar ----
        if pos != 0:
            hit_s = hit_t = hit_1 = False
            if pos == 1:
                if l[i] <= stop:
                    hit_s = True
                elif tgt and h[i] >= tgt:
                    hit_t = True
                elif tgt1 and h[i] >= tgt1:
                    hit_1 = True
            else:
                if h[i] >= stop:
                    hit_s = True
                elif tgt and l[i] <= tgt:
                    hit_t = True
                elif tgt1 and l[i] <= tgt1:
                    hit_1 = True

            if hit_s or hit_t:
                px = stop if hit_s else tgt
                if hit_s:                       # gap beyond the level fills at the open
                    px = min(px, o[i]) if pos == 1 else max(px, o[i])
                else:
                    px = max(px, o[i]) if pos == 1 else min(px, o[i])
                add_r = rem * (((px - entry) * pos) / risk_dist - cost_r)
                r = booked_r + add_r
                eq *= 1.0 + risk * add_r
                why = ("stop" if hit_s else "target") + ("_ps" if booked_r else "")
                trades.append((i, pos, r, r * risk_dist / pip, bars_held, why))
                pos = 0
                booked_r = 0.0
                rem = 1.0
            elif hit_1:
                px = max(tgt1, o[i]) if pos == 1 else min(tgt1, o[i])
                bank = scale * (((px - entry) * pos) / risk_dist - cost_r)
                eq *= 1.0 + risk * bank
                booked_r, rem, tgt1 = bank, 1.0 - scale, 0.0
                if to_be:
                    stop = max(stop, entry) if pos == 1 else min(stop, entry)
                if (l[i] <= stop) if pos == 1 else (h[i] >= stop):
                    gx = min(stop, o[i]) if pos == 1 else max(stop, o[i])
                    add_r = rem * (((gx - entry) * pos) / risk_dist - cost_r)
                    r = booked_r + add_r
                    eq *= 1.0 + risk * add_r
                    trades.append((i, pos, r, r * risk_dist / pip, bars_held,
                                   "be_ps" if to_be else "stop_ps"))
                    pos = 0
                    booked_r = 0.0
                    rem = 1.0
                else:
                    bars_held += 1
                    eqv[i] = eq * (1.0 + risk * rem *
                                   (((c[i] - entry) * pos) / risk_dist - cost_r))
            else:
                bars_held += 1
                open_r = rem * (((c[i] - entry) * pos) / risk_dist - cost_r)
                eqv[i] = eq * (1.0 + risk * open_r)
                if use_exit and ((xl is not None and pos == 1 and xl[i]) or
                                 (xs is not None and pos == -1 and xs[i])):
                    close_pos = 1
                elif max_bars and bars_held >= max_bars:
                    close_pos = 1
                elif trail_atr:
                    a = atrv[i]
                    if a and not np.isnan(a):
                        if pos == 1:
                            stop = max(stop, c[i] - trail_atr * a)
                        else:
                            stop = min(stop, c[i] + trail_atr * a)
        else:
            eqv[i] = eq

        # ---- new setup at this close ----
        if pos == 0 and d[i] != 0:
            pending = int(d[i])
            pending_stop = np.nan if stopp is None else stopp[i]
            pending_tgt = np.nan if tgtarr is None else tgtarr[i]

    open_trade = None
    if pos != 0:
        # what a live trader is actually holding at the end of the series: the barriers are
        # still outstanding, so the engine has to hand the state over instead of hiding it
        open_trade = dict(dir=int(pos), entry=float(entry), stop=float(stop),
                          tgt=float(tgt), bars_held=int(bars_held), rem=float(rem),
                          booked_r=float(booked_r), risk_dist=float(risk_dist),
                          unrealised_r=float(rem * (((c[-1] - entry) * pos) / risk_dist
                                                     - cost_r)),
                          last_close=float(c[-1]), last_bar=str(dates[-1]))

    if pos != 0:                                # mark the open trade closed at the last close
        r = booked_r + rem * (((c[-1] - entry) * pos) / risk_dist - cost_r)
        eq *= 1.0 + risk * (r - booked_r)
        trades.append((n - 1, pos, r, r * risk_dist / pip, bars_held,
                       "eod_ps" if booked_r else "eod"))

    eqs = pd.Series(eqv, index=dates).ffill().bfill()
    eqs.iloc[-1] = eq
    out = trade_stats(trades, dates)
    out["eq"] = eqs
    out["trades"] = trades
    out["nbars"] = n
    out["open"] = open_trade
    return out


def trade_stats(trades, dates):
    """Per-pair summary plus an annualised Sharpe from the (later pooled) curve."""
    if not trades:
        return {"n_trades": 0, "win_rate": 0.0, "pf": 0.0, "exp_r": 0.0,
                "avg_pips": 0.0, "tot_pips": 0.0, "max_bars": 0, "start": None, "end": None}
    r = np.array([t[2] for t in trades], dtype=float)
    pips = np.array([t[3] for t in trades], dtype=float)
    wins, losses = r[r > 0], r[r <= 0]
    gl = abs(losses.sum())
    return {
        "n_trades": len(r),
        "win_rate": 100.0 * len(wins) / len(r),
        "pf": float(wins.sum() / gl) if gl > 1e-9 else (99.0 if wins.sum() > 0 else 0.0),
        "exp_r": float(r.mean()),
        "avg_win_r": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss_r": float(losses.mean()) if len(losses) else 0.0,
        "avg_pips": float(pips.mean()),
        "tot_pips": float(pips.sum()),
        "avg_bars": float(np.mean([t[4] for t in trades])),
        "start": dates[0],
        "end": dates[-1],
    }
