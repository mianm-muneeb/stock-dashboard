"""Helpers for the Kronos Forecast page: forecasting, calibration fixes, backtest.

Pure numpy/pandas apart from data fetching, so the maths can be tested with a
fake predictor. Closes are in price units; internal maths uses log returns
relative to the last close.
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
import pandas as pd
import yfinance as yf


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

def fetch_bars(ticker: str, interval: str = "1d") -> pd.DataFrame:
    period = "5y" if interval == "1d" else "60d"
    df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.dropna()


def next_earnings_date(ticker: str):
    """Next earnings date as a naive Timestamp, or None if unavailable."""
    try:
        cal = yf.Ticker(ticker).calendar
        if not isinstance(cal, dict):
            return None
        d = cal.get("Earnings Date")
        if not d:
            return None
        d = d[0] if isinstance(d, (list, tuple)) else d
        ts = pd.Timestamp(d)
        return ts.tz_localize(None) if ts.tzinfo is not None else ts
    except Exception:
        return None


def earnings_info(ticker: str, bars: pd.DataFrame, limit: int = 24) -> dict:
    """Past earnings reactions and the next report, from yfinance. Never raises.

    Returns {"hist": DataFrame[date, timing, move], "next": Timestamp|None, "next_timing": str}.
    move = log return on the reaction bar (same day for a pre-market report, next day for
    after-close). If the report time is unknown, the larger of the two days is used.
    """
    out = {"hist": pd.DataFrame(columns=["date", "timing", "move"]), "next": None, "next_timing": "bmo"}
    ed = None
    for attempt in range(3):  # Yahoo sometimes rate-limits or returns nothing on the first call
        try:
            ed = yf.Ticker(ticker).get_earnings_dates(limit=limit)
        except Exception:
            ed = None
        if ed is not None and len(ed):
            break
        time.sleep(1.5)
    if ed is None or len(ed) == 0:
        out["next"] = next_earnings_date(ticker)
        return out

    close = bars["close"].astype(float)
    lr = np.log(close).diff()
    idx = close.index
    rows, nxt, nxt_hour = [], None, None
    now = pd.Timestamp.now().normalize()
    for ts, r in ed.iterrows():
        t = pd.Timestamp(ts)
        t = t.tz_localize(None) if t.tzinfo is not None else t  # keep exchange-local clock time
        reported = pd.notna(r.get("Reported EPS"))
        day = t.normalize()
        if not reported and day >= now:
            if nxt is None or day < nxt:
                nxt, nxt_hour = day, t.hour
            continue
        if not reported:
            continue
        pos = int(idx.searchsorted(day))
        if pos < 1 or pos + 1 >= len(idx):
            continue
        if t.hour >= 16:
            timing, mv = "amc", lr.iloc[pos + 1]
        elif 1 <= t.hour < 16:
            timing, mv = "bmo", lr.iloc[pos]
        else:
            a, b = lr.iloc[pos], lr.iloc[pos + 1]
            timing, mv = "unk", (a if abs(a) >= abs(b) else b)
        if np.isfinite(mv):
            rows.append({"date": day, "timing": timing, "move": float(mv)})
    hist = pd.DataFrame(rows, columns=["date", "timing", "move"]).sort_values("date").reset_index(drop=True)
    out["hist"] = hist
    out["next"] = nxt if nxt is not None else next_earnings_date(ticker)
    if nxt is not None and nxt_hour is not None and nxt_hour >= 1:
        out["next_timing"] = "amc" if nxt_hour >= 16 else "bmo"
    elif len(hist):
        known = hist[hist["timing"] != "unk"]["timing"]
        if len(known):
            out["next_timing"] = known.mode().iloc[0]
    return out


def _reaction_pos(future: pd.DatetimeIndex, day: pd.Timestamp, timing: str):
    """Index in `future` of the bar that carries the earnings reaction, or None."""
    pos = int(future.searchsorted(pd.Timestamp(day).normalize()))
    if timing == "amc":
        pos += 1
    return pos if pos < len(future) else None


def earnings_jumps(moves, sigma: float, n: int, rng: np.random.Generator, min_n: int = 4):
    """Draw n earnings-day log jumps from a stock's own past reactions.

    Zero-centred (so we impose size, not direction), with the variance an ordinary day
    already contributes removed. Fewer than min_n past reactions -> fall back to 3 sigma.
    """
    m = np.asarray([x for x in moves if np.isfinite(x)], dtype=float)
    if len(m) >= min_n:
        c = m - m.mean()
        s_tot = float(m.std(ddof=1)) if len(m) > 1 else float(abs(c).mean())
        s = np.sqrt(max(s_tot ** 2 - sigma ** 2, (0.25 * s_tot) ** 2))
        c = c / max(c.std(ddof=1), 1e-9) * s
        draw = rng.choice(c, size=n, replace=True) + rng.normal(0.0, 0.15 * s, size=n)
        return draw, {"n": int(len(m)), "fallback": False, "typical": float(np.mean(np.abs(m)))}
    s = float(np.sqrt(max(8.0 * sigma ** 2, 1e-12)))  # total move ~3 sigma, minus one ordinary day
    return rng.normal(0.0, s, size=n), {"n": int(len(m)), "fallback": True, "typical": float(2.4 * sigma)}


def apply_earnings_jumps(sims: np.ndarray, future: pd.DatetimeIndex, reports, moves, sigma: float,
                         seed: int = 7):
    """Add an earnings jump on each report date inside the forecast window.

    reports: list of (date, timing). Returns (sims, info) where info has n, fallback,
    typical (mean |move|), p10 (adverse 10% jump) and marks (bar dates). sims unchanged
    (info None) if no report lands inside the window.
    """
    rng = np.random.default_rng(seed)
    out = sims.copy()
    info, marks, p10s = None, [], []
    for day, timing in reports:
        pos = _reaction_pos(future, day, timing)
        if pos is None:
            continue
        j, info = earnings_jumps(moves, sigma, len(out), rng)
        out[:, pos:] = out[:, pos:] * np.exp(j)[:, None]
        marks.append(future[pos])
        p10s.append(float(np.percentile(j, 10)))
    if info is None:
        return sims, None
    info = dict(info, marks=marks, p10=float(min(p10s)))
    return out, info


def future_index(last: pd.Timestamp, n: int, interval: str) -> pd.DatetimeIndex:
    if interval == "1d":
        return pd.bdate_range(last + pd.Timedelta(days=1), periods=n)
    step = pd.Timedelta(hours=1)
    idx = pd.date_range(last + step, periods=n * 3, freq="h")
    idx = idx[(idx.weekday < 5) & (idx.hour >= 9) & (idx.hour <= 16)]
    return idx[:n]


# ----------------------------------------------------------------------------
# Forecasting
# ----------------------------------------------------------------------------

def forecast_paths(predictor, hist: pd.DataFrame, future: pd.DatetimeIndex,
                   paths: int, temp: float, top_p: float) -> np.ndarray:
    """Independent single-sample paths as a batch. Returns closes, shape (paths, horizon).

    KronosPredictor.predict() averages samples, so independent paths need
    predict_batch with identical copies of the series.
    """
    x_df = hist[["open", "high", "low", "close", "volume"]].reset_index(drop=True)
    x_ts = pd.Series(hist.index)
    y_ts = pd.Series(future)
    outs = predictor.predict_batch(
        df_list=[x_df] * paths,
        x_timestamp_list=[x_ts] * paths,
        y_timestamp_list=[y_ts] * paths,
        pred_len=len(future),
        T=temp,
        top_p=top_p,
        sample_count=1,
        verbose=False,
    )
    return np.stack([o["close"].values for o in outs])


def daily_sigma(close: pd.Series, n: int = 60) -> float:
    """Realised per-bar volatility of log returns over the last n bars."""
    lr = np.log(close.astype(float)).diff().dropna().iloc[-n:]
    return float(lr.std())


def anchor_first_bar(sims: np.ndarray, last_close: float, sigma: float, cap_mult: float = 1.0) -> np.ndarray:
    """Cap the median first-bar jump at cap_mult * one-bar volatility.

    Shifts every path by the same amount so path dispersion is preserved.
    """
    lr = np.log(sims / last_close)
    first_med = float(np.median(lr[:, 0]))
    cap = cap_mult * sigma
    excess = first_med - float(np.clip(first_med, -cap, cap))
    return last_close * np.exp(lr - excess)


def rescale_to_vol(sims: np.ndarray, last_close: float, sigma: float,
                   drift_keep: float = 0.5, k_min: float = 0.25, k_max: float = 8.0) -> np.ndarray:
    """Stretch/shrink path dispersion to realised volatility and shrink the drift.

    At each horizon step t the spread across paths is scaled so its std equals
    sigma * sqrt(t). The model's median drift is multiplied by drift_keep
    (0 = random walk centred on the last close, 1 = trust the model fully).
    """
    lr = np.log(sims / last_close)
    horizon = lr.shape[1]
    med = np.median(lr, axis=0)
    dev = lr - med
    spread = dev.std(axis=0)
    target = sigma * np.sqrt(np.arange(1, horizon + 1))
    k = np.clip(target / np.maximum(spread, 1e-8), k_min, k_max)
    return last_close * np.exp(drift_keep * med + dev * k)


def adjust(sims: np.ndarray, last_close: float, sigma: float, *, anchor: bool = True,
           rescale: bool = True, drift_keep: float = 0.5) -> np.ndarray:
    # Drop broken paths (NaN or non-positive prices) so one bad path cannot turn the whole fan into NaN
    ok = np.isfinite(sims).all(axis=1) & (sims > 0).all(axis=1)
    out = sims[ok] if ok.sum() >= 3 else sims
    if anchor:
        out = anchor_first_bar(out, last_close, sigma)
    if rescale:
        out = rescale_to_vol(out, last_close, sigma, drift_keep=drift_keep)
    return out


def consensus(sims_a: np.ndarray, sims_b: np.ndarray, last_close: float, sigma: float) -> dict:
    """Do two models agree on direction and roughly on size?"""
    ea = float(np.median(np.log(sims_a[:, -1] / last_close)))
    eb = float(np.median(np.log(sims_b[:, -1] / last_close)))
    horizon = sims_a.shape[1]
    tol = 0.5 * sigma * np.sqrt(horizon)
    same_sign = np.sign(ea) == np.sign(eb)
    close_size = abs(ea - eb) <= tol
    return {"agree": bool(same_sign and close_size), "a": ea, "b": eb, "tol": float(tol),
            "direction": "up" if (ea + eb) > 0 else "down"}


# ----------------------------------------------------------------------------
# Calibration backtest
# ----------------------------------------------------------------------------

def score_origin(sims_raw: np.ndarray, sims_adj: np.ndarray, actual_end: float, last_close: float) -> dict:
    rows = {}
    for name, s in (("raw", sims_raw), ("adjusted", sims_adj)):
        p05, p25, p50, p75, p95 = np.percentile(s[:, -1], [5, 25, 50, 75, 95])
        rows[name] = {
            "in90": bool(p05 <= actual_end <= p95),
            "in50": bool(p25 <= actual_end <= p75),
            "dir_hit": bool(np.sign(p50 - last_close) == np.sign(actual_end - last_close)),
            "abs_err": float(abs(np.log(p50 / actual_end))),
        }
    rows["naive_abs_err"] = float(abs(np.log(last_close / actual_end)))
    return rows


def run_calibration(predictor, bars_by_ticker: dict[str, pd.DataFrame], *, horizon: int, lookback: int,
                    origins: int, spacing: int, paths: int, temp: float, top_p: float,
                    drift_keep: float, progress: Callable[[float, str], None] | None = None) -> pd.DataFrame:
    """Forecast from several past origins per ticker and score against what happened.

    Origins are the most recent windows that still leave `horizon` bars of
    ground truth, so they sit after the model's training data.
    """
    records = []
    total = max(1, len(bars_by_ticker) * origins)
    done = 0
    for ticker, bars in bars_by_ticker.items():
        for k in range(origins):
            end = len(bars) - horizon - k * spacing  # exclusive end of history
            start = end - lookback
            if start < 0:
                done += 1
                continue
            hist = bars.iloc[start:end]
            actual = bars.iloc[end:end + horizon]
            if len(actual) < horizon:
                done += 1
                continue
            last_close = float(hist["close"].iloc[-1])
            sigma = daily_sigma(hist["close"])
            sims = forecast_paths(predictor, hist, actual.index, paths, temp, top_p)
            adj = adjust(sims, last_close, sigma, drift_keep=drift_keep)
            sc = score_origin(sims, adj, float(actual["close"].iloc[-1]), last_close)
            for name in ("raw", "adjusted"):
                records.append({"ticker": ticker, "origin": hist.index[-1].date(), "variant": name,
                                **sc[name], "naive_abs_err": sc["naive_abs_err"]})
            done += 1
            if progress:
                progress(done / total, f"{ticker} origin {k + 1}/{origins}")
    return pd.DataFrame(records)


def summarise_calibration(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    g = df.groupby("variant")
    out = pd.DataFrame({
        "n forecasts": g.size(),
        "actual inside 90% band": g["in90"].mean() * 100,
        "actual inside 50% band": g["in50"].mean() * 100,
        "direction right": g["dir_hit"].mean() * 100,
        "median error": g["abs_err"].mean() * 100,
        "'no change' error": g["naive_abs_err"].mean() * 100,
    })
    return out.round(1)
