"""Descriptive technical/fundamental scoring.

These scores (0-100) and the "at a glance" buckets are explicitly
NON-RECOMMENDATIONS - they describe where a stock currently sits relative to
its own history and common reference ranges, using neutral language only.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def compute_rsi(close: pd.Series, period: int = 14) -> float | None:
    if close is None or len(close) < period + 1:
        return None
    delta = close.diff().dropna()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    last_gain = avg_gain.iloc[-1]
    last_loss = avg_loss.iloc[-1]
    if last_loss == 0:
        return 100.0
    rs = last_gain / last_loss
    rsi = 100 - (100 / (1 + rs))
    if np.isnan(rsi):
        return None
    return float(rsi)


def _clip(value, lo=0, hi=100):
    return max(lo, min(hi, value))


def _valid_num(value) -> bool:
    """True if value is a usable number (not None, NaN, or a non-numeric type).

    yfinance fundamentals occasionally come back as NaN or stray strings for a
    given ticker; comparisons/arithmetic on those raise TypeError, so every
    fundamentals field must be checked with this instead of `is not None`.
    """
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return not math.isnan(value)


# ---------------------------------------------------------------------------
# Chart indicator series (used by lib.charts.render_technical_chart)
# ---------------------------------------------------------------------------

def moving_average(close: pd.Series, window: int) -> pd.Series:
    """Simple moving average of `close` over `window` periods."""
    return close.rolling(window).mean()


def ema(close: pd.Series, window: int) -> pd.Series:
    """Exponential moving average of `close` over `window` periods."""
    return close.ewm(span=window, adjust=False).mean()


def stochastic_oscillator(df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
    """Return (%K, %D) stochastic oscillator series."""
    low_min = df["Low"].rolling(k_period).min()
    high_max = df["High"].rolling(k_period).max()
    rng = (high_max - low_min).replace(0, np.nan)
    percent_k = ((df["Close"] - low_min) / rng) * 100
    percent_d = percent_k.rolling(d_period).mean()
    return percent_k.fillna(50), percent_d.fillna(50)


def atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range series."""
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def obv_series(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume series."""
    direction = np.sign(df["Close"].diff()).fillna(0)
    return (direction * df["Volume"]).cumsum()


def max_pain_strike(calls: pd.DataFrame, puts: pd.DataFrame) -> float | None:
    """Return the strike price that minimizes total option-holder payout at expiry."""
    strikes = sorted(set(calls["strike"]).union(puts["strike"])) if not calls.empty or not puts.empty else []
    if not strikes:
        return None

    best_strike, best_payout = None, None
    for s_test in strikes:
        call_payout = ((s_test - calls["strike"]).clip(lower=0) * calls["openInterest"].fillna(0)).sum()
        put_payout = ((puts["strike"] - s_test).clip(lower=0) * puts["openInterest"].fillna(0)).sum()
        total = call_payout + put_payout
        if best_payout is None or total < best_payout:
            best_payout, best_strike = total, s_test
    return best_strike


def bs_delta(spot: float, strike: float, years: float, iv: float, option_type: str, rate: float = 0.045) -> float | None:
    """Black-Scholes delta for a call or put. `option_type` is "call" or "put"."""
    if not spot or not strike or not iv or years is None or years <= 0 or iv <= 0:
        return None
    d1 = (math.log(spot / strike) + (rate + 0.5 * iv ** 2) * years) / (iv * math.sqrt(years))
    n_d1 = 0.5 * (1 + math.erf(d1 / math.sqrt(2)))
    if option_type == "put":
        return n_d1 - 1
    return n_d1


def bollinger_bands(close: pd.Series, window: int = 20, num_std: float = 2.0):
    """Return (middle, upper, lower) Bollinger Band series."""
    middle = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return middle, upper, lower


def vwap_series(df: pd.DataFrame) -> pd.Series:
    """Cumulative volume-weighted average price over the given DataFrame."""
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    cum_vol = df["Volume"].cumsum()
    cum_vol_price = (typical * df["Volume"]).cumsum()
    return cum_vol_price / cum_vol.replace(0, np.nan)


def rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    """Rolling RSI series (Wilder-style smoothing via simple rolling means)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def macd_series(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Return (macd_line, signal_line, histogram) series."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


# ---------------------------------------------------------------------------
# Technical score
# ---------------------------------------------------------------------------

def technical_score(history: pd.DataFrame) -> tuple[float | None, list[str]]:
    """Return (score 0-100, driver bullet strings) from price history.

    Components: position vs 50/200-day averages, RSI, 52-week range
    position, and 3-month momentum.
    """
    if history is None or history.empty or len(history) < 30:
        return None, []

    close = history["Close"].dropna()
    last = float(close.iloc[-1])
    drivers = []
    components = []

    # vs 200DMA (weight 25)
    if len(close) >= 200:
        sma200 = close.rolling(200).mean().iloc[-1]
        if not np.isnan(sma200):
            pct = (last / sma200 - 1) * 100
            comp = _clip(50 + pct * 4)
            components.append((comp, 25))
            drivers.append(
                f"Price is {'above' if last >= sma200 else 'below'} its 200-day average "
                f"({pct:+.1f}%)"
            )

    # vs 50DMA (weight 20)
    if len(close) >= 50:
        sma50 = close.rolling(50).mean().iloc[-1]
        if not np.isnan(sma50):
            pct = (last / sma50 - 1) * 100
            comp = _clip(50 + pct * 5)
            components.append((comp, 20))
            drivers.append(
                f"Price is {'above' if last >= sma50 else 'below'} its 50-day average "
                f"({pct:+.1f}%)"
            )

    # RSI (weight 25)
    rsi = compute_rsi(close)
    if rsi is not None:
        comp = _clip(rsi)
        components.append((comp, 25))
        if rsi >= 70:
            label = "Stretched momentum reading"
        elif rsi >= 55:
            label = "Firm momentum"
        elif rsi >= 45:
            label = "Neutral momentum"
        elif rsi >= 30:
            label = "Soft momentum"
        else:
            label = "Subdued momentum reading"
        drivers.append(f"{label} (RSI {rsi:.0f})")

    # 52-week range position (weight 20)
    window = close.tail(252) if len(close) >= 252 else close
    lo, hi = float(window.min()), float(window.max())
    if hi > lo:
        pos = (last - lo) / (hi - lo) * 100
        components.append((pos, 20))
        drivers.append(f"Trading at {pos:.0f}% of its 52-week range")

    # 3-month momentum (weight 10)
    if len(close) >= 63:
        mom = (last / close.iloc[-63] - 1) * 100
        comp = _clip(50 + mom * 2)
        components.append((comp, 10))
        drivers.append(f"3-month price change of {mom:+.1f}%")

    if not components:
        return None, drivers

    total_weight = sum(w for _, w in components)
    score = sum(c * w for c, w in components) / total_weight
    return round(score, 1), drivers


# ---------------------------------------------------------------------------
# Fundamental score
# ---------------------------------------------------------------------------

def fundamental_score(fund: dict) -> tuple[float | None, list[str]]:
    """Return (score 0-100, driver bullet strings) from fundamentals dict
    (lib.market_data.get_stock_fundamentals output).
    """
    components = []
    drivers = []

    profit_margin = fund.get("profit_margin")
    if _valid_num(profit_margin):
        comp = _clip(50 + profit_margin * 200)
        components.append((comp, 20))
        drivers.append(f"Net margin of {profit_margin * 100:.1f}%")

    roe = fund.get("return_on_equity")
    if _valid_num(roe):
        comp = _clip(50 + roe * 150)
        components.append((comp, 20))
        drivers.append(f"Return on equity of {roe * 100:.1f}%")

    rev_growth = fund.get("revenue_growth")
    if _valid_num(rev_growth):
        comp = _clip(50 + rev_growth * 200)
        components.append((comp, 15))
        drivers.append(f"Revenue growth of {rev_growth * 100:.1f}% year over year")

    earn_growth = fund.get("earnings_growth")
    if _valid_num(earn_growth):
        comp = _clip(50 + earn_growth * 150)
        components.append((comp, 15))
        drivers.append(f"Earnings growth of {earn_growth * 100:.1f}% year over year")

    dte = fund.get("debt_to_equity")
    if _valid_num(dte):
        ratio = dte / 100 if dte > 10 else dte  # yfinance reports as a percent-like number
        comp = _clip(100 - ratio * 25)
        components.append((comp, 15))
        drivers.append(f"Debt-to-equity ratio of {ratio:.2f}")

    current_ratio = fund.get("current_ratio")
    if _valid_num(current_ratio):
        # ~2.0 is considered comfortable; too high or too low both pull the score down slightly
        comp = _clip(100 - abs(current_ratio - 2.0) * 25)
        components.append((comp, 15))
        drivers.append(f"Current ratio of {current_ratio:.2f}")

    if not components:
        return None, drivers

    total_weight = sum(w for _, w in components)
    score = sum(c * w for c, w in components) / total_weight
    return round(score, 1), drivers


# ---------------------------------------------------------------------------
# "At a glance" descriptive buckets
# ---------------------------------------------------------------------------

def at_a_glance(history: pd.DataFrame, fund: dict) -> list[tuple[str, str]]:
    """Return a list of (label, value) chips with neutral, factual language."""
    chips = []
    close = history["Close"].dropna() if history is not None and not history.empty else pd.Series(dtype=float)
    last = float(close.iloc[-1]) if not close.empty else None

    # Trend vs 200DMA
    if last is not None and len(close) >= 200:
        sma200 = close.rolling(200).mean().iloc[-1]
        if not np.isnan(sma200):
            chips.append(("Trend", "Above 200-day average" if last >= sma200 else "Below 200-day average"))

    # Momentum / RSI bucket
    rsi = compute_rsi(close) if not close.empty else None
    if rsi is not None:
        if rsi >= 70:
            label = "Stretched reading"
        elif rsi >= 55:
            label = "Firm momentum"
        elif rsi >= 45:
            label = "Neutral momentum"
        elif rsi >= 30:
            label = "Soft momentum"
        else:
            label = "Subdued reading"
        chips.append(("Momentum", f"{label} (RSI {rsi:.0f})"))

    # 52-week range position
    if last is not None:
        window = close.tail(252) if len(close) >= 252 else close
        lo, hi = float(window.min()), float(window.max())
        if hi > lo:
            pos = (last - lo) / (hi - lo) * 100
            if pos >= 90:
                label = "Near 52-week high"
            elif pos >= 50:
                label = "Upper half of 52-week range"
            elif pos >= 10:
                label = "Lower half of 52-week range"
            else:
                label = "Near 52-week low"
            chips.append(("52-week range", label))

    # Profitability: ROE tier
    roe = fund.get("return_on_equity")
    if _valid_num(roe):
        if roe < 0:
            label = "Negative"
        elif roe < 0.10:
            label = "Modest"
        elif roe < 0.20:
            label = "Solid"
        else:
            label = "High"
        chips.append(("Profitability", f"{label} (ROE {roe * 100:.1f}%)"))

    # Leverage: D/E tier
    dte = fund.get("debt_to_equity")
    if _valid_num(dte):
        ratio = dte / 100 if dte > 10 else dte
        if ratio < 0.5:
            label = "Low"
        elif ratio < 1.5:
            label = "Moderate"
        elif ratio < 3:
            label = "Elevated"
        else:
            label = "High"
        chips.append(("Leverage", f"{label} (D/E {ratio:.2f})"))

    # Volatility: beta vs market
    beta = fund.get("beta")
    if _valid_num(beta):
        if beta < 0.8:
            label = "Lower than market"
        elif beta <= 1.2:
            label = "In line with market"
        else:
            label = "Higher than market"
        chips.append(("Volatility", f"{label} (beta {beta:.2f})"))

    # Valuation: P/E tier
    pe = fund.get("trailing_pe")
    if _valid_num(pe) and pe > 0:
        if pe < 15:
            label = "Low multiple"
        elif pe < 25:
            label = "Moderate multiple"
        elif pe < 40:
            label = "High multiple"
        else:
            label = "Very high multiple"
        chips.append(("Valuation", f"{label} (P/E {pe:.1f})"))
    else:
        chips.append(("Valuation", "Not meaningful (no positive earnings)"))

    return chips
