"""Simple long-only single-position strategy backtester (pure pandas).

Strategies generate boolean buy/sell signal series from price history; the
engine simulates going all-in on a buy signal (when flat) and selling out
entirely on a sell signal (when in a position). No leverage, shorting, fees,
or slippage are modeled - this is an educational approximation, not a
production trading simulator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from lib.market_data import _YF_SESSION
from lib.signals import bollinger_bands, ema, macd_series, rsi_series

TRADING_DAYS_PER_YEAR = 252

STRATEGIES = {
    "RSI Oversold": "Buy when RSI drops below the oversold threshold; sell when it rises above the overbought threshold.",
    "Golden Cross": "Buy when the fast EMA crosses above the slow EMA; sell on the opposite crossover.",
    "Bollinger Band Bounce": "Buy when price touches the lower Bollinger Band; sell when it touches the upper band.",
    "MACD Crossover": "Buy when the MACD line crosses above its signal line; sell on a crossunder.",
    "Custom": "Set your own RSI and EMA parameters.",
}

DEFAULT_PARAMS = {
    "RSI Oversold": {"rsi_period": 14, "buy_threshold": 30, "sell_threshold": 70},
    "Golden Cross": {"fast_ema": 50, "slow_ema": 200},
    "Bollinger Band Bounce": {"bb_window": 20, "bb_std": 2.0},
    "MACD Crossover": {"macd_fast": 12, "macd_slow": 26, "macd_signal": 9},
    "Custom": {
        "rsi_period": 14, "buy_threshold": 30, "sell_threshold": 70,
        "fast_ema": 20, "slow_ema": 50, "use_ema_filter": True,
    },
}


@st.cache_data(ttl=900, show_spinner=False)
def fetch_price_history(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Daily OHLCV history for `ticker` between `start` and `end` (YYYY-MM-DD)."""
    try:
        df = yf.Ticker(ticker, session=_YF_SESSION).history(start=start, end=end, interval="1d")
    except Exception:
        df = pd.DataFrame()
    if df is None:
        df = pd.DataFrame()
    return df.dropna(subset=["Close"]) if not df.empty else df


def _crossover_up(fast: pd.Series, slow: pd.Series) -> pd.Series:
    return (fast > slow) & (fast.shift(1) <= slow.shift(1))


def _crossover_down(fast: pd.Series, slow: pd.Series) -> pd.Series:
    return (fast < slow) & (fast.shift(1) >= slow.shift(1))


def generate_signals(df: pd.DataFrame, strategy: str, params: dict) -> pd.DataFrame:
    """Return a copy of `df` with boolean `buy` / `sell` columns added."""
    close = df["Close"]
    out = df.copy()
    buy = pd.Series(False, index=df.index)
    sell = pd.Series(False, index=df.index)

    if strategy == "RSI Oversold":
        rsi = rsi_series(close, params.get("rsi_period", 14))
        buy = rsi < params.get("buy_threshold", 30)
        sell = rsi > params.get("sell_threshold", 70)

    elif strategy == "Golden Cross":
        fast = ema(close, params.get("fast_ema", 50))
        slow = ema(close, params.get("slow_ema", 200))
        buy = _crossover_up(fast, slow)
        sell = _crossover_down(fast, slow)

    elif strategy == "Bollinger Band Bounce":
        _, upper, lower = bollinger_bands(close, params.get("bb_window", 20), params.get("bb_std", 2.0))
        buy = close <= lower
        sell = close >= upper

    elif strategy == "MACD Crossover":
        macd_line, signal_line, _ = macd_series(
            close, params.get("macd_fast", 12), params.get("macd_slow", 26), params.get("macd_signal", 9)
        )
        buy = _crossover_up(macd_line, signal_line)
        sell = _crossover_down(macd_line, signal_line)

    elif strategy == "Custom":
        rsi = rsi_series(close, params.get("rsi_period", 14))
        buy = rsi < params.get("buy_threshold", 30)
        sell = rsi > params.get("sell_threshold", 70)
        if params.get("use_ema_filter", True):
            fast = ema(close, params.get("fast_ema", 20))
            slow = ema(close, params.get("slow_ema", 50))
            buy = buy & (fast > slow)

    out["buy"] = buy.fillna(False)
    out["sell"] = sell.fillna(False)
    return out


def run_backtest(df: pd.DataFrame, strategy: str, params: dict, initial_capital: float = 10_000.0) -> dict:
    """Simulate a long-only, fully-invested-or-flat strategy.

    Returns a dict with `equity_curve` (DataFrame indexed by date with
    `Strategy` and `Buy & Hold` columns), `trades` (list of dicts), and
    `metrics` (dict).
    """
    signals = generate_signals(df, strategy, params)
    close = signals["Close"]
    dates = signals.index

    cash = initial_capital
    shares = 0.0
    in_position = False
    entry_price = None
    entry_date = None

    equity = []
    trades = []

    for date, row in signals.iterrows():
        price = float(row["Close"])

        if not in_position and row["buy"]:
            shares = cash / price
            entry_price = price
            entry_date = date
            cash = 0.0
            in_position = True
        elif in_position and row["sell"]:
            exit_value = shares * price
            pnl_dollars = exit_value - (shares * entry_price)
            pnl_pct = (price / entry_price - 1) * 100
            trades.append({
                "Entry Date": entry_date,
                "Exit Date": date,
                "Entry Price": entry_price,
                "Exit Price": price,
                "Shares": shares,
                "P&L ($)": pnl_dollars,
                "P&L (%)": pnl_pct,
                "Days Held": (date - entry_date).days,
            })
            cash = exit_value
            shares = 0.0
            in_position = False
            entry_price = None
            entry_date = None

        total_value = cash + shares * price
        equity.append(total_value)

    equity_curve = pd.Series(equity, index=dates, name="Strategy")

    # Buy & hold comparison: invest the full initial capital on day 1.
    bh_shares = initial_capital / float(close.iloc[0])
    buy_hold = close * bh_shares
    buy_hold.name = "Buy & Hold"

    curve_df = pd.concat([equity_curve, buy_hold], axis=1)

    metrics = compute_metrics(equity_curve, trades, initial_capital)

    return {"equity_curve": curve_df, "trades": trades, "metrics": metrics}


def compute_metrics(equity_curve: pd.Series, trades: list[dict], initial_capital: float) -> dict:
    if equity_curve.empty:
        return {}

    final_value = float(equity_curve.iloc[-1])
    total_return = (final_value / initial_capital - 1) * 100

    days = (equity_curve.index[-1] - equity_curve.index[0]).days
    years = days / 365.25 if days > 0 else 0
    if years > 0 and final_value > 0:
        annual_return = ((final_value / initial_capital) ** (1 / years) - 1) * 100
    else:
        annual_return = 0.0

    daily_returns = equity_curve.pct_change().dropna()
    if daily_returns.std() > 0:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)
    else:
        sharpe = 0.0

    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = float(drawdown.min()) * 100

    closed_trades = trades
    if closed_trades:
        wins = sum(1 for t in closed_trades if t["P&L ($)"] > 0)
        win_rate = wins / len(closed_trades) * 100
    else:
        win_rate = 0.0

    return {
        "Total Return %": total_return,
        "Annual Return %": annual_return,
        "Sharpe Ratio": float(sharpe),
        "Max Drawdown %": max_drawdown,
        "Win Rate %": win_rate,
        "Total Trades": len(closed_trades),
        "Final Value": final_value,
    }
