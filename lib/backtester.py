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

import concurrent.futures

from lib.market_data import _YF_SESSION
from lib.screener import SCREENER_UNIVERSE
from lib.signals import bollinger_bands, ema, macd_series, rsi_series

TRADING_DAYS_PER_YEAR = 252

# Watchlist used by the "BB Mid-Cross Reversion" strategy, which tests a
# pattern across many names rather than a single user-entered ticker. Reuses
# the same curated large/mega-cap universe as the Stock Screener page.
WATCHLIST = SCREENER_UNIVERSE

STRATEGIES = {
    "RSI Oversold": "Buy when RSI drops below the oversold threshold; sell when it rises above the overbought threshold.",
    "Golden Cross": "Buy when the fast EMA crosses above the slow EMA; sell on the opposite crossover.",
    "Bollinger Band Bounce": "Buy when price touches the lower Bollinger Band; sell when it touches the upper band.",
    "MACD Crossover": "Buy when the MACD line crosses above its signal line; sell on a crossunder.",
    "BB Mid-Cross Reversion": (
        "Buy when price closes above the 20-day SMA (Bollinger mid band) after closing "
        "below it the prior bar. Exit at a target gain, a stop loss, or after 10 bars "
        f"held - whichever comes first. Runs across a curated {len(WATCHLIST)}-stock "
        "watchlist instead of a single ticker."
    ),
    "Custom": "Set your own RSI and EMA parameters.",
}

DEFAULT_PARAMS = {
    "RSI Oversold": {"rsi_period": 14, "buy_threshold": 30, "sell_threshold": 70},
    "Golden Cross": {"fast_ema": 50, "slow_ema": 200},
    "Bollinger Band Bounce": {"bb_window": 20, "bb_std": 2.0},
    "MACD Crossover": {"macd_fast": 12, "macd_slow": 26, "macd_signal": 9},
    "BB Mid-Cross Reversion": {
        "bb_window": 20, "take_profit_pct": 1.5, "stop_loss_pct": 1.0, "max_bars_held": 10,
    },
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
        win_pcts = [t["P&L (%)"] for t in closed_trades if t["P&L ($)"] > 0]
        loss_pcts = [t["P&L (%)"] for t in closed_trades if t["P&L ($)"] <= 0]
        avg_win = sum(win_pcts) / len(win_pcts) if win_pcts else 0.0
        avg_loss = sum(loss_pcts) / len(loss_pcts) if loss_pcts else 0.0
    else:
        win_rate = 0.0
        avg_win = 0.0
        avg_loss = 0.0

    return {
        "Total Return %": total_return,
        "Annual Return %": annual_return,
        "Sharpe Ratio": float(sharpe),
        "Max Drawdown %": max_drawdown,
        "Win Rate %": win_rate,
        "Avg Win %": avg_win,
        "Avg Loss %": avg_loss,
        "Total Trades": len(closed_trades),
        "Final Value": final_value,
    }


# ---------------------------------------------------------------------------
# BB Mid-Cross Reversion
#
# Entry/exit are trade-managed (depend on the entry price and holding time),
# not a simple per-bar sell signal, so this strategy uses a dedicated bar-by-
# bar loop rather than the generic generate_signals()/run_backtest() path.
# ---------------------------------------------------------------------------

def run_bb_mid_cross_single(df: pd.DataFrame, params: dict, initial_capital: float = 10_000.0) -> dict:
    """Simulate the BB Mid-Cross Reversion strategy on one ticker's history.

    Entry: close crosses above the 20-day SMA (Bollinger mid band, reused
    from lib.signals.bollinger_bands) from below. Exit: whichever comes
    first - a target gain, a stop loss, or `max_bars_held` bars held.

    Returns `equity_curve` (Series), `trades` (list of dicts), and
    `open_position` (dict with Entry Date/Price, or None) for a trade still
    open at the end of the data.
    """
    close = df["Close"]
    mid, _, _ = bollinger_bands(close, params.get("bb_window", 20), 2.0)

    take_profit = params.get("take_profit_pct", 1.5) / 100
    stop_loss = params.get("stop_loss_pct", 1.0) / 100
    max_bars_held = params.get("max_bars_held", 10)

    cash = initial_capital
    shares = 0.0
    in_position = False
    entry_price = None
    entry_date = None
    bars_held = 0

    prev_close = None
    prev_mid = None

    equity = []
    trades = []

    for date, price, mid_val in zip(df.index, close, mid):
        price = float(price)
        mid_val = float(mid_val) if pd.notna(mid_val) else None

        if in_position:
            bars_held += 1
            change = price / entry_price - 1
            exit_reason = None
            if change >= take_profit:
                exit_reason = "Target"
            elif change <= -stop_loss:
                exit_reason = "Stop"
            elif bars_held >= max_bars_held:
                exit_reason = "Time"

            if exit_reason:
                exit_value = shares * price
                trades.append({
                    "Entry Date": entry_date,
                    "Exit Date": date,
                    "Entry Price": entry_price,
                    "Exit Price": price,
                    "Shares": shares,
                    "P&L ($)": exit_value - shares * entry_price,
                    "P&L (%)": change * 100,
                    "Days Held": (date - entry_date).days,
                    "Exit Reason": exit_reason,
                })
                cash = exit_value
                shares = 0.0
                in_position = False
                entry_price = None
                entry_date = None
                bars_held = 0

        if not in_position and prev_close is not None and prev_mid is not None and mid_val is not None:
            if prev_close < prev_mid and price > mid_val:
                shares = cash / price
                entry_price = price
                entry_date = date
                cash = 0.0
                in_position = True
                bars_held = 0

        total_value = cash + (shares * price if in_position else 0.0)
        equity.append(total_value)

        if mid_val is not None:
            prev_close = price
            prev_mid = mid_val

    equity_curve = pd.Series(equity, index=df.index, name="Strategy")

    open_position = None
    if in_position:
        open_position = {"Entry Date": entry_date, "Entry Price": entry_price}

    return {"equity_curve": equity_curve, "trades": trades, "open_position": open_position}


def bb_mid_cross_signals(df: pd.DataFrame, params: dict | None = None) -> dict:
    """Return chart-overlay entry/exit points for the BB Mid-Cross Reversion
    strategy: {"entries": [(date, price), ...], "exits": [(date, price), ...]}.

    Reuses run_bb_mid_cross_single() so the overlay always matches what the
    backtester itself would have traded.
    """
    params = params or DEFAULT_PARAMS["BB Mid-Cross Reversion"]
    window = params.get("bb_window", 20)
    if df is None or df.empty or len(df) < window + 1:
        return {"entries": [], "exits": []}

    result = run_bb_mid_cross_single(df, params)
    entries = [(t["Entry Date"], t["Entry Price"]) for t in result["trades"]]
    exits = [(t["Exit Date"], t["Exit Price"]) for t in result["trades"]]
    if result["open_position"]:
        entries.append((result["open_position"]["Entry Date"], result["open_position"]["Entry Price"]))
    return {"entries": entries, "exits": exits}


def _bb_mid_cross_for_ticker(args) -> dict | None:
    ticker, start, end, params, per_ticker_capital = args
    hist = fetch_price_history(ticker, start, end)
    if hist.empty or len(hist) < params.get("bb_window", 20) + 5:
        return None

    result = run_bb_mid_cross_single(hist, params, per_ticker_capital)
    for t in result["trades"]:
        t["Ticker"] = ticker

    bh_shares = per_ticker_capital / float(hist["Close"].iloc[0])
    buy_hold = hist["Close"] * bh_shares

    return {
        "ticker": ticker,
        "equity_curve": result["equity_curve"],
        "trades": result["trades"],
        "buy_hold": buy_hold,
    }


def run_bb_mid_cross_watchlist(
    tickers: list, start: str, end: str, params: dict, initial_capital: float = 10_000.0,
) -> dict:
    """Run the BB Mid-Cross Reversion strategy across `tickers` and combine
    the results into one equal-weighted portfolio.

    Each ticker gets an equal slice of `initial_capital` and is simulated
    independently (single position at a time, per ticker); the per-ticker
    equity curves are then summed into one portfolio-level curve. Trades
    from every ticker are pooled for the win-rate / avg-win / avg-loss stats.
    """
    tickers = list(tickers)
    empty = {"equity_curve": pd.DataFrame(), "trades": [], "metrics": {}, "tickers_scanned": len(tickers), "tickers_traded": 0}
    if not tickers:
        return empty

    per_ticker_capital = initial_capital / len(tickers)
    call_args = [(t, start, end, params, per_ticker_capital) for t in tickers]

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        for res in executor.map(_bb_mid_cross_for_ticker, call_args):
            if res is not None:
                results.append(res)

    if not results:
        return empty

    all_trades = []
    strategy_curves = []
    bh_curves = []
    for res in results:
        all_trades.extend(res["trades"])
        strategy_curves.append(res["equity_curve"])
        bh_curves.append(res["buy_hold"])

    strategy_combined = pd.concat(strategy_curves, axis=1).sort_index().ffill().bfill()
    portfolio = strategy_combined.sum(axis=1)
    portfolio.name = "Strategy"

    bh_combined = pd.concat(bh_curves, axis=1).sort_index().ffill().bfill()
    buy_hold = bh_combined.sum(axis=1)
    buy_hold.name = "Buy & Hold"

    curve_df = pd.concat([portfolio, buy_hold], axis=1)

    all_trades.sort(key=lambda t: t["Entry Date"])
    metrics = compute_metrics(portfolio, all_trades, initial_capital)

    return {
        "equity_curve": curve_df,
        "trades": all_trades,
        "metrics": metrics,
        "tickers_scanned": len(tickers),
        "tickers_traded": len({t["Ticker"] for t in all_trades}) if all_trades else 0,
    }
