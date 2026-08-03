import datetime as dt

import plotly.graph_objects as go
import streamlit as st

from lib.backtester import DEFAULT_PARAMS, STRATEGIES, fetch_price_history, run_backtest
from lib.config import APP_NAME, inject_base_style, render_footer
from utils.theme import apply_dark_plotly

st.set_page_config(page_title=f"Backtester - {APP_NAME}", page_icon="🔬", layout="wide")
inject_base_style()

st.title("🔬 Strategy Backtester")
st.caption(
    "Test simple rules-based strategies against historical price data. Long-only, "
    "fully-invested-or-flat simulation - no leverage, shorting, fees, or slippage. "
    "Past performance does not predict future results."
)

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    ticker = st.text_input("Ticker", value="AAPL", key="bt_ticker").strip().upper()
with col2:
    default_start = dt.date.today() - dt.timedelta(days=365 * 3)
    date_range = st.date_input(
        "Date range",
        value=(default_start, dt.date.today()),
        max_value=dt.date.today(),
        key="bt_dates",
    )
with col3:
    initial_capital = st.number_input("Starting capital ($)", min_value=100.0, value=10_000.0, step=1000.0)

if not isinstance(date_range, tuple) or len(date_range) != 2:
    st.info("Select a start and end date.")
    render_footer()
    st.stop()

start_date, end_date = date_range

strategy = st.selectbox("Strategy", list(STRATEGIES.keys()), key="bt_strategy")
st.caption(STRATEGIES[strategy])

params = dict(DEFAULT_PARAMS[strategy])

with st.expander("Strategy parameters", expanded=(strategy == "Custom")):
    if strategy == "RSI Oversold":
        c1, c2, c3 = st.columns(3)
        params["rsi_period"] = c1.number_input("RSI period", min_value=2, max_value=100, value=params["rsi_period"])
        params["buy_threshold"] = c2.number_input("Buy below RSI", min_value=1, max_value=99, value=params["buy_threshold"])
        params["sell_threshold"] = c3.number_input("Sell above RSI", min_value=1, max_value=99, value=params["sell_threshold"])

    elif strategy == "Golden Cross":
        c1, c2 = st.columns(2)
        params["fast_ema"] = c1.number_input("Fast EMA", min_value=2, max_value=200, value=params["fast_ema"])
        params["slow_ema"] = c2.number_input("Slow EMA", min_value=5, max_value=400, value=params["slow_ema"])

    elif strategy == "Bollinger Band Bounce":
        c1, c2 = st.columns(2)
        params["bb_window"] = c1.number_input("Band window", min_value=5, max_value=100, value=params["bb_window"])
        params["bb_std"] = c2.number_input("Std deviations", min_value=0.5, max_value=4.0, value=float(params["bb_std"]), step=0.1)

    elif strategy == "MACD Crossover":
        c1, c2, c3 = st.columns(3)
        params["macd_fast"] = c1.number_input("Fast EMA", min_value=2, max_value=100, value=params["macd_fast"])
        params["macd_slow"] = c2.number_input("Slow EMA", min_value=5, max_value=200, value=params["macd_slow"])
        params["macd_signal"] = c3.number_input("Signal EMA", min_value=2, max_value=100, value=params["macd_signal"])

    elif strategy == "Custom":
        c1, c2, c3 = st.columns(3)
        params["rsi_period"] = c1.number_input("RSI period", min_value=2, max_value=100, value=params["rsi_period"])
        params["buy_threshold"] = c2.number_input("Buy below RSI", min_value=1, max_value=99, value=params["buy_threshold"])
        params["sell_threshold"] = c3.number_input("Sell above RSI", min_value=1, max_value=99, value=params["sell_threshold"])
        c4, c5, c6 = st.columns(3)
        params["use_ema_filter"] = c4.checkbox("Require EMA trend filter", value=params["use_ema_filter"])
        params["fast_ema"] = c5.number_input("Fast EMA", min_value=2, max_value=200, value=params["fast_ema"], disabled=not params["use_ema_filter"])
        params["slow_ema"] = c6.number_input("Slow EMA", min_value=5, max_value=400, value=params["slow_ema"], disabled=not params["use_ema_filter"])

run = st.button("Run backtest", type="primary")

# ---------------------------------------------------------------------------
# Run + display
# ---------------------------------------------------------------------------


def style_trades(df):
    display = df.copy()
    display["Entry Date"] = display["Entry Date"].dt.strftime("%Y-%m-%d")
    display["Exit Date"] = display["Exit Date"].dt.strftime("%Y-%m-%d")
    display["Entry Price"] = display["Entry Price"].map(lambda v: f"${v:,.2f}")
    display["Exit Price"] = display["Exit Price"].map(lambda v: f"${v:,.2f}")
    display["Shares"] = display["Shares"].map(lambda v: f"{v:,.2f}")
    display["P&L ($)"] = display["P&L ($)"].map(lambda v: f"${v:,.2f}")
    display["P&L (%)"] = display["P&L (%)"].map(lambda v: f"{v:+.2f}%")

    def highlight(row):
        raw = df.loc[row.name, "P&L ($)"]
        color = "#2ecc71" if raw >= 0 else "#e74c3c"
        styles = [""] * len(row)
        styles[row.index.get_loc("P&L ($)")] = f"color: {color}; font-weight: 600;"
        styles[row.index.get_loc("P&L (%)")] = f"color: {color}; font-weight: 600;"
        return styles

    return display.style.apply(highlight, axis=1)


def metric_row(metrics, container):
    cols = container.columns(6)
    cols[0].metric("Total Return", f"{metrics['Total Return %']:+.1f}%")
    cols[1].metric("Annual Return", f"{metrics['Annual Return %']:+.1f}%")
    cols[2].metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
    cols[3].metric("Max Drawdown", f"{metrics['Max Drawdown %']:.1f}%")
    cols[4].metric("Win Rate", f"{metrics['Win Rate %']:.0f}%")
    cols[5].metric("Total Trades", f"{metrics['Total Trades']:.0f}")


if run:
    if not ticker:
        st.warning("Enter a ticker.")
        render_footer()
        st.stop()

    with st.spinner(f"Loading price history for {ticker}..."):
        history = fetch_price_history(ticker, start_date.isoformat(), end_date.isoformat())

    if history.empty or len(history) < 50:
        st.warning(f"Not enough price history for {ticker} in this date range to run a backtest.")
        render_footer()
        st.stop()

    result = run_backtest(history, strategy, params, initial_capital)
    st.session_state["bt_last_result"] = {
        "ticker": ticker, "strategy": strategy, "result": result, "history": history,
    }

last = st.session_state.get("bt_last_result")

if last:
    result = last["result"]
    metrics = result["metrics"]
    curve = result["equity_curve"]
    trades = result["trades"]

    st.divider()
    st.subheader(f"{last['strategy']} on {last['ticker']}")
    metric_row(metrics, st)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve.index, y=curve["Strategy"], name="Strategy", line=dict(color="#3498db", width=2)))
    fig.add_trace(go.Scatter(x=curve.index, y=curve["Buy & Hold"], name="Buy & Hold", line=dict(color="#7f8c8d", width=2, dash="dot")))
    fig.update_layout(
        height=420, margin=dict(l=10, r=10, t=30, b=10),
        yaxis_title="Portfolio value ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    apply_dark_plotly(fig)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.subheader("Trade log")
    if not trades:
        st.info("No completed trades in this date range - the strategy never produced both an entry and an exit signal.")
    else:
        import pandas as pd
        trades_df = pd.DataFrame(trades)
        st.dataframe(style_trades(trades_df), use_container_width=True, hide_index=True, height=320)

    # -------------------------------------------------------------------
    # Strategy comparison
    # -------------------------------------------------------------------
    st.divider()
    st.subheader("Compare strategies")
    st.caption("Run every built-in strategy (default parameters) against the same ticker and date range.")

    if st.button("Compare all strategies"):
        history = last["history"]
        rows = []
        compare_curves = {}
        for strat_name in STRATEGIES:
            if strat_name == "Custom":
                continue
            res = run_backtest(history, strat_name, DEFAULT_PARAMS[strat_name], initial_capital)
            m = res["metrics"]
            rows.append({
                "Strategy": strat_name,
                "Total Return %": m["Total Return %"],
                "Annual Return %": m["Annual Return %"],
                "Sharpe Ratio": m["Sharpe Ratio"],
                "Max Drawdown %": m["Max Drawdown %"],
                "Win Rate %": m["Win Rate %"],
                "Total Trades": m["Total Trades"],
            })
            compare_curves[strat_name] = res["equity_curve"]["Strategy"]

        import pandas as pd
        compare_df = pd.DataFrame(rows)
        display = compare_df.copy()
        for col in ("Total Return %", "Annual Return %", "Max Drawdown %", "Win Rate %"):
            display[col] = display[col].map(lambda v: f"{v:+.1f}%" if "Drawdown" not in col and "Win" not in col else f"{v:.1f}%")
        display["Sharpe Ratio"] = display["Sharpe Ratio"].map(lambda v: f"{v:.2f}")
        st.dataframe(display, use_container_width=True, hide_index=True)

        fig2 = go.Figure()
        bh = compare_curves[next(iter(compare_curves))].index
        bh_curve = history["Close"] / float(history["Close"].iloc[0]) * initial_capital
        fig2.add_trace(go.Scatter(x=bh_curve.index, y=bh_curve, name="Buy & Hold", line=dict(color="#7f8c8d", width=2, dash="dot")))
        palette = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6"]
        for i, (name, series) in enumerate(compare_curves.items()):
            fig2.add_trace(go.Scatter(x=series.index, y=series, name=name, line=dict(color=palette[i % len(palette)], width=2)))
        fig2.update_layout(
            height=420, margin=dict(l=10, r=10, t=30, b=10),
            yaxis_title="Portfolio value ($)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        apply_dark_plotly(fig2)
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
else:
    st.info("Configure a strategy and click **Run backtest** to see results.")

render_footer()
