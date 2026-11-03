"""Kronos Forecast page: multi-path price forecast fan chart for any ticker.

Drop into the dashboard's pages/ folder. Needs kronos_model/ (vendored, MIT)
at the repo root and the extra requirements in requirements_kronos.txt.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.config import APP_NAME, inject_base_style, render_footer
from utils.theme import apply_dark_plotly

st.set_page_config(page_title=f"Kronos Forecast - {APP_NAME}", page_icon="🔮", layout="wide")
inject_base_style()

MODELS = {
    "Kronos-small (25M, fast)": "NeoQuasar/Kronos-small",
    "Kronos-base (102M, slower)": "NeoQuasar/Kronos-base",
}
BLUE = "#3fa3f5"


@st.cache_resource(show_spinner="Loading Kronos model (first run downloads weights)...")
def load_predictor(model_id: str):
    import torch
    from kronos_model import Kronos, KronosPredictor, KronosTokenizer

    torch.set_num_threads(max(1, torch.get_num_threads()))
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained(model_id)
    return KronosPredictor(model, tokenizer, device="cpu", max_context=512)


@st.cache_data(ttl=3600, show_spinner=False)
def load_bars(ticker: str, interval: str) -> pd.DataFrame:
    period = "5y" if interval == "1d" else "60d"
    df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.dropna()


def future_index(last: pd.Timestamp, n: int, interval: str) -> pd.DatetimeIndex:
    if interval == "1d":
        return pd.bdate_range(last + pd.Timedelta(days=1), periods=n)
    step = pd.Timedelta(hours=1)
    idx = pd.date_range(last + step, periods=n * 3, freq="h")
    idx = idx[(idx.weekday < 5) & (idx.hour >= 9) & (idx.hour <= 16)]
    return idx[:n]


def run_forecast(predictor, hist: pd.DataFrame, future: pd.DatetimeIndex, paths: int, temp: float, top_p: float):
    x_df = hist[["open", "high", "low", "close", "volume"]].reset_index(drop=True)
    x_ts = pd.Series(hist.index)
    y_ts = pd.Series(future)
    # predict() averages samples, so run independent single-sample paths as a batch
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
    return np.stack([o["close"].values for o in outs])  # (paths, horizon)


st.title("🔮 Kronos Forecast")
st.caption("Open-source candlestick foundation model (Kronos, MIT). Output is a spread of simulated paths, not a price target.")

with st.sidebar:
    ticker = st.text_input("Ticker", "NVDA").strip().upper()
    interval = st.selectbox("Bar size", ["1d", "1h"], index=0)
    model_label = st.selectbox("Model", list(MODELS))
    horizon = st.slider("Forecast horizon (bars)", 5, 60, 20)
    lookback = st.slider("Lookback (bars)", 100, 512, 400)
    paths = st.slider("Simulated paths", 5, 60, 20)
    temp = st.slider("Temperature", 0.5, 1.5, 1.0, 0.05)
    top_p = st.slider("Top-p", 0.5, 1.0, 0.9, 0.05)
    holdout = st.checkbox("Reality check: hide the last N bars and compare", value=False)
    go_btn = st.button("Run forecast", type="primary")

if not go_btn:
    st.info("Pick a ticker (ASX: BHP.AX) and hit **Run forecast**.")
    render_footer()
    st.stop()

bars = load_bars(ticker, interval)
if bars.empty or len(bars) < lookback + (horizon if holdout else 0) + 10:
    st.error(f"Not enough data for {ticker}. Lower the lookback or try another ticker.")
    st.stop()

actual = None
if holdout:
    actual = bars.iloc[-horizon:]
    bars = bars.iloc[:-horizon]

hist = bars.iloc[-lookback:]
future = actual.index if holdout else future_index(hist.index[-1], horizon, interval)

predictor = load_predictor(MODELS[model_label])
with st.spinner(f"Simulating {paths} paths..."):
    sims = run_forecast(predictor, hist, future, paths, temp, top_p)

last_close = float(hist["close"].iloc[-1])
p05, p25, p50, p75, p95 = np.percentile(sims, [5, 25, 50, 75, 95], axis=0)
final = sims[:, -1]
prob_up = float((final > last_close).mean())

fig = go.Figure()
tail = hist.iloc[-min(120, len(hist)):]
fig.add_trace(go.Scatter(x=tail.index, y=tail["close"], name="History", line=dict(color="#e6e6e6", width=2)))
for i in range(min(paths, 25)):
    fig.add_trace(go.Scatter(x=future, y=sims[i], mode="lines", line=dict(color="rgba(63,163,245,.18)", width=1), showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=list(future) + list(future[::-1]), y=list(p95) + list(p05[::-1]), fill="toself", fillcolor="rgba(63,163,245,.12)", line=dict(width=0), name="5-95%"))
fig.add_trace(go.Scatter(x=list(future) + list(future[::-1]), y=list(p75) + list(p25[::-1]), fill="toself", fillcolor="rgba(63,163,245,.25)", line=dict(width=0), name="25-75%"))
fig.add_trace(go.Scatter(x=future, y=p50, name="Median path", line=dict(color=BLUE, width=3)))
if actual is not None:
    fig.add_trace(go.Scatter(x=actual.index, y=actual["close"], name="Actual", line=dict(color="#ffb020", width=3)))
apply_dark_plotly(fig)
fig.update_layout(height=560, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"), yaxis_title="Close")
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

c1, c2, c3, c4 = st.columns(4)
c1.metric("Last close", f"{last_close:,.2f}")
c2.metric("Median end", f"{p50[-1]:,.2f}", f"{(p50[-1] / last_close - 1) * 100:+.1f}%")
c3.metric("Paths ending higher", f"{prob_up * 100:.0f}%")
c4.metric("5-95% end range", f"{p05[-1]:,.0f} - {p95[-1]:,.0f}")

if actual is not None:
    act_end = float(actual["close"].iloc[-1])
    inside = p05[-1] <= act_end <= p95[-1]
    st.write(
        f"Actual end **{act_end:,.2f}** vs median **{p50[-1]:,.2f}** "
        f"({(p50[-1] / act_end - 1) * 100:+.1f}% off). Inside the 5-95% band: **{'yes' if inside else 'no'}**."
    )

render_footer()
