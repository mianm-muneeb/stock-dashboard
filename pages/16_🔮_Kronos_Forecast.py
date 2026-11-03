"""Kronos Forecast: multi-path candlestick forecast with calibration fixes and a backtest tab."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Streamlit Community Cloud mounts apps under /mount/src; its shared CPU gets throttled by heavy runs
ON_CLOUD = ROOT.as_posix().startswith("/mount/src")

from lib import kronos_utils as ku
from lib.config import APP_NAME, inject_base_style, render_footer
from utils.theme import apply_dark_plotly

st.set_page_config(page_title=f"Kronos Forecast - {APP_NAME}", page_icon="🔮", layout="wide")
inject_base_style()

MODEL_IDS = {"small": "NeoQuasar/Kronos-small", "base": "NeoQuasar/Kronos-base"}
BLUE = "#3fa3f5"


@st.cache_resource(show_spinner="Loading Kronos model (first run downloads weights)...")
def load_predictor(model_id: str):
    from kronos_model import Kronos, KronosPredictor, KronosTokenizer

    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained(model_id)
    return KronosPredictor(model, tokenizer, device="cpu", max_context=512)


@st.cache_data(ttl=3600, show_spinner=False)
def load_bars(ticker: str, interval: str) -> pd.DataFrame:
    return ku.fetch_bars(ticker, interval)


@st.cache_data(ttl=3600, show_spinner=False)
def load_earnings(ticker: str):
    return ku.next_earnings_date(ticker)


st.title("🔮 Kronos Forecast")
st.caption(
    "Open-source candlestick model (Kronos, MIT). The fan is a spread of simulated paths, "
    "rescaled to the stock's real volatility. It is not a price target."
)

tab_fc, tab_cal = st.tabs(["Forecast", "Calibration backtest"])

# ---------------------------------------------------------------------------
# Sidebar (forecast settings)
# ---------------------------------------------------------------------------
with st.sidebar:
    ticker = st.text_input("Ticker", "NVDA").strip().upper()
    interval = st.selectbox("Bar size", ["1d", "1h"], index=0)
    mode = st.radio("Model", ["Ensemble (small + base)", "Kronos-small", "Kronos-base"], index=1 if ON_CLOUD else 0)
    if ON_CLOUD:
        st.caption("Cloud CPU is shared and gets throttled. Use Kronos-small here; run Ensemble, Kronos-base and Calibration on your own PC.")
    horizon = st.slider("Forecast horizon (bars)", 5, 60, 20)
    lookback = st.slider("Lookback (bars)", 100, 512, 400)
    paths = st.slider("Simulated paths per model", 10, 80, 20, help="Ensemble on CPU takes a few minutes; Kronos-base is the slow one.")
    temp = st.slider("Temperature", 0.5, 1.5, 1.0, 0.05)
    top_p = st.slider("Top-p", 0.5, 1.0, 0.9, 0.05)
    st.markdown("**Calibration fixes**")
    use_anchor = st.checkbox("Cap the first-bar jump", value=True)
    use_rescale = st.checkbox("Rescale fan to realised volatility", value=True)
    drift_keep = st.slider("Trust in model drift", 0.0, 1.0, 0.5, 0.05,
                           help="0 = random walk from last close, 1 = trust the model fully. Use the Calibration tab to choose.")
    show_raw = st.checkbox("Show raw model band", value=False)
    holdout = st.checkbox("Reality check: hide the last N bars and compare", value=False)
    go_btn = st.button("Run forecast", type="primary")

PARAM_KEYS = ["ticker", "interval", "mode", "horizon", "lookback", "paths", "temp", "top_p",
              "use_anchor", "use_rescale", "drift_keep", "show_raw", "holdout"]
if go_btn:
    # A button is only True for one rerun; keep the settings so other widgets
    # (like "Show it anyway") don't wipe the forecast.
    _g = globals()
    st.session_state["kr_params"] = {k: _g[k] for k in PARAM_KEYS}


def models_for(mode_label: str):
    if mode_label.startswith("Ensemble"):
        return ["small", "base"]
    return ["small"] if mode_label.endswith("small") else ["base"]


# ---------------------------------------------------------------------------
# Forecast tab
# ---------------------------------------------------------------------------
def forecast_tab():
    P = st.session_state.get("kr_params")
    if not P:
        st.info("Pick a ticker (ASX: BHP.AX) and hit **Run forecast**.")
        return
    (ticker, interval, mode, horizon, lookback, paths, temp, top_p,
     use_anchor, use_rescale, drift_keep, show_raw, holdout) = (P[k] for k in PARAM_KEYS)

    bars = load_bars(ticker, interval)
    if bars.empty or len(bars) < lookback + (horizon if holdout else 0) + 10:
        st.error(f"Not enough data for {ticker}. Lower the lookback or try another ticker.")
        return

    actual = None
    if holdout:
        actual = bars.iloc[-horizon:]
        bars = bars.iloc[:-horizon]

    hist = bars.iloc[-lookback:]
    future = actual.index if holdout else ku.future_index(hist.index[-1], horizon, interval)
    last_close = float(hist["close"].iloc[-1])
    sigma = ku.daily_sigma(hist["close"])

    # Earnings gate: the model cannot see events, so forecasts across one are not meaningful
    blocked = False
    if not holdout and interval == "1d":
        edate = load_earnings(ticker)
        if edate is not None and hist.index[-1] < edate <= future[-1]:
            st.warning(
                f"{ticker} reports earnings on {edate.date()}, inside this forecast window. "
                "Kronos cannot see events, so treat the fan as noise."
            )
            blocked = not st.checkbox("Show it anyway", value=False, key="kr_anyway")
    if blocked:
        return

    which = models_for(mode)
    results = {}
    for name in which:
        predictor = load_predictor(MODEL_IDS[name])
        with st.spinner(f"Kronos-{name}: simulating {paths} paths..."):
            results[name] = ku.forecast_paths(predictor, hist, future, paths, temp, top_p)

    raw = np.concatenate(list(results.values()), axis=0)
    adj_parts = [ku.adjust(s, last_close, sigma, anchor=use_anchor, rescale=use_rescale, drift_keep=drift_keep)
                 for s in results.values()]
    sims = np.concatenate(adj_parts, axis=0)

    p05, p25, p50, p75, p95 = np.percentile(sims, [5, 25, 50, 75, 95], axis=0)
    prob_up = float((sims[:, -1] > last_close).mean())

    fig = go.Figure()
    tail = hist.iloc[-min(120, len(hist)):]
    fig.add_trace(go.Scatter(x=tail.index, y=tail["close"], name="History", line=dict(color="#e6e6e6", width=2)))
    for i in range(min(len(sims), 25)):
        fig.add_trace(go.Scatter(x=future, y=sims[i], mode="lines", line=dict(color="rgba(63,163,245,.15)", width=1),
                                 showlegend=False, hoverinfo="skip"))
    if show_raw:
        r05, r95 = np.percentile(raw, [5, 95], axis=0)
        fig.add_trace(go.Scatter(x=list(future) + list(future[::-1]), y=list(r95) + list(r05[::-1]), fill="toself",
                                 fillcolor="rgba(255,176,32,.08)", line=dict(color="rgba(255,176,32,.5)", dash="dot", width=1),
                                 name="Raw model 5-95%"))
    fig.add_trace(go.Scatter(x=list(future) + list(future[::-1]), y=list(p95) + list(p05[::-1]), fill="toself",
                             fillcolor="rgba(63,163,245,.12)", line=dict(width=0), name="5-95%"))
    fig.add_trace(go.Scatter(x=list(future) + list(future[::-1]), y=list(p75) + list(p25[::-1]), fill="toself",
                             fillcolor="rgba(63,163,245,.25)", line=dict(width=0), name="25-75%"))
    fig.add_trace(go.Scatter(x=future, y=p50, name="Median path", line=dict(color=BLUE, width=3)))
    if actual is not None:
        fig.add_trace(go.Scatter(x=actual.index, y=actual["close"], name="Actual", line=dict(color="#ffb020", width=3)))
    apply_dark_plotly(fig)
    fig.update_layout(height=540, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"), yaxis_title="Close")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last close", f"{last_close:,.2f}")
    c2.metric("Median end", f"{p50[-1]:,.2f}", f"{(p50[-1] / last_close - 1) * 100:+.1f}%")
    c3.metric("Paths ending higher", f"{prob_up * 100:.0f}%")
    c4.metric("5-95% end range", f"{p05[-1]:,.0f} - {p95[-1]:,.0f}")

    # Model agreement
    if len(results) == 2:
        cons = ku.consensus(adj_parts[0], adj_parts[1], last_close, sigma)
        if cons["agree"]:
            st.success(f"Models agree: {cons['direction']} (small {cons['a'] * 100:+.1f}%, base {cons['b'] * 100:+.1f}% median).")
        else:
            st.error(f"No consensus: small says {cons['a'] * 100:+.1f}%, base says {cons['b'] * 100:+.1f}%. Treat as no view.")

    # What the fixes did
    sims_raw_end = np.log(raw[:, -1] / last_close)
    st.caption(
        f"Realised one-bar volatility {sigma * 100:.2f}%. Raw model spread at the end: {sims_raw_end.std() * 100:.1f}% "
        f"vs {sigma * np.sqrt(len(future)) * 100:.1f}% implied by realised volatility."
    )

    if actual is not None:
        act_end = float(actual["close"].iloc[-1])
        inside = p05[-1] <= act_end <= p95[-1]
        st.write(
            f"Actual end **{act_end:,.2f}** vs median **{p50[-1]:,.2f}** "
            f"({(p50[-1] / act_end - 1) * 100:+.1f}% off). Inside the 5-95% band: **{'yes' if inside else 'no'}**."
        )


# ---------------------------------------------------------------------------
# Calibration tab
# ---------------------------------------------------------------------------
def calibration_tab():
    if ON_CLOUD:
        st.warning("Calibration is CPU-heavy and Streamlit Cloud throttles heavy apps. Run this tab on your own PC (streamlit run app.py).")
    st.markdown(
        "Forecasts from several **past** dates, scored against what actually happened. "
        "A well-calibrated 90% band contains the real price about 90% of the time; "
        "direction needs to beat a coin flip by more than the noise margin shown below."
    )
    c1, c2, c3 = st.columns(3)
    tickers_txt = c1.text_input("Tickers (comma separated)", "VST, ORCL, NFLX")
    cal_model = c2.selectbox("Model", ["small", "base"], index=0)
    cal_h = c3.slider("Horizon (bars)", 5, 40, 10, key="cal_h")
    d1, d2, d3 = st.columns(3)
    origins = d1.slider("Past dates per ticker", 3, 20, 5)
    spacing = d2.slider("Bars between dates", 5, 30, 10)
    cal_paths = d3.slider("Paths per forecast", 10, 40, 15, key="cal_paths")
    tick_list = [t.strip().upper() for t in tickers_txt.split(",") if t.strip()]
    n_runs = len(tick_list) * origins
    st.caption(f"{n_runs} forecasts. On CPU expect roughly {max(1, round(n_runs * 0.4))}-{max(2, round(n_runs * 1.0))} minutes with Kronos-small (Kronos-base is several times slower). "
               "Dates are the most recent windows only, so they sit after the model's training data.")

    if not st.button("Run calibration"):
        return

    bars_by = {}
    for t in tick_list:
        b = load_bars(t, "1d")
        if len(b) >= lookback + cal_h + origins * spacing + 10:
            bars_by[t] = b
        else:
            st.warning(f"Skipping {t}: not enough data.")
    if not bars_by:
        return

    predictor = load_predictor(MODEL_IDS[cal_model])
    bar = st.progress(0.0, text="Starting...")
    df = ku.run_calibration(
        predictor, bars_by, horizon=cal_h, lookback=lookback, origins=origins, spacing=spacing,
        paths=cal_paths, temp=temp, top_p=top_p, drift_keep=drift_keep,
        progress=lambda f, msg: bar.progress(min(1.0, f), text=msg),
    )
    bar.empty()
    if df.empty:
        st.error("No forecasts scored.")
        return

    summ = ku.summarise_calibration(df)
    n = int(summ["n forecasts"].iloc[0])
    margin = 1.96 * np.sqrt(0.25 / max(n, 1)) * 100
    st.subheader("Results (percent)")
    st.dataframe(summ, use_container_width=True)
    st.caption(
        f"n = {n} per variant. Direction hit rate is only meaningful if it is above 50% by more than about ±{margin:.0f} points. "
        "'Median error' is the average miss of the median end price; compare it with the 'no change' error. "
        "If the model does not beat 'no change', its median adds nothing."
    )
    adj = summ.loc["adjusted"] if "adjusted" in summ.index else None
    if adj is not None:
        cov = adj["actual inside 90% band"]
        if 82 <= cov <= 96:
            st.success(f"Adjusted 90% band is roughly calibrated ({cov:.0f}% coverage).")
        else:
            st.warning(f"Adjusted 90% band covers {cov:.0f}% (target 90%). Don't trust the fan width yet.")
        if adj["direction right"] < 50 + margin:
            st.warning("Direction accuracy is within coin-flip noise: set 'Trust in model drift' to 0.")
        else:
            st.success("Direction beats coin-flip noise on this sample. Re-run on more tickers before relying on it.")
    with st.expander("Every forecast"):
        st.dataframe(df, use_container_width=True)


with tab_fc:
    forecast_tab()
with tab_cal:
    calibration_tab()

render_footer()
