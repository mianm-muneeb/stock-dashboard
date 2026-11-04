"""Kronos Forecast: multi-path candlestick forecast with calibration fixes and a backtest tab."""
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

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

    import torch

    # Uses the GPU automatically when this machine has a CUDA build of PyTorch (Streamlit Cloud has none)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained(model_id)
    return KronosPredictor(model, tokenizer, device=device, max_context=512)


@st.cache_data(ttl=3600, show_spinner=False)
def load_bars(ticker: str, interval: str) -> pd.DataFrame:
    return ku.fetch_bars(ticker, interval)


@st.cache_data(ttl=3600, show_spinner=False)
def load_earnings(ticker: str):
    return ku.next_earnings_date(ticker)


# Full-model results are computed on GitHub Actions (free 4-core runners) and published to the
# kronos-data branch; this page only reads them, so the app never runs the heavy models.
SAVED_BASE = "https://raw.githubusercontent.com/muneeb7777/stock-dashboard/kronos-data/data/kronos"
SAVED_LOCAL = ROOT / "data" / "kronos"


@st.cache_data(ttl=600, show_spinner=False)
def _fetch_saved(rel: str):
    # Raises on failure, and Streamlit does not cache exceptions, so a miss is retried on the next run
    req = urllib.request.Request(f"{SAVED_BASE}/{rel}", headers={"User-Agent": "stock-dashboard"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode("utf-8"))


def load_saved(rel: str):
    """Saved JSON from the kronos-data branch (falls back to ./data/kronos). None if unavailable."""
    try:
        return _fetch_saved(rel)
    except Exception:
        p = SAVED_LOCAL / rel
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None


st.title("🔮 Kronos Forecast")
st.caption(
    "Open-source candlestick model (Kronos, MIT). The fan is a spread of simulated paths, "
    "rescaled to the stock's real volatility. It is not a price target."
)

with st.form("kr_form", border=False):
    _c1, _c2 = st.columns([4, 1], vertical_alignment="bottom")
    ticker = _c1.text_input("Ticker (ASX: BHP.AX)", "", placeholder="e.g. VST").strip().upper()
    go_btn = _c2.form_submit_button("Run forecast", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Sidebar (forecast settings)
# ---------------------------------------------------------------------------
with st.sidebar:
    interval = st.selectbox("Bar size", ["1d", "1h"], index=0)
    mode = st.radio("Model", ["Ensemble (small + base)", "Kronos-small", "Kronos-base"], index=1 if ON_CLOUD else 0)
    if ON_CLOUD:
        st.caption("These settings drive the Live forecast tab. Cloud CPU is shared and throttled, so use Kronos-small here. "
                   "Ensemble and Kronos-base results are precomputed nightly in the Full model (saved) tab.")
    horizon = st.slider("Forecast horizon (bars)", 5, 252, 126, help="126 daily bars is about 6 months.")
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

PARAM_KEYS = ["ticker", "interval", "mode", "horizon", "lookback", "paths", "temp", "top_p",
              "use_anchor", "use_rescale", "drift_keep", "show_raw", "holdout"]
if go_btn and ticker:
    # A button is only True for one rerun; keep the settings so other widgets
    # (like "Show it anyway") don't wipe the forecast.
    _g = globals()
    st.session_state["kr_params"] = {k: _g[k] for k in PARAM_KEYS}


def models_for(mode_label: str):
    if mode_label.startswith("Ensemble"):
        return ["small", "base"]
    return ["small"] if mode_label.endswith("small") else ["base"]


def _animated_fan(ticker, hist, future, sims, raw, p05, p25, p50, p75, p95, last_close, show_raw, actual):
    """Animated, auto-playing fan chart (plotly.js frames in an iframe)."""
    tail = hist.iloc[-min(260, len(hist)):]
    x0 = hist.index[-1]
    fx = [x0] + list(future)
    pre = lambda a: np.concatenate([[last_close], np.asarray(a, dtype=float)])
    P05, P25, P50, P75, P95 = (pre(a) for a in (p05, p25, p50, p75, p95))
    n_paths = min(len(sims), 30)
    paths = [pre(sims[i]) for i in range(n_paths)]
    H = len(future)
    steps = min(H, 60)
    idxs = sorted(set(int(v) for v in np.linspace(1, H, steps)))

    CY, MG = "0,229,255", "255,64,200"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tail.index, y=tail["close"], name="History", mode="lines",
                             line=dict(color="rgba(255,255,255,.55)", width=5), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=tail.index, y=tail["close"], name="History", mode="lines",
                             line=dict(color="#f2f6ff", width=2), hovertemplate="%{x|%d %b %Y}: %{y:,.2f}<extra></extra>"))
    first_path = len(fig.data)
    for p in paths:
        fig.add_trace(go.Scatter(x=[x0], y=[last_close], mode="lines", line=dict(color=f"rgba({CY},.16)", width=1),
                                 hoverinfo="skip", showlegend=False))
    i95 = len(fig.data)
    fig.add_trace(go.Scatter(x=[x0], y=[last_close], fill="toself", fillcolor=f"rgba({CY},.10)",
                             line=dict(color=f"rgba({CY},.35)", width=1, dash="dot"), name="5-95%", hoverinfo="skip"))
    i75 = len(fig.data)
    fig.add_trace(go.Scatter(x=[x0], y=[last_close], fill="toself", fillcolor=f"rgba({CY},.22)",
                             line=dict(width=0), name="25-75%", hoverinfo="skip"))
    iglow = len(fig.data)
    fig.add_trace(go.Scatter(x=[x0], y=[last_close], mode="lines", line=dict(color=f"rgba({CY},.28)", width=12),
                             hoverinfo="skip", showlegend=False))
    imed = len(fig.data)
    fig.add_trace(go.Scatter(x=[x0], y=[last_close], mode="lines", name="Median path", line=dict(color=f"rgb({CY})", width=3),
                             hovertemplate="%{x|%d %b %Y}: %{y:,.2f}<extra>median</extra>"))
    ihg = len(fig.data)
    fig.add_trace(go.Scatter(x=[x0], y=[last_close], mode="markers", marker=dict(size=26, color=f"rgba({MG},.25)"),
                             hoverinfo="skip", showlegend=False))
    ihd = len(fig.data)
    fig.add_trace(go.Scatter(x=[x0], y=[last_close], mode="markers+text", text=[""], textposition="middle right",
                             textfont=dict(color="#ffffff", size=15), marker=dict(size=10, color=f"rgb({MG})",
                             line=dict(color="#fff", width=1.5)), hoverinfo="skip", showlegend=False))
    if actual is not None:
        fig.add_trace(go.Scatter(x=actual.index, y=actual["close"], name="Actual", line=dict(color="#ffb020", width=3)))
    if show_raw:
        r05, r95 = np.percentile(raw, [5, 95], axis=0)
        fig.add_trace(go.Scatter(x=list(future) + list(future[::-1]), y=list(r95) + list(r05[::-1]), fill="toself",
                                 fillcolor="rgba(255,176,32,.06)", line=dict(color="rgba(255,176,32,.5)", dash="dot", width=1),
                                 name="Raw model 5-95%"))

    traces = [first_path + i for i in range(n_paths)] + [i95, i75, iglow, imed, ihg, ihd]
    frames = []
    for k in idxs:
        xs = fx[:k + 1]
        data = [go.Scatter(x=xs, y=p[:k + 1]) for p in paths]
        data.append(go.Scatter(x=xs + xs[::-1], y=list(P95[:k + 1]) + list(P05[:k + 1][::-1])))
        data.append(go.Scatter(x=xs + xs[::-1], y=list(P75[:k + 1]) + list(P25[:k + 1][::-1])))
        data.append(go.Scatter(x=xs, y=P50[:k + 1]))
        data.append(go.Scatter(x=xs, y=P50[:k + 1]))
        data.append(go.Scatter(x=[xs[-1]], y=[P50[k]]))
        chg = (P50[k] / last_close - 1) * 100
        data.append(go.Scatter(x=[xs[-1]], y=[P50[k]], text=[f"  {P50[k]:,.2f}  ({chg:+.1f}%)"]))
        frames.append(go.Frame(data=data, traces=traces, name=str(k)))
    fig.frames = frames

    ylo = float(min(tail["close"].min(), P05.min())); yhi = float(max(tail["close"].max(), P95.max()))
    pad = (yhi - ylo) * 0.06
    span = (future[-1] - tail.index[0])
    fig.update_layout(
        height=700, margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="#07090f", plot_bgcolor="#07090f", font=dict(color="#9fb3c8"),
        title=dict(text=f"<b>{ticker}</b>  ·  Kronos projection  ·  next {H} bars", x=0.01, font=dict(size=20, color="#e8f4ff")),
        xaxis=dict(range=[tail.index[0], future[-1] + span * 0.12], showgrid=True, gridcolor="rgba(0,229,255,.06)",
                   zeroline=False, showspikes=True, spikecolor="rgba(0,229,255,.4)", spikethickness=1),
        yaxis=dict(range=[ylo - pad, yhi + pad], showgrid=True, gridcolor="rgba(0,229,255,.06)", zeroline=False,
                   side="right", tickformat=",.0f"),
        legend=dict(orientation="h", y=1.04, x=1, xanchor="right", bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
        shapes=[dict(type="line", x0=x0, x1=x0, y0=0, y1=1, yref="paper", line=dict(color="rgba(255,255,255,.25)", width=1, dash="dash"))],
        annotations=[dict(x=x0, y=1, yref="paper", text="NOW", showarrow=False, xanchor="right", font=dict(size=11, color="#7e93a8"))],
        updatemenus=[dict(type="buttons", showactive=False, x=0.01, y=0.99, xanchor="left", yanchor="top",
                          bgcolor="#0f1722", bordercolor="rgba(0,229,255,.5)", font=dict(color="#00e5ff"),
                          buttons=[dict(label="⟲ Replay", method="animate",
                                        args=[None, dict(frame=dict(duration=20, redraw=True), fromcurrent=False,
                                                         transition=dict(duration=0), mode="immediate")])])],
    )
    html = fig.to_html(include_plotlyjs="cdn", full_html=False, auto_play=True, config={"displayModeBar": False},
                       default_height="700px")
    components.html(f'<div style="background:#07090f">{html}</div>', height=715)


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

    _animated_fan(ticker, hist, future, sims, raw, p05, p25, p50, p75, p95, last_close, show_raw, actual)
    if len(future) > 60:
        st.caption("Past about 60 bars the fan is mostly volatility, not signal. Read it as a range of outcomes.")

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
        st.warning("Running calibration here is CPU-heavy and Streamlit Cloud throttles heavy apps. "
                   "The weekly saved results are above; run your own on a local copy (streamlit run app.py).")
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


# ---------------------------------------------------------------------------
# Saved full-model tab (Ensemble / small / base, computed nightly on GitHub Actions)
# ---------------------------------------------------------------------------
def _hours_since(iso: str) -> float:
    t = pd.Timestamp(iso)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return (pd.Timestamp.now(tz="UTC") - t).total_seconds() / 3600


def _ago(hours: float) -> str:
    return f"{hours:.0f} h ago" if hours < 48 else f"{hours / 24:.0f} days ago"


def saved_tab():
    idx = load_saved("index.json")
    if not idx or not idx.get("forecasts"):
        st.info("No saved run yet. On GitHub open Actions, pick **Kronos forecast (nightly)** and press Run workflow. "
                "The first run takes about 15 minutes.")
        return
    rows = idx["forecasts"]
    hrs = _hours_since(idx["generated_at"])
    st.caption(
        "Ensemble = Kronos-small + Kronos-base, computed on GitHub's free 4-core runners every weekday after the US close. "
        f"Last update {_ago(hrs)}."
    )
    if hrs > 80:
        st.warning("These results are more than 3 days old, so the nightly job may have failed.")

    table = pd.DataFrame([{
        "Ticker": r["ticker"],
        "Last close": r["last_close"],
        "Last bar": r["last_date"],
        "Ensemble median end": f"{r['ensemble_median_end_pct']:+.1f}%",
        "Paths ending higher": f"{r['ensemble_prob_up'] * 100:.0f}%",
        "Small vs base": "agree" if r["agree"] else "disagree",
        "Earnings in window": (r["earnings_date"] or "yes") if r["earnings_in_window"] else "no",
    } for r in rows])
    st.dataframe(table, hide_index=True, use_container_width=True)

    c1, c2 = st.columns([1, 2])
    pick = c1.selectbox("Ticker", [r["ticker"] for r in rows], key="kr_saved_ticker")
    which = c2.radio("Model", ["Ensemble", "Small", "Base"], horizontal=True, key="kr_saved_model")
    row = next(r for r in rows if r["ticker"] == pick)
    d = load_saved("forecasts/" + urllib.parse.quote(row["file"]))
    if not d:
        st.error("Could not load that forecast file.")
        return

    if d.get("earnings_in_window"):
        st.warning(
            f"{pick} reports earnings on {d['earnings_date']}, inside this forecast window. "
            "Kronos cannot see events, so treat the fan as noise."
        )
        if not st.checkbox("Show it anyway", value=False, key=f"kr_saved_anyway_{pick}"):
            return

    v = d["variants"][which.lower()]
    hist_x = pd.to_datetime(d["history"]["dates"])
    fut = pd.to_datetime(d["future_dates"])
    last_close = d["last_close"]
    show_raw_saved = st.checkbox("Show raw model band", value=False, key="kr_saved_raw")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_x, y=d["history"]["close"], name="History", line=dict(color="#e6e6e6", width=2)))
    for path in v["paths"]:
        fig.add_trace(go.Scatter(x=fut, y=path, mode="lines", line=dict(color="rgba(63,163,245,.15)", width=1),
                                 showlegend=False, hoverinfo="skip"))
    if show_raw_saved:
        fig.add_trace(go.Scatter(x=list(fut) + list(fut[::-1]), y=v["raw_p95"] + v["raw_p05"][::-1], fill="toself",
                                 fillcolor="rgba(255,176,32,.08)", line=dict(color="rgba(255,176,32,.5)", dash="dot", width=1),
                                 name="Raw model 5-95%"))
    fig.add_trace(go.Scatter(x=list(fut) + list(fut[::-1]), y=v["p95"] + v["p05"][::-1], fill="toself",
                             fillcolor="rgba(63,163,245,.12)", line=dict(width=0), name="5-95%"))
    fig.add_trace(go.Scatter(x=list(fut) + list(fut[::-1]), y=v["p75"] + v["p25"][::-1], fill="toself",
                             fillcolor="rgba(63,163,245,.25)", line=dict(width=0), name="25-75%"))
    fig.add_trace(go.Scatter(x=fut, y=v["p50"], name="Median path", line=dict(color=BLUE, width=3)))
    apply_dark_plotly(fig)
    fig.update_layout(height=540, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"), yaxis_title="Close")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Last close", f"{last_close:,.2f}")
    m2.metric("Median end", f"{v['p50'][-1]:,.2f}", f"{v['median_end_pct']:+.1f}%")
    m3.metric("Paths ending higher", f"{v['prob_up'] * 100:.0f}%")
    m4.metric("5-95% end range", f"{v['p05'][-1]:,.0f} - {v['p95'][-1]:,.0f}")

    cons = d["consensus"]
    if cons["agree"]:
        st.success(f"Models agree: {cons['direction']} (small {cons['a'] * 100:+.1f}%, base {cons['b'] * 100:+.1f}% median).")
    else:
        st.error(f"No consensus: small says {cons['a'] * 100:+.1f}%, base says {cons['b'] * 100:+.1f}%. Treat as no view.")

    st.caption(
        f"{d['horizon']} bars ahead from {d['last_date']}. Realised one-bar volatility {d['sigma_pct']:.2f}%. "
        f"{d['paths_per_model']} paths per model, drift trust {d['drift_keep']}, lookback {d['lookback']} bars."
    )


def _cal_verdict(summ: pd.DataFrame, margin: float):
    if "adjusted" not in summ.index:
        return
    adj = summ.loc["adjusted"]
    cov = adj["actual inside 90% band"]
    if 82 <= cov <= 96:
        st.success(f"Adjusted 90% band is roughly calibrated ({cov:.0f}% coverage).")
    else:
        st.warning(f"Adjusted 90% band covers {cov:.0f}% (target 90%). Don't trust the fan width yet.")
    if adj["direction right"] < 50 + margin:
        st.warning("Direction accuracy is within coin-flip noise: keep 'Trust in model drift' low.")
    else:
        st.success("Direction beats coin-flip noise on this sample.")
    if adj["median error"] >= adj["'no change' error"]:
        st.warning("The model's median is no closer to the real price than 'no change'.")
    else:
        st.success("The model's median lands closer to the real price than 'no change'.")


def saved_calibration():
    idx = load_saved("index.json")
    cals = (idx or {}).get("calibrations") or []
    if not cals:
        st.info("No saved calibration yet. It runs every Saturday on GitHub Actions.")
        return
    st.subheader("Saved calibration (weekly, GitHub runner)")
    for fn in cals:
        c = load_saved(fn)
        if not c:
            continue
        summ = pd.DataFrame(c["summary"]).set_index("variant")
        n = int(summ["n forecasts"].iloc[0])
        margin = 1.96 * np.sqrt(0.25 / max(n, 1)) * 100
        st.markdown(
            f"**Kronos-{c['model']}** on {', '.join(c['tickers'])}: {c['horizon']}-bar forecasts from the last "
            f"{c['origins']} non-overlapping dates each, {c['paths']} paths, drift trust {c['drift_keep']}. "
            f"Run {_ago(_hours_since(c['generated_at']))}."
        )
        st.dataframe(summ, use_container_width=True)
        st.caption(
            f"n = {n} per variant. Direction hit rate is only meaningful if it is above 50% by more than about ±{margin:.0f} points. "
            "'Median error' is the average miss of the median end price; compare it with the 'no change' error."
        )
        _cal_verdict(summ, margin)
        with st.expander(f"Every forecast (Kronos-{c['model']})"):
            st.dataframe(pd.DataFrame(c["records"]), use_container_width=True)


forecast_tab()
with st.expander("Calibration (optional)"):
    calibration_tab()

render_footer()
