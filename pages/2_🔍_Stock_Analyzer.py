import datetime as dt
import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import claude_analyst
from lib.charts import OVERLAY_OPTIONS, PANEL_OPTIONS, render_gauge, render_technical_chart
from lib.config import APP_NAME, fmt_large_number, fmt_num, fmt_pct, get_plotly_theme, inject_base_style, render_footer
from lib.logos import get_logo_or_placeholder
from lib.market_data import (
    TIMEFRAMES,
    get_extended_hours_data,
    get_history,
    get_history_tf,
    get_option_chain,
    get_option_expirations,
    get_prev_close,
    get_stock_fundamentals,
    get_valuation_data,
    is_etf,
)
from lib.news import ticker_news, time_ago
from lib.signals import at_a_glance, bs_delta, fundamental_score, max_pain_strike, technical_score
from utils.theme import apply_dark_plotly


def fmt(val, decimals=2, prefix="", suffix=""):
    """Safely format a numeric metric value, tolerating None, NaN, and non-numeric types."""
    try:
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return "–"
        return f"{prefix}{val:.{decimals}f}{suffix}"
    except (TypeError, ValueError):
        return "–"


st.set_page_config(page_title=f"Stock Analyzer - {APP_NAME}", page_icon="🔍", layout="wide")
inject_base_style()

st.title("🔍 Stock Analyzer")

col_input, col_period = st.columns([2, 3])
with col_input:
    default_ticker = st.session_state.pop("switch_ticker", None) or st.query_params.get("ticker", "AAPL")
    ticker = st.text_input("Ticker", value=default_ticker).strip().upper()
with col_period:
    timeframe = st.segmented_control(
        "Timeframe", options=list(TIMEFRAMES.keys()), default="1d", key="analyzer_timeframe"
    )
    timeframe = timeframe or "1d"

if not ticker:
    st.stop()

if is_etf(ticker):
    st.warning(f"{ticker} looks like an ETF. Try the ETF Analyzer page for fund-level data.")

try:
    fund = get_stock_fundamentals(ticker)
except Exception:
    st.error("Yahoo Finance is rate-limiting right now. This is temporary — try again in 30 seconds.")
    if st.button("Retry"):
        st.rerun()
    st.stop()

if not fund.get("price"):
    st.error("Yahoo Finance is rate-limiting right now. This is temporary — try again in 30 seconds.")
    if st.button("Retry"):
        st.rerun()
    st.stop()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

header_cols = st.columns([1, 4, 2, 2, 2, 2])
with header_cols[0]:
    st.image(get_logo_or_placeholder(ticker), width=64)
with header_cols[1]:
    st.markdown(f"### {fund['name']} ({ticker})")
    sector_industry = " · ".join(filter(None, [fund.get("sector"), fund.get("industry")]))
    st.caption(sector_industry or "—")
with header_cols[2]:
    st.metric("Price", fmt(fund.get("price"), prefix="$"))
with header_cols[3]:
    st.metric("Market Cap", fmt_large_number(fund.get("market_cap")))
with header_cols[4]:
    pe = fund.get("trailing_pe")
    st.metric("Trailing P/E", fmt(pe))
with header_cols[5]:
    beta = fund.get("beta")
    st.metric("Beta", fmt(beta))

# ---------------------------------------------------------------------------
# Extended hours banner (shown only outside regular market hours)
# ---------------------------------------------------------------------------

_ext = get_extended_hours_data(ticker)
if _ext:
    _session_label = "PRE-MARKET" if _ext["session"] == "pre" else "AFTER-HOURS"
    _pct = _ext["pct_change"]
    _chg = _ext["change"]
    _color = "#2ecc71" if (_pct or 0) >= 0 else "#e74c3c"
    _banner_cls = "tv-banner-bull" if (_pct or 0) >= 0 else "tv-banner-bear"

    _price_str = f"${_ext['price']:,.2f}"
    _pct_str   = f"{_pct:+.2f}%" if _pct is not None else "—"
    _chg_str   = f"${_chg:+.2f}" if _chg is not None else "—"
    _vol_str   = f"{_ext['volume']:,.0f}" if _ext["volume"] else "—"

    _time_str = "—"
    if _ext["last_trade_time"]:
        _t = _ext["last_trade_time"]
        _h = _t.hour % 12 or 12
        _time_str = f"{_h}:{_t.minute:02d} {'AM' if _t.hour < 12 else 'PM'} ET"

    st.markdown(
        f"<div class='{_banner_cls}' style='border:1px solid {_color};border-radius:10px;"
        f"padding:12px 20px;margin:10px 0 4px;display:flex;align-items:center;gap:28px'>"
        f"<div>"
        f"  <div style='font-size:10px;color:{_color};font-weight:700;letter-spacing:1.5px;"
        f"  text-transform:uppercase;margin-bottom:3px'>{_session_label}</div>"
        f"  <div style='font-size:24px;font-weight:800;color:#fff;font-family:monospace'>{_price_str}</div>"
        f"</div>"
        f"<div>"
        f"  <div style='font-size:18px;font-weight:700;color:{_color}'>{_pct_str}</div>"
        f"  <div style='font-size:13px;color:{_color};opacity:.85'>{_chg_str}</div>"
        f"</div>"
        f"<div style='margin-left:auto;text-align:right;font-size:12px;line-height:1.8'>"
        f"  <span style='color:#666'>Volume</span> <span style='color:#aaa'>{_vol_str}</span><br>"
        f"  <span style='color:#666'>Last trade</span> <span style='color:#aaa'>{_time_str}</span>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Big chart
# ---------------------------------------------------------------------------

st.divider()

view = st.segmented_control(
    "Chart view", options=["Performance", "Price", "Candlestick", "Area"], default="Performance", key="analyzer_view"
)
view = view or "Performance"

control_cols = st.columns([3, 2, 2])
with control_cols[0]:
    overlays = st.multiselect(
        "Overlays", options=OVERLAY_OPTIONS, default=["EMA50", "EMA200"], key="analyzer_overlays",
    )
with control_cols[1]:
    panels = st.multiselect(
        "Indicator panels", options=PANEL_OPTIONS, default=["Volume", "RSI", "MACD"], key="analyzer_panels",
    )
with control_cols[2]:
    show_bb_signals = st.checkbox("Show BB Mid-Cross signals", value=False, key="analyzer_bb_signals")

history = get_history_tf(ticker, timeframe)
baseline = get_prev_close(ticker) if timeframe == "1d" else None
n_sub_panels = len([p for p in panels if p != "Volume"])
chart_height = max(650, 500 + 150 * n_sub_panels)
fig = render_technical_chart(
    history, view=view, baseline_price=baseline, title=ticker,
    overlays=overlays, panels=panels, height=chart_height,
    show_bb_signals=show_bb_signals,
)
st.plotly_chart(
    fig, use_container_width=True,
    config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False},
)

# ---------------------------------------------------------------------------
# Options chain
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Options")

expirations = get_option_expirations(ticker)
if not expirations:
    st.info("No options data available for this ticker.")
else:
    expiration = st.selectbox("Expiration", options=expirations, key="analyzer_expiration")
    chain = get_option_chain(ticker, expiration)
    calls = chain.get("calls", pd.DataFrame())
    puts = chain.get("puts", pd.DataFrame())

    spot = fund.get("price")
    try:
        years = (dt.date.fromisoformat(expiration) - dt.date.today()).days / 365.0
    except ValueError:
        years = None

    # -- Summary bar -------------------------------------------------------
    call_volume = calls["volume"].fillna(0).sum() if not calls.empty else 0
    put_volume = puts["volume"].fillna(0).sum() if not puts.empty else 0
    total_volume = call_volume + put_volume
    pc_ratio = (put_volume / call_volume) if call_volume else None
    pain_strike = max_pain_strike(calls, puts)

    sum_cols = st.columns(3)
    sum_cols[0].metric("Put / Call ratio", fmt(pc_ratio))
    sum_cols[1].metric("Total volume", fmt(total_volume, decimals=0))
    sum_cols[2].metric("Max pain strike", fmt(pain_strike, prefix="$"))

    display_cols = ["strike", "lastPrice", "bid", "ask", "impliedVolatility", "volume", "openInterest"]
    rename = {
        "strike": "Strike",
        "lastPrice": "Last",
        "bid": "Bid",
        "ask": "Ask",
        "impliedVolatility": "IV%",
        "volume": "Volume",
        "openInterest": "Open Interest",
        "delta": "Delta",
    }

    def prep_chain(df: pd.DataFrame, option_type: str) -> pd.DataFrame | None:
        if df is None or df.empty:
            return None
        df = df[[c for c in display_cols if c in df.columns]].copy()
        if "impliedVolatility" in df.columns:
            df["impliedVolatility"] = (df["impliedVolatility"] * 100).round(2)
        if spot is not None and years is not None and years > 0:
            df["delta"] = [
                bs_delta(spot, strike, years, iv / 100, option_type)
                for strike, iv in zip(df["strike"], df["impliedVolatility"])
            ]
            df["delta"] = df["delta"].round(3)
        df["_itm"] = (df["strike"] < spot) if option_type == "call" else (df["strike"] > spot)
        df = df.rename(columns=rename)
        return df

    def style_chain(df: pd.DataFrame):
        itm = df.pop("_itm")
        light = st.query_params.get("theme") == "Light"

        def highlight(row):
            if light:
                bg = "#d4edda" if itm.loc[row.name] else "#ffffff"
                return [f"background-color: {bg}; color: #131722; border-color: #e0e3eb"] * len(row)
            color = "background-color: rgba(46, 204, 113, 0.12)" if itm.loc[row.name] else ""
            return [color] * len(row)

        return df.style.apply(highlight, axis=1)

    calls_col, puts_col = st.columns(2)

    with calls_col:
        st.markdown("##### Calls")
        calls_disp = prep_chain(calls, "call")
        if calls_disp is None:
            st.caption("No call data available.")
        else:
            st.dataframe(style_chain(calls_disp), use_container_width=True, hide_index=True)

    with puts_col:
        st.markdown("##### Puts")
        puts_disp = prep_chain(puts, "put")
        if puts_disp is None:
            st.caption("No put data available.")
        else:
            st.dataframe(style_chain(puts_disp), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Snapshot")
st.caption(
    "A descriptive, point-in-time read of where this stock sits relative to "
    "its own recent history and common reference ranges. These scores "
    "describe current conditions only - they are not recommendations."
)

# Use a longer history window so moving averages and 52-week range are reliable.
scoring_history = get_history(ticker, "1Y")
tech_score, tech_drivers = technical_score(scoring_history)
fund_score, fund_drivers = fundamental_score(fund)
chips = at_a_glance(scoring_history, fund)

snap_cols = st.columns(3)

with snap_cols[0]:
    st.markdown("##### At a glance")
    with st.container(border=True):
        for label, value in chips:
            st.markdown(f"**{label}:** {value}")

with snap_cols[1]:
    st.markdown("##### Technical strength")
    st.plotly_chart(render_gauge(tech_score, "red_green"), use_container_width=True, config={"displayModeBar": False})
    if tech_score is not None:
        st.markdown(f"<h3 style='text-align:center'>{tech_score:.0f} / 100</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='text-align:center'>—</h3>", unsafe_allow_html=True)
    st.caption("Trend, momentum, position vs averages")
    for d in tech_drivers:
        st.markdown(f"- {d}")

with snap_cols[2]:
    st.markdown("##### Fundamental quality")
    st.plotly_chart(render_gauge(fund_score, "red_green"), use_container_width=True, config={"displayModeBar": False})
    if fund_score is not None:
        st.markdown(f"<h3 style='text-align:center'>{fund_score:.0f} / 100</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='text-align:center'>—</h3>", unsafe_allow_html=True)
    st.caption("Margins, returns, leverage, growth")
    for d in fund_drivers:
        st.markdown(f"- {d}")

# ---------------------------------------------------------------------------
# Key statistics
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Key statistics")


def stat_group(title, rows):
    with st.container(border=True):
        st.markdown(f"**{title}**")
        for label, value in rows:
            c1, c2 = st.columns([2, 1])
            c1.markdown(label)
            c2.markdown(f"<div style='text-align:right'>{value}</div>", unsafe_allow_html=True)


stat_cols = st.columns(3)

with stat_cols[0]:
    stat_group("Valuation", [
        ("Trailing P/E", fmt_num(fund.get("trailing_pe"))),
        ("Forward P/E", fmt_num(fund.get("forward_pe"))),
        ("PEG ratio", fmt_num(fund.get("peg_ratio"))),
        ("Price / Book", fmt_num(fund.get("price_to_book"))),
        ("Price / Sales", fmt_num(fund.get("price_to_sales"))),
    ])
    stat_group("Balance sheet", [
        ("Debt / Equity", fmt_num(fund.get("debt_to_equity"))),
        ("Current ratio", fmt_num(fund.get("current_ratio"))),
        ("Quick ratio", fmt_num(fund.get("quick_ratio"))),
    ])

with stat_cols[1]:
    stat_group("Profitability", [
        ("Gross margin", fmt_pct(fund.get("gross_margin"))),
        ("Operating margin", fmt_pct(fund.get("operating_margin"))),
        ("Profit margin", fmt_pct(fund.get("profit_margin"))),
        ("Return on equity", fmt_pct(fund.get("return_on_equity"))),
        ("Return on assets", fmt_pct(fund.get("return_on_assets"))),
    ])
    stat_group("Income", [
        ("Revenue", fmt_large_number(fund.get("revenue"))),
        ("Revenue growth (YoY)", fmt_pct(fund.get("revenue_growth"))),
        ("Earnings growth (YoY)", fmt_pct(fund.get("earnings_growth"))),
        ("EPS (TTM)", fmt_num(fund.get("eps_ttm"))),
        ("EPS (forward)", fmt_num(fund.get("eps_forward"))),
    ])

with stat_cols[2]:
    stat_group("Trading", [
        ("Beta", fmt_num(fund.get("beta"))),
        ("52-week low", fmt_num(fund.get("fifty_two_week_low"))),
        ("52-week high", fmt_num(fund.get("fifty_two_week_high"))),
        ("50-day average", fmt_num(fund.get("fifty_day_avg"))),
        ("200-day average", fmt_num(fund.get("two_hundred_day_avg"))),
        ("Volume", fmt_large_number(fund.get("volume"), currency="")),
        ("Avg. volume", fmt_large_number(fund.get("avg_volume"), currency="")),
    ])
    stat_group("Analyst", [
        ("Target mean price", fmt_num(fund.get("target_mean_price"))),
        ("Recommendation mean (1=Strong Buy, 5=Strong Sell)", fmt_num(fund.get("recommendation_mean"))),
        ("Number of analysts", fmt_num(fund.get("number_of_analysts"), decimals=0)),
    ])

# ---------------------------------------------------------------------------
# Business summary
# ---------------------------------------------------------------------------

st.divider()
with st.expander("Business summary"):
    st.write(fund.get("summary") or "No summary available.")

# ---------------------------------------------------------------------------
# Valuation
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Valuation")

_val = get_valuation_data(ticker)
_vi = _val.get("info", {})
_q_inc = _val.get("quarterly_income", pd.DataFrame())
_q_cf = _val.get("quarterly_cashflow", pd.DataFrame())
_q_bs = _val.get("quarterly_balance", pd.DataFrame())
_a_inc = _val.get("annual_income", pd.DataFrame())
_ph = _val.get("price_history", pd.DataFrame())


def _vrow(df: pd.DataFrame, keys: list):
    """Return the first matching row from a financial statement, or None."""
    if df is None or df.empty:
        return None
    for k in keys:
        if k in df.index:
            return df.loc[k]
    return None


_cp = fund.get("price")
_eps = fund.get("eps_ttm") or _vi.get("trailingEps")
_bvps = _vi.get("bookValue")
_shares = _vi.get("sharesOutstanding") or fund.get("shares_outstanding")

# ── TTM FCF ─────────────────────────────────────────────────────────────────
_fcf_ttm = None
try:
    _ocf_q = _vrow(_q_cf, ["Operating Cash Flow", "Total Cash From Operating Activities",
                             "Cash From Operating Activities"])
    _cap_q = _vrow(_q_cf, ["Capital Expenditure", "Capital Expenditures"])
    if _ocf_q is not None:
        _ocf_sum = float(_ocf_q.iloc[:4].sum())
        _cap_sum = float(_cap_q.iloc[:4].sum()) if _cap_q is not None else 0.0
        _fcf_ttm = _ocf_sum + _cap_sum  # capex is stored negative in yfinance
except Exception:
    pass

# ── 5-yr revenue CAGR from annual income ────────────────────────────────────
_growth_rate = fund.get("revenue_growth") or 0.05
try:
    _rev_a = _vrow(_a_inc, ["Total Revenue", "Revenue"])
    if _rev_a is not None:
        _rv = _rev_a.dropna().sort_index()
        if len(_rv) >= 2:
            _n = min(len(_rv) - 1, 5)
            _r0, _rn = float(_rv.iloc[0]), float(_rv.iloc[-1])
            if _r0 > 0:
                _growth_rate = max(-0.5, min(0.5, (_rn / _r0) ** (1 / _n) - 1))
except Exception:
    pass

_DISC = 0.10

# ── DCF per share (allow negative FCF — show with explanation) ──────────────
_dcf = None
_dcf_note = None
if _fcf_ttm is None:
    _dcf_note = "No free cash flow data available"
elif not (_shares and _shares > 0):
    _dcf_note = "Share count unavailable"
elif _DISC <= _growth_rate:
    _dcf_note = f"Growth ({_growth_rate:.1%}) ≥ discount rate ({_DISC:.0%}) — model invalid"
else:
    try:
        _dcf = (_fcf_ttm * (1 + _growth_rate) / (_DISC - _growth_rate)) / _shares
        if _fcf_ttm < 0:
            _dcf_note = f"Negative FCF (${_fcf_ttm / 1e9:.2f}B TTM) — company burning cash"
    except Exception:
        _dcf_note = "Calculation error"

# ── Graham Number ────────────────────────────────────────────────────────────
_graham = None
if _eps and _eps > 0 and _bvps and _bvps > 0:
    try:
        _graham = (22.5 * _eps * _bvps) ** 0.5
    except Exception:
        pass

_analyst_tgt = fund.get("target_mean_price") or _vi.get("targetMeanPrice")
_high_52     = fund.get("fifty_two_week_high")
_low_52      = fund.get("fifty_two_week_low")
_n_analysts  = fund.get("number_of_analysts")

# ── Model table ──────────────────────────────────────────────────────────────
# Each entry: (label, description, value, note_if_unavailable)
_MODELS = [
    (
        "DCF Value",
        f"FCF · (1 + {_growth_rate:.1%}) ÷ ({_DISC:.0%} − {_growth_rate:.1%})  |  10% discount rate",
        _dcf,
        _dcf_note,
    ),
    (
        "Graham Number",
        "√(22.5 × EPS × BVPS) — Benjamin Graham's classic intrinsic value formula",
        _graham,
        None if _graham else (
            "Requires positive EPS & Book Value per Share"
            if (_eps is not None and _bvps is not None) else "Insufficient data"
        ),
    ),
    (
        "Analyst Target",
        f"Mean of {int(_n_analysts or 0) or '?'} analyst price targets (Wall Street consensus)",
        _analyst_tgt,
        None if _analyst_tgt else "No analyst coverage",
    ),
    (
        "52-Week High",
        "Highest traded price over the past year — recent market upside ceiling",
        _high_52,
        None if _high_52 else "No data",
    ),
]

# ── Valuation verdict ────────────────────────────────────────────────────────
_verdict_vals = [v for _, _, v, _ in _MODELS if v is not None and v > 0]
_verdict_avg  = sum(_verdict_vals) / len(_verdict_vals) if _verdict_vals else None

if _cp and _verdict_avg:
    _vdiff = (_cp - _verdict_avg) / _verdict_avg * 100
    if _vdiff < -20:
        _vlabel, _vcolor, _vcls = "UNDERVALUED", "#2ecc71", "tv-card-bull"
        _vsub = f"Trading {abs(_vdiff):.0f}% below the average of all valuation models"
    elif _vdiff > 20:
        _vlabel, _vcolor, _vcls = "OVERVALUED", "#e74c3c", "tv-card-bear"
        _vsub = f"Trading {_vdiff:.0f}% above the average of all valuation models"
    else:
        _vlabel, _vcolor, _vcls = "FAIRLY VALUED", "#f39c12", "tv-card-amber"
        _vsub = f"Within {abs(_vdiff):.0f}% of the average of all valuation models"
    _used_names = " · ".join(lbl for lbl, _, v, _ in _MODELS if v is not None and v > 0)
    st.markdown(
        f"<div class='{_vcls}' style='border:2px solid {_vcolor};border-radius:12px;"
        f"padding:16px 22px;margin-top:8px;margin-bottom:6px;"
        f"display:flex;align-items:center;justify-content:space-between'>"
        f"<div>"
        f"<div style='font-size:10px;color:#888;text-transform:uppercase;"
        f"letter-spacing:1.5px;margin-bottom:4px'>Valuation Verdict</div>"
        f"<div style='font-size:28px;font-weight:800;color:{_vcolor};"
        f"letter-spacing:1px'>{_vlabel}</div>"
        f"<div style='font-size:12px;color:#bbb;margin-top:4px'>{_vsub}</div>"
        f"</div>"
        f"<div style='text-align:right'>"
        f"<div style='font-size:11px;color:#888;margin-bottom:2px'>Model average</div>"
        f"<div style='font-size:26px;font-weight:700;color:{_vcolor}'>${_verdict_avg:,.2f}</div>"
        f"<div style='font-size:10px;color:#555;margin-top:6px'>{_used_names}</div>"
        f"</div></div>",
        unsafe_allow_html=True,
    )
elif _cp:
    st.info("Insufficient model data for a valuation verdict.")

# ── Bar chart + Bear / Base / Bull range ────────────────────────────────────
_chart_col, _range_col = st.columns([3, 2])

with _chart_col:
    st.markdown("##### Valuation Model Comparison")
    _bar_labels, _bar_vals, _bar_colors, _bar_texts = [], [], [], []
    _bar_notes_html = ""
    for lbl, _, val, note in _MODELS:
        if val is not None and val > 0:
            _bar_labels.append(lbl)
            _bar_vals.append(val)
            _bar_colors.append("#2ecc71" if val >= _cp else "#e74c3c")
            _bar_texts.append(f"${val:,.2f}")
        else:
            _bar_notes_html += (
                f"<div style='font-size:11px;color:#555;margin-top:3px'>"
                f"<span style='color:#444'>{lbl}:</span> {note or 'N/A'}</div>"
            )

    if _bar_labels and _cp:
        _xmax = max(_bar_vals + [_cp]) * 1.28
        _fig_bar = go.Figure(go.Bar(
            y=_bar_labels, x=_bar_vals, orientation="h",
            marker_color=_bar_colors,
            text=_bar_texts, textposition="outside",
            cliponaxis=False,
        ))
        _fig_bar.add_vline(
            x=_cp, line_color="#3498db", line_width=2, line_dash="dash",
            annotation_text=f"Now ${_cp:,.2f}",
            annotation_position="top right",
            annotation_font_color="#3498db",
        )
        if _verdict_avg:
            _fig_bar.add_vline(
                x=_verdict_avg, line_color="#aaa", line_width=1, line_dash="dot",
                annotation_text=f"Avg ${_verdict_avg:,.2f}",
                annotation_position="bottom right",
                annotation_font_color="#888",
            )
        _fig_bar.update_layout(
            height=max(180, 64 * len(_bar_labels) + 40),
            margin=dict(l=0, r=130, t=10, b=10),
            xaxis=dict(title="Estimated Value ($)", range=[0, _xmax]),
            showlegend=False,
        )
        apply_dark_plotly(_fig_bar)
        st.plotly_chart(_fig_bar, use_container_width=True, config={"displayModeBar": False})
        if _bar_notes_html:
            st.markdown(_bar_notes_html, unsafe_allow_html=True)
    else:
        st.info("Insufficient data to render valuation chart.")

with _range_col:
    st.markdown("##### Bear / Base / Bull")
    if _cp:
        _bear = _low_52 or (_cp * 0.75)
        _base = _analyst_tgt or _verdict_avg or _cp
        _bull = max(
            [v for _, _, v, _ in _MODELS if v is not None and v > 0],
            default=_cp * 1.25,
        )

        _bear_pct = (_bear - _cp) / _cp * 100
        _base_pct = (_base - _cp) / _cp * 100
        _bull_pct = (_bull - _cp) / _cp * 100

        _span = max(_bull - _bear, 1.0)
        _pos  = max(0.0, min(100.0, (_cp   - _bear) / _span * 100))
        _bpos = max(0.0, min(100.0, (_base - _bear) / _span * 100))

        st.markdown(f"""
<div style="margin:8px 0 16px">
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;margin-bottom:14px">
    <div class="tv-card-bear" style="border:1px solid #e74c3c;border-radius:8px;padding:8px 6px;text-align:center">
      <div style="font-size:9px;color:#e74c3c;font-weight:700;letter-spacing:1px">BEAR</div>
      <div style="font-size:13px;font-weight:700;margin:2px 0">${_bear:,.2f}</div>
      <div style="font-size:11px;color:#e74c3c">{_bear_pct:+.1f}%</div>
      <div style="font-size:9px;margin-top:2px">52-wk low</div>
    </div>
    <div class="tv-card-neutral" style="border:1px solid #444;border-radius:8px;padding:8px 6px;text-align:center">
      <div style="font-size:9px;font-weight:700;letter-spacing:1px">BASE</div>
      <div style="font-size:13px;font-weight:700;margin:2px 0">${_base:,.2f}</div>
      <div style="font-size:11px;color:#f39c12">{_base_pct:+.1f}%</div>
      <div style="font-size:9px;margin-top:2px">analyst target</div>
    </div>
    <div class="tv-card-bull" style="border:1px solid #2ecc71;border-radius:8px;padding:8px 6px;text-align:center">
      <div style="font-size:9px;color:#2ecc71;font-weight:700;letter-spacing:1px">BULL</div>
      <div style="font-size:13px;font-weight:700;margin:2px 0">${_bull:,.2f}</div>
      <div style="font-size:11px;color:#2ecc71">{_bull_pct:+.1f}%</div>
      <div style="font-size:9px;margin-top:2px">highest model</div>
    </div>
  </div>
  <div style="position:relative;height:18px;border-radius:9px;
              background:linear-gradient(90deg,#7f1d1d 0%,#78350f 45%,#14532d 100%);
              margin:0 2px 6px">
    <div style="position:absolute;top:-5px;left:{_bpos:.1f}%;transform:translateX(-50%);
                width:2px;height:28px;background:#f39c12;opacity:.8;border-radius:1px"></div>
    <div style="position:absolute;top:-7px;left:{_pos:.1f}%;transform:translateX(-50%);
                width:6px;height:32px;background:#3498db;border-radius:3px;
                box-shadow:0 0 8px #3498db88"></div>
  </div>
  <div style="display:flex;justify-content:space-between;font-size:10px;color:#555;padding:0 2px">
    <span>${_bear:,.2f}</span>
    <span style="color:#3498db;font-weight:600">▲ ${_cp:,.2f} now</span>
    <span>${_bull:,.2f}</span>
  </div>
</div>
""", unsafe_allow_html=True)
    else:
        st.info("Price data unavailable.")

# ── Individual Model Cards ───────────────────────────────────────────────────
st.markdown("##### Model Detail")
_mc_cols = st.columns(len(_MODELS))
for _col, (lbl, desc, val, note) in zip(_mc_cols, _MODELS):
    with _col:
        if val is not None and val > 0 and _cp:
            _mpct   = (val - _cp) / _cp * 100
            _mcls   = "tv-card-bull" if _mpct >= 0 else "tv-card-bear"
            _mbdr   = "#2ecc71" if _mpct >= 0 else "#e74c3c"
            _marrow = "▲" if _mpct >= 0 else "▼"
            _mtag   = "upside" if _mpct >= 0 else "downside"
            st.markdown(
                f"<div class='{_mcls}' style='border:1px solid {_mbdr};"
                f"border-radius:10px;padding:14px;height:100%'>"
                f"<div style='font-size:10px;color:{_mbdr};font-weight:700;"
                f"letter-spacing:.8px;text-transform:uppercase;margin-bottom:8px'>{lbl}</div>"
                f"<div style='font-size:22px;font-weight:800;margin-bottom:2px'>"
                f"${val:,.2f}</div>"
                f"<div style='font-size:14px;font-weight:600;color:{_mbdr};margin-bottom:10px'>"
                f"{_marrow} {abs(_mpct):.1f}% {_mtag}</div>"
                f"<div style='font-size:10px;line-height:1.5'>{desc}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        elif val is not None and val <= 0 and _cp:
            st.markdown(
                f"<div class='tv-card-warm' style='border:1px solid #b7791f;"
                f"border-radius:10px;padding:14px;height:100%'>"
                f"<div style='font-size:10px;color:#b7791f;font-weight:700;"
                f"letter-spacing:.8px;text-transform:uppercase;margin-bottom:8px'>{lbl}</div>"
                f"<div style='font-size:22px;font-weight:800;color:#b7791f;margin-bottom:2px'>"
                f"${val:,.2f}</div>"
                f"<div style='font-size:12px;color:#b7791f;margin-bottom:10px'>⚠ Negative value</div>"
                f"<div style='font-size:10px;line-height:1.5'>{note or desc}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='tv-card-neutral' style='border:1px solid #2a2a2a;"
                f"border-radius:10px;padding:14px;height:100%'>"
                f"<div style='font-size:10px;font-weight:700;"
                f"letter-spacing:.8px;text-transform:uppercase;margin-bottom:8px'>{lbl}</div>"
                f"<div style='font-size:20px;font-weight:700;margin-bottom:8px'>N/A</div>"
                f"<div style='font-size:10px;line-height:1.5'>"
                f"{note or 'Not available'}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

# ── Valuation Metrics Grid ───────────────────────────────────────────────────
st.markdown("##### Valuation Metrics")

_ev = _vi.get("enterpriseValue")
_ev_rev = _vi.get("enterpriseToRevenue")
_ev_ebitda = _vi.get("enterpriseToEbitda")
_gross_p = _vi.get("grossProfits")


def _vcard(label: str, value: str) -> None:
    st.markdown(
        f'<div class="tv-vcard-container">'
        f'<div style="font-size:11px;margin-bottom:2px">{label}</div>'
        f'<div style="font-size:15px;font-weight:600">{value}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


_gm1, _gm2, _gm3, _gm4 = st.columns(4)

with _gm1:
    st.markdown("**Multiples**")
    _vcard("Price / Book", fmt_num(fund.get("price_to_book")))
    _vcard("Price / Sales", fmt_num(fund.get("price_to_sales")))
    _vcard("Price / Earnings", fmt_num(fund.get("trailing_pe")))
    _vcard("PEG Ratio", fmt_num(fund.get("peg_ratio")))

with _gm2:
    st.markdown("**Margins & Returns**")
    _vcard("Gross Profit", fmt_large_number(_gross_p))
    _vcard("Operating Margin", fmt_pct(fund.get("operating_margin")))
    _vcard("Net Margin", fmt_pct(fund.get("profit_margin")))
    _vcard("ROE", fmt_pct(fund.get("return_on_equity")))

with _gm3:
    st.markdown("**Enterprise Value**")
    _vcard("Enterprise Value", fmt_large_number(_ev))
    _vcard("EV / Revenue", fmt_num(_ev_rev))
    _vcard("EV / EBITDA", fmt_num(_ev_ebitda))

with _gm4:
    st.markdown("**Financial Health**")
    _vcard("Debt / Equity", fmt_num(fund.get("debt_to_equity")))
    _vcard("Current Ratio", fmt_num(fund.get("current_ratio")))
    _vcard("Quick Ratio", fmt_num(fund.get("quick_ratio")))

# ── What Drives This Stock ───────────────────────────────────────────────────
st.markdown("##### What Drives This Stock")


def _strip_tz(s: pd.Series) -> pd.Series:
    if s.index.tz is not None:
        return s.copy().set_axis(s.index.tz_convert(None))
    return s


# Build the aligned quarterly DataFrame
_hm = {}

_s = _vrow(_q_inc, ["Total Revenue", "Revenue"])
if _s is not None:
    _hm["Revenue"] = _strip_tz(_s)

_s = _vrow(_q_inc, ["Gross Profit"])
if _s is not None:
    _hm["Gross Profit"] = _strip_tz(_s)

_s = _vrow(_q_inc, ["Operating Income", "Total Operating Income As Reported", "EBIT"])
if _s is not None:
    _hm["Operating Income"] = _strip_tz(_s)

_s = _vrow(_q_inc, ["Net Income", "Net Income Common Stockholders",
                     "Net Income Applicable To Common Shares"])
if _s is not None:
    _hm["Net Income"] = _strip_tz(_s)

_s_ocf = _vrow(_q_cf, ["Operating Cash Flow", "Total Cash From Operating Activities"])
_s_cap = _vrow(_q_cf, ["Capital Expenditure", "Capital Expenditures"])
if _s_ocf is not None:
    _fcf_s = (_s_ocf + _s_cap) if _s_cap is not None else _s_ocf
    _hm["FCF"] = _strip_tz(_fcf_s)

_s = _vrow(_q_bs, ["Total Stockholders Equity", "Stockholders Equity", "Common Stock Equity"])
if _s is not None:
    _hm["Book Value"] = _strip_tz(_s)

if not _ph.empty and "Close" in _ph.columns and _shares:
    try:
        _pq = _ph["Close"].resample("QE").last()
    except Exception:
        try:
            _pq = _ph["Close"].resample("Q").last()
        except Exception:
            _pq = None
    if _pq is not None:
        _hm["Market Cap"] = _strip_tz(_pq * _shares)

_hm_df = pd.DataFrame(_hm).sort_index(ascending=True).tail(8).dropna(how="all") if _hm else pd.DataFrame()
_corr = _hm_df.corr() if len(_hm_df) >= 3 else pd.DataFrame()

# Helper to safely pull a correlation between two columns
def _cor(a: str, b: str) -> float | None:
    if _corr.empty or a not in _corr.columns or b not in _corr.columns:
        return None
    v = _corr.loc[a, b]
    return float(v) if pd.notna(v) else None


# ── Correlation progress-bar renderer ───────────────────────────────────────
def _corr_bar(r: float | None) -> str:
    if r is None:
        return "<span style='color:#666'>Not enough data</span>"
    pct = int(abs(r) * 100)
    color = "#2ecc71" if r >= 0.7 else "#f39c12" if r >= 0.4 else "#e74c3c"
    direction = "positive" if r >= 0 else "negative"
    return (
        f"<div style='margin:6px 0'>"
        f"<div style='display:flex;justify-content:space-between;font-size:11px;margin-bottom:3px'>"
        f"<span>{direction}</span><span><b>{r:+.2f}</b></span></div>"
        f"<div class='tv-corr-track'>"
        f"<div style='width:{pct}%;background:{color};border-radius:4px;height:100%'></div>"
        f"</div></div>"
    )


# ── Top Drivers Cards ────────────────────────────────────────────────────────
_r_mc   = _cor("Revenue", "Market Cap")
_r_fcf_ni = _cor("FCF", "Net Income")
_peg    = fund.get("peg_ratio")

_dc1, _dc2, _dc3 = st.columns(3)

# Card 1 — Revenue driver
with _dc1:
    if _r_mc is not None:
        if _r_mc >= 0.7:
            _d1_title = "Revenue is the #1 Driver"
            _d1_body  = "When revenue grows, the stock price tends to follow strongly. Watch quarterly revenue beats and misses closely."
            _d1_cls, _d1_border = "tv-card-bull", "#2ecc71"
        elif _r_mc >= 0.4:
            _d1_title = "Revenue Matters, But Isn't Everything"
            _d1_body  = "Revenue growth influences the stock, but other factors (margins, macro) also move the price meaningfully."
            _d1_cls, _d1_border = "tv-card-amber", "#f39c12"
        else:
            _d1_title = "Stock Decoupled from Revenue"
            _d1_body  = "Price hasn't closely tracked revenue over the past 2 years. Sentiment, buybacks, or macro may be bigger drivers."
            _d1_cls, _d1_border = "tv-card-bear", "#e74c3c"
        st.markdown(
            f"<div class='{_d1_cls}' style='border:1px solid {_d1_border};border-radius:10px;padding:14px'>"
            f"<div style='font-size:13px;font-weight:700;margin-bottom:6px'>{_d1_title}</div>"
            f"<div style='font-size:12px;margin-bottom:8px'>{_d1_body}</div>"
            + _corr_bar(_r_mc) +
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='tv-card-neutral' style='border:1px solid #333;border-radius:10px;padding:14px'>"
            "<div style='font-size:13px;font-weight:700;margin-bottom:6px'>Revenue Driver</div>"
            "<div style='font-size:12px'>Not enough quarterly data to assess.</div>"
            "</div>",
            unsafe_allow_html=True,
        )

# Card 2 — Earnings quality
with _dc2:
    if _r_fcf_ni is not None:
        if _r_fcf_ni >= 0.7:
            _d2_title = "Earnings Quality: Strong"
            _d2_body  = "Cash flow and reported profits move together — earnings are backed by real cash. Lower risk of accounting surprises."
            _d2_cls, _d2_border = "tv-card-bull", "#2ecc71"
        elif _r_fcf_ni >= 0.3:
            _d2_title = "Earnings Quality: Mixed"
            _d2_body  = "Cash flow and profits are somewhat in sync, but diverge at times. Keep an eye on the cash flow statement."
            _d2_cls, _d2_border = "tv-card-amber", "#f39c12"
        else:
            _d2_title = "Earnings Quality: Watch Out"
            _d2_body  = "Profits and cash flow are not moving together. Reported earnings may not be fully converting to cash — dig deeper before investing."
            _d2_cls, _d2_border = "tv-card-bear", "#e74c3c"
        st.markdown(
            f"<div class='{_d2_cls}' style='border:1px solid {_d2_border};border-radius:10px;padding:14px'>"
            f"<div style='font-size:13px;font-weight:700;margin-bottom:6px'>{_d2_title}</div>"
            f"<div style='font-size:12px;margin-bottom:8px'>{_d2_body}</div>"
            + _corr_bar(_r_fcf_ni) +
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='tv-card-neutral' style='border:1px solid #333;border-radius:10px;padding:14px'>"
            "<div style='font-size:13px;font-weight:700;margin-bottom:6px'>Earnings Quality</div>"
            "<div style='font-size:12px'>Not enough cash flow data to assess.</div>"
            "</div>",
            unsafe_allow_html=True,
        )

# Card 3 — Growth vs Value
with _dc3:
    if _peg is not None:
        if _peg < 0:
            _d3_title = "Negative PEG — Interpret Carefully"
            _d3_body  = "A negative PEG usually means negative earnings or negative growth expectations. Focus on the path to profitability."
            _d3_cls, _d3_border = "tv-card-bear", "#e74c3c"
            _d3_badge = "⚠️ Unprofitable / Shrinking"
        elif _peg < 1.0:
            _d3_title = "Value Territory"
            _d3_body  = "PEG below 1 suggests the stock may be undervalued relative to its growth rate — a classic value signal."
            _d3_cls, _d3_border = "tv-card-bull", "#2ecc71"
            _d3_badge = f"PEG {_peg:.2f} — Undervalued vs Growth"
        elif _peg < 2.0:
            _d3_title = "Fairly Valued"
            _d3_body  = "PEG near 1–2 suggests the market is pricing growth reasonably. Not cheap, but not obviously expensive."
            _d3_cls, _d3_border = "tv-card-amber", "#f39c12"
            _d3_badge = f"PEG {_peg:.2f} — Fair Value"
        else:
            _d3_title = "Priced for Growth"
            _d3_body  = "High PEG means investors are paying a premium for expected growth. The stock needs to keep delivering — misses get punished hard."
            _d3_cls, _d3_border = "tv-card-blue", "#3498db"
            _d3_badge = f"PEG {_peg:.2f} — Growth Premium"
        st.markdown(
            f"<div class='{_d3_cls}' style='border:1px solid {_d3_border};border-radius:10px;padding:14px'>"
            f"<div style='font-size:13px;font-weight:700;margin-bottom:6px'>{_d3_title}</div>"
            f"<div style='font-size:12px;margin-bottom:8px'>{_d3_body}</div>"
            f"<div style='font-size:11px;color:{_d3_border};font-weight:600;margin-top:4px'>{_d3_badge}</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='tv-card-neutral' style='border:1px solid #333;border-radius:10px;padding:14px'>"
            "<div style='font-size:13px;font-weight:700;margin-bottom:6px'>Growth vs Value</div>"
            "<div style='font-size:12px'>PEG ratio not available for this ticker.</div>"
            "</div>",
            unsafe_allow_html=True,
        )

# ── Plain-English Insight Boxes ──────────────────────────────────────────────
st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)

_insights = []

# What Wall Street watches most
if _r_mc is not None:
    if _r_mc >= 0.7:
        _insights.append(("green", "Revenue is the #1 driver — watch quarterly revenue beats and misses. A revenue miss often hurts more than an earnings miss for this stock."))
    elif _cor("Net Income", "Market Cap") is not None and (_cor("Net Income", "Market Cap") or 0) >= 0.7:
        _insights.append(("green", "Earnings per share is the #1 driver — the market rewards profit growth. Focus on EPS surprises each quarter."))
    elif _cor("FCF", "Market Cap") is not None and (_cor("FCF", "Market Cap") or 0) >= 0.7:
        _insights.append(("green", "Free cash flow is the #1 driver — investors value cash generation above all. Track FCF yield and buyback capacity."))
    else:
        _insights.append(("yellow", "No single metric dominates — this stock is sensitive to sentiment, macro, or factors not captured in quarterly financials alone."))

# Earnings quality warning
if _r_fcf_ni is not None and _r_fcf_ni < 0.3:
    _insights.append(("red", "⚠️ Cash flow is diverging from reported profits. This can signal aggressive accounting, heavy capex cycles, or working capital strain. Review the cash flow statement before relying on EPS alone."))
elif _r_fcf_ni is not None and _r_fcf_ni >= 0.7:
    _insights.append(("green", "Cash earnings quality is high — reported profits are well-supported by actual cash. Lower risk of earnings restatements or unpleasant surprises."))

# Book value signal
_r_bv_mc = _cor("Book Value", "Market Cap")
if _r_bv_mc is not None and _r_bv_mc >= 0.7:
    _insights.append(("yellow", "Book value moves with market cap — this stock behaves more like a value/asset-heavy name. Price-to-book is a relevant valuation yardstick."))

_BOX = {
    "green":  ("tv-card-bull",  "border:1px solid #2ecc71"),
    "yellow": ("tv-card-amber", "border:1px solid #f39c12"),
    "red":    ("tv-card-bear",  "border:1px solid #e74c3c"),
}

for _tone, _text in _insights:
    _cls, _style = _BOX[_tone]
    st.markdown(
        f"<div class='{_cls}' style='{_style};border-radius:8px;padding:10px 14px;margin-bottom:8px;"
        f"font-size:13px;line-height:1.5'>{_text}</div>",
        unsafe_allow_html=True,
    )

if not _insights and _corr.empty:
    st.info("Not enough quarterly data to generate insights. Try again after the company reports a few more quarters.")

# ── Technical correlation matrix (hidden by default) ────────────────────────
with st.expander("Show technical correlation matrix"):
    if not _corr.empty:
        _fig_hm = go.Figure(go.Heatmap(
            z=_corr.values,
            x=_corr.columns.tolist(),
            y=_corr.index.tolist(),
            colorscale=[[0.0, "#1c3a2e"], [0.5, "#27ae60"], [1.0, "#2ecc71"]],
            zmin=-1, zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in _corr.values],
            texttemplate="%{text}",
            showscale=True,
            colorbar=dict(title="r"),
        ))
        _fig_hm.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
        apply_dark_plotly(_fig_hm)
        st.plotly_chart(_fig_hm, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("Need at least 3 quarters of data to build the correlation matrix.")

# ---------------------------------------------------------------------------
# AI analysis
# ---------------------------------------------------------------------------

st.divider()
st.subheader("AI analysis")

if not claude_analyst.is_configured():
    st.info("AI analysis is unavailable. Set `GROQ_API_KEY` in Streamlit secrets to enable it.")

tab1, tab2, tab3 = st.tabs(["Bull / Bear case", "Deep analysis", "Recent headlines"])

with tab1:
    if claude_analyst.is_configured():
        if st.button("Generate bull / bear case", key="bullbear_btn"):
            recent_news = ticker_news(ticker, limit=5)
            try:
                st.write_stream(claude_analyst.bull_bear_case(ticker, fund, chips, tech_score, fund_score, recent_news))
            except Exception as e:
                st.error(f"AI analysis failed: {e}")
    else:
        st.caption("AI analysis unavailable - no API key configured.")

with tab2:
    if claude_analyst.is_configured():
        if st.button("Generate deep analysis", key="deep_btn"):
            recent_news = ticker_news(ticker, limit=8)
            close = scoring_history["Close"].dropna() if not scoring_history.empty else None
            if close is not None and len(close) > 1:
                period_return = (close.iloc[-1] / close.iloc[0] - 1) * 100
                history_summary = (
                    f"Over the last ~1 year, price moved from {close.iloc[0]:.2f} to "
                    f"{close.iloc[-1]:.2f} ({period_return:+.1f}%)."
                )
            else:
                history_summary = "Limited price history available."
            try:
                st.write_stream(claude_analyst.deep_analysis(ticker, fund, history_summary, recent_news))
            except Exception as e:
                st.error(f"AI analysis failed: {e}")
    else:
        st.caption("AI analysis unavailable - no API key configured.")

with tab3:
    items = ticker_news(ticker, limit=10)
    if not items:
        st.info("No recent headlines available.")
    else:
        for item in items:
            with st.container(border=True):
                st.markdown(f"**{item['title']}**")
                meta = item["publisher"]
                ago = time_ago(item["time"])
                if ago:
                    meta += f" · {ago}"
                st.caption(meta)
                summary = (item.get("summary") or "").strip()
                if summary:
                    st.write(summary[:280] + ("..." if len(summary) > 280 else ""))
                if item.get("link"):
                    st.markdown(f"[Read more]({item['link']})")

render_footer()
