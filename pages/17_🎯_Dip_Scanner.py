"""Dip Scanner: today's buy list and sell checks for the backtested dip-buy rule."""
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib import dip_scanner as ds
from lib.config import APP_NAME, inject_base_style, render_footer

st.set_page_config(page_title=f"Dip Scanner - {APP_NAME}", page_icon="🎯", layout="wide")
inject_base_style()

NY, SYD = ZoneInfo("America/New_York"), ZoneInfo("Australia/Sydney")


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def load_members():
    return ds.fetch_members()


@st.cache_data(ttl=20 * 60, show_spinner=False)
def load_closes(tickers: tuple, stamp: str):
    import yfinance as yf

    frames, chunk = [], 150
    for k in range(0, len(tickers), chunk):
        part = list(tickers[k:k + chunk])
        for attempt in range(3):
            try:
                df = yf.download(part, period="15mo", interval="1d", auto_adjust=True, progress=False, threads=True)
                frames.append(df["Close"] if isinstance(df.columns, pd.MultiIndex) else df[["Close"]].rename(columns={"Close": part[0]}))
                break
            except Exception:
                time.sleep(3 * (attempt + 1))
        time.sleep(1.0)
    C = pd.concat(frames, axis=1)
    return C.loc[:, ~C.columns.duplicated()]


def market_clock():
    now_ny = datetime.now(NY)
    open_t = now_ny.replace(hour=9, minute=30, second=0, microsecond=0)
    close_t = now_ny.replace(hour=16, minute=0, second=0, microsecond=0)
    is_open = now_ny.weekday() < 5 and open_t <= now_ny < close_t
    close_syd = close_t.astimezone(SYD)
    return now_ny, is_open, close_t, close_syd


st.title("🎯 Dip Scanner")
st.caption(
    "Rule: stock above its 200-day average, in the most volatile 20% of the S&P 500/400/600, with RSI(2) below 2 "
    "(a sharp 1-3 day drop). Buy at the close. Sell at the close of the first day that closes higher, or after 10 days. "
    "Hold at most 5 at once."
)

now_ny, is_open, close_ny, close_syd = market_clock()
mins_left = int((close_ny - now_ny).total_seconds() // 60)
if is_open:
    st.info(f"US market is OPEN. It closes at {close_syd:%I:%M %p} Sydney time ({mins_left} min). "
            "Prices below are live, so signals are provisional until the close. Best time to act: the last 10-15 minutes.")
else:
    st.info(f"US market is closed. Next close is at {close_syd:%I:%M %p} Sydney time on trading days. "
            "Signals below are from the last close. If you missed that close, buying at the next open also worked in the test, slightly worse.")

with st.sidebar:
    st.markdown("**Scanner settings**")
    account = st.number_input("Account size ($)", 1000.0, 10_000_000.0, 10_000.0, 1000.0)
    slots = st.slider("Max positions at once", 1, 10, 5)
    rsi_max = st.slider("RSI(2) below", 1.0, 10.0, 2.0, 0.5, help="Tested value: 2. Higher = more signals, smaller edge.")
    vol_pct = st.slider("Volatility percentile at least", 0.5, 0.95, 0.8, 0.05, help="Tested value: 0.8 (top 20%).")
    refresh = st.button("Refresh prices")

members = load_members()
stamp = datetime.now(NY).strftime("%Y-%m-%d %H") + ("-" + str(int(time.time())) if refresh else "")
with st.spinner("Downloading prices for ~1,500 stocks (about a minute the first time, cached for 20 min)..."):
    closes = load_closes(tuple(sorted(members)), stamp)
got = int(closes.iloc[-1].notna().sum()) if len(closes) else 0
last_bar = closes.index[-1].date() if len(closes) else None
st.caption(f"Universe: {len(members)} stocks, prices for {got} on the latest bar ({last_bar}). Yahoo sometimes rate-limits; hit Refresh if the count looks low.")

try:
    sig, near = ds.scan(closes, rsi_max=rsi_max, vol_pct=vol_pct)
except ValueError as e:
    st.error(f"Scan failed: {e}")
    st.stop()

per_pos = account / slots
st.subheader(f"Buy list ({len(sig)})")
if sig.empty:
    st.success("No signals right now. That's normal: in the test, signals came in bursts on market sell-off days.")
else:
    out = pd.DataFrame({
        "Ticker": sig.index,
        "Index": [members.get(t, "") for t in sig.index],
        "Price": sig["price"].round(2).values,
        "RSI(2)": sig["rsi2"].round(2).values,
        "Today %": (sig["chg1"] * 100).round(1).values,
        "3-day %": (sig["chg3"] * 100).round(1).values,
        "Shares": np.floor(per_pos / sig["price"]).astype(int).values,
        "Amount $": (np.floor(per_pos / sig["price"]) * sig["price"]).round(0).values,
    })
    st.dataframe(out, hide_index=True, use_container_width=True)
    st.caption(f"Take the top {slots} (deepest dips first) if you have free slots. Each position = account / {slots} = ${per_pos:,.0f}.")

with st.expander("Almost there (closest to a signal)"):
    if near.empty:
        st.write("None.")
    else:
        st.dataframe(pd.DataFrame({
            "Ticker": near.index, "Index": [members.get(t, "") for t in near.index],
            "Price": near["price"].round(2).values, "RSI(2)": near["rsi2"].round(2).values,
            "Today %": (near["chg1"] * 100).round(1).values,
        }), hide_index=True, use_container_width=True)

st.subheader("Sell check")
st.caption("One holding per line: TICKER, buy date (YYYY-MM-DD). Example: CELH, 2026-11-04")
txt = st.text_area("Your open dip trades", "", height=110, label_visibility="collapsed")
holds = []
for line in txt.splitlines():
    parts = [p.strip() for p in line.replace(";", ",").split(",") if p.strip()]
    if not parts:
        continue
    t = parts[0].upper()
    bought = None
    if len(parts) > 1:
        try:
            bought = pd.Timestamp(parts[1])
        except Exception:
            bought = None
    holds.append((t, bought))
if holds:
    missing = [t for t, _ in holds if t not in closes.columns]
    if missing:
        import yfinance as yf
        extra = yf.download(missing, period="3mo", interval="1d", auto_adjust=True, progress=False)
        extra = extra["Close"] if isinstance(extra.columns, pd.MultiIndex) else extra[["Close"]].rename(columns={"Close": missing[0]})
        closes = pd.concat([closes, extra], axis=1)
    res = ds.sell_check(closes, holds)
    st.dataframe(res.round(2), hide_index=True, use_container_width=True)

st.subheader("How it tested")
st.markdown(
    """
| Out-of-sample 2021-2026 (rule picked on 2014-2020) | Per year | Worst drop | Trades won |
|---|---|---|---|
| **Buy at close, idle cash in SPY, 5 slots** | **22.5%** | -33% | 68% |
| Buy at close, idle cash in cash, 5 slots | 13.5% | -22% | 68% |
| Buy at next open, idle cash in SPY, 5 slots | 21.5% | -44% | 59% |
| SPY buy and hold | 14.6% | -25% | n/a |

Same rule on 2014-2020: 22.3% a year. Average trade +0.69%, average hold about 2 days. Tested Nov 2026 on 1,085 stocks
with 0.1% trading costs. Uses today's index members only (stocks that collapsed and were dropped are missing), so real
results will be somewhat lower. Losses cluster in sharp sell-offs; single trades lost up to about 40%.
"""
)

render_footer()
