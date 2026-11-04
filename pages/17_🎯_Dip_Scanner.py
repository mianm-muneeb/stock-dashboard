"""Dip Scanner: today's buy list and sell checks for the backtested dip-buy rule."""
import sys
import time
from datetime import datetime, timedelta
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
    """Now in New York, whether the market is open, and the next session's open/close (weekends skipped)."""
    now = datetime.now(NY)
    def session(d):
        return d.replace(hour=9, minute=30, second=0, microsecond=0), d.replace(hour=16, minute=0, second=0, microsecond=0)
    o, c = session(now)
    is_open = now.weekday() < 5 and o <= now < c
    d = now
    if now.weekday() >= 5 or now >= c:
        d = now + timedelta(days=1)
        while d.weekday() >= 5:
            d += timedelta(days=1)
    no, nc = session(d)
    return now, is_open, no, nc


st.title("🎯 Dip Scanner")
st.caption(
    "Rule: stock above its 200-day average, in the most volatile 20% of the S&P 500/400/600, with RSI(2) below 2 "
    "(a sharp 1-3 day drop). Buy at the close. Sell at the close of the first day that closes higher, or after 10 days. "
    "Hold at most 5 at once."
)

now_ny, is_open, open_ny, close_ny = market_clock()
open_syd, close_syd = open_ny.astimezone(SYD), close_ny.astimezone(SYD)
check_syd = close_syd - timedelta(minutes=15)

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
st.subheader("What to do now")
if is_open:
    mins_left = int((close_ny - now_ny).total_seconds() // 60)
    when = (f"Buy in the last 10-15 minutes before the US close at **{close_syd:%I:%M %p} Sydney** "
            f"({mins_left} min from now), or place market-on-close orders. Prices are live, so the list can still change.")
else:
    when = (f"These are from the last US close. **Best:** open this page again at **{check_syd:%a %I:%M %p} Sydney** "
            f"(15 min before the next US close) and buy what it lists then. "
            f"**Backup:** buy these at the next US open, **{open_syd:%a %I:%M %p} Sydney**.")
if sig.empty:
    st.success(f"No buys right now. Check again at {check_syd:%a %I:%M %p} Sydney.")
else:
    st.markdown(when)
    lines = []
    for t, row in sig.head(slots).iterrows():
        sh = int(per_pos // row["price"])
        lines.append(f"- **BUY {sh} × {t}** at about **${row['price']:,.2f}** (≈${sh * row['price']:,.0f}). "
                     f"Sell at the close of the first day it closes above the day before. Sell anyway after 10 trading days.")
    st.markdown("\n".join(lines))
    if len(sig) > slots:
        st.caption(f"{len(sig) - slots} more qualified; the list above takes the deepest dips first ({slots} max).")
    st.caption("Each morning (Sydney) after buying, enter the trades under 'Your open trades' below; it tells you SELL or HOLD.")

with st.expander(f"Details: all {len(sig)} signals"):
    if not sig.empty:
        st.dataframe(pd.DataFrame({
            "Ticker": sig.index,
            "Index": [members.get(t, "") for t in sig.index],
            "Price": sig["price"].round(2).values,
            "RSI(2)": sig["rsi2"].round(2).values,
            "Today %": (sig["chg1"] * 100).round(1).values,
            "3-day %": (sig["chg3"] * 100).round(1).values,
            "Shares": np.floor(per_pos / sig["price"]).astype(int).values,
        }), hide_index=True, use_container_width=True)

with st.expander("Almost there (closest to a signal)"):
    if near.empty:
        st.write("None.")
    else:
        st.dataframe(pd.DataFrame({
            "Ticker": near.index, "Index": [members.get(t, "") for t in near.index],
            "Price": near["price"].round(2).values, "RSI(2)": near["rsi2"].round(2).values,
            "Today %": (near["chg1"] * 100).round(1).values,
        }), hide_index=True, use_container_width=True)

st.subheader("Your open trades (from this list only)")
st.caption("One per line: TICKER, buy date (YYYY-MM-DD). Example: DT, 2026-11-05. Don't put long-term holdings here.")
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

with st.expander("How it tested"):
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
