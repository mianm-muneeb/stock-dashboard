import csv
import datetime as dt
from pathlib import Path

import pandas as pd
import streamlit as st

from lib.config import APP_NAME, inject_base_style, render_footer
from lib.market_data import get_research_data
from lib.news import ticker_news, time_ago
from lib.signals import compute_rsi

st.set_page_config(page_title=f"Research Agent - {APP_NAME}", page_icon="🔬", layout="wide")
inject_base_style()

st.title("🔬 Research Agent")
st.caption(
    "Evidence-only pre-trade checklist: earnings timing, volume, moving averages, "
    "RSI, historical earnings-day moves, and headlines. No scores, ratings, or recommendations."
)

LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "research_agent_log.csv"
LOG_FIELDS = [
    "timestamp", "ticker", "earnings_within_14d", "earnings_date",
    "volume_ratio", "price_vs_50ma_pct", "price_vs_200ma_pct", "rsi",
]


def log_query(row: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    is_new = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


ticker = st.text_input("Ticker", value="AAPL", key="ra_ticker").strip().upper()
run = st.button("Analyze", type="primary")

if not (run and ticker):
    render_footer()
    st.stop()

data = get_research_data(ticker)
history = data["history"]

if history.empty or "Close" not in history.columns:
    st.error("Yahoo Finance is rate-limiting right now, or this ticker has no price history. Try again in 30 seconds.")
    if st.button("Retry"):
        st.rerun()
    render_footer()
    st.stop()

close = history["Close"].dropna()
volume = history["Volume"].dropna() if "Volume" in history.columns else pd.Series(dtype=float)
current_price = float(close.iloc[-1])
today = dt.date.today()

st.markdown(f"### {ticker} — ${current_price:,.2f}")
st.caption(f"Last close · {close.index[-1].strftime('%Y-%m-%d')}")

# ---------------------------------------------------------------------------
# a. Upcoming earnings
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Upcoming Earnings")

calendar = data.get("calendar")
earnings_dates_df = data.get("earnings_dates")

has_calendar = isinstance(calendar, dict) and bool(calendar.get("Earnings Date"))
has_earnings_df = isinstance(earnings_dates_df, pd.DataFrame) and not earnings_dates_df.empty

next_earnings_date = None
if has_calendar:
    future = [d for d in calendar["Earnings Date"] if isinstance(d, dt.date) and d >= today]
    if future:
        next_earnings_date = min(future)
if next_earnings_date is None and has_earnings_df:
    future_idx = [ts.date() for ts in earnings_dates_df.index if ts.date() >= today]
    if future_idx:
        next_earnings_date = min(future_idx)

if not has_calendar and not has_earnings_df:
    st.info("Data unavailable.")
    earnings_within_14d = False
elif next_earnings_date is None:
    st.write("No earnings within 14 days.")
    earnings_within_14d = False
else:
    days_until = (next_earnings_date - today).days
    earnings_within_14d = days_until <= 14
    if earnings_within_14d:
        plural = "s" if days_until != 1 else ""
        st.success(f"**{next_earnings_date.strftime('%Y-%m-%d')}** — {days_until} day{plural} away")
    else:
        st.write("No earnings within 14 days.")
        st.caption(f"Next known earnings date: {next_earnings_date.strftime('%Y-%m-%d')} ({days_until} days away)")

# ---------------------------------------------------------------------------
# b. Volume vs 20-day average
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Volume vs 20-Day Average")

if volume.empty:
    st.info("Data unavailable.")
    volume_ratio = None
else:
    latest_volume = float(volume.iloc[-1])
    avg20 = float(volume.tail(20).mean())
    volume_ratio = (latest_volume / avg20) if avg20 else None
    cols = st.columns(3)
    cols[0].metric("Latest Volume", f"{latest_volume:,.0f}")
    cols[1].metric("20-Day Avg Volume", f"{avg20:,.0f}")
    cols[2].metric("Ratio", f"{volume_ratio:.2f}x average" if volume_ratio is not None else "–")

# ---------------------------------------------------------------------------
# c. Price vs 50MA / 200MA
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Price vs Moving Averages")

ma50 = float(close.tail(50).mean()) if len(close) >= 50 else None
ma200 = float(close.tail(200).mean()) if len(close) >= 200 else None

cols = st.columns(2)
with cols[0]:
    if ma50 is None:
        st.info("50MA: Data unavailable.")
        price_vs_50ma_pct = None
    else:
        price_vs_50ma_pct = (current_price - ma50) / ma50 * 100
        direction = "Above" if price_vs_50ma_pct >= 0 else "Below"
        st.metric("50-Day MA", f"${ma50:,.2f}")
        st.write(f"{direction} 50MA by {abs(price_vs_50ma_pct):.1f}%")
with cols[1]:
    if ma200 is None:
        st.info("200MA: Data unavailable.")
        price_vs_200ma_pct = None
    else:
        price_vs_200ma_pct = (current_price - ma200) / ma200 * 100
        direction = "Above" if price_vs_200ma_pct >= 0 else "Below"
        st.metric("200-Day MA", f"${ma200:,.2f}")
        st.write(f"{direction} 200MA by {abs(price_vs_200ma_pct):.1f}%")

# ---------------------------------------------------------------------------
# d. RSI
# ---------------------------------------------------------------------------
st.divider()
st.subheader("RSI (14)")

rsi = compute_rsi(close, period=14)
if rsi is None:
    st.info("Data unavailable.")
else:
    st.metric("RSI", f"{rsi:.1f}")

# ---------------------------------------------------------------------------
# e. Historical same-day earnings moves
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Historical Earnings-Day Moves")

if not has_earnings_df:
    st.info("Data unavailable.")
else:
    past_dates = sorted(
        (ts.date() for ts in earnings_dates_df.index if ts.date() < today),
        reverse=True,
    )[:8]

    date_pos = {d: i for i, d in enumerate(ts.date() for ts in close.index)}

    moves = []
    for d in past_dates:
        pos = date_pos.get(d)
        if pos is None or pos == 0:
            continue
        prev_close = float(close.iloc[pos - 1])
        day_close = float(close.iloc[pos])
        pct = (day_close / prev_close - 1) * 100
        moves.append({"Earnings Date": d.strftime("%Y-%m-%d"), "% Move": round(pct, 2)})

    if not moves:
        st.info("Data unavailable.")
    else:
        if len(moves) < 8:
            st.caption(f"Showing {len(moves)} of up to 8 most recent earnings dates (limited by available price history).")
        st.dataframe(pd.DataFrame(moves), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# f. Recent news headlines
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Recent Headlines")

news_items = ticker_news(ticker, limit=5)
if not news_items:
    st.info("Data unavailable.")
else:
    for item in news_items:
        with st.container(border=True):
            st.markdown(f"**{item['title']}**")
            meta = item["publisher"]
            ago = time_ago(item["time"])
            if ago:
                meta += f" · {ago}"
            st.caption(meta)
            if item.get("link"):
                st.markdown(f"[Read more]({item['link']})")

# ---------------------------------------------------------------------------
# Log the query
# ---------------------------------------------------------------------------
try:
    log_query({
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "ticker": ticker,
        "earnings_within_14d": earnings_within_14d,
        "earnings_date": next_earnings_date.strftime("%Y-%m-%d") if next_earnings_date else "",
        "volume_ratio": round(volume_ratio, 4) if volume_ratio is not None else "",
        "price_vs_50ma_pct": round(price_vs_50ma_pct, 4) if price_vs_50ma_pct is not None else "",
        "price_vs_200ma_pct": round(price_vs_200ma_pct, 4) if price_vs_200ma_pct is not None else "",
        "rsi": round(rsi, 4) if rsi is not None else "",
    })
except Exception:
    pass

render_footer()
