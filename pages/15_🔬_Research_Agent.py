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

GREEN, RED, AMBER, NEUTRAL = "#2ecc71", "#e74c3c", "#f39c12", "#888"


def log_query(row: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    is_new = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def unavailable(reason: str | None) -> None:
    """Show 'Data unavailable', surfacing the real fetch error (if any) instead
    of hiding it - a bare fetch failure looks the same as a ticker that
    legitimately has no data otherwise.
    """
    if reason:
        st.warning(f"Data unavailable — fetch failed: {reason}")
    else:
        st.info("Data unavailable.")


def render_research(ticker: str, data: dict) -> None:
    """Render all six sections for an already-fetched research payload."""
    history = data["history"]
    close = history["Close"].dropna()
    volume = history["Volume"].dropna() if "Volume" in history.columns else pd.Series(dtype=float)
    current_price = float(close.iloc[-1])
    today = dt.date.today()

    st.markdown(f"### {ticker} — ${current_price:,.2f}")
    st.caption(f"Last close · {close.index[-1].strftime('%Y-%m-%d')}")

    # -----------------------------------------------------------------------
    # a. Upcoming earnings
    # -----------------------------------------------------------------------
    st.divider()
    st.subheader("Upcoming Earnings")

    calendar = data.get("calendar")
    calendar_error = data.get("calendar_error")
    earnings_dates_df = data.get("earnings_dates")
    earnings_error = data.get("earnings_error")

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

    with st.container(border=True):
        if not has_calendar and not has_earnings_df:
            unavailable(calendar_error or earnings_error)
            earnings_within_14d = False
        elif next_earnings_date is None:
            st.markdown(f"<span style='color:{NEUTRAL};font-weight:600'>No earnings within 14 days.</span>", unsafe_allow_html=True)
            earnings_within_14d = False
        else:
            days_until = (next_earnings_date - today).days
            earnings_within_14d = days_until <= 14
            color = GREEN if earnings_within_14d else NEUTRAL
            label = "EARNINGS WITHIN 14 DAYS" if earnings_within_14d else "NEXT KNOWN EARNINGS DATE"
            plural = "s" if days_until != 1 else ""
            st.markdown(
                f"<div style='display:flex;align-items:baseline;gap:20px;flex-wrap:wrap'>"
                f"<div>"
                f"<div style='font-size:10px;color:{color};font-weight:700;letter-spacing:1.3px'>{label}</div>"
                f"<div style='font-size:28px;font-weight:800;color:{color}'>{next_earnings_date.strftime('%Y-%m-%d')}</div>"
                f"</div>"
                f"<div style='font-size:14px;color:#aaa'>{days_until} day{plural} away</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            if not earnings_within_14d:
                st.caption("No earnings within 14 days.")

    # -----------------------------------------------------------------------
    # b. Volume vs 20-day average
    # -----------------------------------------------------------------------
    st.divider()
    st.subheader("Volume vs 20-Day Average")

    with st.container(border=True):
        if volume.empty:
            unavailable(None)
            volume_ratio = None
        else:
            latest_volume = float(volume.iloc[-1])
            avg20 = float(volume.tail(20).mean())
            volume_ratio = (latest_volume / avg20) if avg20 else None
            cols = st.columns(3)
            cols[0].metric("Latest Volume", f"{latest_volume:,.0f}")
            cols[1].metric("20-Day Avg Volume", f"{avg20:,.0f}")
            if volume_ratio is not None:
                cols[2].metric("Ratio", f"{volume_ratio:.2f}x average", delta=f"{(volume_ratio - 1) * 100:+.0f}% vs avg")
            else:
                cols[2].metric("Ratio", "–")

    # -----------------------------------------------------------------------
    # c. Price vs 50MA / 200MA
    # -----------------------------------------------------------------------
    st.divider()
    st.subheader("Price vs Moving Averages")

    ma50 = float(close.tail(50).mean()) if len(close) >= 50 else None
    ma200 = float(close.tail(200).mean()) if len(close) >= 200 else None

    with st.container(border=True):
        cols = st.columns(2)
        with cols[0]:
            if ma50 is None:
                unavailable(None)
                price_vs_50ma_pct = None
            else:
                price_vs_50ma_pct = (current_price - ma50) / ma50 * 100
                direction = "Above" if price_vs_50ma_pct >= 0 else "Below"
                st.metric("50-Day MA", f"${ma50:,.2f}", delta=f"{price_vs_50ma_pct:+.1f}%")
                st.caption(f"{direction} 50MA by {abs(price_vs_50ma_pct):.1f}%")
        with cols[1]:
            if ma200 is None:
                unavailable(None)
                price_vs_200ma_pct = None
            else:
                price_vs_200ma_pct = (current_price - ma200) / ma200 * 100
                direction = "Above" if price_vs_200ma_pct >= 0 else "Below"
                st.metric("200-Day MA", f"${ma200:,.2f}", delta=f"{price_vs_200ma_pct:+.1f}%")
                st.caption(f"{direction} 200MA by {abs(price_vs_200ma_pct):.1f}%")

    # -----------------------------------------------------------------------
    # d. RSI
    # -----------------------------------------------------------------------
    st.divider()
    st.subheader("RSI (14)")

    rsi = compute_rsi(close, period=14)

    with st.container(border=True):
        if rsi is None:
            unavailable(None)
        else:
            rsi_color = RED if rsi >= 70 else GREEN if rsi <= 30 else "#d1d4dc"
            st.markdown(
                f"<div style='font-size:10px;color:{NEUTRAL};font-weight:700;letter-spacing:1.3px'>RSI (14)</div>"
                f"<div style='font-size:34px;font-weight:800;color:{rsi_color}'>{rsi:.1f}</div>",
                unsafe_allow_html=True,
            )

    # -----------------------------------------------------------------------
    # e. Historical same-day earnings moves
    # -----------------------------------------------------------------------
    st.divider()
    st.subheader("Historical Earnings-Day Moves")

    with st.container(border=True):
        if not has_earnings_df:
            unavailable(earnings_error)
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
                unavailable(earnings_error)
            else:
                if len(moves) < 8:
                    st.caption(f"Showing {len(moves)} of up to 8 most recent earnings dates (limited by available price history).")

                moves_df = pd.DataFrame(moves)

                def _highlight_move(row):
                    color = GREEN if row["% Move"] >= 0 else RED
                    return [f"color: {color}; font-weight: 600" if col == "% Move" else "" for col in row.index]

                st.dataframe(
                    moves_df.style.apply(_highlight_move, axis=1),
                    column_config={"% Move": st.column_config.NumberColumn("% Move", format="%.2f%%")},
                    use_container_width=True,
                    hide_index=True,
                )

    # -----------------------------------------------------------------------
    # f. Recent news headlines
    # -----------------------------------------------------------------------
    st.divider()
    st.subheader("Recent Headlines")

    with st.spinner("Fetching headlines…"):
        news_items = ticker_news(ticker, limit=5)

    if not news_items:
        unavailable(None)
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

    # -----------------------------------------------------------------------
    # Log the query
    # -----------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Entry point: ticker input + Analyze button gate everything below.
# ---------------------------------------------------------------------------
ticker = st.text_input("Ticker", value="AAPL", key="ra_ticker").strip().upper()
run = st.button("Analyze", type="primary")

if run:
    if not ticker:
        st.warning("Enter a ticker symbol.")
    else:
        try:
            with st.spinner(f"Fetching data for {ticker}…"):
                data = get_research_data(ticker)
                history = data["history"]

            if history.empty or "Close" not in history.columns:
                st.error("Yahoo Finance is rate-limiting right now, or this ticker has no price history. Try again in 30 seconds.")
                if st.button("Retry"):
                    st.rerun()
            else:
                render_research(ticker, data)
        except Exception as e:
            st.error(f"Something went wrong while analyzing {ticker}: {type(e).__name__}: {e}")

render_footer()
