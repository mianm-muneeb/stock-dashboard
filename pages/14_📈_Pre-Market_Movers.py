import datetime as dt

import pandas as pd
import streamlit as st

from lib.config import APP_NAME, fmt_large_number, inject_base_style, render_footer
from lib.premarket import ET, PREMARKET_UNIVERSE, is_premarket_window, scan_premarket, top_headlines_bulk

st.set_page_config(page_title=f"Pre-Market Movers - {APP_NAME}", page_icon="📈", layout="wide")
inject_base_style()

st.title("📈 Pre-Market Movers")
st.caption(
    f"Scans a curated watchlist of {len(PREMARKET_UNIVERSE)} large-cap S&P 500 stocks for "
    "pre-market price action. Data refreshes every 5 minutes."
)
st.info(
    "Pre-market data available 4:00-9:30 AM ET. During market hours shows regular session data.",
    icon="ℹ️",
)

premarket_active = is_premarket_window()

with st.spinner("Scanning watchlist..."):
    universe_df = scan_premarket(tuple(PREMARKET_UNIVERSE))

if universe_df.empty:
    st.warning("No data available right now - try again shortly.")
    render_footer()
    st.stop()

# ---------------------------------------------------------------------------
# Effective price/change: pre-market when in the pre-market window, otherwise
# fall back to the regular session so the page isn't empty all day.
# ---------------------------------------------------------------------------

df = universe_df.copy()
if premarket_active:
    df["Price"] = df["Pre-Market Price"]
    df["Change%"] = df["Pre-Market Change%"]
else:
    df["Price"] = df["Regular Price"]
    df["Change%"] = df["Regular Change%"]

df = df[df["Price"].notna() & df["Change%"].notna()].copy()

has_range = (df["52W High"] > df["52W Low"]) & df["52W High"].notna() & df["52W Low"].notna()
df["Range Position%"] = pd.NA
df.loc[has_range, "Range Position%"] = (
    (df.loc[has_range, "Price"] - df.loc[has_range, "52W Low"])
    / (df.loc[has_range, "52W High"] - df.loc[has_range, "52W Low"]) * 100
)

now_et = dt.datetime.now(ET)
st.caption(
    f"Last updated: {now_et.strftime('%Y-%m-%d %H:%M:%S')} ET · "
    f"Session: {'Pre-Market' if premarket_active else 'Regular'} · "
    f"{len(df)} of {len(universe_df)} stocks have data available"
)

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

f1, f2, f3 = st.columns(3)
with f1:
    mcap_min = st.slider(
        "Minimum market cap ($B)", 1, 500, 10,
        help="Set to 500 for $500B+ (mega-caps, no upper cap)",
    )
with f2:
    change_min = st.slider("Minimum change % (pre-market or regular)", 0.0, 10.0, 0.5, step=0.1)
with f3:
    vol_min = st.slider("Minimum avg daily volume (M)", 0.0, 50.0, 1.0, step=0.5)

filtered = df[
    df["Market Cap"].notna() & (df["Market Cap"] >= mcap_min * 1e9)
    & (df["Change%"] >= change_min)
    & df["Avg Volume"].notna() & (df["Avg Volume"] >= vol_min * 1e6)
].sort_values("Change%", ascending=False)

st.divider()
st.caption(f"{len(filtered)} of {len(df)} stocks with available data match the current filters.")

if filtered.empty:
    st.info("No stocks match the current filters.")
    render_footer()
    st.stop()

# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

price_label = "Pre-Market Price" if premarket_active else "Price (Regular)"
change_label = "Pre-Mkt Change %" if premarket_active else "Change % (Regular)"

headers = ["Ticker", "Company", "Sector", price_label, change_label, "Market Cap", "Avg Volume", "52W Range Pos.", "Why It's Moving", ""]
widths = [0.6, 1.6, 1.1, 0.9, 1.0, 0.8, 0.9, 0.9, 2.2, 1.2]

headlines = top_headlines_bulk(tuple(filtered["Ticker"]))

header_cols = st.columns(widths)
for col, label in zip(header_cols, headers):
    col.markdown(f"**{label}**")
st.divider()

for _, row in filtered.iterrows():
    cols = st.columns(widths)
    change = row["Change%"]
    color = "#2ecc71" if change >= 2 else ("#f1c40f" if change >= 0.5 else "#d1d4dc")
    range_pos = row["Range Position%"]

    cols[0].markdown(f"**{row['Ticker']}**")
    cols[1].write(row["Company"])
    cols[2].write(row["Sector"])
    cols[3].write(f"${row['Price']:,.2f}")
    cols[4].markdown(f"<span style='color:{color}; font-weight:600;'>{change:+.2f}%</span>", unsafe_allow_html=True)
    cols[5].write(fmt_large_number(row["Market Cap"]))
    cols[6].write(fmt_large_number(row["Avg Volume"], currency=""))
    cols[7].write(f"{float(range_pos):.0f}%" if pd.notna(range_pos) else "—")
    cols[8].write(headlines.get(row["Ticker"]) or "—")

    if cols[9].button("Open in Analyzer", key=f"open_{row['Ticker']}"):
        st.session_state["switch_ticker"] = row["Ticker"]
        st.switch_page("pages/2_🔍_Stock_Analyzer.py")

render_footer()
