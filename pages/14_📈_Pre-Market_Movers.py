import datetime as dt

import pandas as pd
import streamlit as st

from lib.config import APP_NAME, inject_base_style, render_footer
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

headlines = top_headlines_bulk(tuple(filtered["Ticker"]))

display = filtered[["Ticker", "Company", "Sector", "Price", "Change%", "Market Cap", "Avg Volume", "Range Position%"]].copy()
display = display.rename(columns={
    "Price": price_label,
    "Change%": change_label,
    "Range Position%": "52W Range Pos.",
})
display["Market Cap"] = display["Market Cap"] / 1e9
display["Avg Volume"] = display["Avg Volume"] / 1e6
display["Why It's Moving"] = display["Ticker"].map(lambda t: headlines.get(t) or "—")
display = display.reset_index(drop=True)


def _highlight_change(row):
    val = row[change_label]
    if val >= 2:
        style = "color: #2ecc71; font-weight: 600;"
    elif val >= 0.5:
        style = "color: #f1c40f; font-weight: 600;"
    else:
        style = ""
    styles = [""] * len(row)
    styles[row.index.get_loc(change_label)] = style
    return styles


styled = display.style.apply(_highlight_change, axis=1)

event = st.dataframe(
    styled,
    column_config={
        "Ticker": st.column_config.TextColumn("Ticker"),
        "Company": st.column_config.TextColumn("Company"),
        "Sector": st.column_config.TextColumn("Sector"),
        price_label: st.column_config.NumberColumn(price_label, format="$%.2f"),
        change_label: st.column_config.NumberColumn(change_label, format="%.2f%%"),
        "Market Cap": st.column_config.NumberColumn("Market Cap ($B)", format="$%.1fB"),
        "Avg Volume": st.column_config.NumberColumn("Avg Volume (M)", format="%.1fM"),
        "52W Range Pos.": st.column_config.NumberColumn("52W Range %", format="%.0f%%"),
        "Why It's Moving": st.column_config.TextColumn("Why It's Moving", width="large"),
    },
    hide_index=True,
    use_container_width=True,
    height=600,
    on_select="rerun",
    selection_mode="single-row",
)

selected_rows = event.selection.rows if event and event.selection else []
if selected_rows:
    sel_ticker = display.iloc[selected_rows[0]]["Ticker"]
    if st.button(f"Open {sel_ticker} in Analyzer"):
        st.session_state["switch_ticker"] = sel_ticker
        st.switch_page("pages/2_🔍_Stock_Analyzer.py")
else:
    st.caption("Select a row above to open it in the Stock Analyzer.")

render_footer()
