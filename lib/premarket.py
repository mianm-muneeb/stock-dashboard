"""Pre-market movers: scans a curated large-cap S&P 500 watchlist for
pre-market price action, falling back to regular-session data outside the
4:00-9:30 AM ET pre-market window (Yahoo only populates preMarketPrice /
preMarketChangePercent during that window).
"""

from __future__ import annotations

import concurrent.futures
import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from lib.market_data import NAME_MAP, _fetch_ticker

ET = ZoneInfo("America/New_York")

# 150 liquid, large-cap S&P 500 constituents across sectors, curated for a
# pre-market watchlist.
PREMARKET_UNIVERSE = [
    # Technology / Communication Services
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA", "AVGO", "ORCL",
    "CRM", "ADBE", "AMD", "INTC", "CSCO", "NFLX", "QCOM", "TXN", "IBM", "NOW",
    "INTU", "AMAT", "MU", "LRCX", "ADI", "PANW", "SNOW", "ABNB", "UBER", "PYPL",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "AXP", "BRK-B", "BLK",
    "SCHW", "C", "SPGI", "ICE", "CME", "PGR", "TRV", "AIG", "MET", "PRU",
    # Healthcare
    "UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV", "TMO", "ABT", "DHR", "BMY",
    "AMGN", "GILD", "CVS", "ELV", "CI", "HUM", "MDT", "ISRG", "SYK", "BSX",
    # Consumer
    "WMT", "COST", "PG", "KO", "PEP", "MCD", "HD", "NKE", "TGT", "SBUX",
    "LOW", "CMCSA", "DIS", "BKNG", "MDLZ", "CL", "KMB", "EL", "YUM", "CMG",
    # Industrials
    "BA", "CAT", "GE", "HON", "LMT", "UPS", "RTX", "DE", "MMM", "UNP",
    "NOC", "GD", "EMR", "ETN", "ITW",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "PSX", "MPC", "VLO", "OXY", "WMB",
    # Utilities
    "NEE", "DUK", "SO", "AEP", "D", "EXC", "SRE", "XEL", "ED", "PEG",
    # Telecom / Other
    "T", "VZ", "TMUS", "F", "GM", "COIN", "RBLX", "RIVN", "SHOP", "PLTR",
    # Materials / Real Estate
    "LIN", "APD", "ECL", "NEM", "FCX", "SHW", "PLD", "AMT", "EQIX", "PSA",
    "O", "SPG", "DLR", "WY", "AVB",
]


def is_premarket_window(now: dt.datetime | None = None) -> bool:
    """True during 4:00-9:30 AM ET on weekdays - Yahoo's pre-market data window."""
    now = now or dt.datetime.now(ET)
    if now.weekday() >= 5:
        return False
    start = now.replace(hour=4, minute=0, second=0, microsecond=0)
    open_ = now.replace(hour=9, minute=30, second=0, microsecond=0)
    return start <= now < open_


def _scan_one(ticker: str) -> dict | None:
    tk = _fetch_ticker(ticker, retries=2)
    if tk is None:
        return None
    info = tk.info

    pre_price = info.get("preMarketPrice")
    pre_pct = info.get("preMarketChangePercent")
    reg_pct = info.get("regularMarketChangePercent")

    return {
        "Ticker": ticker,
        "Company": info.get("shortName") or info.get("longName") or NAME_MAP.get(ticker, ticker),
        "Sector": info.get("sector") or "—",
        "Pre-Market Price": pre_price,
        # Note: yfinance's *ChangePercent fields are already percentages
        # (e.g. 1.89 means +1.89%), not fractions - do not multiply by 100.
        "Pre-Market Change%": pre_pct,
        "Regular Price": info.get("currentPrice") or info.get("regularMarketPrice"),
        "Regular Change%": reg_pct,
        "Market Cap": info.get("marketCap"),
        "Avg Volume": info.get("averageVolume"),
        "52W Low": info.get("fiftyTwoWeekLow"),
        "52W High": info.get("fiftyTwoWeekHigh"),
    }


@st.cache_data(ttl=300, show_spinner=False)
def scan_premarket(tickers: tuple[str, ...]) -> pd.DataFrame:
    """Scan `tickers` for pre-market (or regular-session fallback) price data."""
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        for row in executor.map(_scan_one, tickers):
            if row:
                rows.append(row)
    return pd.DataFrame(rows)


def _one_headline(ticker: str) -> tuple[str, str | None]:
    from lib.news import ticker_news

    items = ticker_news(ticker, limit=1)
    return ticker, (items[0]["title"] if items else None)


@st.cache_data(ttl=300, show_spinner=False)
def top_headlines_bulk(tickers: tuple[str, ...]) -> dict:
    """Return {ticker: latest headline or None} for `tickers`."""
    result = {}
    if not tickers:
        return result
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for ticker, headline in executor.map(_one_headline, tickers):
            result[ticker] = headline
    return result
