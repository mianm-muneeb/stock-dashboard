"""Pre-market movers: scans a curated large-cap S&P 500 watchlist for
pre-market price action, falling back to regular-session data outside the
4:00-9:30 AM ET pre-market window (Yahoo only populates preMarketPrice /
preMarketChangePercent during that window).
"""

from __future__ import annotations

import concurrent.futures
import datetime as dt
import random
import time
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
import yfinance as yf

from lib.market_data import _YF_SESSION, NAME_MAP

ET = ZoneInfo("America/New_York")
_SCAN_WORKERS = 15

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


def _scan_one(ticker: str) -> tuple[dict | None, str | None]:
    """Return (row, error). error holds the last exception text when the
    fetch failed, so the page can show *why* instead of a generic no-data
    message (market_data._fetch_ticker swallows exceptions)."""
    info, err = None, None
    for attempt in range(2):
        try:
            info = yf.Ticker(ticker, session=_YF_SESSION).info
            if info and len(info) > 5:
                break
            info, err = None, "empty response from Yahoo"
        except Exception as e:
            info, err = None, f"{type(e).__name__}: {e}"
        time.sleep((2 ** attempt) + random.uniform(0, 1))
    if info is None:
        return None, f"{ticker}: {err}"

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
    }, None


@st.cache_data(ttl=300, show_spinner=False)
def scan_premarket(tickers: tuple[str, ...]) -> pd.DataFrame:
    """Scan `tickers` for pre-market (or regular-session fallback) price data.

    Raises RuntimeError (with the real underlying error) when no ticker
    returned data; raising also keeps the failure out of the 5-minute cache.
    """
    rows, errors = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=_SCAN_WORKERS) as executor:
        for row, err in executor.map(_scan_one, tickers):
            if row:
                rows.append(row)
            elif err:
                errors.append(err)
    if not rows:
        raise RuntimeError(
            f"0 of {len(tickers)} tickers returned data. First error: "
            f"{errors[0] if errors else 'none recorded'}"
        )
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
