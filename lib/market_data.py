"""yfinance wrappers shared across the app.

All network calls are cached with @st.cache_data so pages stay snappy and
avoid hammering Yahoo Finance.
"""

from __future__ import annotations

import datetime as dt
import logging
import random
import tempfile
import time

import requests
import pandas as pd
import streamlit as st
import yfinance as yf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------

_YF_SESSION = requests.Session()
_YF_SESSION.headers["User-Agent"] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
)

# Cache timezone data in the system temp dir so it's always writable
# (avoids permission errors on Streamlit Cloud / read-only filesystems).
try:
    yf.set_tz_cache_location(tempfile.gettempdir())
except Exception:
    pass


def _fetch_ticker(symbol: str, retries: int = 5) -> yf.Ticker | None:
    for attempt in range(retries):
        try:
            session = requests.Session()
            session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            })
            tk = yf.Ticker(symbol, session=session)
            info = tk.info
            if info and len(info) > 5:
                return tk
        except Exception:
            time.sleep((2 ** attempt) + random.uniform(2, 5))
    return None


def _fetch_history(tk: yf.Ticker, retries: int = 3, **kwargs) -> pd.DataFrame:
    """Call tk.history(**kwargs) with exponential-backoff retries."""
    for attempt in range(retries):
        try:
            df = tk.history(**kwargs)
            if df is not None and not df.empty:
                return df
        except Exception:
            pass
        if attempt < retries - 1:
            time.sleep(2 ** attempt + random.uniform(0, 1))
    return pd.DataFrame()


def _is_not_found_error(e: Exception) -> bool:
    """True for a 404 from Yahoo's quoteSummary API - its deterministic way of
    saying "this symbol has no such data" (e.g. an ETF has no earnings
    calendar). That's a real, permanent answer, not a transient failure, so
    it shouldn't be retried or surfaced as a scary fetch error.
    """
    response = getattr(e, "response", None)
    return response is not None and getattr(response, "status_code", None) == 404


def _retry_call(fn, retries: int = 3, label: str = ""):
    """Call fn() with exponential-backoff retries on exceptions only (a
    legitimate empty/falsy result is not retried - that's real data, not a
    failure). Returns (result, error_message); error_message is None unless
    every attempt raised, in which case it holds the last exception's text
    so callers can surface *why* a fetch failed instead of silently showing
    "unavailable" for both real outages and tickers with no data.
    """
    last_err = None
    for attempt in range(retries):
        try:
            return fn(), None
        except Exception as e:
            if _is_not_found_error(e):
                logger.info("market_data: %s returned 404 (no data for this symbol)", label)
                return None, None
            last_err = f"{type(e).__name__}: {e}"
            logger.warning("market_data: %s failed (attempt %d/%d): %s", label, attempt + 1, retries, last_err)
        if attempt < retries - 1:
            time.sleep(2 ** attempt + random.uniform(0, 1))
    return None, last_err


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Real indices / futures / FX, not ETF proxies - used on Market Pulse + charts.
INDEX_TICKERS = {
    "^GSPC": "S&P 500",
    "^NDX": "Nasdaq 100",
    "^DJI": "Dow Jones",
    "^RUT": "Russell 2000",
    "^VIX": "VIX",
    "^TNX": "10Y Yield",
    "GC=F": "Gold",
    "CL=F": "Crude Oil (WTI)",
    "BTC-USD": "Bitcoin",
    "DX-Y.NYB": "US Dollar Index",
}

# The 11 SPDR sector ETFs.
SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Health Care",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLB": "Materials",
    "XLC": "Communication Services",
}

# Universe of widely-held large/mega-cap tickers used to compute "movers"
# (gainers / losers / most active) on the Market Pulse page.
MOVERS_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO", "ORCL", "ADBE",
    "CRM", "AMD", "INTC", "CSCO", "NFLX", "DIS", "UBER", "PYPL", "SHOP", "PLTR",
    "JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "AXP", "BRK-B", "BLK",
    "UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV", "TMO", "ABT",
    "WMT", "COST", "PG", "KO", "PEP", "MCD", "HD", "NKE", "TGT", "SBUX",
    "XOM", "CVX", "COP", "BA", "CAT", "GE", "HON", "LMT",
    "T", "VZ", "TMUS", "F", "GM", "COIN", "RBLX", "RIVN",
]

# Display names for tickers used across the app (indices, sector ETFs, and
# the movers universe) - avoids a slow per-ticker .info call just for a name.
NAME_MAP = {
    **INDEX_TICKERS,
    **SECTOR_ETFS,
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Alphabet", "AMZN": "Amazon",
    "META": "Meta Platforms", "NVDA": "NVIDIA", "TSLA": "Tesla", "AVGO": "Broadcom",
    "ORCL": "Oracle", "ADBE": "Adobe", "CRM": "Salesforce", "AMD": "AMD",
    "INTC": "Intel", "CSCO": "Cisco", "NFLX": "Netflix", "DIS": "Disney",
    "UBER": "Uber", "PYPL": "PayPal", "SHOP": "Shopify", "PLTR": "Palantir",
    "JPM": "JPMorgan Chase", "BAC": "Bank of America", "WFC": "Wells Fargo",
    "GS": "Goldman Sachs", "MS": "Morgan Stanley", "V": "Visa", "MA": "Mastercard",
    "AXP": "American Express", "BRK-B": "Berkshire Hathaway", "BLK": "BlackRock",
    "UNH": "UnitedHealth", "JNJ": "Johnson & Johnson", "LLY": "Eli Lilly",
    "PFE": "Pfizer", "MRK": "Merck", "ABBV": "AbbVie", "TMO": "Thermo Fisher",
    "ABT": "Abbott Labs",
    "WMT": "Walmart", "COST": "Costco", "PG": "Procter & Gamble", "KO": "Coca-Cola",
    "PEP": "PepsiCo", "MCD": "McDonald's", "HD": "Home Depot", "NKE": "Nike",
    "TGT": "Target", "SBUX": "Starbucks",
    "XOM": "Exxon Mobil", "CVX": "Chevron", "COP": "ConocoPhillips", "BA": "Boeing",
    "CAT": "Caterpillar", "GE": "GE Aerospace", "HON": "Honeywell", "LMT": "Lockheed Martin",
    "T": "AT&T", "VZ": "Verizon", "TMUS": "T-Mobile US", "F": "Ford", "GM": "General Motors",
    "COIN": "Coinbase", "RBLX": "Roblox", "RIVN": "Rivian",
}

# Period label -> yfinance period/interval configuration.
# When "yf_period" is set we use yfinance's built-in period string (cheapest).
# Otherwise we compute a start date "days" ago and request "interval" data.
PERIOD_MAP = {
    "1D": {"yf_period": "1d", "interval": "5m", "days": 1},
    "5D": {"yf_period": "5d", "interval": "15m", "days": 5},
    "1M": {"yf_period": "1mo", "interval": "1d", "days": 30},
    "3M": {"yf_period": "3mo", "interval": "1d", "days": 91},
    "6M": {"yf_period": "6mo", "interval": "1d", "days": 182},
    "YTD": {"yf_period": "ytd", "interval": "1d", "days": None},
    "1Y": {"yf_period": "1y", "interval": "1d", "days": 365},
    "3Y": {"yf_period": None, "interval": "1d", "days": 365 * 3},
    "5Y": {"yf_period": "5y", "interval": "1d", "days": 365 * 5},
    "10Y": {"yf_period": "10y", "interval": "1wk", "days": 365 * 10},
    "20Y": {"yf_period": None, "interval": "1wk", "days": 365 * 20},
    "30Y": {"yf_period": None, "interval": "1wk", "days": 365 * 30},
    "Max": {"yf_period": "max", "interval": "1mo", "days": None},
}

# TradingView-style timeframe selector for the Stock Analyzer chart.
# "resample" (when set) is applied to the raw yfinance bars to build
# intervals yfinance doesn't offer natively.
TIMEFRAMES = {
    "1m": {"yf_period": "7d", "interval": "1m", "resample": None},
    "2m": {"yf_period": "60d", "interval": "2m", "resample": None},
    "5m": {"yf_period": "60d", "interval": "5m", "resample": None},
    "15m": {"yf_period": "60d", "interval": "15m", "resample": None},
    "30m": {"yf_period": "60d", "interval": "30m", "resample": None},
    "60m": {"yf_period": "60d", "interval": "60m", "resample": None},
    "90m": {"yf_period": "60d", "interval": "90m", "resample": None},
    "1d": {"yf_period": "2y", "interval": "1d", "resample": None},
    "1wk": {"yf_period": "10y", "interval": "1wk", "resample": None},
    "1mo": {"yf_period": "max", "interval": "1mo", "resample": None},
}


# ---------------------------------------------------------------------------
# Quotes
# ---------------------------------------------------------------------------

def _empty_quote(ticker: str) -> dict:
    return {
        "ticker": ticker,
        "price": None,
        "prev_close": None,
        "change": None,
        "pct_change": None,
        "name": NAME_MAP.get(ticker, ticker),
        "currency": "USD",
        "volume": None,
    }


def _quote_from_intraday(ticker: str, df: pd.DataFrame | None) -> dict:
    """Derive a quote dict from a slice of intraday OHLCV bars covering the
    last few sessions (as returned by a batched yf.download call).
    """
    if df is None or df.empty:
        return _empty_quote(ticker)

    df = df.dropna(subset=["Close"])
    if df.empty:
        return _empty_quote(ticker)

    sessions = sorted(set(df.index.date))
    last_session = sessions[-1]
    last_df = df[df.index.date == last_session]

    price = float(last_df["Close"].iloc[-1])
    volume = float(last_df["Volume"].sum()) if "Volume" in last_df.columns else None

    prev_close = None
    if len(sessions) >= 2:
        prev_df = df[df.index.date == sessions[-2]]
        if not prev_df.empty:
            prev_close = float(prev_df["Close"].iloc[-1])
    if prev_close is None:
        prev_close = float(last_df["Open"].iloc[0])

    change = price - prev_close
    pct_change = (change / prev_close) * 100 if prev_close else None

    return {
        "ticker": ticker,
        "price": price,
        "prev_close": prev_close,
        "change": change,
        "pct_change": pct_change,
        "name": NAME_MAP.get(ticker, ticker),
        "currency": "USD",
        "volume": volume,
    }


@st.cache_data(ttl=60, show_spinner=False)
def get_quotes_bulk(tickers: tuple) -> dict:
    """Return {ticker: quote_dict} for many tickers via a single batched
    download (fast, even for dozens of tickers).
    """
    tickers = list(dict.fromkeys(tickers))  # de-dupe, preserve order
    if not tickers:
        return {}

    try:
        data = yf.download(
            tickers, period="5d", interval="15m", group_by="ticker",
            threads=True, progress=False, auto_adjust=False,
            session=_YF_SESSION,
        )
    except Exception:
        data = None

    result = {}
    for t in tickers:
        df = None
        if data is not None and not data.empty:
            try:
                df = data[t]
            except (KeyError, TypeError):
                df = None
        result[t] = _quote_from_intraday(t, df)

    return result


@st.cache_data(ttl=60, show_spinner=False)
def get_quote(ticker: str) -> dict:
    """Return a snapshot quote for a single ticker."""
    return get_quotes_bulk((ticker,))[ticker]


@st.cache_data(ttl=60, show_spinner=False)
def get_movers(universe: tuple, top_n: int = 5) -> dict:
    """Return {gainers, losers, most_active} lists of quote dicts for the
    given ticker universe, each sorted appropriately and trimmed to top_n.
    """
    quotes = get_quotes_bulk(universe)
    valid = [q for q in quotes.values() if q.get("price") is not None and q.get("pct_change") is not None]

    gainers = sorted(valid, key=lambda q: q["pct_change"], reverse=True)[:top_n]
    losers = sorted(valid, key=lambda q: q["pct_change"])[:top_n]

    with_volume = [q for q in valid if q.get("volume")]
    most_active = sorted(with_volume, key=lambda q: q["volume"], reverse=True)[:top_n]

    return {"gainers": gainers, "losers": losers, "most_active": most_active}


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def get_history(ticker: str, period_label: str) -> pd.DataFrame:
    """Return OHLCV history for a ticker over the given PERIOD_MAP label."""
    cfg = PERIOD_MAP.get(period_label, PERIOD_MAP["1Y"])
    tk = yf.Ticker(ticker, session=_YF_SESSION)
    if cfg["yf_period"]:
        return _fetch_history(tk, period=cfg["yf_period"], interval=cfg["interval"])
    start = dt.datetime.now() - dt.timedelta(days=cfg["days"])
    return _fetch_history(tk, start=start, interval=cfg["interval"])


@st.cache_data(ttl=3600, show_spinner=False)
def get_history_tf(ticker: str, timeframe: str) -> pd.DataFrame:
    """Return OHLCV history for a ticker over the given TIMEFRAMES label."""
    cfg = TIMEFRAMES.get(timeframe, TIMEFRAMES["1d"])
    tk = yf.Ticker(ticker, session=_YF_SESSION)
    df = _fetch_history(tk, period=cfg["yf_period"], interval=cfg["interval"])

    if not df.empty and cfg["resample"]:
        df = df.resample(cfg["resample"]).agg({
            "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum",
        }).dropna(subset=["Close"])

    return df


@st.cache_data(ttl=3600, show_spinner=False)
def get_prev_close(ticker: str) -> float | None:
    """Return the previous trading day's close - used as 1D chart baseline."""
    try:
        tk = yf.Ticker(ticker, session=_YF_SESSION)
        fast = tk.fast_info
        pc = fast.get("previousClose") or fast.get("previous_close")
        if pc:
            return float(pc)
        hist = _fetch_history(tk, period="5d", interval="1d")
        if len(hist) >= 2:
            return float(hist["Close"].iloc[-2])
        if len(hist) == 1:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return None


@st.cache_data(ttl=60, show_spinner=False)
def get_extended_hours_data(ticker: str) -> dict | None:
    """Return extended-hours snapshot when outside regular market hours.

    Returns None during regular trading hours (9:30 AM–4:00 PM ET, weekdays)
    or when no extended-hours price is available.

    Returned dict keys:
        session        – "pre" or "post"
        price          – last extended-hours price
        change         – change from previous regular close
        pct_change     – % change (e.g. -1.23 means −1.23 %)
        volume         – cumulative extended-hours volume (may be None)
        last_trade_time – tz-aware datetime of last extended trade (may be None)
        prev_close     – previous regular-session close
    """
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    now = dt.datetime.now(et)

    if now.weekday() >= 5:
        return None

    market_open  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0,  second=0, microsecond=0)
    pre_start    = now.replace(hour=4,  minute=0,  second=0, microsecond=0)
    post_end     = now.replace(hour=20, minute=0,  second=0, microsecond=0)

    if market_open <= now <= market_close:
        return None

    if pre_start <= now < market_open:
        session = "pre"
    elif market_close < now <= post_end:
        session = "post"
    else:
        return None

    tk = _fetch_ticker(ticker)
    if tk is None:
        return None
    full_info = tk.info

    prev_close = (
        full_info.get("regularMarketPreviousClose")
        or full_info.get("previousClose")
    )

    price = None
    pct_change = None
    volume = None
    last_trade_time = None

    # Primary: 1-minute bars with prepost=True for price + volume + timestamp
    try:
        bars = _fetch_history(tk, period="1d", interval="1m", prepost=True)
        if bars is not None and not bars.empty:
            if bars.index.tz is None:
                bars.index = bars.index.tz_localize("UTC").tz_convert(et)
            else:
                bars.index = bars.index.tz_convert(et)

            session_bars = (
                bars[bars.index < market_open]
                if session == "pre"
                else bars[bars.index > market_close]
            )
            if not session_bars.empty:
                price = float(session_bars["Close"].iloc[-1])
                volume = float(session_bars["Volume"].sum())
                last_trade_time = session_bars.index[-1].to_pydatetime()
    except Exception:
        pass

    # Fallback: .info fields
    if price is None:
        if session == "pre":
            price = full_info.get("preMarketPrice")
            raw_pct = full_info.get("preMarketChangePercent")
        else:
            price = full_info.get("postMarketPrice")
            raw_pct = full_info.get("postMarketChangePercent")
        if raw_pct is not None:
            pct_change = raw_pct * 100

    if price is None:
        return None

    change = (price - prev_close) if prev_close else None
    if pct_change is None and prev_close and change is not None:
        pct_change = change / prev_close * 100

    return {
        "session": session,
        "price": price,
        "change": change,
        "pct_change": pct_change,
        "volume": volume,
        "last_trade_time": last_trade_time,
        "prev_close": prev_close,
    }


@st.cache_data(ttl=3600, show_spinner=False)
def get_history_bulk(tickers: tuple, period_label: str) -> dict:
    """Return {ticker: DataFrame} of history for many tickers."""
    return {t: get_history(t, period_label) for t in tickers}


# ---------------------------------------------------------------------------
# Fundamentals / ETF details
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def is_etf(ticker: str) -> bool:
    tk = _fetch_ticker(ticker)
    if tk is None:
        return False
    return (tk.info.get("quoteType") or "").upper() == "ETF"


@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_fundamentals(ticker: str) -> dict:
    """Return a curated dict of fundamentals/metadata for a stock."""
    tk = _fetch_ticker(ticker)
    info = tk.info if tk is not None else {}

    return {
        "ticker": ticker,
        "name": info.get("shortName") or info.get("longName") or ticker,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "website": info.get("website"),
        "summary": info.get("longBusinessSummary"),
        "price": info.get("currentPrice") or info.get("regularMarketPrice"),
        "market_cap": info.get("marketCap"),
        "trailing_pe": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "peg_ratio": info.get("trailingPegRatio"),
        "price_to_book": info.get("priceToBook"),
        "price_to_sales": info.get("priceToSalesTrailing12Months"),
        "beta": info.get("beta"),
        "dividend_yield": info.get("dividendYield"),
        "profit_margin": info.get("profitMargins"),
        "operating_margin": info.get("operatingMargins"),
        "gross_margin": info.get("grossMargins"),
        "return_on_equity": info.get("returnOnEquity"),
        "return_on_assets": info.get("returnOnAssets"),
        "debt_to_equity": info.get("debtToEquity"),
        "current_ratio": info.get("currentRatio"),
        "quick_ratio": info.get("quickRatio"),
        "revenue": info.get("totalRevenue"),
        "revenue_growth": info.get("revenueGrowth"),
        "earnings_growth": info.get("earningsGrowth"),
        "eps_ttm": info.get("trailingEps"),
        "eps_forward": info.get("forwardEps"),
        "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
        "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
        "fifty_day_avg": info.get("fiftyDayAverage"),
        "two_hundred_day_avg": info.get("twoHundredDayAverage"),
        "avg_volume": info.get("averageVolume"),
        "volume": info.get("volume") or info.get("regularMarketVolume"),
        "shares_outstanding": info.get("sharesOutstanding"),
        "float_shares": info.get("floatShares"),
        "held_by_insiders": info.get("heldPercentInsiders"),
        "held_by_institutions": info.get("heldPercentInstitutions"),
        "target_mean_price": info.get("targetMeanPrice"),
        "recommendation_mean": info.get("recommendationMean"),
        "number_of_analysts": info.get("numberOfAnalystOpinions"),
        "exchange": info.get("exchange"),
        "currency": info.get("currency", "USD"),
    }


@st.cache_data(ttl=3600, show_spinner=False)
def get_etf_details(ticker: str) -> dict:
    """Return ETF metadata: expense ratio, sector weights, top holdings, returns."""
    tk = _fetch_ticker(ticker)
    info = tk.info if tk is not None else {}

    sector_weights = {}
    top_holdings = []
    try:
        funds = tk.funds_data if tk is not None else None
        sw = funds.sector_weightings if funds is not None else None
        if sw:
            sector_weights = sw
        th = funds.top_holdings
        if th is not None and not th.empty:
            for sym, row in th.iterrows():
                top_holdings.append({
                    "symbol": sym,
                    "name": row.get("Name", sym),
                    "weight": row.get("Holding Percent", 0.0),
                })
    except Exception:
        pass

    return {
        "ticker": ticker,
        "name": info.get("shortName") or info.get("longName") or ticker,
        "category": info.get("category"),
        "expense_ratio": info.get("netExpenseRatio") or info.get("annualReportExpenseRatio"),
        "ytd_return": info.get("ytdReturn"),
        "three_year_avg_return": info.get("threeYearAverageReturn"),
        "five_year_avg_return": info.get("fiveYearAverageReturn"),
        "beta_3y": info.get("beta3Year"),
        "total_assets": info.get("totalAssets"),
        "sector_weights": sector_weights,
        "top_holdings": top_holdings,
        "price": info.get("navPrice") or info.get("regularMarketPrice"),
    }


# ---------------------------------------------------------------------------
# Options chain
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def get_option_expirations(ticker: str) -> tuple:
    """Return available option expiration date strings (YYYY-MM-DD)."""
    try:
        return tuple(yf.Ticker(ticker, session=_YF_SESSION).options)
    except Exception:
        return ()


@st.cache_data(ttl=3600, show_spinner=False)
def get_option_chain(ticker: str, expiration: str) -> dict:
    """Return {"calls": DataFrame, "puts": DataFrame} for the given expiration."""
    try:
        chain = yf.Ticker(ticker, session=_YF_SESSION).option_chain(expiration)
        return {"calls": chain.calls, "puts": chain.puts}
    except Exception:
        return {"calls": pd.DataFrame(), "puts": pd.DataFrame()}


# ---------------------------------------------------------------------------
# Valuation data
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def get_research_data(ticker: str) -> dict:
    """Fetch raw data for the Research Agent page: price/volume history (3y,
    for 200MA + RSI + up to ~8 quarters of earnings-day moves), the earnings
    calendar, and confirmed earnings dates.

    calendar/earnings_dates are fetched with retries so a transient Yahoo
    Finance hiccup doesn't get cached as "no data" for the full 5-minute TTL.
    If every retry still fails, the *_error field carries the last exception
    text so the page can tell callers the difference between "this ticker
    genuinely has no earnings data" (e.g. an ETF) and "the fetch failed".

    Returns a dict with keys:
        history         - 3y daily OHLCV DataFrame (empty if fetch failed)
        calendar        - tk.calendar dict, or None if unavailable/empty
        calendar_error  - last exception text if the calendar fetch raised, else None
        earnings_dates  - DataFrame from tk.get_earnings_dates(), or None if unavailable/empty
        earnings_error  - last exception text if the earnings fetch raised, else None
    """
    tk = _fetch_ticker(ticker)
    if tk is None:
        return {
            "history": pd.DataFrame(),
            "calendar": None, "calendar_error": "Could not authenticate with Yahoo Finance",
            "earnings_dates": None, "earnings_error": "Could not authenticate with Yahoo Finance",
        }

    history = _fetch_history(tk, period="3y", interval="1d")

    calendar, calendar_error = _retry_call(lambda: tk.calendar, label=f"{ticker} calendar")
    if not calendar:
        calendar = None

    earnings_dates, earnings_error = _retry_call(lambda: tk.get_earnings_dates(limit=12), label=f"{ticker} get_earnings_dates")
    if earnings_dates is None or earnings_dates.empty:
        earnings_dates = None

    return {
        "history": history,
        "calendar": calendar, "calendar_error": calendar_error,
        "earnings_dates": earnings_dates, "earnings_error": earnings_error,
    }


@st.cache_data(ttl=3600, show_spinner=False)
def get_valuation_data(ticker: str) -> dict:
    """Fetch raw data needed for the Valuation section.

    Returns a dict with keys:
        info              – yfinance .info dict
        quarterly_income  – quarterly income statement DataFrame
        quarterly_cashflow – quarterly cash flow DataFrame
        quarterly_balance – quarterly balance sheet DataFrame
        annual_income     – annual income statement DataFrame
        price_history     – 2-year daily OHLCV DataFrame
    """
    tk = _fetch_ticker(ticker)
    info = tk.info if tk is not None else {}

    def _safe(fn):
        try:
            result = fn()
            return result if result is not None else pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    if tk is None:
        empty = pd.DataFrame()
        return {
            "info": info,
            "quarterly_income": empty,
            "quarterly_cashflow": empty,
            "quarterly_balance": empty,
            "annual_income": empty,
            "price_history": empty,
        }

    return {
        "info": info,
        "quarterly_income": _safe(lambda: tk.quarterly_income_stmt),
        "quarterly_cashflow": _safe(lambda: tk.quarterly_cashflow),
        "quarterly_balance": _safe(lambda: tk.quarterly_balance_sheet),
        "annual_income": _safe(lambda: tk.income_stmt),
        "price_history": _safe(lambda: _fetch_history(tk, period="2y", interval="1d")),
    }
