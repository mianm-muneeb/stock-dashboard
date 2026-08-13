"""Research Agent batch screener: scans the Pre-Market Movers watchlist for
tickers with upcoming earnings and ranks them on a transparent, visible point
system (earnings-day move consistency, volume, moving-average alignment,
RSI, and earnings urgency). Descriptive only - no buy/sell scoring.
"""

from __future__ import annotations

import concurrent.futures
import csv
import datetime as dt
from pathlib import Path

import pandas as pd
import streamlit as st

from lib.market_data import get_research_data
from lib.premarket import PREMARKET_UNIVERSE
from lib.signals import compute_rsi

# Only screen tickers with earnings inside this window - the whole point of
# this screener is the upcoming earnings event, so anything further out (or
# unknown) is skipped rather than shown with a meaningless blank column.
EARNINGS_WINDOW_DAYS = 21
EARNINGS_URGENT_DAYS = 14

# Minimum number of past earnings-day moves required before we score
# consistency at all - avoids a 1/1 or 2/2 fluke looking like a strong signal.
MIN_MOVES_FOR_CONSISTENCY = 4

LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "research_agent_log.csv"
LOG_FIELDS = ["timestamp", "tickers_scanned", "top_5_tickers"]


def _log_scan(tickers_scanned: int, ranked: pd.DataFrame) -> None:
    """Append one row per actual scan run. Called from inside scan_watchlist()
    so it only fires on a real cache miss, not on every page rerun caused by
    e.g. selecting a row in the results table.
    """
    top5 = ranked.head(5)
    top5_str = "|".join(f"{r.Ticker}:{r.Score}" for r in top5.itertuples())
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        is_new = not LOG_PATH.exists()
        with open(LOG_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            if is_new:
                writer.writeheader()
            writer.writerow({
                "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
                "tickers_scanned": tickers_scanned,
                "top_5_tickers": top5_str,
            })
    except Exception:
        pass


def _scan_one(ticker: str) -> dict | None:
    data = get_research_data(ticker)
    history = data["history"]
    if history.empty or "Close" not in history.columns:
        return None

    close = history["Close"].dropna()
    if close.empty:
        return None
    volume = history["Volume"].dropna() if "Volume" in history.columns else pd.Series(dtype=float)
    current_price = float(close.iloc[-1])
    today = dt.date.today()

    # -- Next earnings date (same lookup as the detail view) ----------------
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

    if next_earnings_date is None:
        return None  # skip: no known upcoming earnings

    days_to_earnings = (next_earnings_date - today).days
    if days_to_earnings > EARNINGS_WINDOW_DAYS:
        return None  # skip: earnings too far out to be a near-term setup

    # -- Historical earnings-day move consistency ----------------------------
    moves = []
    if has_earnings_df:
        past_dates = sorted(
            (ts.date() for ts in earnings_dates_df.index if ts.date() < today),
            reverse=True,
        )[:8]
        date_pos = {d: i for i, d in enumerate(ts.date() for ts in close.index)}
        for d in past_dates:
            pos = date_pos.get(d)
            if pos is None or pos == 0:
                continue
            prev_close = float(close.iloc[pos - 1])
            day_close = float(close.iloc[pos])
            moves.append((day_close / prev_close - 1) * 100)

    n_moves = len(moves)
    consistency_pct = None
    consistency_label = "—"
    if n_moves >= MIN_MOVES_FOR_CONSISTENCY:
        up_count = sum(1 for m in moves if m >= 0)
        down_count = n_moves - up_count
        if up_count >= down_count:
            direction, n_matching = "up", up_count
        else:
            direction, n_matching = "down", down_count
        consistency_pct = n_matching / n_moves * 100
        matching_moves = [m for m in moves if (m >= 0) == (direction == "up")]
        avg_magnitude = sum(matching_moves) / len(matching_moves)
        consistency_label = f"{n_matching}/{n_moves} {direction}, avg {avg_magnitude:+.1f}%"

    # -- Volume vs 20-day average ---------------------------------------------
    volume_ratio = None
    if not volume.empty:
        avg20 = float(volume.tail(20).mean())
        if avg20:
            volume_ratio = float(volume.iloc[-1]) / avg20

    # -- Price vs 50MA / 200MA -------------------------------------------------
    ma50 = float(close.tail(50).mean()) if len(close) >= 50 else None
    ma200 = float(close.tail(200).mean()) if len(close) >= 200 else None
    ma_position = "—"
    ma_aligned = False
    if ma50 is not None and ma200 is not None:
        if current_price > ma50 and current_price > ma200:
            ma_position, ma_aligned = "Bullish", True
        elif current_price < ma50 and current_price < ma200:
            ma_position, ma_aligned = "Bearish", True
        else:
            ma_position = "Mixed"

    rsi = compute_rsi(close, period=14)

    # -- Transparent point score -------------------------------------------
    earnings_within_14d = days_to_earnings <= EARNINGS_URGENT_DAYS
    score = 0
    if consistency_pct is not None and consistency_pct >= 75.0:
        score += 2
    if volume_ratio is not None and volume_ratio > 1.3:
        score += 1
    if ma_aligned:
        score += 1
    if rsi is not None and 40 <= rsi <= 60:
        score += 1
    if earnings_within_14d:
        score += 1

    return {
        "Ticker": ticker,
        "Score": score,
        "Days to Earnings": days_to_earnings,
        "Earnings Within 14D": earnings_within_14d,
        "Historical Consistency": consistency_label,
        "Volume Ratio": volume_ratio,
        "MA Position": ma_position,
        "RSI": rsi,
    }


@st.cache_data(ttl=900, show_spinner=False)
def scan_watchlist(tickers: tuple[str, ...]) -> pd.DataFrame:
    """Scan `tickers` for earnings-driven setups, ranked by Score descending.

    Only includes tickers with a known earnings date within
    EARNINGS_WINDOW_DAYS - everything else is skipped since "Days to
    Earnings" is central to what this screener is evaluating.
    """
    # Each ticker makes several calls to Yahoo's calendar/earnings-dates
    # endpoints, which are far more sensitive to concurrent load than the
    # simple quote lookups other scanners in this app do - too many parallel
    # workers here trips transient rate-limiting that clean sequential calls
    # don't hit, so this stays deliberately more conservative than
    # lib/premarket.py's scan (15 workers) or lib/screener.py's (10).
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for row in executor.map(_scan_one, tickers):
            if row:
                rows.append(row)

    ranked = pd.DataFrame(rows)
    if not ranked.empty:
        ranked = ranked.sort_values(["Score", "Days to Earnings"], ascending=[False, True]).reset_index(drop=True)

    _log_scan(len(tickers), ranked)
    return ranked


__all__ = ["scan_watchlist", "PREMARKET_UNIVERSE", "EARNINGS_WINDOW_DAYS", "EARNINGS_URGENT_DAYS"]
