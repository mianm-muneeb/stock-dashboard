"""Dip-buy scanner logic (rule backtested in scripts/dip_grid2.py).

Rule: stock above its 200-day average, in the top 20% of 60-day volatility across the universe,
RSI(2) below 2. Buy at the close, sell at the close of the first day that closes higher than the
day before, or after 10 trading days. At most 5 positions; idle cash in SPY.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

INDEX_PAGES = {
    "S&P 500": "List_of_S%26P_500_companies",
    "S&P 400": "List_of_S%26P_400_companies",
    "S&P 600": "List_of_S%26P_600_companies",
}
MAX_HOLD = 10


def fetch_members() -> dict[str, str]:
    """Ticker -> index name, from Wikipedia's constituent tables."""
    import requests

    out: dict[str, str] = {}
    for name, page in INDEX_PAGES.items():
        html = requests.get(f"https://en.wikipedia.org/wiki/{page}", headers={"User-Agent": "Mozilla/5.0"},
                            timeout=30).text
        for tb in pd.read_html(io.StringIO(html)):
            col = next((c for c in tb.columns if str(c).strip().lower() in ("symbol", "ticker symbol", "ticker")), None)
            if col is None:
                continue
            for t in tb[col].astype(str).str.replace(".", "-", regex=False).str.strip():
                if t and t != "nan":
                    out.setdefault(t, name)
            break
    return out


def rsi2(df: pd.DataFrame) -> pd.DataFrame:
    d = df.diff()
    up = d.clip(lower=0).ewm(alpha=0.5, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=0.5, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def clean(closes: pd.DataFrame) -> pd.DataFrame:
    """Keep tickers that have a price on the latest bar; fill small interior gaps."""
    closes = closes.dropna(how="all")
    closes = closes.loc[:, closes.iloc[-1].notna()]
    return closes.ffill()


def scan(closes: pd.DataFrame, rsi_max: float = 2.0, vol_pct: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (signals, near misses) for the latest bar."""
    C = clean(closes)
    if len(C) < 210:
        raise ValueError("need at least 210 bars of history")
    vol = C.pct_change().rolling(60).std().iloc[-1]
    df = pd.DataFrame({
        "price": C.iloc[-1],
        "rsi2": rsi2(C).iloc[-1],
        "sma200": C.rolling(200).mean().iloc[-1],
        "vol_rank": vol.rank(pct=True),
        "vol_daily": vol,
        "chg1": C.iloc[-1] / C.iloc[-2] - 1,
        "chg3": C.iloc[-1] / C.iloc[-4] - 1,
    }).dropna(subset=["price", "rsi2", "sma200", "vol_rank"])
    df["above200"] = df["price"] > df["sma200"]
    base = df["above200"] & (df["vol_rank"] >= vol_pct)
    sig = df[base & (df["rsi2"] < rsi_max)].sort_values("rsi2")
    near = df[base & (df["rsi2"] >= rsi_max)].sort_values("rsi2").head(10)
    return sig, near


def sell_check(closes: pd.DataFrame, holdings: list[tuple[str, pd.Timestamp | None]]) -> pd.DataFrame:
    """For each (ticker, buy date): SELL if the latest close is above the prior close or the 10-day limit is hit."""
    rows = []
    for t, bought in holdings:
        if t not in closes.columns or closes[t].dropna().shape[0] < 2:
            rows.append({"ticker": t, "last close": np.nan, "day %": np.nan, "days held": None, "action": "no price data",
                         "next: sell if it closes above": np.nan})
            continue
        s = closes[t].dropna()
        up = bool(s.iloc[-1] > s.iloc[-2])
        held = int((s.index.normalize() > pd.Timestamp(bought).normalize()).sum()) if bought is not None else None
        if held == 0:
            action = "HOLD (bought on the latest bar)"
        elif up:
            action = "SELL at the close (closed up)"
        elif held is not None and held >= MAX_HOLD:
            action = "SELL at the close (10-day limit)"
        else:
            action = "HOLD"
        rows.append({"ticker": t, "last close": float(s.iloc[-1]), "day %": (s.iloc[-1] / s.iloc[-2] - 1) * 100,
                     "days held": held, "action": action,
                     "next: sell if it closes above": float(s.iloc[-1]) if action.startswith("HOLD") else np.nan})
    return pd.DataFrame(rows)
