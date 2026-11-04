"""Cross-sectional factor research on the S&P 500 (price-only factors), monthly rebalance.

For each factor: every month rank all stocks, hold the top quintile for the next month, compare to the equal-weight
universe. Reports excess return per month, t-stat, % of months beating the universe, by-year results, and the result
after trading costs. Survivorship bias: uses TODAY's index members, so results are somewhat flattering.
"""
import io
import sys
import numpy as np
import pandas as pd
import requests
import yfinance as yf

COST_ROUNDTRIP = 0.002  # 0.2% per full turnover (spread + slippage), applied to the share of the book replaced

html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                    headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
tickers = pd.read_html(io.StringIO(html))[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
print("universe:", len(tickers), flush=True)

px = yf.download(tickers, period="13y", interval="1d", auto_adjust=True, progress=False, threads=True)["Close"]
px = px.dropna(axis=1, thresh=int(len(px) * 0.8)).ffill()
print("prices:", px.shape, px.index[0].date(), px.index[-1].date(), flush=True)

m = px.resample("ME").last()
ret1 = m.pct_change()
fwd1 = ret1.shift(-1)
daily = px.pct_change()
vol60 = daily.rolling(60).std().resample("ME").last()
hi52 = px.rolling(252).max().resample("ME").last()

factors = {
    "momentum 12-1": m.shift(1) / m.shift(12) - 1,
    "momentum 6-1": m.shift(1) / m.shift(6) - 1,
    "short-term reversal (1m, losers)": -ret1,
    "low volatility (60d)": -vol60,
    "near 52-week high": m / hi52,
    "high volatility (60d)": vol60,
}

uni = fwd1.mean(axis=1)
rows = []
by_year = {}
for name, f in factors.items():
    top_ret, prev = [], None
    for d in f.index[:-1]:
        s = f.loc[d].dropna()
        if len(s) < 100 or pd.isna(uni.get(d)):
            top_ret.append(np.nan); continue
        top = set(s[s >= s.quantile(0.8)].index)
        r = fwd1.loc[d, list(top)].mean()
        turn = 1.0 if prev is None else len(top - prev) / max(len(top), 1)
        top_ret.append(r - COST_ROUNDTRIP * turn)
        prev = top
    ser = pd.Series(top_ret, index=f.index[:-1])
    ex = (ser - uni.reindex(ser.index)).dropna()
    ex = ex[ex.index >= ex.index[0]]
    t = ex.mean() / (ex.std() / np.sqrt(len(ex)))
    rows.append((name, len(ex), ex.mean() * 100, t, (ex > 0).mean() * 100, ex.mean() * 1200))
    by_year[name] = ex.groupby(ex.index.year).mean().mul(100).round(2).to_dict()

print("\nTop quintile minus equal-weight universe, NET of costs, monthly")
print(f"{'factor':36s} {'n':>4s} {'excess/mo %':>12s} {'t-stat':>7s} {'beats mkt %':>12s} {'per yr %':>9s}")
for r in rows:
    print(f"{r[0]:36s} {r[1]:4d} {r[2]:12.2f} {r[3]:7.2f} {r[4]:12.0f} {r[5]:9.1f}")
print("\nBy year (avg monthly excess %):")
for k, v in by_year.items():
    print(k, v)
print("\n(|t| above about 2 is the usual bar for 'probably not luck')")
