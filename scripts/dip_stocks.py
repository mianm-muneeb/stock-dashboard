"""RSI(2) dip-buy in an uptrend across the S&P 500 (today's members: survivorship-flattered). Per-trade stats."""
import io
import numpy as np
import pandas as pd
import requests
import yfinance as yf

html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
tk = pd.read_html(io.StringIO(html))[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
px = yf.download(tk, period="13y", interval="1d", auto_adjust=True, progress=False, threads=True)["Close"]
px = px.dropna(axis=1, thresh=int(len(px) * 0.8))
COST = 0.001  # 0.1% round trip
MAXH = 10
trades = []
for t in px.columns:
    s = px[t].dropna()
    if len(s) < 400:
        continue
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=0.5, adjust=False).mean(); dn = (-d.clip(upper=0)).ewm(alpha=0.5, adjust=False).mean()
    rsi = (100 - 100 / (1 + up / dn)).values
    sma200 = s.rolling(200).mean().values; sma5 = s.rolling(5).mean().values
    v = s.values; idx = s.index
    i = 200
    while i < len(v) - 1:
        if rsi[i] < 5 and v[i] > sma200[i]:
            e = i + 1  # buy at the next day's close (conservative vs buying the signal close)
            if e + 1 >= len(v):
                break
            j = e + 1
            while j < len(v) - 1 and j - e < MAXH and v[j] <= sma5[j]:
                j += 1
            trades.append((t, idx[e], j - e, v[j] / v[e] - 1 - COST))
            i = j + 1
        else:
            i += 1
tr = pd.DataFrame(trades, columns=["t", "date", "days", "ret"])
tr["year"] = tr.date.dt.year
print(f"trades {len(tr)}  tickers {tr.t.nunique()}  {tr.date.min().date()} to {tr.date.max().date()}")
print(f"win rate {(tr.ret > 0).mean() * 100:.0f}%  avg trade {tr.ret.mean() * 100:+.2f}%  median {tr.ret.median() * 100:+.2f}%  "
      f"avg hold {tr.days.mean():.1f} days  avg per day held {tr.ret.sum() / tr.days.sum() * 100:+.2f}%")
print(f"worst trade {tr.ret.min() * 100:.1f}%  worst 5% of trades avg {tr.ret.nsmallest(max(1, len(tr) // 20)).mean() * 100:.1f}%  best {tr.ret.max() * 100:.1f}%")
print("by year (n / win% / avg%):")
for y, g in tr.groupby("year"):
    print(f"  {y}: {len(g):4d} / {(g.ret > 0).mean() * 100:3.0f}% / {g.ret.mean() * 100:+.2f}%")
# 1 slot at a time, each day pick whichever fired: rough capital growth if you took 1 trade at a time, all-in
tr = tr.sort_values("date")
busy_until = pd.Timestamp("1900-01-01"); eq = 1.0; taken = 0
for _, r in tr.iterrows():
    if r.date > busy_until:
        eq *= 1 + r.ret; taken += 1
        busy_until = r.date + pd.Timedelta(days=r.days * 1.5)
yrs = (tr.date.max() - tr.date.min()).days / 365
print(f"one-trade-at-a-time all-in: {taken} trades, total x{eq:.1f}, about {(eq ** (1 / yrs) - 1) * 100:.0f}% a year (before tax, optimistic: survivorship)")
