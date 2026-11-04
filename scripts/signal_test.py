"""Does a simple trend/momentum filter beat 'always buy' on 3-month direction?  No fitting: fixed rules."""
import sys
import numpy as np
import pandas as pd
import yfinance as yf

TICKERS = ("AAPL MSFT NVDA AMZN GOOGL META TSLA AVGO ORCL NFLX AMD CRM ADBE INTC CSCO QCOM TXN "
           "JPM BAC GS V MA WMT COST HD PG KO PEP JNJ UNH LLY MRK PFE XOM CVX CAT DE BA GE VST CEG NEE").split()
H = 63

px = yf.download(TICKERS, period="11y", interval="1d", auto_adjust=True, progress=False)["Close"].dropna(how="all")
rows = []
for t in px.columns:
    s = px[t].dropna()
    if len(s) < 600:
        continue
    sma200 = s.rolling(200).mean(); sma50 = s.rolling(50).mean()
    mom121 = s.shift(21) / s.shift(252) - 1
    mom3 = s / s.shift(63) - 1
    hi52 = s.rolling(252).max()
    fwd = s.shift(-H) / s - 1
    d = pd.DataFrame({"close": s, "trend": s > sma200, "golden": sma50 > sma200, "mom121": mom121, "mom3": mom3,
                      "near_hi": s / hi52, "fwd": fwd}).dropna()
    d = d.iloc[::21]  # monthly origins
    d["ticker"] = t
    rows.append(d)
df = pd.concat(rows)
df["year"] = df.index.year
df["up"] = df.fwd > 0

rules = {
    "always buy (baseline)": pd.Series(True, index=df.index),
    "above 200d avg": df.trend,
    "above 200d + 12-1m momentum > 0": df.trend & (df.mom121 > 0),
    "above 200d + mom>0 + within 10% of 52w high": df.trend & (df.mom121 > 0) & (df.near_hi > 0.9),
    "above 200d + 3m momentum > 0": df.trend & (df.mom3 > 0),
}
print(f"windows: {len(df)}, tickers: {df.ticker.nunique()}, {df.index.min().date()} to {df.index.max().date()}")
for name, m in rules.items():
    sub = df[m.values]
    print(f"\n{name}: n={len(sub)} ({len(sub)/len(df):.0%} of windows)  hit={sub.up.mean():.1%}  avg 3m return={sub.fwd.mean()*100:+.1f}%  "
          f"worst decile={sub.fwd.quantile(.1)*100:+.1f}%")
    print("  by year:", {int(y): f"{g.up.mean():.0%}/{len(g)}" for y, g in sub.groupby("year")})
print("\nbaseline by year:", {int(y): f"{g.up.mean():.0%}" for y, g in df.groupby("year")})
df.to_csv("signal_test.csv")

# right now
print("\nNOW")
for t in ["VST", "AAPL", "TSLA", "NVDA", "NFLX", "ORCL"]:
    if t in px:
        s = px[t].dropna()
        print(t, f"close={s.iloc[-1]:.2f} above200={s.iloc[-1] > s.rolling(200).mean().iloc[-1]} "
                 f"mom12-1={(s.iloc[-22] / s.iloc[-253] - 1) * 100:+.0f}% near52wHigh={s.iloc[-1] / s.iloc[-252:].max():.0%}")
