"""Battery of documented price-only strategies on liquid ETFs (no survivorship bias), 2007-2026, vs buy and hold."""
import numpy as np
import pandas as pd
import yfinance as yf

ETFS = "SPY QQQ IWM EFA EEM TLT GLD XLK XLF XLE XLV XLY XLP XLI XLU XLB".split()
d = yf.download(ETFS, start="2007-01-01", interval="1d", auto_adjust=True, progress=False)
C, O = d["Close"].dropna(subset=["SPY"]), d["Open"].dropna(subset=["SPY"])
C = C.ffill(); O = O.ffill()
r = C.pct_change().fillna(0)
COST = 0.0002  # 2 bps per switch


def stats(name, ret, invested=None):
    ret = ret.dropna()
    eq = (1 + ret).cumprod()
    yrs = len(ret) / 252
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    vol = ret.std() * np.sqrt(252)
    dd = (eq / eq.cummax() - 1).min()
    sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else 0
    inv = f"{invested.mean() * 100:.0f}%" if invested is not None else "-"
    print(f"{name:42s} CAGR {cagr * 100:5.1f}%  vol {vol * 100:5.1f}%  maxDD {dd * 100:6.1f}%  Sharpe {sharpe:4.2f}  invested {inv}")
    return eq


def apply_costs(pos, ret):
    return ret * pos.shift(1).fillna(0) - COST * pos.diff().abs().fillna(0)


spy = C["SPY"]
print(f"{C.index[0].date()} to {C.index[-1].date()}\n")
stats("SPY buy and hold", r["SPY"])

# 1. 200-day trend timing
sma = spy.rolling(200).mean()
pos = (spy > sma).astype(float)
stats("SPY above 200d average, else cash", apply_costs(pos, r["SPY"]), pos)

# 2. dual momentum (monthly, 12m)
m = C[["SPY", "EFA", "EEM", "TLT", "GLD"]].resample("ME").last()
mom = m.pct_change(12)
pick = mom.idxmax(axis=1).where(mom.max(axis=1) > 0, None)
w = pd.DataFrame(0.0, index=C.index, columns=m.columns)
for dt, p in pick.items():
    if p is not None and isinstance(p, str):
        nxt = C.index[C.index > dt]
        if len(nxt):
            w.loc[nxt[0]:, :] = 0.0
            w.loc[nxt[0]:, p] = 1.0
port = (w.shift(1) * r[m.columns]).sum(axis=1) - COST * w.diff().abs().sum(axis=1)
stats("Dual momentum (best of 5, else cash)", port, (w.sum(axis=1) > 0))

# 3. sector rotation, top 3 by 6m momentum
sect = "XLK XLF XLE XLV XLY XLP XLI XLU XLB".split()
ms = C[sect].resample("ME").last()
mom6 = ms.pct_change(6)
w = pd.DataFrame(0.0, index=C.index, columns=sect)
for dt in mom6.index:
    top = mom6.loc[dt].dropna().nlargest(3).index
    nxt = C.index[C.index > dt]
    if len(top) and len(nxt):
        w.loc[nxt[0]:, :] = 0.0
        w.loc[nxt[0]:, top] = 1 / 3
port = (w.shift(1) * r[sect]).sum(axis=1) - COST * w.diff().abs().sum(axis=1)
stats("Sector rotation (top 3 of 9, 6m momentum)", port[w.sum(axis=1) > 0])
stats("  vs equal-weight all 9 sectors", r[sect].mean(axis=1))

# 4. RSI(2) pullback in uptrend (Connors)
delta = spy.diff()
up = delta.clip(lower=0).ewm(alpha=1 / 2, adjust=False).mean()
dn = (-delta.clip(upper=0)).ewm(alpha=1 / 2, adjust=False).mean()
rsi2 = 100 - 100 / (1 + up / dn)
sma5 = spy.rolling(5).mean()
state = 0; pos = []
for i in range(len(spy)):
    if state == 0 and rsi2.iloc[i] < 10 and spy.iloc[i] > sma.iloc[i] if not np.isnan(sma.iloc[i]) else False:
        state = 1
    elif state == 1 and spy.iloc[i] > sma5.iloc[i]:
        state = 0
    pos.append(state)
pos = pd.Series(pos, index=spy.index, dtype=float)
ret = apply_costs(pos, r["SPY"])
stats("RSI(2)<10 pullback in uptrend (SPY)", ret, pos)
trade_id = (pos.diff() == 1).cumsum()
tr = (1 + ret[pos.shift(1) == 1]).groupby(trade_id[pos.shift(1) == 1]).prod() - 1
if len(tr):
    print(f"   trades {len(tr)}, win rate {(tr > 0).mean() * 100:.0f}%, avg trade {tr.mean() * 100:+.2f}%, worst {tr.min() * 100:.1f}%")

# 5. overnight vs intraday
ov = (O["SPY"] / C["SPY"].shift(1) - 1).fillna(0)
intr = (C["SPY"] / O["SPY"] - 1).fillna(0)
stats("SPY overnight only (close to open)", ov)
stats("SPY intraday only (open to close)", intr)

# 6. turn of month
idx = C.index
dom_pos = pd.Series(0.0, index=idx)
mon = pd.Series(idx.to_period("M"), index=idx)
rank_from_start = mon.groupby(mon).cumcount() + 1
rank_from_end = mon.groupby(mon).cumcount(ascending=False) + 1
dom_pos[(rank_from_end <= 1) | (rank_from_start <= 3)] = 1.0
stats("SPY turn-of-month only (last 1 + first 3 days)", apply_costs(dom_pos, r["SPY"]), dom_pos)
