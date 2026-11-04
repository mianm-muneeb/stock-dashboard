"""Dip-buy research v2: S&P 500 + 400 + 600, entry/exit timing, slot count, idle cash in SPY.

Rules are judged on 2021-2026 only after being picked on 2014-2020 (see dip_grid.py).
Timing A: buy next open, sell next open after the exit signal (fully realistic).
Timing B: buy at the signal day's close, sell at the exit day's close (needs orders placed just before
the bell using the near-close price; slightly optimistic).
Universe = today's index members (survivorship bias flatters results, more so for small caps).
"""
import io
import os
import time

import numpy as np
import pandas as pd

COST = 0.001
MAXH = 10
START = pd.Timestamp("2014-09-01")
SPLIT = pd.Timestamp("2021-01-01")
PAGES = {"sp500": "List_of_S%26P_500_companies", "sp400": "List_of_S%26P_400_companies",
         "sp600": "List_of_S%26P_600_companies"}


def members(page):
    import requests
    html = requests.get(f"https://en.wikipedia.org/wiki/{page}", headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
    for tb in pd.read_html(io.StringIO(html)):
        for col in tb.columns:
            if str(col).strip().lower() in ("symbol", "ticker symbol", "ticker"):
                return tb[col].astype(str).str.replace(".", "-", regex=False).str.strip().tolist()
    return []


def load():
    if os.environ.get("SYNTH"):
        rng = np.random.default_rng(1)
        idx = pd.bdate_range("2013-11-04", "2026-11-03")
        cols = [f"S{i}" for i in range(80)] + ["SPY"]
        r = rng.normal(0.0004, 0.022, (len(idx), len(cols)))
        C = pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=idx, columns=cols)
        O = (C.shift(1) * np.exp(rng.normal(0, 0.006, C.shape))).bfill()
        grp = {c: ("sp500" if i < 30 else "sp400" if i < 55 else "sp600") for i, c in enumerate(cols[:-1])}
        return C, O, grp
    import yfinance as yf
    grp = {}
    for g, page in PAGES.items():
        for t in members(page):
            grp.setdefault(t, g)
    print("members:", {g: sum(v == g for v in grp.values()) for g in PAGES}, flush=True)
    raw = yf.download(list(grp) + ["SPY"], period="13y", interval="1d", auto_adjust=True, progress=False, threads=True)
    return raw["Close"], raw["Open"], grp


def rsi2(s):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=0.5, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=0.5, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


t0 = time.time()
C, O, grp = load()
keep = C.columns[C.notna().sum() > len(C) * 0.8]
C, O = C[keep], O[keep]
dates = C.index
n = len(dates)
spy = C["SPY"]
spy_ret = spy.pct_change().fillna(0).values
stocks = [t for t in keep if t != "SPY"]
Cs, Os = C[stocks], O[stocks]
R = Cs.apply(rsi2)
S200 = Cs.rolling(200).mean()
S5 = Cs.rolling(5).mean()
VR = Cs.pct_change().rolling(60).std().rank(axis=1, pct=True)
print(f"data: {len(stocks)} stocks ({time.time() - t0:.0f}s)", flush=True)

COMBOS = [(x, tm) for x in ("upclose", "sma5") for tm in ("A", "B")]
rows = []
for t in stocks:
    c, o, r = Cs[t].values, Os[t].values, R[t].values
    s200, s5, vr = S200[t].values, S5[t].values, VR[t].values
    sig = np.where((r < 5) & (c > s200) & (vr >= 0.5))[0]
    sig = sig[sig + MAXH + 2 < n]
    for i in sig:
        if not (np.isfinite(o[i + 1]) and o[i + 1] > 0 and np.isfinite(c[i]) and c[i] > 0):
            continue
        cl = c[i + 1:i + MAXH + 1]
        conds = {"upclose": cl > c[i:i + MAXH], "sma5": cl > s5[i + 1:i + MAXH + 1]}
        rec = [t, grp.get(t, "?"), i, dates[i], r[i], vr[i]]
        for x, tm in COMBOS:
            cond = conds[x]
            k = int(np.argmax(cond)) + 1 if cond.any() else MAXH
            if tm == "A":
                rec += [o[i + k + 1] / o[i + 1] - 1 - COST, k, i + 1, i + k + 1]
            else:
                rec += [c[i + k] / c[i] - 1 - COST, k, i, i + k]
        rows.append(rec)
cols = ["t", "grp", "i", "date", "rsi", "volrank"]
for x, tm in COMBOS:
    cols += [f"{x}{tm}_{f}" for f in ("ret", "k", "ei", "xi")]
df = pd.DataFrame(rows, columns=cols)
df = df[df.date >= START]
train = df.date < SPLIT
print(f"candidates: {len(df)} ({time.time() - t0:.0f}s)\n", flush=True)

RULES = {
    "rsi<2, top20% vol": (df.rsi < 2) & (df.volrank >= 0.8),
    "rsi<2, top50% vol": (df.rsi < 2) & (df.volrank >= 0.5),
    "rsi<5, top20% vol": (df.rsi < 5) & (df.volrank >= 0.8),
}
UNIS = {"S&P 500": df.grp == "sp500", "500+400+600": df.grp.isin(["sp500", "sp400", "sp600"])}

print("TRADE LEVEL, TEST 2021-2026 (train in brackets)")
print(f"{'universe':12s} {'rule':20s} {'exit':8s} {'time':4s} {'trades':>7s} {'win%':>6s} {'avg trade%':>11s} {'per day held%':>14s}")
for un, um in UNIS.items():
    for rn, rm in RULES.items():
        for x, tm in COMBOS:
            out = []
            for part in (~train, train):
                s = df[um & rm & part]
                rr = s[f"{x}{tm}_ret"].dropna()
                kk = s.loc[rr.index, f"{x}{tm}_k"]
                out.append((len(rr), (rr > 0).mean() * 100, rr.mean() * 100, rr.sum() / kk.sum() * 100))
            te, tr = out
            print(f"{un:12s} {rn:20s} {x:8s} {tm:4s} {te[0]:7d} {te[1]:5.0f}% {te[2]:+10.2f} ({tr[2]:+.2f}) {te[3]:+8.3f} ({tr[3]:+.3f})")
print(flush=True)

cidx = {t: j for j, t in enumerate(stocks)}
Carr, Oarr = Cs.values, Os.values


def simulate(sub, key, K, overlay):
    sub = sub[sub[f"{key}_ret"].notna()].sort_values("rsi")
    tm = key[-1]
    by_e = {e: g for e, g in sub.groupby(f"{key}_ei")}
    cash, last, pos = 1.0, 1.0, []
    eq = np.empty(n)
    taken = []
    invested_days = 0
    for d in range(n):
        if overlay and d > 0:
            cash *= 1 + spy_ret[d]
        if tm == "A":
            keep_ = []
            for p in pos:
                if p["x"] == d:
                    cash += p["a"] * (1 + p["r"])
                else:
                    keep_.append(p)
            pos = keep_
        else:
            keep_ = []
            for p in pos:
                if p["x"] == d:
                    cash += p["a"] * (1 + p["r"])
                else:
                    keep_.append(p)
            pos = keep_
        g = by_e.get(d)
        if g is not None and len(pos) < K:
            held = {p["t"] for p in pos}
            for row in g.itertuples(index=False):
                if len(pos) >= K:
                    break
                if row.t in held:
                    continue
                j = cidx[row.t]
                a = last / K
                cash -= a
                ep = Oarr[d, j] if tm == "A" else Carr[d, j]
                pos.append({"t": row.t, "j": j, "e": ep, "a": a, "x": int(getattr(row, f"{key}_xi")),
                            "r": getattr(row, f"{key}_ret")})
                held.add(row.t)
                taken.append(getattr(row, f"{key}_ret"))
        val = cash
        for p in pos:
            px = Carr[d, p["j"]]
            val += p["a"] * (px / p["e"] if np.isfinite(px) and p["e"] > 0 else 1.0)
        invested_days += bool(pos)
        eq[d] = val
        last = val
    return pd.Series(eq, index=dates), np.array(taken)


def perf(e, lo, hi):
    e = e[(e.index >= lo) & (e.index < hi)]
    r = e.pct_change().dropna()
    yrs = len(r) / 252
    cagr = (e.iloc[-1] / e.iloc[0]) ** (1 / yrs) - 1
    dd = (e / e.cummax() - 1).min()
    sh = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
    return cagr * 100, dd * 100, r.mean() * 100, sh


END = dates[-1] + pd.Timedelta(days=1)
spy_e = pd.Series(np.cumprod(1 + spy_ret), index=dates)
b = perf(spy_e, SPLIT, END)
print(f"SPY buy and hold, TEST: CAGR {b[0]:.1f}%  maxDD {b[1]:.1f}%  avg day {b[2]:+.3f}%  Sharpe {b[3]:.2f}\n")
print("PORTFOLIO, TEST 2021-2026 (train CAGR in brackets)")
print(f"{'universe':12s} {'rule':20s} {'exit+time':10s} {'slots':>5s} {'idle cash':>9s} {'CAGR%':>14s} {'maxDD%':>7s} {'avg day%':>9s} {'Sharpe':>6s} {'trades':>6s} {'win%':>5s}")
for un in UNIS:
    for rn in ("rsi<2, top20% vol", "rsi<2, top50% vol"):
        for key in ("upcloseB", "sma5B", "upcloseA"):
            for K in (3, 5, 10):
                for ov in (False, True):
                    e, tk = simulate(df[UNIS[un] & RULES[rn]], key, K, ov)
                    te = perf(e, SPLIT, END)
                    tr = perf(e, START, SPLIT)
                    print(f"{un:12s} {rn:20s} {key:10s} {K:5d} {'SPY' if ov else 'cash':>9s} {te[0]:6.1f} ({tr[0]:5.1f}) {te[1]:7.1f} {te[2]:+9.3f} {te[3]:6.2f} {len(tk):6d} {(tk > 0).mean() * 100:5.0f}", flush=True)
print(f"\ndone in {time.time() - t0:.0f}s")
