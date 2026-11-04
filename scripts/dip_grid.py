"""Dip-buy research on S&P 500 stocks.

Grid of filters and exits is ranked on 2014-2020 (train) and judged on 2021-2026 (test, unseen).
Entry: next day's open after the signal. Exit: next open after the exit signal. 0.1% round-trip cost.
Universe: today's S&P 500 members (survivorship bias makes results somewhat better than reality).
Set SYNTH=1 to run on fake prices (code check only).
"""
import io
import itertools
import os
import time

import numpy as np
import pandas as pd

COST = 0.001
MAXH = 10
STOP = 0.08
START = pd.Timestamp("2014-09-01")
SPLIT = pd.Timestamp("2021-01-01")
EXITS = [f"{a}_{b}" for a in ("sma5", "rsi70", "upclose") for b in ("nostop", "stop8")]


def load():
    if os.environ.get("SYNTH"):
        rng = np.random.default_rng(0)
        idx = pd.bdate_range("2013-11-04", "2026-11-03")
        cols = [f"S{i}" for i in range(60)] + ["SPY"]
        r = rng.normal(0.0004, 0.02, (len(idx), len(cols)))
        C = pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=idx, columns=cols)
        O = C.shift(1) * np.exp(rng.normal(0, 0.005, C.shape))
        return C, O.bfill()
    import requests
    import yfinance as yf
    html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                        headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
    tk = pd.read_html(io.StringIO(html))[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
    raw = yf.download(tk + ["SPY"], period="13y", interval="1d", auto_adjust=True, progress=False, threads=True)
    return raw["Close"], raw["Open"]


def rsi2(s):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=0.5, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=0.5, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


t0 = time.time()
C, O = load()
keep = C.columns[C.notna().sum() > len(C) * 0.8]
C, O = C[keep], O[keep]
dates = C.index
n = len(dates)
spy = C["SPY"]
spy_up = (spy > spy.rolling(200).mean()).values
stocks = [t for t in keep if t != "SPY"]
Cs, Os = C[stocks], O[stocks]
R = Cs.apply(rsi2)
S200 = Cs.rolling(200).mean()
S5 = Cs.rolling(5).mean()
VR = Cs.pct_change().rolling(60).std().rank(axis=1, pct=True)
print(f"data: {len(stocks)} stocks, {dates[0].date()} to {dates[-1].date()} ({time.time() - t0:.0f}s)", flush=True)

# ---------------------------------------------------------------- candidates
rows = []
for t in stocks:
    c, o, r = Cs[t].values, Os[t].values, R[t].values
    s200, s5, vr = S200[t].values, S5[t].values, VR[t].values
    sig = np.where((r < 10) & (c > s200))[0]
    sig = sig[sig + MAXH + 2 < n]
    for i in sig:
        e = o[i + 1]
        if not (np.isfinite(e) and e > 0):
            continue
        cl = c[i + 1:i + MAXH + 1]
        conds = {"sma5": cl > s5[i + 1:i + MAXH + 1], "rsi70": r[i + 1:i + MAXH + 1] > 70, "upclose": cl > c[i:i + MAXH]}
        stop = cl <= e * (1 - STOP)
        rec = [t, i, dates[i], r[i], vr[i], bool(spy_up[i])]
        for ex in EXITS:
            name, st = ex.split("_")
            cond = conds[name] | stop if st == "stop8" else conds[name]
            k = int(np.argmax(cond)) + 1 if cond.any() else MAXH
            rec += [o[i + k + 1] / e - 1 - COST, k]
        rows.append(rec)
cols = ["t", "i", "date", "rsi", "volrank", "spy_up"] + [f"{ex}_{f}" for ex in EXITS for f in ("ret", "k")]
df = pd.DataFrame(rows, columns=cols)
df = df[df.date >= START]
train = df.date < SPLIT
print(f"candidates: {len(df)} ({time.time() - t0:.0f}s)", flush=True)


def filt(rth, mkt, vol):
    m = df.rsi < rth
    if mkt == "spy_up":
        m &= df.spy_up
    if vol == "top50":
        m &= df.volrank >= 0.5
    elif vol == "top20":
        m &= df.volrank >= 0.8
    return m


def tstats(sub, ex):
    rr, kk = sub[f"{ex}_ret"].dropna(), sub.loc[sub[f"{ex}_ret"].notna(), f"{ex}_k"]
    if len(rr) == 0:
        return 0, np.nan, np.nan, np.nan
    return len(rr), (rr > 0).mean() * 100, rr.mean() * 100, rr.sum() / kk.sum() * 100


grid = []
for rth, mkt, vol, ex in itertools.product([10, 5, 2], ["any", "spy_up"], ["any", "top50", "top20"], EXITS):
    m = filt(rth, mkt, vol)
    a = tstats(df[m & train], ex)
    b = tstats(df[m & ~train], ex)
    grid.append(dict(rsi=rth, mkt=mkt, vol=vol, exit=ex, tr_n=a[0], tr_win=a[1], tr_avg=a[2], tr_day=a[3],
                     te_n=b[0], te_win=b[1], te_avg=b[2], te_day=b[3]))
G = pd.DataFrame(grid)
G.to_csv("dip_grid_results.csv", index=False)
ok = G[G.tr_n >= 500].sort_values("tr_day", ascending=False)
pd.set_option("display.width", 200)
print("\nTOP 10 BY TRAIN (2014-2020) return per day held  ->  same rules on TEST (2021-2026)")
print(ok.head(10).round(2).to_string(index=False))

# ---------------------------------------------------------------- portfolio simulation
cidx = {t: j for j, t in enumerate(stocks)}
Carr, Oarr = Cs.values, Os.values


def simulate(sub, ex, K=10, rank="rsi"):
    sub = sub[sub[f"{ex}_ret"].notna()].sort_values(rank, ascending=(rank == "rsi"))
    by_day = {i: g for i, g in sub.groupby("i")}
    cash, last, pos = 1.0, 1.0, []
    eq = np.full(n, np.nan)
    rets = []
    for d in range(n):
        still = []
        for p in pos:
            if p["x"] == d:
                cash += p["a"] * (1 + p["r"])
            else:
                still.append(p)
        pos = still
        g = by_day.get(d - 1)
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
                pos.append({"t": row.t, "j": j, "e": Oarr[d, j], "a": a, "x": d + int(getattr(row, f"{ex}_k")),
                            "r": getattr(row, f"{ex}_ret")})
                held.add(row.t)
                rets.append(getattr(row, f"{ex}_ret"))
        val = cash
        for p in pos:
            px = Carr[d, p["j"]]
            val += p["a"] * (px / p["e"] if np.isfinite(px) else 1.0)
        eq[d] = val
        last = val
    return pd.Series(eq, index=dates), np.array(rets)


def perf(e, lo, hi):
    e = e[(e.index >= lo) & (e.index < hi)].dropna()
    r = e.pct_change().dropna()
    yrs = len(r) / 252
    cagr = (e.iloc[-1] / e.iloc[0]) ** (1 / yrs) - 1
    dd = (e / e.cummax() - 1).min()
    sh = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
    return cagr * 100, dd * 100, r.mean() * 100, (r > 0).mean() * 100, sh


END = dates[-1] + pd.Timedelta(days=1)
print("\nPORTFOLIO (10 equal slots, deepest dip first)        TRAIN 2014-2020                       TEST 2021-2026")
print(f"{'rules':44s} {'CAGR':>6s} {'maxDD':>6s} {'avg day':>8s} {'up days':>8s}   {'CAGR':>6s} {'maxDD':>6s} {'avg day':>8s} {'up days':>8s} {'Sharpe':>6s} {'trades':>6s} {'win%':>5s}")
picks = [dict(rsi=10, mkt="any", vol="any", exit="sma5_nostop")] + ok.head(4)[["rsi", "mkt", "vol", "exit"]].to_dict("records")
for pk in picks:
    sub = df[filt(pk["rsi"], pk["mkt"], pk["vol"])]
    e, rr = simulate(sub, pk["exit"])
    a = perf(e, START, SPLIT)
    b = perf(e, SPLIT, END)
    label = f"rsi<{pk['rsi']} {pk['mkt']} vol:{pk['vol']} {pk['exit']}"
    print(f"{label:44s} {a[0]:6.1f} {a[1]:6.1f} {a[2]:8.3f} {a[3]:8.0f}   {b[0]:6.1f} {b[1]:6.1f} {b[2]:8.3f} {b[3]:8.0f} {b[4]:6.2f} {len(rr):6d} {(rr > 0).mean() * 100:5.0f}")

best = picks[1] if len(picks) > 1 else picks[0]
print("\nBEST RULE, number of slots (TEST 2021-2026):")
for K in (5, 10, 20):
    e, rr = simulate(df[filt(best["rsi"], best["mkt"], best["vol"])], best["exit"], K=K)
    b = perf(e, SPLIT, END)
    print(f"  K={K:2d}: CAGR {b[0]:6.1f}%  maxDD {b[1]:6.1f}%  avg day {b[2]:+.3f}%  Sharpe {b[4]:.2f}")
print(f"\ndone in {time.time() - t0:.0f}s")
