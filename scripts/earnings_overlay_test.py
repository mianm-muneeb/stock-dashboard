"""Does the earnings-jump overlay make the 90% band honest across earnings?

Walk-forward, no lookahead: at each origin the jump is sampled only from reports that
happened before it. The fan is the vol-rescaled shape the page produces (zero drift,
spread = sigma*sqrt(t)), so Kronos itself is not needed to test the overlay.

Run: python scripts/earnings_overlay_test.py [TICKER ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from lib import kronos_utils as ku  # noqa: E402

TICKERS = sys.argv[1:] or ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL", "NFLX", "AMD", "CRM",
                           "ADBE", "ORCL", "COST", "WMT", "HD", "NKE", "DIS", "BA", "JPM", "GS", "PYPL",
                           "UBER", "SHOP", "VST", "CELH", "PLTR", "COIN", "SNOW", "INTC", "MU"]
N_PATHS = 400
rng = np.random.default_rng(11)
rows = []

for t in TICKERS:
    try:
        bars = ku.fetch_bars(t, "1d")
        ei = ku.earnings_info(t, bars)
    except Exception as e:
        print(t, "skip", e)
        continue
    hist = ei["hist"]
    if len(hist) < 8 or len(bars) < 400:
        print(t, "skip: reports", len(hist), "bars", len(bars))
        continue
    idx = bars.index
    close = bars["close"].astype(float)
    for H in (21, 63):
        # origins every 10 bars, need H bars ahead and 60 behind
        for o in range(80, len(bars) - H - 1, 10):
            d0 = idx[o]
            fut = pd.bdate_range(d0 + pd.Timedelta(days=1), periods=H)
            prior = hist[hist["date"] <= d0]
            if len(prior) < 6:
                continue
            win = hist[(hist["date"] > d0) & (hist["date"] <= fut[-1])]
            has_e = len(win) > 0
            sigma = ku.daily_sigma(close.iloc[: o + 1])
            last = float(close.iloc[o])
            steps = rng.normal(0, sigma, size=(N_PATHS, H))
            base = last * np.exp(np.cumsum(steps, axis=1))
            reports = [(r.date, r.timing) for r in win.itertuples()]
            jumped, info = ku.apply_earnings_jumps(base, fut, reports, prior["move"].tolist(), sigma, seed=o)
            actual = float(close.iloc[o + H])
            for name, sims in (("plain", base), ("overlay", jumped)):
                lo, hi = np.percentile(sims[:, -1], [5, 95])
                rows.append({"ticker": t, "H": H, "origin": str(d0.date()), "earnings_in_window": has_e,
                             "model": name, "inside": bool(lo <= actual <= hi),
                             "width": float(np.log(hi / lo))})
    print(t, "done", len(rows))

df = pd.DataFrame(rows)
summ = (df.groupby(["H", "earnings_in_window", "model"])
          .agg(windows=("inside", "size"), coverage=("inside", "mean"), width=("width", "mean"))
          .reset_index())
summ["coverage"] = (summ["coverage"] * 100).round(1)
summ["width"] = (summ["width"] * 100).round(1)
print(summ.to_string(index=False))
out = ROOT / "scripts" / "earnings_overlay_result.json"
out.write_text(json.dumps(summ.to_dict("records"), indent=1))
print("saved", out)
