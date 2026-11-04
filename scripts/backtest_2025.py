"""Backtest Kronos on 2025: forecast from monthly origins, score against what really happened.

Run:  python scripts/backtest_2025.py AAPL TSLA
Scores the ensemble (small + base, with the calibration fixes) at 1, 3 and 6 months ahead.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from lib import kronos_utils as ku  # noqa: E402

HORIZON, LOOKBACK, PATHS, SPACING = 126, 400, 20, 21
CHECK = [21, 63, 126]
MODELS = {"small": "NeoQuasar/Kronos-small", "base": "NeoQuasar/Kronos-base"}


def load(name):
    import torch
    from kronos_model import Kronos, KronosPredictor, KronosTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    return KronosPredictor(Kronos.from_pretrained(MODELS[name]), tok, device=dev, max_context=512)


def main():
    tickers = [t.upper() for t in sys.argv[1:]] or ["AAPL", "TSLA"]
    preds = {n: load(n) for n in MODELS}
    rows = []
    for t in tickers:
        bars = ku.fetch_bars(t, "1d")
        # origins: one per ~month across 2025, needing HORIZON bars of ground truth after
        pos = [i for i, d in enumerate(bars.index) if d.year == 2025]
        origins = list(range(pos[0], pos[-1] + 1, SPACING))
        for o in origins:
            end = o + 1
            if end - LOOKBACK < 0 or end + HORIZON > len(bars):
                continue
            hist = bars.iloc[end - LOOKBACK:end]
            actual = bars.iloc[end:end + HORIZON]
            last = float(hist["close"].iloc[-1])
            sigma = ku.daily_sigma(hist["close"])
            t0 = time.time()
            parts = [ku.adjust(ku.forecast_paths(p, hist, actual.index, PATHS, 1.0, 0.9), last, sigma,
                               drift_keep=0.5) for p in preds.values()]
            sims = np.concatenate(parts, axis=0)
            for h in CHECK:
                col = sims[:, h - 1]
                cons = ku.consensus(parts[0][:, :h], parts[1][:, :h], last, sigma)
                p05, p25, p50, p75, p95 = np.percentile(col, [5, 25, 50, 75, 95])
                act = float(actual["close"].iloc[h - 1])
                rows.append(dict(ticker=t, origin=str(hist.index[-1].date()), h=h, last=last,
                                 pred_med_pct=(p50 / last - 1) * 100, act_pct=(act / last - 1) * 100,
                                 agree=bool(cons["agree"]), prob_up=float((col > last).mean()),
                                 in90=bool(p05 <= act <= p95), in50=bool(p25 <= act <= p75),
                                 dir_hit=bool((p50 > last) == (act > last)),
                                 err=abs(p50 / last - act / last) * 100, naive_err=abs(act / last - 1) * 100,
                                 p05_pct=(p05 / last - 1) * 100, p95_pct=(p95 / last - 1) * 100))
            print(f"{t} {hist.index[-1].date()} done in {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    out = ROOT / ("backtest_2025_" + "_".join(tickers) + ".json")
    df.to_json(out, orient="records", indent=1)
    g = df.groupby(["ticker", "h"]).agg(n=("in90", "size"), in90=("in90", "mean"), in50=("in50", "mean"),
                                         dir_hit=("dir_hit", "mean"), err=("err", "mean"),
                                         naive_err=("naive_err", "mean"),
                                         pred=("pred_med_pct", "mean"), actual=("act_pct", "mean")).round(2)
    print(g.to_string())
    print(df.groupby("h").agg(n=("in90", "size"), in90=("in90", "mean"), in50=("in50", "mean"),
                              dir_hit=("dir_hit", "mean"), err=("err", "mean"),
                              naive_err=("naive_err", "mean")).round(2).to_string())
    df["conf_dir"] = np.where(df.prob_up >= 0.7, 1, np.where(df.prob_up <= 0.3, -1, 0))
    c = df[df.agree & (df.conf_dir != 0)].copy()
    c["hit"] = (np.sign(c.act_pct) == c.conf_dir)
    c["signed_ret"] = c.act_pct * c.conf_dir
    print("CONFIDENT CALLS (models agree and >=70% of paths one way)")
    if len(c):
        print(c.groupby("h").agg(n=("hit", "size"), hit=("hit", "mean"), avg_signed_ret=("signed_ret", "mean")).round(2).to_string())
        print("long-only:", c[c.conf_dir == 1].groupby("h").agg(n=("hit", "size"), hit=("hit", "mean"), avg_ret=("act_pct", "mean")).round(2).to_string())
    else:
        print("none")
    print("ALL forecasts hit rate by h:", (np.sign(df.act_pct) == np.where(df.pred_med_pct > 0, 1, -1)).groupby(df.h).mean().round(2).to_dict())
    print("RESULTS_WRITTEN", out)


if __name__ == "__main__":
    main()
