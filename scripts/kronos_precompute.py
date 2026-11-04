#!/usr/bin/env python
"""Precompute Kronos results on a GitHub Actions runner.

  forecast   Ensemble (Kronos-small + Kronos-base) paths per ticker -> <out>/forecasts/<TICKER>.json
  calibrate  Past-origin backtest for one model                     -> <out>/calibration_<model>.json

The Streamlit page only reads these files, so the app itself never runs the heavy models.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_IDS = {"small": "NeoQuasar/Kronos-small", "base": "NeoQuasar/Kronos-base"}
_UTILS = None


def _utils():
    """Load lib/kronos_utils.py by path (lib/__init__ may import Streamlit)."""
    global _UTILS
    if _UTILS is None:
        import importlib.util

        spec = importlib.util.spec_from_file_location("kronos_utils", ROOT / "lib" / "kronos_utils.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _UTILS = mod
    return _UTILS


def load_predictor(name: str):
    import torch

    torch.set_num_threads(os.cpu_count() or 2)
    from kronos_model import Kronos, KronosPredictor, KronosTokenizer

    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained(MODEL_IDS[name])
    return KronosPredictor(model, tokenizer, device="cpu", max_context=512)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def safe_name(ticker: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", ticker)


def read_watchlist(path: str) -> list[str]:
    out: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0]
        out += [x.upper() for x in re.split(r"[,\s]+", line) if x]
    return list(dict.fromkeys(out))


def r2(a) -> list[float]:
    return [round(float(x), 2) for x in a]


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (pd.Timestamp, dt.datetime, dt.date)):
        return o.isoformat()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def fetch_bars_retry(ku, ticker: str, tries: int = 4) -> pd.DataFrame:
    for i in range(tries):
        try:
            df = ku.fetch_bars(ticker, "1d")
        except Exception as e:  # yfinance throttling etc.
            print(f"fetch {ticker} attempt {i + 1} failed: {e}", flush=True)
            df = pd.DataFrame()
        if df is not None and not df.empty:
            return df
        time.sleep(5 * (i + 1))
    return pd.DataFrame()


def drop_partial_bar(ticker: str, bars: pd.DataFrame) -> pd.DataFrame:
    """Drop today's bar if the exchange is still open (a half-finished candle skews the forecast)."""
    if ticker.upper().endswith(".AX"):
        tz, close = "Australia/Sydney", (16, 10)
    else:
        tz, close = "America/New_York", (16, 5)
    now = pd.Timestamp.now(tz=tz)
    if bars.index[-1].date() == now.date() and now.weekday() < 5 and (now.hour, now.minute) < close:
        return bars.iloc[:-1]
    return bars


# ----------------------------------------------------------------------------
# forecast
# ----------------------------------------------------------------------------

def variant_block(sims: np.ndarray, raw: np.ndarray, last_close: float) -> dict:
    p05, p25, p50, p75, p95 = np.percentile(sims, [5, 25, 50, 75, 95], axis=0)
    r05, r95 = np.percentile(raw, [5, 95], axis=0)
    pick = np.linspace(0, len(sims) - 1, min(25, len(sims))).astype(int)
    return {
        "p05": r2(p05), "p25": r2(p25), "p50": r2(p50), "p75": r2(p75), "p95": r2(p95),
        "raw_p05": r2(r05), "raw_p95": r2(r95),
        "paths": [r2(sims[i]) for i in pick],
        "prob_up": round(float((sims[:, -1] > last_close).mean()), 3),
        "median_end_pct": round(float((p50[-1] / last_close - 1) * 100), 2),
    }


def forecast_one(ku, preds: dict, ticker: str, a) -> dict:
    bars = fetch_bars_retry(ku, ticker)
    if bars.empty:
        raise RuntimeError(f"no price data for {ticker}")
    bars = drop_partial_bar(ticker, bars)
    if len(bars) < a.lookback + 10:
        raise RuntimeError(f"{ticker}: only {len(bars)} bars, need {a.lookback + 10}")

    hist = bars.iloc[-a.lookback:]
    future = ku.future_index(hist.index[-1], a.horizon, "1d")
    last_close = float(hist["close"].iloc[-1])
    sigma = ku.daily_sigma(hist["close"])

    edate = ku.next_earnings_date(ticker)
    in_window = bool(edate is not None and hist.index[-1] < edate <= future[-1])

    raw: dict[str, np.ndarray] = {}
    adj: dict[str, np.ndarray] = {}
    for name, pred in preds.items():
        t0 = time.time()
        raw[name] = ku.forecast_paths(pred, hist, future, a.paths, a.temp, a.top_p)
        adj[name] = ku.adjust(raw[name], last_close, sigma, anchor=True, rescale=True, drift_keep=a.drift_keep)
        print(f"{ticker} Kronos-{name}: {a.paths} paths in {time.time() - t0:.0f}s", flush=True)
    raw["ensemble"] = np.concatenate([raw["small"], raw["base"]], axis=0)
    adj["ensemble"] = np.concatenate([adj["small"], adj["base"]], axis=0)

    tail = hist.iloc[-120:]
    return {
        "ticker": ticker,
        "generated_at": now_iso(),
        "interval": "1d",
        "horizon": a.horizon,
        "lookback": a.lookback,
        "paths_per_model": a.paths,
        "temp": a.temp,
        "top_p": a.top_p,
        "drift_keep": a.drift_keep,
        "last_date": hist.index[-1].strftime("%Y-%m-%d"),
        "last_close": round(last_close, 2),
        "sigma_pct": round(sigma * 100, 3),
        "earnings_date": edate.strftime("%Y-%m-%d") if edate is not None else None,
        "earnings_in_window": in_window,
        "history": {"dates": [d.strftime("%Y-%m-%d") for d in tail.index], "close": r2(tail["close"])},
        "future_dates": [d.strftime("%Y-%m-%d") for d in future],
        "variants": {k: variant_block(adj[k], raw[k], last_close) for k in ("ensemble", "small", "base")},
        "consensus": ku.consensus(adj["small"], adj["base"], last_close, sigma),
    }


def cmd_forecast(a) -> int:
    ku = _utils()
    tickers = [t.strip().upper() for t in a.tickers.split(",") if t.strip()]
    if not tickers:
        print("no tickers given")
        return 2
    preds = {name: load_predictor(name) for name in ("small", "base")}
    out_dir = Path(a.out) / "forecasts"
    out_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for t in tickers:
        try:
            payload = forecast_one(ku, preds, t, a)
            path = out_dir / f"{safe_name(t)}.json"
            path.write_text(json.dumps(payload, default=_json_default, separators=(",", ":")), encoding="utf-8")
            e = payload["variants"]["ensemble"]
            print(f"{t}: wrote {path.name}; ensemble median {e['median_end_pct']:+.1f}%, "
                  f"up {e['prob_up'] * 100:.0f}%, small/base agree={payload['consensus']['agree']}", flush=True)
        except Exception as e:
            print(f"{t}: FAILED: {e}", flush=True)
            failed.append(t)
    return 1 if failed else 0


# ----------------------------------------------------------------------------
# calibrate
# ----------------------------------------------------------------------------

def cmd_calibrate(a) -> int:
    ku = _utils()
    tickers = [t.strip().upper() for t in a.tickers.split(",") if t.strip()]
    if a.watchlist:
        tickers = read_watchlist(a.watchlist)
    tickers = tickers[: a.max_tickers]
    if not tickers:
        print("no tickers given")
        return 2

    need = a.lookback + a.horizon + a.origins * a.spacing + 10
    bars_by = {}
    for t in tickers:
        b = fetch_bars_retry(ku, t)
        if b.empty:
            print(f"skip {t}: no data", flush=True)
            continue
        b = drop_partial_bar(t, b)
        if len(b) < need:
            print(f"skip {t}: {len(b)} bars, need {need}", flush=True)
            continue
        bars_by[t] = b
    if not bars_by:
        print("nothing to calibrate")
        return 1

    predictor = load_predictor(a.model)
    t0 = time.time()
    df = ku.run_calibration(
        predictor, bars_by, horizon=a.horizon, lookback=a.lookback, origins=a.origins, spacing=a.spacing,
        paths=a.paths, temp=a.temp, top_p=a.top_p, drift_keep=a.drift_keep,
        progress=lambda f, msg: print(f"[{f * 100:5.1f}%] {msg} ({time.time() - t0:.0f}s)", flush=True),
    )
    if df.empty:
        print("no forecasts scored")
        return 1

    summ = ku.summarise_calibration(df)
    df = df.assign(origin=df["origin"].astype(str))
    payload = {
        "generated_at": now_iso(),
        "model": a.model,
        "tickers": list(bars_by),
        "horizon": a.horizon,
        "lookback": a.lookback,
        "origins": a.origins,
        "spacing": a.spacing,
        "paths": a.paths,
        "drift_keep": a.drift_keep,
        "summary": summ.reset_index().to_dict("records"),
        "records": df.to_dict("records"),
    }
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"calibration_{a.model}.json"
    path.write_text(json.dumps(payload, default=_json_default, separators=(",", ":")), encoding="utf-8")
    print(summ.to_string(), flush=True)
    print(f"wrote {path}", flush=True)
    return 0


# ----------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("forecast")
    f.add_argument("--tickers", required=True, help="comma separated")
    f.add_argument("--out", default="out")
    f.add_argument("--horizon", type=int, default=20)
    f.add_argument("--lookback", type=int, default=400)
    f.add_argument("--paths", type=int, default=40, help="paths per model")
    f.add_argument("--temp", type=float, default=1.0)
    f.add_argument("--top-p", dest="top_p", type=float, default=0.9)
    f.add_argument("--drift-keep", dest="drift_keep", type=float, default=0.5)

    c = sub.add_parser("calibrate")
    c.add_argument("--tickers", default="", help="comma separated")
    c.add_argument("--watchlist", default="", help="file with one or more tickers per line")
    c.add_argument("--max-tickers", dest="max_tickers", type=int, default=6)
    c.add_argument("--model", choices=sorted(MODEL_IDS), default="small")
    c.add_argument("--out", default="out")
    c.add_argument("--horizon", type=int, default=20)
    c.add_argument("--lookback", type=int, default=400)
    c.add_argument("--origins", type=int, default=8)
    c.add_argument("--spacing", type=int, default=20, help="bars between origins (20 = non-overlapping outcomes)")
    c.add_argument("--paths", type=int, default=20)
    c.add_argument("--temp", type=float, default=1.0)
    c.add_argument("--top-p", dest="top_p", type=float, default=0.9)
    c.add_argument("--drift-keep", dest="drift_keep", type=float, default=0.5)

    a = ap.parse_args(argv)
    return cmd_forecast(a) if a.cmd == "forecast" else cmd_calibrate(a)


if __name__ == "__main__":
    sys.exit(main())
