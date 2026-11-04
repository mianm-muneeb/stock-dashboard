#!/usr/bin/env bash
# Publish ./incoming (forecasts/*.json, calibration_*.json) to data/kronos on the kronos-data branch.
# Keeps results out of main so nightly updates never redeploy or clutter the app's history.
# Run from the repo root inside GitHub Actions (the checkout token needs contents: write).
set -euo pipefail

git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"

if [ ! -d incoming ] || [ -z "$(ls -A incoming 2>/dev/null)" ]; then
  echo "Nothing to publish."
  exit 0
fi

for attempt in 1 2 3 4; do
  if git ls-remote --exit-code --heads origin kronos-data >/dev/null 2>&1; then
    git fetch origin kronos-data
    git checkout -f -B kronos-data FETCH_HEAD
  else
    git checkout --orphan kronos-data
    git rm -rf --quiet . || true
  fi

  mkdir -p data/kronos/forecasts
  if [ -d incoming/forecasts ]; then
    cp -r incoming/forecasts/. data/kronos/forecasts/
  fi
  cp incoming/calibration_*.json data/kronos/ 2>/dev/null || true

  python3 - <<'PY'
import datetime, glob, json, os

rows = []
for p in sorted(glob.glob("data/kronos/forecasts/*.json")):
    try:
        d = json.load(open(p, encoding="utf-8"))
        e = d["variants"]["ensemble"]
        rows.append({
            "ticker": d["ticker"],
            "file": os.path.basename(p),
            "generated_at": d["generated_at"],
            "last_date": d["last_date"],
            "last_close": d["last_close"],
            "horizon": d["horizon"],
            "ensemble_median_end_pct": e["median_end_pct"],
            "ensemble_prob_up": e["prob_up"],
            "agree": d["consensus"]["agree"],
            "earnings_date": d["earnings_date"],
            "earnings_in_window": d["earnings_in_window"],
        })
    except Exception as exc:
        print("skipping", p, exc)

cals = [os.path.basename(p) for p in sorted(glob.glob("data/kronos/calibration_*.json"))]
index = {
    "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    "forecasts": rows,
    "calibrations": cals,
}
json.dump(index, open("data/kronos/index.json", "w", encoding="utf-8"), indent=1)
print(f"index: {len(rows)} forecasts, {len(cals)} calibrations")
PY

  git add data/kronos
  if git diff --cached --quiet; then
    echo "No changes to publish."
    exit 0
  fi
  git commit -q -m "kronos: update saved results $(date -u +%Y-%m-%dT%H:%MZ)"
  if git push origin kronos-data; then
    echo "Published."
    exit 0
  fi
  echo "Push rejected (attempt $attempt), refetching..."
  sleep $((attempt * 5))
done

echo "Could not publish after retries" >&2
exit 1
