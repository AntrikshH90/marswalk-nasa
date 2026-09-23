#!/usr/bin/env bash
# MARSWALK one-shot launcher.
# Assumes:  pip install -r requirements.txt
set -e
cd "$(dirname "$0")"

if [ ! -f data/jezero_dem_20m.tif ]; then
  echo "[1/4] fetching the source CTX DEM (90 MB - this is the slow step)…"
  python3 scripts/fetch_data.py
fi

if [ ! -f layers/meta.json ] || [ ! -f layers/objective.npy ]; then
  echo "[2/4] preprocessing layers…"
  python3 scripts/preprocess.py
fi

if [ ! -f models/metrics.json ] || [ ! -f models/verification.json ]; then
  echo "[3/4] training ML models + building the verification record…"
  python3 scripts/train_ml.py
  python3 scripts/build_verification.py || true
fi

if [ ! -f data/fallback_photos/index.jsonl ]; then
  echo "[3b] caching offline fallbacks (imagery, weather, space weather)…"
  python3 scripts/bake_offline_fallbacks.py || true
fi

echo "[4/4] starting MARSWALK on :8000"
exec python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level info
