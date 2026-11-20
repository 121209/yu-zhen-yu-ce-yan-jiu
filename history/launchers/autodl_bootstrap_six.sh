#!/usr/bin/env bash
# Launch four-scheme training on the six remaining EarthquakeNPP datasets.
set -euo pipefail
OUT="${STNKF_GPU_OUT:-/root/autodl-tmp/st-nkf-gpu-six}"
REPO="${REPO:-/root/autodl-tmp/ST-NKF-pytorch}"
BUDGET_HOURS="${BUDGET_HOURS:-0}"
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1
export STNKF_GPU_OUT="$OUT"
# pandas may be missing on a fresh AutoDL image
/root/miniconda3/bin/python3 - <<'PY' || python3 - <<'PY'
import importlib,sys,subprocess
need=[]
for m in ("pandas","numpy","torch"):
    try:
        importlib.import_module(m)
    except ImportError:
        need.append("pandas" if m=="pandas" else m)
if need:
    subprocess.check_call([sys.executable,"-m","pip","install","-q",*need])
print("ok", need or "all present")
PY
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "$REPO/models.py" && -f "$SCRIPT_DIR/ST-NKF-pytorch/models.py" ]]; then
  REPO="$SCRIPT_DIR/ST-NKF-pytorch"
fi
cd "$REPO"
# Prefer miniconda python (AutoDL default PATH often has no python3)
PY=/root/miniconda3/bin/python3
command -v "$PY" >/dev/null || PY=python3
exec "$PY" -u "$SCRIPT_DIR/run_six_datasets_gpu.py" \
  --out "$OUT" \
  --epochs "${EPOCHS:-50}" \
  --patience "${PATIENCE:-0}" \
  --budget-hours "$BUDGET_HOURS" \
  "$@"
