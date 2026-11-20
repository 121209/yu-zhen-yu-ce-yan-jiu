#!/usr/bin/env bash
# Unpack and start four-scheme GPU training on an AutoDL instance.
set -euo pipefail
OUT="${STNKF_GPU_OUT:-/root/autodl-tmp/st-nkf-gpu}"
REPO="${STNKF_REPO:-/root/autodl-tmp/ST-NKF-pytorch}"
BUDGET_HOURS="${BUDGET_HOURS:-1.4}"
mkdir -p "$OUT" "$(dirname "$REPO")"
cd "$(dirname "$REPO")"

# Install Python deps if the image is missing them. Prefer the image's torch.
python3 - << 'PY'
import importlib, sys, subprocess
need = []
for m in ("numpy", "pandas", "torch"):
    try:
        importlib.import_module(m)
    except ImportError:
        need.append(m)
if need:
    # Never replace a preinstalled CUDA torch with a CPU wheel.
    pkgs = [p for p in need if p != "torch"]
    if "torch" in need:
        sys.stderr.write("ERROR: torch is not installed in this image.\n")
        sys.exit(1)
    if pkgs:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit("CUDA not available")
print("gpu", torch.cuda.get_device_name(0))
PY

# The training script looks for ST-NKF-pytorch next to itself or at $REPO.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "$REPO/models.py" && -f "$SCRIPT_DIR/ST-NKF-pytorch/models.py" ]]; then
  REPO="$SCRIPT_DIR/ST-NKF-pytorch"
fi
export PYTHONUNBUFFERED=1
export STNKF_GPU_OUT="$OUT"
cd "$REPO"
# SCHEME=scheme2 (etc.) runs a single scheme on this container.
SCHEME_ARGS=()
if [[ -n "${SCHEME:-}" ]]; then
  SCHEME_ARGS=(--schemes "$SCHEME")
fi
exec python3 -u "$SCRIPT_DIR/run_four_schemes_gpu.py" \
  --out "$OUT" \
  --epochs "${EPOCHS:-50}" \
  --patience "${PATIENCE:-0}" \
  --budget-hours "$BUDGET_HOURS" \
  "${SCHEME_ARGS[@]}" \
  --shutdown-when-done \
  "$@"
