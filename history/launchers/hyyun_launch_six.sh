#!/bin/bash
# Launch four-scheme training on all six remaining EarthquakeNPP datasets.
# Run this inside tmux so an SSH drop does not kill the job.
set -euo pipefail
export PYTHONUNBUFFERED=1
export TZ=Asia/Shanghai
OUT=/hy-tmp/st-nkf-gpu-six
PACK=/hy-tmp/pack
PY=/usr/local/bin/python3
LOG="$OUT/console.log"

mkdir -p "$OUT" "$PACK"
if [ ! -f "$PACK/run_six_datasets_gpu.py" ] || [ ! -d "$PACK/ST-NKF-pytorch/Datasets" ]; then
  tar xzf /hy-tmp/st-nkf-gpu-six-pack.tgz -C "$PACK"
fi

cd "$PACK"
{
  echo "===== launch $(date '+%Y-%m-%d %H:%M:%S %Z') ====="
  echo "host=$(hostname) pwd=$(pwd)"
  "$PY" -c "import torch,pandas; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_GPU', 'pandas', pandas.__version__)"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  echo "starting run_six_datasets_gpu.py  epochs=50 patience=0 budget=22.5h  H=128 official aux"
} | tee -a "$LOG"

# Resume-safe: completed schemes/datasets are skipped.
set +e
"$PY" -u "$PACK/run_six_datasets_gpu.py" \
  --out "$OUT" \
  --epochs 50 \
  --patience 0 \
  --budget-hours 22.5 \
  --gpu-id 0 \
  2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
echo "===== exit rc=$rc $(date '+%Y-%m-%d %H:%M:%S %Z') =====" | tee -a "$LOG"
if [ -f "$OUT/DONE" ]; then
  echo "DONE marker:" | tee -a "$LOG"
  cat "$OUT/DONE" | tee -a "$LOG"
fi
exit "$rc"
