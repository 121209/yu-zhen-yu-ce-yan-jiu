#!/bin/bash
# Train scheme5 on all seven catalogs, persist, wait for a confirmed local
# pull (PULLED), then shut down from inside the instance. Do not 释放.
set -u
export PYTHONUNBUFFERED=1
export TZ=Asia/Shanghai
PACK=/hy-tmp/st-nkf-scheme5-pack
OUT=/hy-tmp/st-nkf-scheme5-spatial-tf-20261117
PY=/usr/local/bin/python3
LOG="$OUT/console.log"

mkdir -p "$OUT"
cd "$PACK" || exit 1

{
  echo "===== launch $(date '+%Y-%m-%d %H:%M:%S %Z') ====="
  echo "host=$(hostname) pwd=$(pwd)"
  "$PY" -c "import torch,pandas; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_GPU', 'pandas', pandas.__version__)"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  echo "scheme5: Temporal FCNN, Spatial Transformer, Kappa FCNN"
  echo "protocol: H=128 official splits batch=16 lr=1e-4 seed=1 epochs=50 patience=0 budget=6.5h"
} | tee -a "$LOG"

mkdir -p "$OUT/code"
cp -a "$PACK/run_scheme5_spatial_tf_gpu.py" "$PACK/launch_scheme5_4090.sh" \
  "$PACK/run_four_schemes_gpu.py" "$PACK/run_six_datasets_gpu.py" "$OUT/code/" 2>/dev/null || true
if [ -d "$PACK/ST-NKF-pytorch" ]; then
  cp -a "$PACK/ST-NKF-pytorch/models.py" "$PACK/ST-NKF-pytorch/run_models.py" \
    "$PACK/ST-NKF-pytorch/config.py" "$PACK/ST-NKF-pytorch/layers.py" \
    "$PACK/ST-NKF-pytorch/layers_transformer.py" "$OUT/code/" 2>/dev/null || true
fi

set +e
"$PY" -u "$PACK/run_scheme5_spatial_tf_gpu.py" \
  --out "$OUT" \
  --epochs 50 \
  --patience 0 \
  --budget-hours 6.5 \
  --gpu-id 0 \
  --schemes scheme5 \
  2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
echo "===== python exit rc=$rc $(date '+%Y-%m-%d %H:%M:%S %Z') =====" | tee -a "$LOG"
echo "exit=$rc" > "$OUT/process-exit.txt"
echo "finished=$(date '+%Y-%m-%d %H:%M:%S %Z')" >> "$OUT/process-exit.txt"
if [ ! -f "$OUT/DONE" ]; then
  printf 'done=%s\nfinished=%s\nexit=%s\n' \
    "$([ "$rc" -eq 0 ] && echo 1 || echo 0)" \
    "$(date '+%Y-%m-%d %H:%M:%S %Z')" \
    "$rc" > "$OUT/DONE"
fi
echo "DONE marker:" | tee -a "$LOG"
cat "$OUT/DONE" | tee -a "$LOG"

# Best-effort persist to 个人数据; do not fail the handshake if oss is unavailable.
if command -v oss >/dev/null 2>&1; then
  echo "OSS_UPLOAD_START $(date '+%Y-%m-%d %H:%M:%S %Z')" | tee -a "$LOG"
  oss cp -r "$OUT" "oss://personal/st-nkf-scheme5-spatial-tf-20261117/" >>"$LOG" 2>&1 \
    || oss --help >>"$LOG" 2>&1 || true
  echo "OSS_UPLOAD_END $(date '+%Y-%m-%d %H:%M:%S %Z')" | tee -a "$LOG"
fi

echo "WAITING_FOR_PULLED $(date '+%Y-%m-%d %H:%M:%S %Z')" | tee -a "$LOG"
# Wait until the box writes PULLED after a verified complete pull.
# Do not shut down early. Cap the wait so a dead puller cannot bill forever;
# 90 minutes is long enough for a normal handshake even after a short stall.
wait_s=0
max_wait=5400
while [ ! -f "$OUT/PULLED" ]; do
  if [ "$wait_s" -ge "$max_wait" ]; then
    echo "PULLED_TIMEOUT after ${wait_s}s $(date '+%Y-%m-%d %H:%M:%S %Z')" | tee -a "$LOG"
    echo "timeout=1" >> "$OUT/process-exit.txt"
    break
  fi
  sleep 15
  wait_s=$((wait_s + 15))
  if [ $((wait_s % 300)) -eq 0 ]; then
    echo "still waiting for PULLED ${wait_s}s $(date '+%Y-%m-%d %H:%M:%S %Z')" | tee -a "$LOG"
  fi
done

if [ -f "$OUT/PULLED" ]; then
  echo "PULLED_SEEN $(date '+%Y-%m-%d %H:%M:%S %Z')" | tee -a "$LOG"
  cat "$OUT/PULLED" | tee -a "$LOG"
  echo "ISSUING_SHUTDOWN $(date '+%Y-%m-%d %H:%M:%S %Z')" | tee -a "$LOG"
  /usr/local/bin/shutdown || shutdown || true
else
  echo "NO_PULLED_SKIP_SHUTDOWN $(date '+%Y-%m-%d %H:%M:%S %Z')" | tee -a "$LOG"
  echo "leaving instance up so results can still be pulled; please shut down after pull" | tee -a "$LOG"
fi
exit "$rc"
