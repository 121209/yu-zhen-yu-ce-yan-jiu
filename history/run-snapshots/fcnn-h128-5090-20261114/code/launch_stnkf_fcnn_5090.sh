#!/bin/bash
set -u
export PYTHONUNBUFFERED=1
PACK=/hy-tmp/st-nkf-fcnn-h128-pack
OUT=/hy-tmp/st-nkf-fcnn-h128-20261114
mkdir -p "$OUT"
cd "$PACK/ST-NKF-pytorch" || exit 1

{
  echo "host=$(hostname)"
  echo "date=$(date '+%Y-%m-%d %H:%M:%S %Z')"
  python3 - <<'PY'
import json, os, platform, torch
print(json.dumps({
  "hostname": os.uname().nodename,
  "python": platform.python_version(),
  "torch": torch.__version__,
  "cuda": torch.version.cuda,
  "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}, indent=2))
PY
  python3 -m pip freeze
} > "$OUT/environment.json" 2>&1

# snapshot of the code actually used
mkdir -p "$OUT/code"
cp -a "$PACK/run_stnkf_fcnn_h128_gpu.py" "$PACK/launch_stnkf_fcnn_5090.sh" "$OUT/code/" 2>/dev/null || true
cp -a models.py run_models.py config.py "$OUT/code/" 2>/dev/null || true

echo "LAUNCH_START $(date '+%Y-%m-%d %H:%M:%S %Z')" | tee -a "$OUT/console.log"
python3 -u "$PACK/run_stnkf_fcnn_h128_gpu.py" --out "$OUT" --gpu-id 0 --epochs 50 --patience 20
rc=$?
echo "exit=$rc" > "$OUT/process-exit.txt"
echo "finished=$(date '+%Y-%m-%d %H:%M:%S %Z')" >> "$OUT/process-exit.txt"
if [ "$rc" -eq 0 ]; then
  printf 'done=1\nfinished=%s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" > "$OUT/DONE"
else
  printf 'done=0\nfinished=%s\nexit=%s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$rc" > "$OUT/DONE"
fi
echo "LAUNCH_END rc=$rc $(date '+%Y-%m-%d %H:%M:%S %Z')" | tee -a "$OUT/console.log"
# Give the local puller a short window, then shut down from inside the instance.
echo "SHUTDOWN_IN 180s $(date '+%Y-%m-%d %H:%M:%S %Z')" | tee -a "$OUT/console.log"
sleep 180
echo "ISSUING_SHUTDOWN $(date '+%Y-%m-%d %H:%M:%S %Z')" | tee -a "$OUT/console.log"
/usr/local/bin/shutdown || shutdown || true
exit "$rc"
