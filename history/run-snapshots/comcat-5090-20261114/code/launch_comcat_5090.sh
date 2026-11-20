#!/bin/bash
set -euo pipefail
export TZ=Asia/Shanghai
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
cd /hy-tmp/pack
OUT=/hy-tmp/st-nkf-comcat-20261114
mkdir -p "$OUT/experiment_snapshot"
# save exact code/params used (no catalog copies)
python3 - << 'PY'
import json, hashlib, os, sys, platform
from pathlib import Path
from datetime import datetime
out = Path("/hy-tmp/st-nkf-comcat-20261114/experiment_snapshot")
out.mkdir(parents=True, exist_ok=True)
pack = Path("/hy-tmp/pack")
for rel in [
    "run_four_schemes_gpu.py", "run_six_datasets_gpu.py", "launch_comcat_5090.sh", "EXPERIMENT.json",
    "ST-NKF-pytorch/models.py", "ST-NKF-pytorch/layers_transformer.py", "ST-NKF-pytorch/layers.py",
    "ST-NKF-pytorch/run_models.py", "ST-NKF-pytorch/config.py", "ST-NKF-pytorch/utils.py", "ST-NKF-pytorch/batch.py",
]:
    src = pack / rel
    dst = out / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())
info = {
  "started": datetime.now().strftime("%Y-%m-%d %H:%M:%S Asia/Shanghai"),
  "hostname": platform.node(),
  "python": sys.version,
  "platform": platform.platform(),
}
try:
    import torch
    info.update({
      "torch": torch.__version__,
      "cuda": torch.version.cuda,
      "gpu": torch.cuda.get_device_name(0),
      "gpu_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
    })
except Exception as e:
    info["torch_error"] = repr(e)
(out / "environment.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
os.system("/usr/local/bin/python3 -m pip freeze > /hy-tmp/st-nkf-comcat-20261114/experiment_snapshot/pip-freeze.txt")
PY
echo "launch $(date '+%F %T %Z')" | tee -a "$OUT/console.log"
/usr/local/bin/python3 -u run_six_datasets_gpu.py \
  --datasets ComCat \
  --epochs 50 --patience 0 --budget-hours 0 --gpu-id 0 \
  --out "$OUT" >> "$OUT/console.log" 2>&1
ec=$?
printf 'exit=%s\nfinished=%s\n' "$ec" "$(date '+%F %T %Z')" | tee "$OUT/process-exit.txt"
# DONE marker even if non-zero, so backups can finish; do not shutdown here
if [ "$ec" -eq 0 ]; then
  printf 'done=1\nfinished=%s\n' "$(date '+%F %T %Z')" > "$OUT/DONE"
else
  printf 'done=0\nexit=%s\nfinished=%s\n' "$ec" "$(date '+%F %T %Z')" > "$OUT/DONE"
fi
exit "$ec"
