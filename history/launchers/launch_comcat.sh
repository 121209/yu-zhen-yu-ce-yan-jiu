#!/bin/bash
set -euo pipefail
cd /hy-tmp/pack
OUT=/hy-tmp/st-nkf-comcat-20261114
mkdir -p "$OUT"
BUDGET=$(/usr/local/bin/python3 -c 'from datetime import datetime; now=datetime.now(); end=now.replace(hour=12,minute=35,second=0,microsecond=0); print(max(.01,(end-now).total_seconds()/3600))')
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/usr/local/bin/python3 -u run_six_datasets_gpu.py --datasets ComCat --epochs 50 --patience 0 --budget-hours "$BUDGET" --out "$OUT" > "$OUT/console.log" 2>&1
printf 'exit=0\n' > "$OUT/process-exit.txt"
