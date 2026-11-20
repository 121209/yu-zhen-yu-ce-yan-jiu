#!/bin/bash
set -u
OUT=/workspace/st-nkf-diagrams/st-nkf-ll-decomposition-20261117
LOG="$OUT/scheme5-followon.log"
PY=/workspace/st-nkf-torch-cpu/bin/python
SCRIPT=/workspace/st-nkf-diagrams/eval_ll_decomposition.py
WAIT_PID=372254
mkdir -p "$OUT"
echo "$(date '+%F %T') Asia/Shanghai waiting for decomp PID $WAIT_PID before scheme5" >> "$LOG"
while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 30
done
echo "$(date '+%F %T') Asia/Shanghai PID $WAIT_PID exited; starting scheme5 resume on $OUT" >> "$LOG"
# Append to the same run.log; resume skips completed rows.
"$PY" -u "$SCRIPT" \
  --out "$OUT" \
  --datasets SCEDC_30 SCEDC_25 SCEDC_20 SaltonSea SanJac WHITE ComCat \
  --models scheme5 \
  --device cpu \
  --batch-size 16 \
  >> "$OUT/run.log" 2>&1
ec=$?
echo "$(date '+%F %T') Asia/Shanghai scheme5 follow-on exit=$ec" >> "$LOG"
exit $ec
