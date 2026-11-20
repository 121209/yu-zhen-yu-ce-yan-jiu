#!/bin/bash
set -u
OUT=/workspace/st-nkf-diagrams/st-nkf-ll-decomposition-20261117
LOG="$OUT/resume-after-etas.log"
PY=/workspace/st-nkf-torch-cpu/bin/python
SCRIPT=/workspace/st-nkf-diagrams/eval_ll_decomposition.py
WAIT_PID=279328
mkdir -p "$OUT"
echo "$(date '+%F %T') Asia/Shanghai waiting for ETAS PID $WAIT_PID before resuming LL decomposition" >> "$LOG"
while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 30
done
echo "$(date '+%F %T') Asia/Shanghai ETAS PID $WAIT_PID exited; resuming FCNN + schemes 1-4 on $OUT" >> "$LOG"
"$PY" -u "$SCRIPT" \
  --out "$OUT" \
  --datasets SCEDC_30 SCEDC_25 SCEDC_20 SaltonSea SanJac WHITE ComCat \
  --models fcnn scheme3 scheme2 scheme1 scheme4 \
  --device cpu \
  --batch-size 16 \
  >> "$OUT/run.log" 2>&1
ec1=$?
echo "$(date '+%F %T') Asia/Shanghai FCNN+schemes1-4 resume exit=$ec1; starting scheme5" >> "$LOG"
"$PY" -u "$SCRIPT" \
  --out "$OUT" \
  --datasets SCEDC_30 SCEDC_25 SCEDC_20 SaltonSea SanJac WHITE ComCat \
  --models scheme5 \
  --device cpu \
  --batch-size 16 \
  >> "$OUT/run.log" 2>&1
ec2=$?
echo "$(date '+%F %T') Asia/Shanghai scheme5 follow-on exit=$ec2" >> "$LOG"
exit $ec2
