#!/bin/bash
# Incremental pull of scheme5 results from the 华北 4090, then write PULLED
# so the instance can shut down. Never prints the SSH password.
set -u
export SSH_ASKPASS=<REDACTED_SSH_CTL_OR_IDENTITY>
export SSH_ASKPASS_REQUIRE=force
export DISPLAY=:0
export TZ=Asia/Shanghai

SSH=(ssh -p <REDACTED_PORT>
  -o StrictHostKeyChecking=yes
  -o UserKnownHostsFile=<REDACTED_SSH_CTL_OR_IDENTITY>
  -o HostKeyAlias=<REDACTED_SSH_CTL_OR_IDENTITY>
  -o PreferredAuthentications=password
  -o PubkeyAuthentication=no
  -o ConnectTimeout=25
  <REDACTED_USER_AT_HOST>)

REMOTE_OUT=/hy-tmp/st-nkf-scheme5-spatial-tf-20261117
LOCAL_OUT=/workspace/st-nkf-diagrams/st-nkf-scheme5-spatial-tf-20261117
LOG=/workspace/st-nkf-diagrams/st-nkf-scheme5-spatial-tf-20261117-sync.log
mkdir -p "$LOCAL_OUT"

remote() {
  setsid -w "${SSH[@]}" "$@"
}

pull_once() {
  mkdir -p "$LOCAL_OUT"
  # Pull whatever currently exists under the output dir.
  if ! remote "test -d $REMOTE_OUT"; then
    return 1
  fi
  # Use a tmp dir then swap so a dropped transfer cannot leave a half-written tree
  # as the only copy. Keep the previous successful tree until the new archive extracts.
  tmp=$(mktemp -d /tmp/scheme5-pull.XXXXXX)
  if remote "tar czf - -C /hy-tmp st-nkf-scheme5-spatial-tf-20261117" | tar xzf - -C "$tmp"; then
    if [ -d "$tmp/st-nkf-scheme5-spatial-tf-20261117" ]; then
      # merge/replace
      rm -rf "$LOCAL_OUT.next"
      mv "$tmp/st-nkf-scheme5-spatial-tf-20261117" "$LOCAL_OUT.next"
      rm -rf "$tmp"
      # keep previous as .prev only for one generation
      if [ -d "$LOCAL_OUT" ] && [ "$(ls -A "$LOCAL_OUT" 2>/dev/null)" ]; then
        rm -rf "$LOCAL_OUT.prev"
        mv "$LOCAL_OUT" "$LOCAL_OUT.prev"
      else
        rm -rf "$LOCAL_OUT"
      fi
      mv "$LOCAL_OUT.next" "$LOCAL_OUT"
      return 0
    fi
  fi
  rm -rf "$tmp"
  return 1
}

verify_complete() {
  local json="$LOCAL_OUT/six-datasets-gpu-ll.json"
  local done="$LOCAL_OUT/DONE"
  [ -f "$done" ] || return 1
  [ -f "$json" ] || return 1
  python3 - "$json" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
data = json.loads(p.read_text())
want = ["SCEDC_30", "SanJac", "WHITE", "SCEDC_25", "SaltonSea", "ComCat", "SCEDC_20"]
ds = data.get("datasets") or {}
prot = data.get("protocol") or {}
outcome = prot.get("outcome")
ok = all((ds.get(name) or {}).get("status") == "ok" and
         ((ds.get(name) or {}).get("schemes") or {}).get("scheme5", {}).get("status") == "ok"
         for name in want)
# Also require best weights for completed catalogs
weights_ok = True
root = p.parent
for name in want:
    w = root / name / "weights"
    # common names used by run_scheme
    candidates = list(w.glob("scheme5_best.pt")) + list(w.glob("*scheme5*best*.pt"))
    if not candidates:
        weights_ok = False
        break
sys.exit(0 if ok and weights_ok and outcome == "completed" else 1)
PY
}

echo "$(date '+%Y-%m-%d %H:%M:%S %Z') watcher start" | tee -a "$LOG"
fail=0
while true; do
  if pull_once; then
    fail=0
    echo "$(date '+%Y-%m-%d %H:%M:%S %Z') pull ok" | tee -a "$LOG"
    ls -l "$LOCAL_OUT/DONE" "$LOCAL_OUT/six-datasets-gpu-ll.json" 2>>"$LOG" || true
    if remote "test -f $REMOTE_OUT/DONE" && verify_complete; then
      stamp=$(date '+%Y-%m-%d %H:%M:%S %Z')
      echo "pulled=$stamp" | tee "$LOCAL_OUT/PULLED" | tee -a "$LOG"
      # write PULLED on the instance so launch script can shut down
      remote "printf 'pulled=%s\nhost=box\n' '$stamp' > $REMOTE_OUT/PULLED"
      echo "$(date '+%Y-%m-%d %H:%M:%S %Z') wrote remote PULLED, handshake complete" | tee -a "$LOG"
      exit 0
    fi
  else
    fail=$((fail + 1))
    echo "$(date '+%Y-%m-%d %H:%M:%S %Z') pull miss fail=$fail" | tee -a "$LOG"
  fi
  sleep 60
done
