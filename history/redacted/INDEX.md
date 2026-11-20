# Redacted / reviewed derivatives

Original sensitive files were **not** included. Each sanitized file has a sibling `*.REDACTIONS.md`.

- `comcat-fix/HANDOFF.md`: **sanitized** — Scientific handoff kept; SSH/host/port/control-socket connection details replaced with placeholders.
- `comcat-5090/EXPERIMENT.sanitized.json`: **sanitized** — Dropped keys ssh, instance, billing. Kept created, purpose, gpu, image, region, and code_files sha256/bytes. Three original copies on the box are byte-identical; one sanitized file represents all three.
- `fcnn-h128-5090/environment.sanitized.txt`: **sanitized** — Redacted host= and JSON hostname only. File is a host header plus pip-style package==version list; nvidia version strings that look like dotted quads were kept as package versions.
- `comcat-fix/supervisor-latest.json`: **sanitized** — Reviewed: no SSH/host/billing values. instance_online is a boolean. Scientific epoch/loss/mu/protocol fields kept unchanged. No placeholders inserted.
- `comcat-fix/remote_status.py`: **sanitized** — Reviewed: local status helper (hashlib/json/subprocess/Path); no SSH, host, or credential literals found. Copied unchanged as a reviewed derivative.
- `comcat-fix/supervisor-error.json`: **sanitized** — Redacted IP/host in the error string; kept timestamp and failure count.
- `launchers/watch_pull_scheme5_4090.sanitized.sh`: **sanitized** — Kept pull/completeness logic and comments; replaced SSH host, port, user, and control-socket/identity parameters.
- `comcat-fix/supervise_comcat.sanitized.py`: **sanitized** — Kept supervision logic; replaced SSH/host/control-socket parameters.
- `comcat-fix/sync_checkpoints.sanitized.py`: **sanitized** — Kept checkpoint sync logic; replaced SSH/host/control-socket parameters.
- `comcat-fix/supervisor.sanitized.log`: **sanitized** — Kept timestamps and event outcomes; replaced connection endpoints.
- `comcat-fix/sync.sanitized.log`: **sanitized** — Kept timestamps and sync outcomes; replaced connection endpoints.
- `scheme5/retry-pull.sanitized.log`: **sanitized** — Kept pull outcomes by dataset; replaced IPv4/host connection details.
- `fcnn-h128-5090/sync.sanitized.log`: **WITHHELD_cannot_sanitize_confidently** — Large paramiko/SSH transfer logs. Scientific delay is already documented in history/logs/etas-h128-aligned-final/box-stall-record.md from phase 2. Original remains on the box only.
- `fcnn-h128-5090/watcher.sanitized.stdout`: **WITHHELD_cannot_sanitize_confidently** — Large paramiko/SSH transfer logs. Scientific delay is already documented in history/logs/etas-h128-aligned-final/box-stall-record.md from phase 2. Original remains on the box only.
