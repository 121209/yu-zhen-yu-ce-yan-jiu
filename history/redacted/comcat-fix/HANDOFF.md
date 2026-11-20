# ComCat rerun — 2026-11-14 Asia/Shanghai

GPU connected and idle at 09:39; prepaid expiry reported around 12:41 (not independently rechecked in billing console). No renewal, spending, shutdown, release, Git push, PR, clone, or changes to five completed datasets.

## Fix and tests
- Repository: /workspace/ST-NKF-pytorch/models.py; backup: backup/models.py.
- Nonconvex simple polygons: integrate every positive-ray signed crossing, exits +F(r), entries -F(r), orientation corrected. Strict side straddling handles vertex tangencies. Differentiability to CDF parameters retained. Existing convex implementation remains unchanged.
- /workspace/st-nkf-diagrams/run_four_schemes_gpu.py: one-line resume fix avoids undefined payload if last checkpoint already reached epoch cap. Backup under backup/.
- Added /workspace/ST-NKF-pytorch/tests/test_nonconvex_boundary.py.
- Exact diff: comcat-fix.patch. GPU originals backed up alongside each deployed file as .pre-comcat-20261114.
- Tests: existing Transformer/model smoke 24 PASS; port 11 PASS; new nonconvex 3 PASS.
- Independent 300k-draw radial Monte Carlo, 4096-angle check: maximum 128-angle error vs MC 0.00062660 on U polygon and 0.00096424 on actual ComCat. CW/CCW match exactly. Finite-difference gradient max error 1.68e-11. Convex original-vs-general path error <1e-12. Four real GPU model loss/backward/Adam smoke tests all finite. Full evidence: comcat-validation.{json,log}, comcat-{regression,port-tests,unit-tests}.log.

## Running task
- tmux: st-nkf-comcat (separate from old st-nkf).
- Started 09:41 Asia/Shanghai, order 2,1,4,3.
- Remote output /hy-tmp/st-nkf-comcat-20261114; box mirror /workspace/st-nkf-diagrams/st-nkf-comcat-20261114.
- Official aux/train/val/test = 14933/40701/14741/21885; m0=2.5; H128 batch16 lr1e-4 seed1 boundary_angles128 epochs50 patience0. Existing mu update, Adam exclusions, CDF differentiation, jitter catalog and best-validation evaluation unchanged.
- Launch /hy-tmp/launch_comcat.sh, copied here as launch_comcat.sh.
- Hard wall-clock target 12:35 plus existing Budget reserves (6min + 1.4 epoch), thus orderly budget pause expected earlier, before prepaid expiry. No cap reduction: incomplete runs are marked partial and resume to 50.
- First actual full epoch (including train, val and analytic mu) scheme2: 76.37384 s. End 09:42:19; train loss 7.595529, val loss 8.398263; mu 1.831749e-5. About 64min for scheme2 at this pace. Four schemes roughly 4–5h plus final evaluation, hence morning/prepaid completion not assured and likely impossible without further speedup; do not claim full results.
- First best and last checkpoints already pulled successfully at 09:43. ComCat/checkpoints/scheme2_{best,last}.pt and scheme2_epochs.jsonl. Later schemes same convention. Last includes optimizer, epoch, shuffle RNG and best state.
- Local watcher sync_checkpoints.py running, Shell735899 (PID112023 at launch), sync.log; copies only this rerun every90s and at final DONE, until12:39. It never deletes remote files or stops the machine. Watcher errors need followup.
- Query GPU: ssh <REDACTED_SSH_CTL_OR_IDENTITY> -p <REDACTED_PORT> <REDACTED_USER_AT_HOST> ... ; existing authenticated mux. Usual <REDACTED_SSH_CTL_OR_IDENTITY> known-hosts settings if reconnect needed. Never print authentication environment.
- On completion evaluate recorded best-validation weights; JSON test.LL_ts is mean, multiply by21885 for total. Partial evaluation must be labelled partial. Final requested all-four 50ep LL, best epochs, runtime are not yet available.
- Parent should inform user promptly of timing risk, monitor at least first scheme completion and before12:25, and verify final checkpoints are local before12:41. Do not renew or shut down without authorization.

## Follow-up supervision (09:46 Asia/Shanghai)
Existing 90s backup watcher checked healthy; latest local checkpoint/curve at epoch3. One separate READ-ONLY status supervisor is running (Shell735903, PID114344 at launch); it does not duplicate transfers or modify training. Logs: supervisor.log, supervisor-latest.json; per-scheme completion/evaluation events: scheme-events.jsonl. It queries every120s, records completion/faults, verifies actual epoch50 eligibility (status ok + epochs_ran50 + completed/already_finished), and calculates test total LL using21885. At12:20 it audits local checkpoint archive CRC/staleness plus remote trusted checkpoint deserialization (optimizer, shuffle RNG, next_epoch, best state, finite weights), recording predeadline-backup-verification.json. At final DONE/process exit it waits for existing watcher to finish exact SHA256 mirroring of ALL result files, then writes FINAL-REPORT.json and exits to produce background completion notification; deadline12:38. Final report includes tmux process state and instance reachability. If network failure persists to deadline, report preserves last observed state rather than asserting completion. The stale initial supervisor Shell735902 was stopped and observed; ONLY735903 remains running.
