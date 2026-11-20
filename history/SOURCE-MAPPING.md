# Source and placement mapping (phase 2)

Extract both phase-2 ZIPs at the repository root (`unzip -o ...`). Paths inside the archives are already repo-relative.

## Priority 1 — training/eval code and runtime scripts (part1 zip)

| Bundle path | Meaning | Suggested repo action |
|---|---|---|
| `tests/test_nonconvex_boundary.py` | Nonconvex boundary tests added with the 2026-11-14 ComCat fix | Add if missing |
| `tests/test_transformer_smoke.py` | Transformer smoke tests | Add if missing |
| `tests/test_port.py` | Upstream port tests (ST-NKF-pytorch) | Add if missing |
| `tests/test_plot_notebooks.py` | Plot notebook tests | Add if missing |
| `history/scripts-exact-snapshots/` | Native box/GPU-run script copies, including hardcoded `/workspace` or historical `/root/autodl-tmp` and `/hy-tmp` path strings | Keep under `history/`; do **not** replace portable `scripts/` or portable `eval_ll_decomposition.py` |
| `history/launchers/` | Clean launch/bootstrap/watch scripts (no SSH) | Add under `history/launchers/` or `scripts/launchers/` |
| `history/comcat-nonconvex-fix/` | Patch, validation, unit/port/regression logs, `validate_comcat.py`, source backups, FINAL-REPORT.json | Add as history of the ComCat fix |
| `history/run-snapshots/` | Training-time copies of `models.py` / launchers / `pip-freeze.txt` from FCNN, scheme5, and ComCat 5090 runs | History only |
| `history/env-snapshots/` | `requirements.txt`, `environment.yml`, plotting requirements (same bytes as repo root copies) | Comparison only |

## Priority 2 — history docs and logs (part2 zip)

| Bundle path | Meaning | Suggested repo action |
|---|---|---|
| `history/EXPERIMENT-LEDGER.md` | Evidence-backed chronology | Add and link from README |
| `history/*sha256.csv` and `WITHHELD-PUBLIC-SAFETY.csv` | Hash inventories for remote comparison | Add |
| `history/logs/` | Full official and historical training logs, status, epoch jsonl, stall record, final aligned ETAS logs/json | Add; reconcile ETAS ll/logs if remote still has the 6-dataset snapshot |
| `history/docs/` | Week plan, CPU progress, 4090 plots | Add |
| `history/early-four-schemes-20epoch/` | 20-epoch / patience-8 SanJac run (not official 50/0) | Label historical |
| `history/weights-historical/` | Only small superseded weight sets (20-epoch four schemes; pre-aligned ETAS). Official 49 are **not** included | Label historical; do not use as official best |

## Do not extract-over

- `results/decomposition/` (phase 1 canonical 42/42)
- `results/decomposition_partial/` (historical 24-row)
- `weights/**` official 49 best snapshots
- `Datasets/` official jitter trees
- Portable `eval_ll_decomposition.py` and portable `scripts/*.py` already in the repo (except you may add new launchers in a subfolder)
- Stale `payloads/` must not be rematerialized onto these trees

## Official script path policy

If a file exists both in `scripts/` (archive/portable) and `history/scripts-exact-snapshots/` (native), keep `scripts/` as the runnable default. The snapshots are the exact training-time sources.
