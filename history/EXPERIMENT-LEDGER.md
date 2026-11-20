# ST-NKF official experiment ledger (box evidence)

Compiled 2026-11-20, Asia/Shanghai (UTC+8), from files on this box only.  
No git clone of yu-zhen-yu-ce-yan-jiu, no repository write, no GPU, no user computer.  
Remote state used only as parent-reported facts: public main `8cf4ea08525a49e0bc0e81a4ff95dff4b285aac8` already has canonical 42/42 decomposition results and 49 official weights. This executor did not independently fetch that commit.

Times below are Asia/Shanghai when the source file or log says so. Many early files under `/workspace/st-nkf-diagrams` share mtime 2026-11-12 19:37–19:38, which looks like a bulk copy and is **not** treated as the experiment event time. Internal `Updated:` / log timestamps are preferred.

Labels: **canonical** = official H=128, seed=1, 50-epoch, patience=0, official jitter catalog and config.py splits, best-validation snapshot. **historical** = earlier protocol, failed, partial, mid-run, or superseded.

## 0. Official protocol (canonical)

Recorded consistently in official run status files (e.g. `st-nkf-gpu-4090/four-schemes-gpu-status.md`, `st-nkf-gpu-six/six-datasets-gpu-status.md`):

- H=128, batch=16, lr=1e-4, seed=1, boundary_angles=128
- Epoch cap 50, patience 0 (disabled)
- Official EarthquakeNPP / `config.py` auxiliary/train/val/test windows
- Official `*_catalog_jitter.csv` (upstream NumPy seed 42; see `Datasets/SOURCE.txt`); no extra train-time jitter
- μ initialized from train event rate / area, analytic update after each epoch
- Evaluation always restores the best-validation `save_weights` snapshot
- Seven datasets: SCEDC_30, SCEDC_25, SCEDC_20, SaltonSea, SanJac, WHITE, ComCat
- Six neural models: ST-NKF FCNN baseline + scheme1–5; plus protocol-aligned ETAS (empirical), **not** included in the 42-row decomposition

Background μ: a scalar per model, not shared across models; frozen at the checkpoint value at test time. **No separate background-rate evaluation or fixed-μ ablation was found on this box.**

## 1. Chronology (evidence-backed)

### Planning and CPU smoke (historical)

- `st-nkf-week-plan.md`: plan to reproduce ST-NKF-pytorch then replace FCNN kernels with Transformer. Local git tip of `/workspace/ST-NKF-pytorch` is `eb4ddba3660531445ffe0376125709fd2457c990` (`main`, merge of PR #2). Working tree has **uncommitted** experiment mods (`models.py`, `run_models.py`, `tests/test_port.py`) plus untracked `layers_transformer.py`, `tests/test_nonconvex_boundary.py`, `tests/test_transformer_smoke.py`. Those working-tree files are **byte-identical** to the official archive copies of the same names (see `SOURCE-HASHES.sha256.csv`). Git HEAD alone is **not** the official experiment source.
- `st-nkf-cpu-progress.md` (box, dated 2026-11-10 in the document): `tests.test_port` 10/10; SanJac empirical and FCNN 1-epoch CPU smokes completed; Transformer default batch-128 OOM; truncated H=128 1-epoch completed; full-H batch-8 run stopped mid-epoch by a 25-minute timeout. Logs: `/workspace/st-nkf-cpu-logs/` (copied into this bundle).

### Early four-scheme SanJac, 20 epochs / patience 8 (historical, not official)

- `four-schemes-status.md` internal timestamp **2026-11-11 23:38:29**. Protocol: SanJac official splits, jitter catalog, H=128, batch=16, **epochs=20, patience=8**, lr=1e-4, seed=1. Not the official 50/0 protocol. Weights under `four-schemes-weights/` (packed here, labeled historical). Driver: `run_four_schemes.py`.

### CDF-v2 / diagram work (historical)

- Logs and JSON `cdf-v2-*` exist under `/workspace/st-nkf-diagrams/`. Event times are not independently re-derived beyond file contents; packed as early logs. This is not the official 50-epoch contrast.

### Official SanJac schemes 1–4 on RTX 4090 (canonical for SanJac scheme1–4)

- `four-schemes-gpu-status.md` updated **2026-11-12 21:18:41**. Official protocol, device NVIDIA GeForce RTX 4090, order scheme2→1→4→3, all four `ok`, 50 epochs. Best epochs 41/42/42/35. Test LL_ts: scheme2 -4.6665, scheme1 -4.6601, scheme4 -4.6612, scheme3 -4.7581. Logs and epoch jsonl packed. Best weights match the official 49.

### Official schemes 1–4 on six datasets, same 4090 (canonical for five datasets; ComCat failed)

- `six-datasets-gpu-status.md` updated **2026-11-14 05:44:07**. Same official protocol. Dataset order SCEDC_30 → SCEDC_25 → WHITE → SaltonSea → **ComCat** → SCEDC_20.
- SCEDC_30/25, WHITE, SaltonSea, SCEDC_20: all four schemes `ok`.
- **ComCat all four schemes failed** with `ValueError('Boundary-mass quadrature assumes a convex study region (SanJac is a projected rectangle).')`. Those failures are **not** the canonical ComCat weights.
- A mid-run snapshot exists at `st-nkf-gpu-hyyun-six/` (`SNAPSHOT-2026-11-13-1851.md` and a 9.8 MiB tgz). Treated as historical mid-run; binaries inventoried, not re-packed.

### Nonconvex ComCat fix (2026-11-14 morning)

- `/workspace/ST-NKF-pytorch/models.py` mtime 2026-11-14 09:39:37; same bytes as the official archive `models.py`.
- `comcat-fix-20261114/`: `comcat-fix.patch`, validation JSON/log, unit/port/regression logs, `validate_comcat.py`, `tests/test_nonconvex_boundary.py`. Scientific summary from those files and the (withheld) handoff text, without connection details:
  - Nonconvex simple polygons: integrate every positive-ray signed crossing; existing convex path unchanged.
  - Tests cited in that material: existing Transformer/model smoke 24 PASS; port 11 PASS; new nonconvex 3 PASS; Monte Carlo / gradient checks recorded in `comcat-validation.{json,log}`.
- SSH/supervisor/sync files from that folder are **withheld** from this public bundle (see `WITHHELD-PUBLIC-SAFETY.csv`).

### ComCat 4090 rerun, budget-partial (historical)

- `st-nkf-comcat-20261114/six-datasets-gpu-status.md` updated **2026-11-14 10:41:39**. ComCat only; scheme2 `ok` (test LL_ts -7.2127, best epoch 30); scheme1/4/3 marked **partial**. Not the canonical ComCat set. Logs packed; partial checkpoints inventoried only.

### ComCat schemes 1–4 on 5090 (canonical ComCat scheme1–4)

- Run dir `st-nkf-comcat-5090-20261114/`. Best weights under `run/ComCat/weights/` are byte-identical to the official archive. Provenance `comcat-5090-four-schemes-gpu-ll.json` is already in the earlier archive. `FINAL-PULL.json` is packed (no ssh/account fields in the classification scan). `EXPERIMENT.json` (ssh/instance/billing) is withheld. `pip-freeze.txt` packed as the GPU Python environment snapshot.

### ST-NKF FCNN baseline H=128 seed=1 on 5090 (canonical FCNN × 7)

- Script `run_stnkf_fcnn_h128_gpu.py` (native snapshot packed). Launch: `launch_stnkf_fcnn_5090.sh`.
- `FINAL-PULL-20261116.txt` at **2026-11-16 10:43:28 CST** records an **incomplete mid-run pull** (SCEDC_20 incomplete at epoch 18; WHITE and ComCat not started). That note is historical.
- Training later finished: `run/DONE` and `process-exit.txt` say **2026-11-16 14:05:20 CST**, exit=0. Training log last lines: ComCat status=ok, best_epoch=47, test LL_ts=-7.435534, total=-162726.67, n=21885. All 7 official FCNN best weights match the archive. `stnkf-fcnn-h128-ll.json` matches `results/provenance/stnkf-fcnn-h128-ll.json` in the earlier archive.
- `box-stall-record.md` states the 5090 result pull was delayed by box VM pauses and recovered **2026-11-17 08:58**.

### Pre-aligned ETAS CPU run (historical; not official protocol)

- `st-nkf-etas-h128-cpu-20261114/etas-h128-cpu-ll.json` protocol: epoch_cap **1000**, patience **20**, lr **0.005**, `mu_init` left at 1 (paper path), `boundary_correction` false, `upgrades_applied` false. Started **2026-11-16 10:43:52**, finished **2026-11-16 11:09:31**, outcome completed. SanJac entry is `reused` from `etas-h128-full-result.json` (no SanJac weight file in that weights directory; six other `.pt` files present). **Do not substitute these weights for official aligned ETAS.**

### Protocol-aligned ETAS CPU (canonical ETAS × 7)

- Native dir `st-nkf-etas-h128-aligned-cpu-20261116/`. Script `run_etas_h128_aligned_cpu.py`. All 7 `etas_*_H128_seed1_aligned_best.pt` match the official archive (6) plus the 2026-11-20 ComCat overlay (SHA256 `db1df156c56f14913dba0951f5724acbd5aa5f689619d3f388ebe606e03f7f0d`, 64029 bytes).
- Training log / stdout in the overlay match the native final files (46469 bytes). The **older archive** `logs/etas/` copies are shorter (44785 bytes) — a stale pre-ComCat-finish snapshot.
- Native `etas-h128-aligned-ll.json` (12308 bytes) differs in bytes from the overlay JSON (13072) and from the old archive JSON (11384). Cloud agent must compare remote ll.json against these three hashes and keep the final completed 7-dataset record. Do not restore the 6-dataset archive copy.
- Previous ETAS overlay handoff: finished **2026-11-20 08:34:32**, best epoch 47, test LL_ts -7.460639476776123, n=21885 (see packed `etas-final-archive-HANDOFF-20261120.txt`).
- `box-stall-record.md` (written 2026-11-17 09:05:21, while some datasets were still running) documents VM pause/snapshot stalls, not a training-code hang.

### Scheme 5 (canonical scheme5 × 7)

- Dir `st-nkf-scheme5-spatial-tf-20261117/`. Launchers `launch_scheme5_4090.sh` and `launch_scheme5_scedc20_resume.sh`. All 7 official scheme5 best weights match the archive.
- `six-datasets-gpu-ll.json` lists all seven datasets and matches archive provenance `scheme5-six-datasets-gpu-ll.json`.
- `six-datasets-gpu-status.md` updated **2026-11-17 20:14:45** shows **only SCEDC_20 / scheme5** in the table (resume run). The status table is incomplete as a 7-dataset summary; use the ll.json as the complete result record. This is a documentation gap, not a missing-weight gap.
- `.prev` copy is identical except `PULLED` and `console.log` (those two files packed). Watcher script `watch_pull_scheme5_4090.sh` withheld (ssh).

### Log-likelihood decomposition (canonical 42/42; already on remote)

- Native `/workspace/st-nkf-diagrams/st-nkf-ll-decomposition-20261117/`. Finished **2026-11-20 15:38:17**, `summary.csv` rows=42 missing=0, 126 event files. Already pushed in phase 1; **not re-packed here**.
- ETAS is excluded from the 42 rows (eval `MODEL_ORDER`); expected.
- Native `eval_ll_decomposition.py` hardcodes `/workspace/ST-NKF-pytorch`. The official archive copy adds `--repo` / `ST_NKF_REPO` and optional payload unpack. Keep the portable repo script as the default; do not change numerical protocol. Native snapshot is under `history/scripts-exact-snapshots/`.

### Archive assembly and materialize hazard

- Local archive staging `/workspace/yu-zhen-yu-ce-yan-jiu-archive` was assembled 2026-11-19 (README still describes 24/42 decomp and missing ComCat ETAS — stale relative to current remote).
- `.github/workflows/materialize-binaries.yml` unpacks `payloads/` (a 2026-11-19 text-safe snapshot) and can **commit those older files over newer README, logs, results, and decomposition**. On a public repo this is still a data-loss risk if `payloads/**` is touched or the workflow is dispatched. Completeness scope for the cloud agent: disable or guard this workflow, and do not re-run `unpack_archive.py` / `restore_binaries.py` against the stale payloads in a way that overwrites `results/decomposition/` or final ETAS files.

## 2. Reproducible commands (only where a file records them)

These are the commands or entry points actually present in packed scripts/status; they are not a newly invented recipe.

Official four schemes / six datasets (from launchers and status protocol):

```text
python -u run_six_datasets_gpu.py --datasets <NAME> --epochs 50 --patience 0 --out <OUT>
# scheme order recorded: scheme2 → scheme1 → scheme4 → scheme3
# launch_comcat.sh uses the same with --datasets ComCat and a wall-clock --budget-hours
```

Official FCNN (from `launch_stnkf_fcnn_5090.sh` and the training log):

```text
# launcher on the 5090 machine used run_stnkf_fcnn_h128_gpu.py
# log records official H=128 seed=1, 50 epochs, ComCat finished 2026-11-16 14:05:20 CST
```

Official scheme5:

```text
# launch_scheme5_4090.sh / launch_scheme5_scedc20_resume.sh wrap run_scheme5_spatial_tf_gpu.py
# official protocol as in the scheme5 status header (H=128, 50 epochs, patience 0)
```

Official aligned ETAS:

```text
python run_etas_h128_aligned_cpu.py
# exact CLI flags: see history/scripts-exact-snapshots/run_etas_h128_aligned_cpu.py
```

Decomposition (portable form already in the repo; do not switch back to hardcoded roots):

```text
python eval_ll_decomposition.py --repo /path/to/yu-zhen-yu-ce-yan-jiu
```

CPU smoke (historical, from `st-nkf-cpu-progress.md`):

```text
python -m unittest tests.test_port
python run_models.py 1 -1 SanJac empirical empirical empirical --epochs 1 --no-patterns
python run_models.py 1 -1 SanJac neural neural neural --epochs 1 --no-patterns
python run_models.py 1 -1 SanJac neural neural neural --epochs 1 --no-patterns --backbone transformer
```

## 3. Source mapping (current vs snapshots)

See `SOURCE-HASHES.sha256.csv` for bytes and SHA256.

- Training core (`models.py`, `run_models.py`, `layers.py`, `layers_transformer.py`, `config.py`, `utils.py`, `batch.py`, `requirements.txt`, `environment.yml`): native working tree **=** official archive. Already expected on remote. Not re-packed as a second copy of the repo root (env snapshots only under `history/env-snapshots/`).
- Official runtime scripts already in `scripts/` on the archive differ from the **native diagrams copies** only by repo-root path portability (`_HERE.parent`, `ST_NKF_REPO`, comments that `/root/autodl-tmp` and `/hy-tmp` are historical paths, not credentials). Inspected diffs; no numerical-protocol change found. **Keep the portable repo scripts as the default.** Native exact snapshots are in `history/scripts-exact-snapshots/`.
- `eval_ll_decomposition.py`: three local versions exist — diagrams (hardcoded roots, 30567 bytes), archive portable (34224), and `/workspace/yu-zhen-yu-ce-yan-jiu/eval_ll_decomposition.py` (32328, a third intermediate). Hashes in `SOURCE-HASHES.sha256.csv` and the handoff. Cloud agent should keep the portable `--repo` version unless a later review finds a protocol bug.
- Tests `tests/test_*.py` were **not** in the official archive tree. Packed at repo-relative `tests/`.
- Launch/watch/bootstrap scripts were **not** in the official archive, except path strings in README. Clean launchers packed under `history/launchers/`. SSH-bearing watchers withheld.

## 4. Already on remote (do not overwrite; hashes for comparison)

- 42/42 decomposition under `results/decomposition/` (phase 1).
- 49 official best weights: all 49 native official best `.pt` files are byte-identical to the local archive (42 neural + 6 ETAS) plus the ComCat ETAS overlay. See `CANONICAL-49-WEIGHTS.sha256.csv`.
- Official jitter catalogs + shapes + `SOURCE.txt` (non-jitter catalogs were never in the official archive). See `DATASETS.sha256.csv`.
- Core training source listed above, and portable official scripts.
- Partial provenance JSON under `results/provenance/` (several ll.json files match native logs; included in archive). Full training logs and epoch jsonl were not in that archive and are packed here.
- `results/decomposition_partial/` is the historical 24-row snapshot; leave it labeled historical.

## 5. Explicit gaps (not inferred as complete)

- This executor did not clone or list current remote trees; cloud agent must compare these hashes to main `8cf4ea08…`.
- Upstream preprocessing notebook (`NeuralKernelFunction` / `Data processing.ipynb`) is not in this bundle. Jitter is documented as already applied upstream (NumPy seed 42) in `Datasets/SOURCE.txt`. The full upstream TF repo (~394 MiB) was not packaged.
- Non-jitter `*_catalog.csv` exist in `/workspace/ST-NKF-pytorch/Datasets/` (~32 MiB) and are inventoried, not packed (not official eval inputs).
- No separate background μ evaluation or fixed-μ ablation outputs were found.
- Scheme5 `six-datasets-gpu-status.md` table covers only the SCEDC_20 resume; use ll.json for all seven.
- Pre-aligned ETAS is a different protocol; SanJac weight in that folder is reused from an earlier SanJac ETAS result file, not stored as `etas_SanJac_H128_seed1.pt` there.
- Official last-epoch checkpoints (`*_last.pt`), optimizer states, and mid-run snapshots (~33.7 MiB unique historical `.pt` besides official-best duplicates) are **inventoried only** (`HISTORICAL-WEIGHTS.sha256.csv`), not packed, to avoid duplicating large binaries and the official 49. Optional later GitHub Release if desired.
- Duplicate dataset trees in `st-nkf-gpu-pack/` and `st-nkf-gpu-six-pack/` (~70 MiB) are not packed.
- Python venvs `st-nkf-torch-cpu` (1.1 GiB) and `st-nkf-diagram-env` (146 MiB) are environments, not experiment artifacts; not packed. GPU `environment.json` withheld (hostname/ssh). `pip-freeze.txt` from the ComCat 5090 snapshot is packed.
- AutoDL/hyyun instance identifiers, billing, and SSH connection details are not included. Files that contain them remain on the box only.
- Plot notebooks in ST-NKF-pytorch were not the official training protocol; `tests/test_plot_notebooks.py` is included with tests, notebooks themselves are already in ST-NKF-pytorch and were not copied again.
- Whether remote README still says “private / 24/42 / ComCat ETAS unfinished” is for the cloud agent to check and update.

## 6. Payload materialize hazard (completeness / safety)

`.github/workflows/materialize-binaries.yml` in the 2026-11-19 archive unpacks `payloads/` and can commit that older tree onto main (`contents: write`). That snapshot predates: final 42/42 decomposition, final ComCat ETAS overlay/logs, and any README updates. Cloud agent should disable or tightly guard this workflow on the now-public repo, and must not re-run the stale materialize path.
