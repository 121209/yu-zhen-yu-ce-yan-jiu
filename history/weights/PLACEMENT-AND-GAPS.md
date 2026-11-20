# Placement, dedup, and missing originals (phase 2c)

## Safe git-relative placement

Extract zips at the **repository root**. Do not write into official `weights/baseline_stnkf_fcnn/`, `weights/schemes/`, or `weights/etas/` (the 49 official best snapshots on remote main must stay unchanged).

| Bundle path | Action |
|---|---|
| `history/weights/gpu-six-schemes1-4/{dataset}/checkpoints/*_{best,last}.pt` | Add. These are trainer checkpoint blobs (often including optimizer), **not** the official `save_weights` best files. |
| `history/weights/scheme5-20261117/{dataset}/checkpoints/scheme5_{best,last}.pt` | Add. Checkpoint-format; official scheme5 best remains under `weights/schemes/scheme5/`. |
| `history/weights/comcat-4090-partial-20261114/ComCat/checkpoints/` | Add. Scheme1 is partial (jsonl 10/50). Scheme2 last/best here are byte-identical to the 5090 run's scheme2 checkpoints (5090 reused the finished 4090 scheme2); packed once (see MAPPING.csv). |
| `history/weights/comcat-5090-20261114/ComCat/checkpoints/` | Add unique scheme1/3/4 last and checkpoint-best. |
| `history/weights/hyyun-midrun-20261113-1851/SaltonSea/checkpoints/scheme2_{best,last}.pt` | Add. Unique mid-run (jsonl epoch 19 at 2026-11-13 18:51:12). Other hyyun snapshot weights are hash-identical to later `gpu-six-schemes1-4` files and were **not** packed twice. |
| `history/weights/MAPPING.csv` | Source, status, epoch evidence, identical additional sources. |
| `history/weights/INVENTORY-152-DEDUP.csv` | All 152 rows from the phase-2 historical-weight inventory and what happened to each. |
| `Datasets/**/*_catalog.csv` | Add **next to** existing official `*_catalog_jitter.csv`. Do not replace jitter files. Byte-identical in local ST-NKF-pytorch and NeuralKernelFunction; packed once. |
| `history/preprocessing/upstream-NeuralKernelFunction/Data processing.ipynb` | Add. Upstream jitter notebook (seed 42). |
| `history/redacted/` | Sanitized derivatives only, plus `*.REDACTIONS.md` and INDEX.md. Never the originals. |

The 12 already-on-remote historical files (`history/weights-historical/early-four-schemes-20epoch/` and `.../etas-h128-cpu-pre-aligned/`) are **not** re-packed. The 48 inventory rows that are byte-identical to official 49 best weights are **not** packed.

## Missing original artifacts (not recreated)

- Official SanJac 4090 run: epoch jsonl and official best weights exist; **`scheme{1-4}_last.pt` trainer checkpoints were not found** under `st-nkf-gpu-4090/checkpoints/` on this box (only official `weights/scheme*_H128_seed1_best.pt`).
- FCNN official run: only official `*_best.pt` files were found; no `*_last.pt` trainer checkpoints on this box.
- Aligned ETAS official run: only official `*_aligned_best.pt`; no separate last checkpoints found.
- First six-dataset ComCat failure produced no usable ComCat best/last under `st-nkf-gpu-six/ComCat/` (status failed on convex-region error); no failed-run weight blobs to pack.
- Large FCNN `sync.log` / `watcher.stdout` remain on the box if listed as withheld in `history/redacted/INDEX.md`.
