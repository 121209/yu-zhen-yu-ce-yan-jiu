# ST-NKF CPU progress (box, 2026-11-10)

Workdir: `/workspace/ST-NKF-pytorch`  
Env: `/workspace/ST-NKF-pytorch/.venv` (Python 3.13, torch 2.15.0+cu132, **CPU only**, numpy 2.5.4, pandas 3.1.1)  
Times below are Asia/Shanghai (UTC+8). No AutoDL use, no git push, no PR, PR #2 was not touched.

## Git tip

Clone of https://github.com/121209/ST-NKF-pytorch on `main`.

- Plan’s known tip (2026-11-03): `c8fe20ab0fa7d58bdb2486f9d77c66286121ba24` (PR #1).
- **Rechecked tip of origin/main:** `eb4ddba3660531445ffe0376125709fd2457c990` — *Merge pull request #2 from 121209/codex/add-plot-notebooks*, author date 2026-11-10 09:26:04 +0800.
- Local branch: `main`, tracking `origin/main`. Uncommitted local work only; nothing pushed.

## 1) Original-model CPU smoke (before any Transformer edits)

`python -m unittest tests.test_port` — **10/10 OK** in 1.486 s.

### SanJac empirical 1-epoch (`gpu_id=-1`, `--epochs 1 --no-patterns`)

Log: `/workspace/st-nkf-cpu-logs/sanjac-empirical-1epoch.log`  
~8 s, 21:05:08–21:05:13.

- Splits: `auxiliary=1672, training=11770, validation=3449, testing=4400`, `area_km2=10310.1`, device=cpu
- Epoch 1/1: `loss=1599.522345` `val_loss=2181.748215` `mu=5.336053e-04` (all finite; epoch-1 loss large as expected)
- Test results: `LL_ts=-5.533785343170166`, `LL_t=0.9201478958129883`, `LL_s=-6.453933238983154`
- Weights: `/workspace/ST-NKF-pytorch/weights/SanJac_empirical_empirical_empirical.pt`

### SanJac neural FCNN 1-epoch (default backbone)

Log: `/workspace/st-nkf-cpu-logs/sanjac-fcnn-neural-1epoch.log`  
~78 s, 21:05:18–21:06:33.

- Same splits and area as above
- Epoch 1/1: `loss=1599.724039` `val_loss=2181.526327` `mu=5.504783e-04` (finite)
- Test results: `LL_ts=-7.042505264282227`, `LL_t=0.9834831357002258`, `LL_s=-8.025988578796387`
- Weights: `/workspace/ST-NKF-pytorch/weights/SanJac_neural_neural_neural.pt`

`ll_seismic_model.log` lines match those test LLs (seed=1). Default path is unchanged FCNN.

## 2) Files changed (local only)

| Path | Change |
|---|---|
| `/workspace/ST-NKF-pytorch/layers_transformer.py` | **new** — `TemporalKernelTransformer`, `SpatialKernelTransformer` |
| `/workspace/ST-NKF-pytorch/models.py` | `backbone="fcnn"|"transformer"` (default `fcnn`); kappa always `KappaNetwork` |
| `/workspace/ST-NKF-pytorch/run_models.py` | `--backbone` flag; weight/csv/log names tagged only when not `fcnn` |
| `/workspace/ST-NKF-pytorch/tests/test_transformer_smoke.py` | **new** — shape/backward/Kappa-is-FCNN/default-flag tests |

Outer `KernelPointProcess` intensity, mu update, empirical path, and default FCNN path were not redesigned.

## Transformer design choices

- I/O stays `(B, H, 1) → (B, H, 1)`.
- `Linear(1 → d_model)` with `d_model=size_nn` (32 in `run_models.py`), 1-layer `TransformerEncoder` (pre-norm, GELU, dropout 0, `nhead` the largest of 8/4/2/1 that divides `d_model`), then `Linear(d_model → 1)`.
- Self-attention is along the history axis `H` so events can attend to each other (this is the point vs pointwise FCNN).
- Output-side CDF constraint: `F(x) = tanh(softplus(raw) * x)`, so `F(0)=0` and for `x>=0` the value is in `[0, 1)`. No hard box clamp. Attention / projection weights are **not** clamped ≥ 0; `project()` is a no-op. Kappa’s FCNN `project()` (clamp weights ≥ 0, bias + softplus) is unchanged.
- To keep `autograd.grad` PDFs **nonnegative and finite**, the encoder sees `x.detach()`. Then `dF/dx = sech²(s x) * s ≥ 0`. Without this, negative temporal PDFs made `l_ts < 0` and `log(l_ts)` NaN on some SanJac windows. Parameters still get gradients through `s` in both the CDF integral and the PDF term.
- CPU SDPA flash attention has no second backward (`create_graph=True` is required for CDF→PDF). Forward wraps the encoder in `sdpa_kernel(SDPBackend.MATH)`.
- Kappa stays `_NonnegativeMLP` / `KappaNetwork` for both backbones.

## 3) Transformer tests and smokes

`python -m unittest tests.test_transformer_smoke tests.test_port` — **19/19 OK** in 1.193 s (8 transformer tests including Kappa-is-still-FCNN, default backbone `fcnn`, finite backward, nonnegative PDF; all 10 original port tests still pass).

### Official full-H command (batch 128) — OOM, no LL

Command: `python run_models.py 1 -1 SanJac neural neural neural --epochs 1 --no-patterns --backbone transformer`  
Log: `/workspace/st-nkf-cpu-logs/sanjac-transformer-1epoch-default-bs128.log`

Splits printed correctly, then the process was OOM-killed at 21:09:41 (Asia/Shanghai). dmesg: `python` pid 135331, anon-rss ≈ 12.5 GiB, VM ≈ 27.5 GiB, box limit 15.6 GiB. Earlier probe: B=8 H=1672 was safe; B=32 already killed. No transformer weights were written.

### Truncated-history 1-epoch (completed, printed LL)

Log: `/workspace/st-nkf-cpu-logs/sanjac-transformer-truncated-H128.log`  
SanJac catalog, `H=128`, 256/128/128 train/val/test targets, batch 8, backbone=transformer, kappa=FCNN. 21:11:12, **2.2 s**.

- Epoch 1/1: `loss=2284.351622` `val_loss=972.517395` `mu=4.378206e-04` (finite)
- Test results: `LL_ts=-8.394207000732422`, `LL_t=0.8393619656562805`, `LL_s=-9.233570098876953` (all finite)

(Scales on this prefix: `max_time=36.4197`, `max_mag=3.02`, `max_dis=141.161`. Not comparable to the full-catalog FCNN LLs.)

### Bounded full-H SanJac train (not a full epoch)

Because default batch 128 OOMs and a full epoch at a safe batch is clearly >30 min, training was rerun at **batch 8** for a **25 min cap**.  
Log: `/workspace/st-nkf-cpu-logs/sanjac-transformer-fullH-bounded-bs8.log`  
21:11:33–21:36:29 (Asia/Shanghai).

- Full splits, `H=1672`, `n_train=11770`, 1472 steps/epoch
- Stopped mid-epoch by the 25 min `timeout` (exit 124) after **980 / 1472 steps**, **7840 / 11770** samples (~66.6% of one epoch)
- Every logged step loss was finite. Step 1: `loss=2566.443359` (2.13 s). Last logged: step 980 `loss=2528.830322`, running `mean_loss=1593.894134` at 1485.6 s (~1.5 s/step)
- Projected full 1-epoch train (no val / mu update / test eval): **~37 minutes** at batch 8. Adding val, mu, and train/val/test eval would push this well past 30 min, so the epoch was not finished and **full-H Test LL was not printed**
- Mean train loss after 66% of the epoch (~1594) is in the same ballpark as the FCNN neural epoch-1 loss (1599.72)

## Blockers / follow-ups

- **Memory:** full-H Transformer + `create_graph` at the default batch 128 does not fit in 15.6 GiB RAM. Need a smaller batch (8 works) or a GPU with more memory.
- **Time:** a complete SanJac 1-epoch Transformer run on this CPU is roughly 40+ minutes even at batch 8 (train only ~37 min). Paper-scale 1000-epoch training needs a GPU; not started.
- **No default-path breakage:** FCNN 1-epoch artifacts and CLI default `--backbone fcnn` are intact.
- Local uncommitted files only; do not push / open a PR unless asked. Existing main already contains merged PR #2 — left alone.

