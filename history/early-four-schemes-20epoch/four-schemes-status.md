# Four-scheme ST-NKF status

Updated: 2026-11-11 23:38:29 Asia/Shanghai

## Protocol

- Dataset: SanJac official splits from config.py, jitter catalog from NKF (no extra jitter).
- H=128, batch=16, epochs=20, patience=8, lr=0.0001, seed=1, boundary_angles=128.
- Loss: spatiotemporal point-process NLL. mu initialized from rate/area, analytic update after each epoch.
- Default FCNN/ETAS path left bitwise unchanged (not used by these four schemes).

## Results

| Scheme | Architecture | Split | LL_ts | LL_t | LL_s | Epoch (best) | Seconds | Notes |
|---|---|---|---:|---:|---:|---|---:|---|
| scheme1 | Temporal + Spatial + Kappa Transformer | VAL | -5.6184 | 0.7440 | -6.3625 | 14/20 | 1290.6 | mu=5.726342e-05 |
|  |  | TEST | -4.6846 | 1.2478 | -5.9325 |  | 0.0 | mu=5.726342e-05 |
| scheme2 | Temporal + Spatial Transformer, Kappa 5x32 FCNN | VAL | -5.6233 | 0.7328 | -6.3561 | 13/20 | 903.7 | mu=5.946291e-05 |
|  |  | TEST | -4.6903 | 1.2315 | -5.9218 |  | 0.0 | mu=5.946291e-05 |
| scheme3 | Temporal Transformer only; Spatial and Kappa FCNN | VAL | -5.7282 | 0.8750 | -6.6033 | 20/20 | 6218.3 | mu=1.031121e-04 |
|  |  | TEST | -4.7671 | 1.3747 | -6.1417 |  | 0.0 | mu=1.031121e-04 |
| scheme4 | Same as scheme 2, Kappa deeper MLP (4x64 GELU) | VAL | -5.6175 | 0.7373 | -6.3548 | 17/20 | 1006.6 | mu=6.001539e-05 |
|  |  | TEST | -4.6873 | 1.2383 | -5.9256 |  | 0.0 | mu=6.001539e-05 |

Max RSS so far: 7334 MB.

Logs: `four-schemes-training.log`, `four-schemes-ll.json`. Weights under `four-schemes-weights/`.
Not pushed, no PR, no GPU rental.
