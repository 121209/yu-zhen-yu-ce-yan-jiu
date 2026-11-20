# Six-dataset GPU training (four schemes each)

Updated: 2026-11-17 20:14:45 Asia/Shanghai

## Protocol (same on every dataset, matching the SanJac 4090 run)

- H=128, batch=16, lr=0.0001, seed=1, boundary_angles=128, epochs=50, patience=0 (disabled).
- Official EarthquakeNPP / config.py splits; NKF jitter catalog, no extra train-time jitter.
- μ initialized from train event rate / area, analytic update after each epoch.
- Evaluation always restores the best-validation snapshot.
- Device: NVIDIA GeForce RTX 4090. Scheme order: scheme5.
- Dataset order: SCEDC_20.
- Time budget: 0.0 hours (0 means no limit). After this process exits, shut down on the AutoDL console.

| Dataset | Scheme | Status | VAL LL_ts | TEST LL_ts | TEST total LL_ts | Best ep | Seconds |
|---|---|---|---:|---:|---:|---|---:|
| SCEDC_20 | scheme5 | ok | -6.5435 | -5.2680 | -68837.54 | 42 | 5587.7 |

