# Six-dataset GPU training (four schemes each)

Updated: 2026-11-14 10:41:39 Asia/Shanghai

## Protocol (same on every dataset, matching the SanJac 4090 run)

- H=128, batch=16, lr=0.0001, seed=1, boundary_angles=128, epochs=50, patience=0 (disabled).
- Official EarthquakeNPP / config.py splits; NKF jitter catalog, no extra train-time jitter.
- μ initialized from train event rate / area, analytic update after each epoch.
- Evaluation always restores the best-validation snapshot.
- Device: NVIDIA GeForce RTX 4090. Scheme order: scheme2 → scheme1 → scheme4 → scheme3.
- Dataset order: ComCat.
- Time budget: 2.900428865 hours (0 means no limit). After this process exits, shut down on the AutoDL console.

| Dataset | Scheme | Status | VAL LL_ts | TEST LL_ts | TEST total LL_ts | Best ep | Seconds |
|---|---|---|---:|---:|---:|---|---:|
| ComCat | scheme2 | ok | -7.8014 | -7.2127 | -157849.09 | 30 | 3639.0 |
| ComCat | scheme1 | partial |  |  |  |  |  |
| ComCat | scheme4 | partial |  |  |  |  |  |
| ComCat | scheme3 | partial |  |  |  |  |  |

