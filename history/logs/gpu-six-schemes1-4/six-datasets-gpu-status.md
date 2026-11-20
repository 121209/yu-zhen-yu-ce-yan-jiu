# Six-dataset GPU training (four schemes each)

Updated: 2026-11-14 05:44:07 Asia/Shanghai

## Protocol (same on every dataset, matching the SanJac 4090 run)

- H=128, batch=16, lr=0.0001, seed=1, boundary_angles=128, epochs=50, patience=0 (disabled).
- Official EarthquakeNPP / config.py splits; NKF jitter catalog, no extra train-time jitter.
- μ initialized from train event rate / area, analytic update after each epoch.
- Evaluation always restores the best-validation snapshot.
- Device: NVIDIA GeForce RTX 4090. Scheme order: scheme2 → scheme1 → scheme4 → scheme3.
- Dataset order: SCEDC_30 → SCEDC_25 → WHITE → SaltonSea → ComCat → SCEDC_20.
- Time budget: 22.5 hours (0 means no limit). After this process exits, shut down on the AutoDL console.

| Dataset | Scheme | Status | VAL LL_ts | TEST LL_ts | TEST total LL_ts | Best ep | Seconds |
|---|---|---|---:|---:|---:|---|---:|
| SCEDC_30 | scheme2 | ok | -8.0504 | -5.8711 | -11143.40 | 49 | 650.2 |
| SCEDC_30 | scheme1 | ok | -8.0472 | -5.8616 | -11125.28 | 49 | 659.0 |
| SCEDC_30 | scheme4 | ok | -8.0496 | -5.8417 | -11087.57 | 49 | 641.8 |
| SCEDC_30 | scheme3 | ok | -8.1701 | -5.9718 | -11334.41 | 45 | 664.0 |
| SCEDC_25 | scheme2 | ok | -7.2304 | -5.6102 | -28051.07 | 49 | 2426.1 |
| SCEDC_25 | scheme1 | ok | -7.2334 | -5.6031 | -28015.53 | 39 | 2430.2 |
| SCEDC_25 | scheme4 | ok | -7.2320 | -5.6101 | -28050.27 | 49 | 2341.8 |
| SCEDC_25 | scheme3 | ok | -7.3488 | -5.7772 | -28885.85 | 38 | 2460.9 |
| WHITE | scheme2 | ok | -3.8486 | -2.5796 | -62116.33 | 45 | 1899.4 |
| WHITE | scheme1 | ok | -3.8476 | -2.5759 | -62028.68 | 45 | 1953.2 |
| WHITE | scheme4 | ok | -3.8489 | -2.5805 | -62139.41 | 45 | 1855.9 |
| WHITE | scheme3 | ok | -3.9178 | -2.6741 | -64391.55 | 45 | 1825.8 |
| SaltonSea | scheme2 | ok | -2.3713 | -0.1573 | -645.46 | 25 | 2951.2 |
| SaltonSea | scheme1 | ok | -2.3779 | -0.1643 | -674.15 | 25 | 3003.6 |
| SaltonSea | scheme4 | ok | -2.3781 | -0.1712 | -702.68 | 25 | 3137.1 |
| SaltonSea | scheme3 | ok | -2.6180 | -0.3846 | -1578.47 | 36 | 3172.2 |
| ComCat | scheme2 | failed |  |  |  |  | ValueError('Boundary-mass quadrature assumes a convex study region (SanJac is a projected rectangle).') |
| ComCat | scheme1 | failed |  |  |  |  | ValueError('Boundary-mass quadrature assumes a convex study region (SanJac is a projected rectangle).') |
| ComCat | scheme4 | failed |  |  |  |  | ValueError('Boundary-mass quadrature assumes a convex study region (SanJac is a projected rectangle).') |
| ComCat | scheme3 | failed |  |  |  |  | ValueError('Boundary-mass quadrature assumes a convex study region (SanJac is a projected rectangle).') |
| SCEDC_20 | scheme2 | ok | -6.5374 | -5.2947 | -69185.74 | 32 | 6763.7 |
| SCEDC_20 | scheme1 | ok | -6.5369 | -5.3046 | -69315.03 | 42 | 7094.9 |
| SCEDC_20 | scheme4 | ok | -6.5368 | -5.2917 | -69146.18 | 32 | 6912.1 |
| SCEDC_20 | scheme3 | ok | -6.6511 | -5.4954 | -71808.73 | 42 | 7279.5 |

