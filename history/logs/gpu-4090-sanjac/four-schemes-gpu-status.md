# Four-scheme GPU training

Updated: 2026-11-12 21:18:41 Asia/Shanghai

## Protocol

- Dataset: SanJac official splits, NKF jitter catalog, H=128, batch=16, lr=0.0001, seed=1, boundary_angles=128.
- Epoch cap 50, patience 0; evaluation always uses best-val weights.
- Device: NVIDIA GeForce RTX 4090. Run order: scheme2 → scheme1 → scheme4 → scheme3.
- Time budget: 10.0 hours.

| Scheme | Status | VAL LL_ts | TEST LL_ts | TEST LL_t | TEST LL_s | Best ep | Reason | Seconds |
|---|---|---:|---:|---:|---:|---|---|---:|
| scheme2 | ok | -5.6097 | -4.6665 | 1.2647 | -5.9311 | 41 | completed | 1537.4 |
| scheme1 | ok | -5.6094 | -4.6601 | 1.2639 | -5.9239 | 42 | completed | 1571.2 |
| scheme4 | ok | -5.6088 | -4.6612 | 1.2708 | -5.9320 | 42 | completed | 1517.7 |
| scheme3 | ok | -5.7142 | -4.7581 | 1.4251 | -6.1832 | 35 | completed | 1544.2 |

