# Upstream catalog preprocessing

Source (local box copy, not re-fetched): `/workspace/NeuralKernelFunction/Data processing.ipynb`  
Upstream project: https://github.com/zhancx-edu/NeuralKernelFunction  
This notebook is the documented origin of the official `*_catalog_jitter.csv` files (also cited in `Datasets/SOURCE.txt`).

Recorded in the notebook (do not treat this file as a new preprocessing run):

- `np.random.seed(42)`
- Spatial jitter: Uniform[-0.5, 0.5] km (`spatial_jitter_range = 1.0`)
- Temporal jitter: Uniform[-0.05, 0.05] seconds (`time_jitter_range = 0.1` seconds, converted to days)

The five non-jitter `*_catalog.csv` inputs are packed next to the official jitter files under `Datasets/` (they are byte-identical in the local ST-NKF-pytorch and NeuralKernelFunction copies; packed once). They do **not** replace `*_catalog_jitter.csv`.

The rest of the upstream TensorFlow repository (models, plots, environment, method.jpg, etc.) is not packed here; this file is the preprocessing source actually used for the official catalogs.
