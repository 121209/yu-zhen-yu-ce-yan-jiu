# 余震预测研究

本仓库存放余震预测（ST-NKF）官方对照评测用的**代码工作副本**，方便在本地运行对数似然分解评估。

GitHub 仓库名不能使用中文，因此地址是 [yu-zhen-yu-ce-yan-jiu](https://github.com/121209/yu-zhen-yu-ce-yan-jiu)，含义即「余震预测研究」。仓库为私有。

## 这里有什么

- `models.py`、`run_models.py`：云端评测用的工作副本（相对原仓库有本地修改）
- `layers_transformer.py`：Transformer 层实现（当时尚未合入原仓库）
- `layers.py`、`config.py`、`utils.py`、`batch.py`
- `eval_ll_decomposition.py`：分解评估脚本（已支持 `--repo`）

本包**不含**目录数据 `Datasets/`，也不含 42 个官方对照权重（权重请继续使用此前发过的压缩包）。

## 使用方法

先拉取本仓库，再从已有的 [ST-NKF-pytorch](https://github.com/121209/ST-NKF-pytorch) 取得 `Datasets/`（官方评估使用其中的 `*_catalog_jitter.csv` 和 shape 文件）：

```powershell
git clone https://github.com/121209/yu-zhen-yu-ce-yan-jiu.git
git clone https://github.com/121209/ST-NKF-pytorch.git
Copy-Item -Recurse .\ST-NKF-pytorch\Datasets .\yu-zhen-yu-ce-yan-jiu\Datasets
```

然后在评估脚本所在目录运行，例如：

```powershell
python .\eval_ll_decomposition.py --repo "C:\Users\admin\Desktop\yu-zhen-yu-ce-yan-jiu"
```

把 `--repo` 指到同时包含 `models.py`、`run_models.py`、`layers_transformer.py` 和 `Datasets/` 的目录即可。若你使用自己改过的 `Eval_ll_Decomposition.py`，同样把 `--repo` 指向这个目录。

请使用本仓库中的 `models.py` / `run_models.py` / `layers_transformer.py`，不要只用 GitHub 上 ST-NKF-pytorch 里尚未包含这些改动的版本来评那 42 个官方对照权重。
