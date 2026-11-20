# 余震预测研究

本仓库是这次 **ST-NKF 官方对照实验** 的完整归档：五个 Transformer 方案、ST-NKF / FCNN 基线，以及协议对齐的 ETAS（经验核）。仓库里同时保存评测与训练脚本、官方对照权重、说明文档、已有实验日志和输出结果，以及官方评估用的 `Datasets/`。

GitHub 不能使用中文仓库名，因此地址是 [yu-zhen-yu-ce-yan-jiu](https://github.com/121209/yu-zhen-yu-ce-yan-jiu)，含义即「余震预测研究」。**本仓库为私有**，请不要改为公开，除非你明确决定公开这些实验材料。

克隆（需要有权限的 GitHub 账号）：

```bash
git clone https://github.com/121209/yu-zhen-yu-ce-yan-jiu.git
```

请使用本仓库中的 `models.py`、`run_models.py` 和 `layers_transformer.py` 来评这批官方对照权重，不要只用 [ST-NKF-pytorch](https://github.com/121209/ST-NKF-pytorch) 里尚未包含这些改动的版本。

## 实验协议

全部官方对照快照均为：

- 历史长度 **H = 128**
- 随机种子 **seed = 1**
- 官方 EarthquakeNPP / `config.py` 时间窗（auxiliary / train / validation / test）
- 官方 jitter 目录（`*_catalog_jitter.csv`），训练时不再额外 jitter
- batch = 16，lr = 1e-4，`boundary_angles = 128`
- μ 由训练集事件率 / 面积初始化，并在每个 epoch 解析更新
- 评测始终恢复 **best-validation** 快照（`save_weights` 格式）

对照结果页（已补齐七个目录的最终 ETAS 列，2026-11-20 更新）：

```text
https://st-nkf-h-128-en3c.pages.grok.bot/
```

## 目录结构

```text
.
├── models.py, run_models.py, layers.py, layers_transformer.py
├── config.py, utils.py, batch.py
├── eval_ll_decomposition.py          # 分解评估入口，--repo 指向本仓库根目录
├── requirements.txt, environment.yml, LICENSE
├── Datasets/                         # 官方评估用 jitter 目录与 shape
│   ├── SOURCE.txt
│   ├── ComCat/   ComCat_catalog_jitter.csv, california_shape.npy
│   ├── SCEDC/    SCEDC_catalog_jitter.csv, SCEDC_shape.npy
│   ├── QTM/      SaltonSea_*_jitter.csv, SanJac_*_jitter.csv, 对应 shape
│   └── WHITE/    WHITE_catalog_jitter.csv, WHITE_shape.npy
├── payloads/                         # 文本安全分卷（展开后即得到完整目录树）
├── scripts/                          # 本次官方对照训练 / 评估脚本
│   ├── run_stnkf_fcnn_h128_gpu.py
│   ├── run_four_schemes_gpu.py
│   ├── run_six_datasets_gpu.py
│   ├── run_scheme5_spatial_tf_gpu.py     # 包装器，调用上面两个模块
│   ├── run_etas_h128_aligned_cpu.py
│   ├── evaluate_four_schemes_region.py
│   ├── restore_binaries.py              # 从 *.b64 还原官方 .pt / .npy
│   └── unpack_archive.py                # 展开 payloads/parts 分卷
├── weights/
│   ├── baseline_stnkf_fcnn/          # ST-NKF FCNN 基线 × 7 个数据集
│   ├── schemes/scheme{1-5}/{dataset}/
│   └── etas/                         # 已完成的 7 个 ETAS 权重（含 ComCat）
├── logs/
│   ├── etas/                         # 对齐协议 ETAS 最终完整训练日志
│   └── decomposition/                # 42/42 分解评估最终日志（2026-11-20 15:38:17）
└── results/
    ├── provenance/                   # 各次训练写出的 ll.json
    ├── decomposition/                # 正式 42/42 分解结果（以 summary.csv 为准）
    ├── decomposition_partial/        # 历史 24/42 快照，保留，不作为最终对照
    ├── etas-h128-aligned-ll.json     # 全部 7 个数据集最终结果
    ├── CHECKPOINT_SOURCES.json
    └── MANIFEST.sha256.csv
```

`results/decomposition/` 是 2026-11-20 15:38:17（Asia/Shanghai）完成的 42/42 正式结果。新的分解评估仍默认写到这个目录，不会覆盖 `results/decomposition_partial/` 里的历史 24 组快照。

## 权重清单

| 模型 | 时间核 | 空间核 | κ | 权重数量 |
|---|---|---|---|---|
| ST-NKF FCNN 基线（`weights/baseline_stnkf_fcnn/`） | FCNN | FCNN | FCNN | 7 / 7 |
| scheme1 | Transformer | Transformer | Transformer | 7 / 7 |
| scheme2 | Transformer | Transformer | FCNN | 7 / 7 |
| scheme3 | Transformer | FCNN | FCNN | 7 / 7 |
| scheme4 | Transformer | Transformer | MLP | 7 / 7 |
| scheme5 | FCNN | Transformer | FCNN | 7 / 7 |
| ETAS（经验核，`weights/etas/`） | 经验 | 经验 | 经验 | **7 / 7（全部完成）** |

七个数据集均为：`SCEDC_30`、`SCEDC_25`、`SCEDC_20`、`SaltonSea`、`SanJac`、`WHITE`、`ComCat`。

- **42 个** FCNN + 五个方案权重，全部是官方 H=128、seed=1、best-validation 的 `save_weights` 快照。
- **7 个** 已完成的 ETAS 权重：`SCEDC_30`、`SCEDC_25`、`SCEDC_20`、`SaltonSea`、`SanJac`、`WHITE`、`ComCat`。
- **ETAS 七个目录已于 2026-11-20 08:34:32（Asia/Shanghai，北京时间）全部完成，共 42+7=49 个官方快照。** ComCat 跑满 50 轮，best_epoch=47；test LL_ts=-7.460639476776123，n_test=21885，总量=-163276.09494924545。不要用旧快照或论文路径 ETAS 顶替。

哈希与来源见 `results/MANIFEST.sha256.csv` 和 `results/CHECKPOINT_SOURCES.json`。

## 二进制与完整树如何入库

GitHub 的文本接口不能无损上传 `.pt` / `.npy`，有时对超大文件也不方便一次提交整棵目录树。因此本仓库用两层文本安全载荷保存官方材料：

1. `payloads/parts/part-*.b64`：整棵归档树（不含原始 `.pt` / `.npy`）的分卷 `tar.gz`。
2. 同目录下的 `*.pt.b64` / `*.npy.b64`：官方二进制快照，可用 `results/MANIFEST.sha256.csv` 校验还原后的哈希。

如果克隆下来还看不到 `Datasets/` 或 `weights/*.pt`，先在仓库根目录执行：

```bash
python scripts/unpack_archive.py --root .
python scripts/restore_binaries.py --root .
```

也可以直接运行 `python eval_ll_decomposition.py --repo <仓库根>`：脚本会在需要时自动解包并还原。仓库还配置了 GitHub Actions（`.github/workflows/materialize-binaries.yml`）；若该工作流已成功执行，之后的克隆会直接带有完整目录树和二进制权重。

请不要用其他旧快照替换这些官方文件。**历史 `payloads/` 分卷是 2026-11-19 的初始快照；仓库已物化时不要重新执行 `unpack_archive.py`，否则可能用旧 README/日志/JSON 覆盖最终增量更新。以 main 中的最终文件和 MANIFEST 为准。

## 如何运行对数似然分解

在已安装依赖的环境中，把 `--repo` 指到本仓库根目录（同时包含 `models.py`、`run_models.py`、`layers_transformer.py`、`Datasets/` 和 `weights/`）：

```bash
python eval_ll_decomposition.py --repo /path/to/yu-zhen-yu-ce-yan-jiu
```

Windows 示例：

```powershell
python .\eval_ll_decomposition.py --repo "C:\Users\admin\Desktop\yu-zhen-yu-ce-yan-jiu"
```

脚本会按仓库内相对路径寻找 42 个权重：

- FCNN：`weights/baseline_stnkf_fcnn/stnkf_fcnn_{dataset}_H128_seed1_best.pt`
- scheme5：`weights/schemes/scheme5/{dataset}/scheme5_H128_seed1_best.pt`
- scheme1–4：`weights/schemes/{scheme}/{dataset}/{scheme}_H128_seed1_best.pt`

默认输出目录是 `results/decomposition/`。该目录的 `summary.csv` 已有完整 42 行，再次运行会跳过已完成的行。历史 24 组在 `results/decomposition_partial/`，不要用新结果覆盖它。

**当前 `MODEL_ORDER` 不含 ETAS。** `eval_ll_decomposition.py` 还不能对经验 ETAS 核做同样的分解；ETAS 的权重、训练脚本、日志和已有 ll 结果仍收录在本仓库中，只是不要指望这条评估脚本直接读 `weights/etas/`。这 42 组分解结果同样不含 ETAS。

## 分解评估进度（42 / 42）

正式结果在 `results/decomposition/`，于 **2026-11-20 15:38:17（Asia/Shanghai，北京时间）** 完成，exit=0，missing=0。请以该目录的 `summary.csv` 和 `comparison.csv` 为准。`results/decomposition_partial/` 是更早的 24 组历史快照，保留不动；其中的 `comparison.csv` 仍只覆盖 SanJac 上的部分对比，不要把它当作完整对照。

已完成（42 组，7 个数据集 × 6 个神经网络模型，seed=1）：

- 数据集：`SCEDC_30`、`SCEDC_25`、`SCEDC_20`、`SaltonSea`、`SanJac`、`WHITE`、`ComCat`
- 模型：FCNN 与 scheme1–5

**这 42 组不含 ETAS。** `MODEL_ORDER` 仍然没有 ETAS；经验 ETAS 核不在本次分解里。ETAS 的 7 个权重和 ll 结果另见 `weights/etas/` 与 `results/etas-h128-aligned-ll.json`。

全档案哈希仍以 `results/MANIFEST.sha256.csv` 为准。仓库根目录的 `MANIFEST.sha256.csv` 和 `results/decomposition/MANIFEST.sha256.csv` 只覆盖这次分解增量，不能代替全档案清单。`summary.csv` 里的 `checkpoint_path` 保留原始本机路径。

## 训练脚本说明

`scripts/` 里是这次官方对照实际使用的训练入口，不是烟测或早期 CDF 试验脚本：

- `run_stnkf_fcnn_h128_gpu.py`：ST-NKF FCNN 基线
- `run_four_schemes_gpu.py` + `run_six_datasets_gpu.py`：scheme1–4
- `run_scheme5_spatial_tf_gpu.py`：scheme5 的薄包装，调用上述两个模块并注册 scheme5 骨干
- `run_etas_h128_aligned_cpu.py`：协议对齐的 ETAS（CPU）
- `evaluate_four_schemes_region.py`：历史有限区域分解辅助脚本

归档副本里，脚本会优先把「本仓库根目录」（`scripts/` 的上一级）当作代码与 `Datasets/` 所在位置。脚本中若仍出现 `/root/autodl-tmp` 或 `/hy-tmp`，只是当时云主机上的路径字符串，**不含登录凭据**。默认输出目录可能仍指向历史云路径，重跑时请显式传入 `--out`。

## 数据说明

官方评估使用 jitter 目录：`Datasets/**/*_catalog_jitter.csv` 以及对应的 `*_shape.npy`（ComCat 为 `california_shape.npy`）。这些文件已收录，`config.py` 中的路径是 `./Datasets/...`。

未收录非 jitter 的 `*_catalog.csv`（其中 SCEDC 最大，约 11 MB）。它们不是这套官方对照评估的输入。若你需要原始未 jitter 目录，可以从 [ST-NKF-pytorch](https://github.com/121209/ST-NKF-pytorch) 的 `Datasets/` 复制。

## ETAS / 日志快照说明

- `results/etas-h128-aligned-ll.json` 含全部 7 个数据集最终结果；原始训练 JSON 另存 `results/provenance/etas-h128-aligned-final-native.json`。
- `logs/etas/` 已更新为最终完整日志，包含 ComCat 第 50 轮、恢复第 47 轮最佳验证快照和最终 completed 记录。
- 本次只归档已完成文件，没有重启任何训练、评估、GPU 或分解 watcher。

## 许可

与上游 ST-NKF-pytorch 工作副本一致，见 `LICENSE`。
