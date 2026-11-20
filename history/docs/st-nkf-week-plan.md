# ST-NKF PyTorch 本周计划（复现 + Transformer 换 FCNN）

> 面向：xuan de  
> 仓：https://github.com/121209/ST-NKF-pytorch  
> main tip：`c8fe20ab0fa7d58bdb2486f9d77c66286121ba24`（2026-11-03 合并 PR #1）  
> 上游 TF：https://github.com/zhancx-edu/NeuralKernelFunction（本地已有 `/workspace/NeuralKernelFunction`）  
> 论文：Zhan et al., JGR: Machine Learning and Computation, https://doi.org/10.1029/2025JH001110  

**原则：先复现、再换核；外层点过程公式不动，只换三个神经核里的 FCNN。**

---

## 0. 仓与 PR 现状

| 项 | 状态 |
|---|---|
| PR #1 Port to PyTorch | **已合并** → main `c8fe20ab` |
| PR #2 六个论文绘图 Notebook | **仍 open**（`codex/add-plot-notebooks` @ `b08854ca`，自 11-04 闲置）；**不挡复现/换核** |
| Issues | 无独立 issue；open_issues=1 即 PR #2 |
| 121209 其它相关仓 | 检索仅此一个；上游在 `zhancx-edu/NeuralKernelFunction` |

关键入口：

- 训练：`run_models.py`（CLI：`seed gpu_id dataset kappa temporal spatial`）
- 批量：`batch.py`（默认 dry-run，加 `--run` 才执行）
- 模型：`models.py` → `KernelPointProcess`
- 核层：`layers.py`（empirical + neural FCNN）
- 切分：`config.py`（UTC 零点，与 README history 长度一致）
- 数据：`Datasets/*/*_catalog_jitter.csv`（已随仓；说明见 `Datasets/SOURCE.txt`）
- 单测：`python -m unittest tests.test_port`

指标（测试集平均）：`LL_ts` / `LL_t` / `LL_s`，写入 `ll_seismic_model.log`。  
仓内**没有**填论文表；对照用上游 TF 的 `ll_seismic_model.log`（本机副本有 3×seed×7 目录的 neural/empirical 结果）。

---

## 1. FCNN 在哪、吃什么、吐什么

### 位置（要换的就是这些）

| 模块 | 文件 | 角色 | 激活 / 约束 |
|---|---|---|---|
| `_NonnegativeMLP` | `layers.py` | 三个神经核的公共全连接栈 | 隐层 `tanh`；权非负（`abs(Glorot)` 初值 + `project()` clamp≥0） |
| `TemporalKernelNetwork` | `layers.py` | 时间 **CDF** \(H_\theta(\Delta t)\) | 无 bias，输出 `tanh`；PDF 靠 `autograd.grad` |
| `SpatialKernelNetwork` | `layers.py` | 空间 **CDF** \(W_\phi(r)\) | 同上；PDF = \(\partial W/\partial r / (2\pi(r+\varepsilon))\) |
| `KappaNetwork` | `layers.py` | 激发产能 \(\kappa_\psi(m)\) | **有 bias**，输出 `softplus` |
| 接线 | `models.py` `KernelPointProcess.forward` | `*_id=="neural"` 时走上述网络，否则走 empirical | 外层强度公式不变 |
| 超参 | `run_models.py` | `size_layer=5`, `size_nn=32`（论文选定） | Adam lr=0.005；μ **不进 Adam**，每 epoch 末 declustering 覆盖 |

### 张量形状（未在 docstring 写死，由 `utils.process_data` + forward 推断）

对每个目标事件，历史窗长度 \(H=\) 辅助窗事件数（SanJac **1672**）：

- 网络输入 / 输出均为 **逐历史事件、标量特征**：`(B, H, 1) → (B, H, 1)`
  - temporal：`t1 / max_time`（年龄归一化）→ CDF；`pdf = dCDF/dt`
  - spatial：`dis / max_dis`（距离归一化）→ CDF；再变 PDF
  - kappa：`mags / max_mag` → softplus 产能
- `kappa` 还对去掉最新一条的历史再算一次 `(B, H-1, 1)`，用于补偿积分差分
- **没有**跨历史事件的消息传递：三个 FCNN 都是 **pointwise MLP**（同一套权对每个历史事件独立算）

经验核对照（不换）：`ProductionFunction` / `OmoriFunction` / `SpatialFunction` / `BackgroundFunction`。

---

## 2. 「复现」清单

### 环境

```bash
git clone https://github.com/121209/ST-NKF-pytorch.git && cd ST-NKF-pytorch
# CPU smoke：
pip install -r requirements.txt
# AutoDL / GPU：按驱动从 https://pytorch.org 装 CUDA 版 torch，不要用 environment.yml 里的 cpuonly
python -m unittest tests.test_port
```

数据已在仓内 `Datasets/`，**无需另下**（来源 EarthquakeNPP + 上游 jitter；MIT 只管代码，目录请引用 Zhan et al. / EarthquakeNPP）。

### Smoke（本机 / box CPU 即可）

```bash
python run_models.py 1 -1 SanJac empirical empirical empirical --epochs 1 --no-patterns
python run_models.py 1 -1 SanJac neural neural neural --epochs 1 --no-patterns
```

验收：切分 `auxiliary=1672, training=11770, validation=3449, testing=4400`；loss/val_loss/mu 有限；打印 `Test results: LL_ts/LL_t/LL_s`。**第 1 个 epoch loss 会很大**（μ 初值 1），属正常。

### 论文级复现（需要 AutoDL GPU）

建议顺序（由小到大）：

1. **SanJac** seed 1：ETAS 全 empirical + ST-NKF 全 neural（默认 1000 epoch / patience 20）
2. 同目录 seed 2、3 → 与上游 TF 日志均值±std 对照数量级
3. 再扩 `SCEDC_30` / `WHITE` / `SaltonSea`；**ComCat、SCEDC_20 历史窗很长，稠密距离矩阵可达数 GB**，内存不够勿硬跑

对照锚点（上游 TF 本机 `ll_seismic_model.log`，SanJac neural 三 seed 约）：

- `LL_ts ≈ -4.48 ~ -4.49`，`LL_t ≈ 1.11`，`LL_s ≈ -5.60`
- SanJac empirical：`LL_ts ≈ -4.64`

**注意**：PyTorch 与 TF 不会逐位相同（初始化/shuffle/float 归约）；目标是 **同量级、同相对结论（NKF ≈ ETAS）**，不是 bit-match。

产出：`weights/*.pt`、`csv/*.csv`、`ll_seismic_model.log`、`etas_parameters.log`、可选 `results/*.npy`。

---

## 3. Transformer 替换草图（外层不动）

**一句话**：保留 `KernelPointProcess` 的强度分解 \(\lambda=\mu+\sum\kappa\cdot h\cdot w\)、μ 更新、empirical 路径与训练循环；只把 `layers.py` 里三个 `_NonnegativeMLP` 子类换成 **同 I/O 形状** 的 Transformer 核；CLI 加 `--backbone fcnn|transformer`（或新 mode `transformer`），默认仍 FCNN 以免打断复现。

建议最小改动文件：

1. **新建** `layers_transformer.py`（或 `layers.py` 内并列）  
   - `TemporalKernelTransformer` / `SpatialKernelTransformer` / `KappaTransformer`  
   - 输入仍 `(B,H,1)`：先 `Linear(1→d_model)`，再 **沿 H 维** self-attention（让历史事件可互看——这才是相对 FCNN 的实质增益），最后 `Linear(d_model→1)`  
   - kappa：输出 `softplus`；temporal/spatial：输出需仍可当作 CDF（至少保证可导、非负 PDF；单调性可用 softplus 累积或排序投影等简版约束，首版可先不硬保 CDF∈[0,1]，但必须保证 `autograd.grad` 得到有限 PDF）
2. **改** `models.py`：按 `backbone` 实例化三网；`project_constraints` 对 Transformer 改为「输出端非负 / softplus」类约束（权重全非负对 attention 不自然，勿照搬 FCNN 的 `clamp≥0`）
3. **改** `run_models.py`：`--backbone`；权重文件名带上 backbone 标签
4. **单测** `tests/test_transformer_smoke.py`：随机小窗 `(B=2,H=8,1)` forward + 一步 backward，检查 shape 与有限 loss（**CPU 即可**）

风险：

- 论文 FCNN 的「全非负权 + CDF 求导」保证密度性质；Transformer 破坏后易出现负密度 / 爆炸 → 先 SanJac 短训看 LL 是否有限
- Attention 对 H=1672 的 \(O(H^2)\)：SanJac 可试；ComCat/SCEDC_20 需截断、局部窗或线性 attention
- 参数量与 epoch 时间明显高于 FCNN → 必须 GPU

---

## 4. 本周日序（复现优先，架构可并行）

| 日 | 做什么 | 机器 |
|---|---|---|
| **D1** | clone；unittest；SanJac 两路 1-epoch smoke；确认切分与有限 LL | CPU / box |
| **D2** | AutoDL 起机；SanJac ETAS + ST-NKF **完整训练** seed=1；记下 LL 与耗时 | **AutoDL GPU** |
| **D2–D3（并行）** | Transformer 模块骨架 + shape/backward 单测；不改默认 FCNN 路径 | CPU / box |
| **D3–D4** | SanJac seed 2/3（FCNN）凑齐三 seed；与上游 TF 日志对照 | AutoDL |
| **D4–D5** | Transformer SanJac 短训（如 50–100 epoch）→ 再完整跑一 seed；对比 FCNN | AutoDL |
| **D6** | 可选：`SCEDC_30` / `WHITE`；整理 `ll_seismic_model.log` 对比表 | AutoDL |
| **D7** | 写实验笔记：复现结论 + Transformer 是否保住有限 LL / 是否逼近 ETAS；PR 策略（换核单独开 PR，勿绑 #2） | — |

**PR #2（绘图）**：本周可不碰；真要出图等 FCNN 复现 CSV 齐了再合或 rebase。

---

## 5. 风险一览

| 风险 | 影响 | 缓解 |
|---|---|---|
| 无个人 GPU | 完整 neural 训不动 | AutoDL（与课程 H20 分开）；CPU 只做 smoke + 架构 |
| ComCat / SCEDC_20 内存 | 稠密窗数 GB | 本周先 SanJac → SCEDC_30 |
| PT≠TF 数值 | 「对不上论文表」焦虑 | 对上游同仓 TF 日志与相对结论，不追 bit-match |
| Transformer 破 CDF/非负 | NaN / 负 PDF | 短训门禁；输出 softplus；必要时先只换 kappa |
| Attention \(O(H^2)\) | 大目录 OOM/极慢 | 小目录验证；大目录再上局部/线性 attention |
| PR #2 闲置 | 无 | 与复现解耦 |

---

## 6. 需用户拍板的两点（最多）

1. **换几个核？** 三个都换（kappa + temporal + spatial），还是先只换其中一个（更稳：建议先 **kappa**，因无 CDF 求导约束）？  
2. **本周 AutoDL**：已有实例/预算吗？优先卡型（论文参考 RTX 3090；4090/3090 均可）？若暂时没有 GPU，本周只能做 CPU smoke + Transformer 骨架，完整复现顺延。
