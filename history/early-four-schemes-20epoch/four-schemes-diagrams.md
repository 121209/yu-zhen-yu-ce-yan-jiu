# 四个方案的网络结构图

公式全部用 pdflatex（amsmath + lmodern）按 600 dpi 渲染为透明 PNG，再与 Noto Sans CJK 标签一起排入 16×10 英寸、220 dpi 的图。旧版 v2 的指数 / Rayleigh 轻尾公式未再出现。

重跑（需本机已有 TinyTeX，脚本会把 `~/.TinyTeX/bin/x86_64-linux` 加入 PATH）：

```bash
cd /workspace/st-nkf-diagrams
python3 render_four_schemes.py
```

辅助脚本：`latex_math.py`（公式渲染与缓存）、`render_four_schemes.py`（四张总图）。缓存目录 `_math_cache/`。

## 文件

| 方案 | PNG / SVG / PDF |
|---|---|
| 1. 时间核、空间核、激发产能全部使用 Transformer | `scheme1-all-transformer.{png,svg,pdf}` |
| 2. 时空核 Transformer，κ 保持 5×32 FCNN | `scheme2-st-transformer-kappa-fcnn.{png,svg,pdf}` |
| 3. 仅时间核 Transformer；空间核与 κ 为论文 FCNN | `scheme3-temporal-transformer.{png,svg,pdf}` |
| 4. 与方案 2 相同，κ 改为 4×64 GELU MLP | `scheme4-st-transformer-kappa-mlp.{png,svg,pdf}` |

绝对路径前缀：`/workspace/st-nkf-diagrams/`。

四张图共用同一套框位（时间核、空间核、κ、强度、补偿项、μ），用颜色区分模块：蓝 = Transformer，青绿 = 论文 5×32 FCNN，紫 = KappaMLP 4×64 GELU，金 = 闭式 CDF / 共享公式。

## 公式与代码一致（当前本地实现）

- 历史上下文只含 \((t_{\mathrm{prev}}-t_j)/\mathrm{max\_time}\) 与 \(m_j/\mathrm{max\_mag}\)，形状 \((B,H,2)\)；查询不进入编码器。补偿项两端复用同一套参数和 \(H\) 条历史，最新事件下限 \(F_t(0)=0\)。
- Transformer 核：Linear \(2\to 32\)，Encoder ×1，d_model=32，8 heads，FFN 128，GELU，dropout 0，pre-norm，`SDPBackend.MATH`。输出头 Linear \(32\to 12\)（4 个权重 + 4 个尺度 + 4 个指数）。时间 / 空间 / κ 编码器权重互不共享。
- 时间 Omori 混合：\(F_t(u)=\sum_k w_k\bigl(1-[c_k/(u+c_k)]^{p_k-1}\bigr)\)，\(s=\mathrm{logspace}(-5,0,4)\)。
- 空间 ETAS 径向混合（仅 Transformer 空间核）：\(F_r(u)=\sum_k w_k\bigl(1-[D_k/(u^2+D_k)]^{q_k-1}\bigr)\)，\(s=\mathrm{logspace}(-8,-1,4)\)（与时间核的尺度网格不同，以代码为准）。
- 方案 3 空间核是论文 5×32 非负 tanh FCNN CDF，输入为 \(r/\mathrm{max\_dis}\)，不是 ETAS 混合；密度使用 \(\varepsilon=10^{-5}\) 雅可比。
- PDF 只由 CDF 的 autograd 得到。Transformer 空间核在 \(r=0\) 用 \(F_r''(0)/(2\pi)\)，无 \(\varepsilon\)。
- 强度：\(\lambda=\mu+\sum_i\kappa_i h(\Delta t_i)w(\Delta x_i,\Delta y_i)\)，全平面密度，不对研究区重归一化。
- 补偿项含边界质量：\(\Lambda=\mu A\Delta t+\sum_i\kappa_i[F_t(u_i^{\mathrm{up}})-F_t(u_i^{\mathrm{lo}})]m_i\)，\(m_i\) 为 SanJac 凸多边形上 128 角极坐标求积。
- 损失仅为联合时空 NLL，图中不分解 \(\mathrm{LL}_t/\mathrm{LL}_s\)。
- \(\mu\) 由训练事件率 / 面积初始化，不进入 Adam，每 epoch 结束后按 declustering 公式解析更新。

对照源码：`/workspace/ST-NKF-pytorch/layers_transformer.py`、`layers.py`、`models.py`。
