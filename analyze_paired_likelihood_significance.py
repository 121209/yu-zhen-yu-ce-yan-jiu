#!/usr/bin/env python3
"""实验 1B-1：逐事件配对的似然增益、bootstrap 区间和 Wilcoxon 检验。

在仓库根目录运行（依赖 numpy、pandas、scipy，无需 torch）：
  python analyze_paired_likelihood_significance.py
  python analyze_paired_likelihood_significance.py --block-days 7 --include-etas
  python analyze_paired_likelihood_significance.py --check-only

默认 10,000 次逐事件配对 bootstrap，输出均值增益的百分位 95% CI。
bootstrap p 值使用零假设下重新中心化的分布，为双侧近似检验，采用 +1 修正。
--block-days 改为重采样固定 UTC 日历块（保留内部空块）；整块内事件共同抽取。
块模式仍估计每事件平均增益。Wilcoxon 改用非空块的平均增益，检验对象不同。
Wilcoxon 使用双侧、Pratt 零差处理和 SciPy method=auto。
各指标的配对差值统一保留 12 位小数，避免浮点舍入制造虚假的秩差异。
其零假设是独立单位的差值分布关于零对称，并非一般意义的均值相等。
两种 p 值分别对本次输出的全部目录/对比/指标作 Holm 校正。
CI 为逐项区间，不是多重比较同时置信区间。

结果只描述这些 seed=1 固定权重，不能量化重新训练的随机性。
事件模式依赖事件独立假设；时间块也需检查块长敏感性。
SCEDC 三个阈值目录相互重叠，不合并为独立地区重复。
SciPy 定义：https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATASETS = ("SCEDC_30", "SCEDC_25", "SCEDC_20", "SaltonSea", "SanJac", "WHITE", "ComCat")
MODELS = ("fcnn", "scheme3", "scheme5", "scheme2", "scheme4", "scheme1")
METRICS = ("LL_ts", "LL_t_region", "LL_s_given_t_region")
# 对比方向均为公式中的前者减后者，正值表示前者得分更高。
CONTRASTS = (
    ("time_space", "S3_minus_FCNN", {"scheme3": 1, "fcnn": -1}),
    ("time_space", "S5_minus_FCNN", {"scheme5": 1, "fcnn": -1}),
    ("time_space", "S2_minus_S3", {"scheme2": 1, "scheme3": -1}),
    ("time_space", "S2_minus_S5", {"scheme2": 1, "scheme5": -1}),
    ("time_space", "S2_minus_FCNN", {"scheme2": 1, "fcnn": -1}),
    ("interaction", "S2_minus_S3_minus_S5_plus_FCNN",
     {"scheme2": 1, "scheme3": -1, "scheme5": -1, "fcnn": 1}),
    ("kappa", "S4_minus_S2", {"scheme4": 1, "scheme2": -1}),
    ("kappa", "S1_minus_S2", {"scheme1": 1, "scheme2": -1}),
    ("kappa", "S1_minus_S4", {"scheme1": 1, "scheme4": -1}),
)
EVENT_COLUMNS = ("event_id", "time", "log_lambda_ts", "log_lambda_t_region",
                 "log_conditional_space_region", "compensator_increment")


def load_events(repo, dataset, model):
    folder = "etas_decomposition" if model == "etas" else "decomposition"
    path = repo / "results" / folder / "events" / f"{dataset}_{model}_seed1_events.csv"
    table = pd.read_csv(path, usecols=EVENT_COLUMNS).sort_values("event_id")
    values = table[list(EVENT_COLUMNS)].to_numpy(dtype=float)
    if (len(table) == 0 or not np.isfinite(values).all()
            or not np.array_equal(table.event_id.to_numpy(), np.arange(len(table)))
            or np.any(np.diff(table.time.to_numpy()) < 0)):
        raise ValueError(f"事件编号、顺序或数值异常：{path}")
    # 两种评估均从原始公共列重构 float64 得分，避免混用附加的 float32 LL 列。
    joint = table.log_lambda_ts.to_numpy() - table.compensator_increment.to_numpy()
    temporal = table.log_lambda_t_region.to_numpy() - table.compensator_increment.to_numpy()
    space = table.log_conditional_space_region.to_numpy()
    if not np.allclose(joint, temporal + space, rtol=0, atol=1e-8):
        raise ValueError(f"区域分解恒等式未通过：{path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return table.time.to_numpy(), np.column_stack((joint, temporal, space)), digest


def bootstrap_statistics(differences, times, n_bootstrap, rng, block_days=None):
    """三项指标共用每次抽样索引，保持事件配对及分解关系，限制批量内存。"""
    observed = differences.mean(axis=0)
    if block_days is None:
        sums = differences
        counts = np.ones(len(differences), dtype=np.int64)
    else:
        index = np.floor(times / block_days).astype(np.int64)
        index -= index.min()
        if index.max() > 1_000_000:
            raise ValueError("时间块过细，会产生超过一百万个块；请增大 --block-days")
        counts = np.bincount(index)
        sums = np.column_stack([np.bincount(index, weights=differences[:, j],
                                           minlength=len(counts)) for j in range(3)])
    n_units = len(counts)
    if n_units < 2:
        return observed, np.full((2, 3), np.nan), np.full(3, np.nan), n_units, sums, counts
    batch = max(1, min(128, (32 * 1024**2) // (n_units * 32)))
    boot = np.empty((n_bootstrap, 3))
    for start in range(0, n_bootstrap, batch):
        stop = min(start + batch, n_bootstrap)
        index = rng.integers(n_units, size=(stop - start, n_units))
        denominators = counts[index].sum(axis=1)
        # 含空日历块时，少量重采样可能整组无事件，此时重新抽取该组。
        while np.any(denominators == 0):
            empty = denominators == 0
            index[empty] = rng.integers(n_units, size=(int(empty.sum()), n_units))
            denominators = counts[index].sum(axis=1)
        boot[start:stop] = sums[index].sum(axis=1) / denominators[:, None]
    ci = np.quantile(boot, (0.025, 0.975), axis=0)
    null = boot - observed
    p = (1 + (np.abs(null) >= np.abs(observed)).sum(axis=0)) / (n_bootstrap + 1)
    return observed, ci, p, n_units, sums, counts


def holm(values):
    values = np.asarray(values, dtype=float)
    adjusted = np.full(len(values), np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    order = valid[np.argsort(values[valid])]
    adjusted[order] = np.minimum(1, np.maximum.accumulate(
        values[order] * np.arange(len(order), 0, -1)))
    return adjusted


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--block-days", type=float, help="改用时间块重采样，例如 1、7、30 天")
    parser.add_argument("--include-etas", action="store_true", help="另加六个神经模型减 ETAS 的对比")
    parser.add_argument("--out", type=Path, help="全新输出目录，默认 results/paired_likelihood_significance")
    parser.add_argument("--check-only", action="store_true", help="只核对所有输入与事件配对，不统计、不写文件")
    args = parser.parse_args()
    if (args.n_bootstrap < 100 or args.seed < 0 or
            (args.block_days is not None and
             (not np.isfinite(args.block_days) or args.block_days <= 0))):
        parser.error("重采样次数至少 100，种子非负，时间块长度须为正数")
    repo = args.repo.resolve()
    destination = (args.out or repo / "results/paired_likelihood_significance").resolve()
    if not args.check_only and destination.exists():
        parser.error(f"目录已存在，请通过 --out 另选目录：{destination}")
    models = MODELS + (("etas",) if args.include_etas else ())
    contrasts = list(CONTRASTS)
    if args.include_etas:
        contrasts.extend(("etas_comparison", f"{m}_minus_ETAS", {m: 1, "etas": -1}) for m in MODELS)
    if not args.check_only:
        try:
            from scipy.stats import wilcoxon
        except ImportError:
            parser.error("缺少 SciPy。请使用同一个解释器执行 python -m pip install scipy")
    rows, hashes = [], {}
    for dataset in dict.fromkeys(args.datasets):
        all_scores, reference_times = {}, None
        for model in models:
            times, scores, digest = load_events(repo, dataset, model)
            if reference_times is not None and (len(times) != len(reference_times)
                    or not np.allclose(times, reference_times, rtol=0, atol=1e-9)):
                raise ValueError(f"{dataset}/{model} 与 FCNN 的测试事件不一致")
            reference_times = times if reference_times is None else reference_times
            all_scores[model] = scores
            hashes[f"{dataset}/{model}"] = digest
        print(f"{dataset}: 已配对 {len(times):,} 个事件 × {len(models)} 个模型", flush=True)
        if args.check_only:
            continue
        for contrast_index, (family, name, weights) in enumerate(contrasts):
            diff = np.round(sum(weight * all_scores[model]
                                for model, weight in weights.items()), decimals=12)
            rng = np.random.default_rng(np.random.SeedSequence(
                [args.seed, DATASETS.index(dataset), contrast_index]))
            mean, ci, p, n_units, sums, counts = bootstrap_statistics(
                diff, times, args.n_bootstrap, rng, args.block_days)
            unit_means = np.round(sums[counts > 0] / counts[counts > 0, None], decimals=12)
            for j, metric in enumerate(METRICS):
                unit_values = unit_means[:, j]
                stat, p_w = np.nan, np.nan
                if len(unit_values) >= 2:
                    if np.all(unit_values == 0):
                        stat, p_w = 0.0, 1.0
                    else:
                        test = wilcoxon(unit_values, zero_method="pratt",
                                        alternative="two-sided", method="auto")
                        stat, p_w = float(test.statistic), float(test.pvalue)
                rows.append(dict(
                    dataset=dataset, family=family, contrast=name, metric=metric,
                    seed_model=1, n_events=len(diff), resampling_unit=(
                        "event" if args.block_days is None else "calendar_time_block"),
                    block_days=args.block_days, n_resampling_units=n_units,
                    n_wilcoxon_units=len(unit_values), n_bootstrap=args.n_bootstrap,
                    mean_gain_nats_per_event=mean[j], median_event_gain=np.median(diff[:, j]),
                    positive_event_fraction=np.mean(diff[:, j] > 0),
                    ci95_low=ci[0, j], ci95_high=ci[1, j],
                    p_bootstrap_centered=p[j], wilcoxon_statistic=stat, p_wilcoxon=p_w,
                    wilcoxon_p_underflow=bool(p_w == 0), small_unit_count=n_units < 20))
            print(f"  {name}: bootstrap 完成", flush=True)
    if args.check_only:
        print("全部输入检查通过；没有生成结果文件。")
        return
    result = pd.DataFrame(rows)
    result["p_bootstrap_holm"] = holm(result.p_bootstrap_centered)
    result["p_wilcoxon_holm"] = holm(result.p_wilcoxon)
    metadata = dict(
        arguments=vars(args), metrics=list(METRICS), unit="nats/event",
        paired_difference_rounding_decimals=12,
        contrasts={name: weights for _, name, weights in contrasts}, source_sha256=hashes,
        ci="percentile 95%, marginal intervals; centered-null two-sided bootstrap p with +1 correction",
        multiplicity="Holm across every emitted dataset/contrast/metric; each p-value method separately",
        wilcoxon="two-sided, Pratt zeros, SciPy auto; event differences or nonempty-block means",
        limitations=["fixed seed1 weights, no training-seed uncertainty",
                     "event IID assumption; blocks require block-length sensitivity",
                     "calendar blocks cover first through last event; internal empty blocks retained; all-empty draws redrawn",
                     "SCEDC magnitude-threshold catalogs overlap; no pooling as independent repeats",
                     "Wilcoxon p=0 can indicate numerical underflow; see flag"],
        versions={p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scipy")})
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".paired-") as tmp:
        output = Path(tmp) / "output"
        output.mkdir()
        result.to_csv(output / "paired_significance.csv", index=False, encoding="utf-8-sig")
        (output / "analysis_settings.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        output.rename(destination)
    print(f"完成：{destination}", flush=True)


if __name__ == "__main__":
    main()
