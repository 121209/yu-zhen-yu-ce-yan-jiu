#!/usr/bin/env python3
"""实验 1B-2：按最近前序中强震的时间间隔和震中距离，切片比较似然增益。

依赖 numpy、pandas；无需 torch、权重或重新推理。示例：
  python analyze_aftershock_time_distance_slices.py
  python analyze_aftershock_time_distance_slices.py --include-etas --export-event-groups
  python analyze_aftershock_time_distance_slices.py --strong-quantile 0.95 --out results/slices_q95
  python analyze_aftershock_time_distance_slices.py --check-only

默认参考事件为本目录辅助期起点以来、严格早于目标事件的最近 M>=4 地震。
参考事件可在辅助/训练/验证/测试期，当前事件不充当自己的参考事件。
可改用训练期震级的 95% 分位阈值；阈值不使用测试期未来地震。
这是一种“距参考中强震的时距分组”，未作触发父子关系或独立主震序列识别。
重叠震群按时间上最近的合格参考地震重新计时。
时间阶段为 <1h、[1h,24h)、[1d,7d]、>7d。
距离使用经纬度球面震中距（地球半径 6371 km），不是带深度的震源距。
默认距离箱 [0,10)、[10,30)、[30,100)、>=100 km。
输出包括总体、时间切片、距离切片及时间×距离联合切片。
无前序参考事件单列，空分组保留 n=0 和空得分，不以零代替缺失值。
每事件得分从两种评估公共 CSV 列重构，统一区域一致分解口径。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import runpy
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATASETS = ("SCEDC_30", "SCEDC_25", "SCEDC_20", "SaltonSea", "SanJac", "WHITE", "ComCat")
MODELS = ("fcnn", "scheme1", "scheme2", "scheme3", "scheme4", "scheme5")
METRICS = ("LL_ts", "LL_t_region", "LL_s_given_t_region")
PHASES = ("lt_1h", "1h_to_lt_24h", "1d_to_7d_inclusive", "gt_7d")
EVENT_COLUMNS = ("event_id", "time", "log_lambda_ts", "log_lambda_t_region",
                 "log_conditional_space_region", "compensator_increment")


def load_events(repo, dataset, model):
    folder = "etas_decomposition" if model == "etas" else "decomposition"
    path = repo / "results" / folder / "events" / f"{dataset}_{model}_seed1_events.csv"
    frame = pd.read_csv(path, usecols=EVENT_COLUMNS).sort_values("event_id")
    if (len(frame) == 0 or not np.isfinite(frame[list(EVENT_COLUMNS)].to_numpy(dtype=float)).all()
            or not np.array_equal(frame.event_id.to_numpy(), np.arange(len(frame)))
            or np.any(np.diff(frame.time.to_numpy()) < 0)):
        raise ValueError(f"事件编号、顺序或数值异常：{path}")
    joint = frame.log_lambda_ts.to_numpy() - frame.compensator_increment.to_numpy()
    temporal = frame.log_lambda_t_region.to_numpy() - frame.compensator_increment.to_numpy()
    space = frame.log_conditional_space_region.to_numpy()
    if not np.allclose(joint, temporal + space, rtol=0, atol=1e-8):
        raise ValueError(f"区域分解恒等式未通过：{path}")
    return frame.time.to_numpy(), np.column_stack((joint, temporal, space)), hashlib.sha256(path.read_bytes()).hexdigest()


def read_catalog(repo, dataset, config):
    path = repo / config["catalog_path"]
    catalog = pd.read_csv(path)
    catalog["catalog_row"] = np.arange(len(catalog))
    if dataset.startswith("SCEDC_"):
        catalog = catalog.loc[catalog.magnitude >= float(dataset.split("_")[1]) / 10].copy()
    columns = ["time_days", "magnitude", "latitude", "longitude"]
    values = catalog[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(np.diff(catalog.time_days) < 0):
        raise ValueError(f"目录时间排序或数值异常：{path}")
    if (np.any(np.abs(catalog.latitude) > 90) or np.any(np.abs(catalog.longitude) > 180)):
        raise ValueError(f"经纬度异常：{path}")
    targets = catalog.loc[(catalog.time_days >= config["testing_start"])
                          & (catalog.time_days <= config["testing_end"])].copy()
    # 与原 prepare_run 的按计数切片再交叉核对，防止错配同一时刻的多个事件。
    t = catalog.time_days.to_numpy()
    start = sum(np.count_nonzero((t >= config[a]) & (t < config[b]))
                for a, b in (("auxiliary_start", "training_start"),
                             ("training_start", "validation_start"),
                             ("validation_start", "testing_start")))
    if not np.array_equal(catalog.iloc[start:start + len(targets)].catalog_row,
                          targets.catalog_row):
        raise ValueError(f"{dataset}: 官方按计数测试窗口与按日期窗口不一致")
    return catalog, targets.reset_index(drop=True), hashlib.sha256(path.read_bytes()).hexdigest()


def assign_reference_events(catalog, targets, threshold, auxiliary_start, distance_edges):
    """用原始 float64 时刻选参考事件，避免模型导出时刻的 float32 舍入错配。"""
    sources = catalog.loc[(catalog.magnitude >= threshold)
                          & (catalog.time_days >= auxiliary_start)].copy()
    target_t = targets.time_days.to_numpy(dtype=float)
    source_t = sources.time_days.to_numpy(dtype=float)
    index = np.searchsorted(source_t, target_t, side="left") - 1
    valid = index >= 0
    dt = np.full(len(targets), np.nan)
    distance = np.full(len(targets), np.nan)
    reference_row = np.full(len(targets), -1, dtype=np.int64)
    reference_time = np.full(len(targets), np.nan)
    reference_magnitude = np.full(len(targets), np.nan)
    if valid.any():
        refs = sources.iloc[index[valid]]
        dt[valid] = target_t[valid] - refs.time_days.to_numpy()
        reference_row[valid] = refs.catalog_row.to_numpy()
        reference_time[valid] = refs.time_days.to_numpy()
        reference_magnitude[valid] = refs.magnitude.to_numpy()
        lat1 = np.radians(targets.latitude.to_numpy()[valid])
        lat2 = np.radians(refs.latitude.to_numpy())
        delta_lon = np.radians(targets.longitude.to_numpy()[valid] - refs.longitude.to_numpy())
        h = np.sin((lat1-lat2)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(delta_lon/2)**2
        distance[valid] = 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
    phase = np.full(len(targets), -1, dtype=int)
    phase[valid] = np.select((dt[valid] < 1/24, dt[valid] < 1, dt[valid] <= 7),
                             (0, 1, 2), default=3)
    distance_bin = np.full(len(targets), -1, dtype=int)
    distance_bin[valid] = np.searchsorted(distance_edges, distance[valid], side="right")
    return pd.DataFrame(dict(
        event_id=np.arange(len(targets)), catalog_row=targets.catalog_row.to_numpy(),
        time_days_raw=target_t, magnitude=targets.magnitude.to_numpy(),
        reference_catalog_row=reference_row, reference_time_days=reference_time,
        reference_magnitude=reference_magnitude, elapsed_days=dt,
        epicentral_distance_km=distance, phase_code=phase, distance_bin_code=distance_bin))


def aggregate(dataset, assignments, model_scores, baseline, distance_labels):
    phase = assignments.phase_code.to_numpy()
    distance = assignments.distance_bin_code.to_numpy()
    groups = [("overall", "all", "all", np.ones(len(phase), dtype=bool))]
    groups += [("time", label, "all", phase == j) for j, label in enumerate(PHASES)]
    groups += [("distance", "all", label, distance == j) for j, label in enumerate(distance_labels)]
    groups += [("time_distance", time, dist, (phase == j) & (distance == k))
               for j, time in enumerate(PHASES) for k, dist in enumerate(distance_labels)]
    groups += [("no_prior_reference", "unassigned", "unassigned", phase == -1)]
    rows = []
    for kind, phase_label, distance_label, mask in groups:
        n = int(mask.sum())
        references = assignments.loc[mask, "reference_catalog_row"].to_numpy()
        for model, scores in model_scores.items():
            selected = scores[mask]
            diff = selected - model_scores[baseline][mask]
            row = dict(dataset=dataset, model=model, baseline=baseline, seed=1,
                       slice_type=kind, time_phase=phase_label, distance_bin=distance_label,
                       n_events=n, event_fraction=n / len(phase),
                       n_reference_events=len(np.unique(references[references >= 0])))
            for j, metric in enumerate(METRICS):
                row[metric + "_mean"] = float(selected[:, j].mean()) if n else np.nan
                row[metric + "_total"] = float(selected[:, j].sum()) if n else np.nan
                row["delta_" + metric + "_mean"] = float(diff[:, j].mean()) if n else np.nan
            row["joint_positive_event_fraction"] = float(np.mean(diff[:, 0] > 0)) if n else np.nan
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    strong = parser.add_mutually_exclusive_group()
    strong.add_argument("--strong-magnitude", type=float, default=4.0)
    strong.add_argument("--strong-quantile", type=float, help="仅训练期估计的震级分位阈值，例如 0.95")
    parser.add_argument("--distance-edges-km", nargs="+", type=float, default=[10, 30, 100])
    parser.add_argument("--baseline", choices=("fcnn", "etas"), default="fcnn")
    parser.add_argument("--include-etas", action="store_true")
    parser.add_argument("--export-event-groups", action="store_true", help="另存每个事件的参考事件和分组，供追溯")
    parser.add_argument("--out", type=Path, help="全新输出目录，默认 results/aftershock_time_distance_slices")
    parser.add_argument("--check-only", action="store_true", help="只检查配对和分组，不写文件")
    args = parser.parse_args()
    edges = np.asarray(args.distance_edges_km)
    if (not np.isfinite(edges).all() or np.any(edges <= 0) or np.any(np.diff(edges) <= 0)
            or not np.isfinite(args.strong_magnitude)
            or (args.strong_quantile is not None and not 0 < args.strong_quantile < 1)):
        parser.error("距离边界须为递增正数；震级阈值须有限；分位数须在 (0,1) 内")
    repo = args.repo.resolve()
    destination = (args.out or repo / "results/aftershock_time_distance_slices").resolve()
    if not args.check_only and destination.exists():
        parser.error(f"目录已存在，请通过 --out 另选目录：{destination}")
    configs = runpy.run_path(str(repo / "config.py"))["DATASET_CONFIGS"]
    models = MODELS + (("etas",) if args.include_etas or args.baseline == "etas" else ())
    distance_labels = [f"{a:g}_to_lt_{b:g}_km" for a, b in zip([0] + edges.tolist(), edges)]
    distance_labels.append(f"ge_{edges[-1]:g}_km")
    rows, audits, event_groups = [], {}, {}
    for dataset in dict.fromkeys(args.datasets):
        config = configs[dataset]
        catalog, targets, digest = read_catalog(repo, dataset, config)
        cutoff = args.strong_magnitude
        if args.strong_quantile is not None:
            train = catalog.loc[(catalog.time_days >= config["training_start"])
                                & (catalog.time_days < config["validation_start"]), "magnitude"]
            if train.empty:
                raise ValueError(f"{dataset}: 没有训练期事件用于估计阈值")
            cutoff = float(train.quantile(args.strong_quantile))
        assignments = assign_reference_events(catalog, targets, cutoff,
                                              config["auxiliary_start"], edges)
        scores, hashes = {}, {}
        exported_times = targets.time_days.to_numpy().astype(np.float32).astype(np.float64)
        for model in models:
            times, values, file_hash = load_events(repo, dataset, model)
            if len(times) != len(targets) or not np.allclose(times, exported_times, rtol=0, atol=1e-9):
                raise ValueError(f"{dataset}/{model}: 测试事件与原始目录的顺序不一致")
            scores[model], hashes[model] = values, file_hash
        assignments["time_days_exported"] = exported_times
        n_unassigned = int((assignments.phase_code < 0).sum())
        print(f"{dataset}: n={len(targets):,}, M阈值={cutoff:g}, 无前序参考={n_unassigned}", flush=True)
        audits[dataset] = dict(threshold_magnitude=cutoff, n_test=len(targets),
                               no_prior_reference=n_unassigned, catalog_sha256=digest,
                               events_sha256=hashes)
        if not args.check_only:
            rows.extend(aggregate(dataset, assignments, scores, args.baseline, distance_labels))
            if args.export_event_groups:
                event_groups[dataset] = assignments
    if args.check_only:
        print("全部输入和事件分组检查通过；没有生成结果文件。")
        return
    metadata = dict(
        arguments=vars(args), datasets=audits, unit="nats/event",
        phase_codes=dict(enumerate(PHASES)), distance_codes=dict(enumerate(distance_labels)),
        assignment="latest strictly earlier qualifying event since auxiliary_start; same-time events excluded",
        distance="great-circle epicentral distance, Earth radius 6371 km; no depth used",
        missing="phase/distance code -1, reference_catalog_row -1 means no prior reference",
        limitations=["descriptive slices, no parent-offspring identification",
                     "group events are clustered, n_events is not independent replication",
                     "training-only magnitude quantiles can include more than the top 5% because of ties",
                     "strong events themselves may be scored relative to an earlier reference"],
        versions={p: importlib.metadata.version(p) for p in ("numpy", "pandas")})
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".slices-") as tmp:
        output = Path(tmp) / "output"
        output.mkdir()
        pd.DataFrame(rows).to_csv(output / "time_distance_slices.csv", index=False, encoding="utf-8-sig")
        (output / "analysis_settings.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        if args.export_event_groups:
            (output / "event_groups").mkdir()
            for dataset, assignments in event_groups.items():
                assignments.to_csv(output / "event_groups" / f"{dataset}_event_groups.csv",
                                   index=False, encoding="utf-8-sig")
        output.rename(destination)
    print(f"完成：{destination}", flush=True)


if __name__ == "__main__":
    main()
