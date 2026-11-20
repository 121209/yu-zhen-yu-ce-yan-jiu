#!/usr/bin/env python3
"""Export paired, per-event ETAS test scores from the aligned H128/seed1 weights.

Run in the archived ST-NKF repository with Python 3.10+ and a PyTorch
environment that can import its models.py. Uses the original CPU protocol.
Examples:
    python eval_etas_ll_decomposition.py
    python eval_etas_ll_decomposition.py --datasets SanJac --out results/etas_sanjac
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DATASETS = ("SCEDC_30", "SCEDC_25", "SCEDC_20", "SaltonSea", "SanJac", "WHITE", "ComCat")
MEAN_ATOL = 5e-6  # nats/event; original reference uses float32 mean then multiplies by n
IDENTITY_ATOL = 1e-5  # nats/event, including the original float32 joint score


def evaluate(dataset, record, manifest, events_dir, batch_size):
    from scripts.run_etas_h128_aligned_cpu import build_model, prepare_run
    from eval_ll_decomposition import predict_split_with_region

    ref = record["test"]
    weight = Path("weights", "etas", f"etas_{dataset}_H128_seed1_aligned_best.pt")
    digest = hashlib.sha256(weight.read_bytes()).hexdigest()
    if digest != manifest[weight.as_posix()]:
        raise ValueError(f"{dataset}: checkpoint differs from the repository manifest")
    if (record["status"], record["history"], record["seed"], record["boundary_angles"],
            record["boundary_correction"]) != ("ok", 128, 1, 128, True):
        raise ValueError(f"{dataset}: incompatible ETAS reference protocol")

    prep = prepare_run(dataset, history=128)
    if prep["input_dim_test"] != 128 or prep["testing_num"] != ref["n"]:
        raise ValueError(f"{dataset}: test split or history differs from reference")
    model = build_model(prep)
    model.load_weights(str(weight))
    model.eval()
    mu = float(model.bac_func.mu.detach())
    if not np.isclose(mu, record["mu"], rtol=1e-6, atol=0):
        raise ValueError(f"{dataset}: loaded background rate differs from reference")

    # ETAS also differentiates its temporal CDF in forward(). Keep autograd enabled.
    predictions = predict_split_with_region(model, "test", batch_size)
    model._store_eval("test", predictions)
    def vector(values):
        return np.asarray(values, dtype=np.float64).reshape(-1)

    times = vector(model.cur_t_test)
    n = len(times)
    ids = np.arange(n)
    # Compare the actual target order, not just the number of earthquakes.
    baseline = ROOT / "results/decomposition/events" / f"{dataset}_fcnn_seed1_events.csv"
    baseline_events = np.loadtxt(baseline, delimiter=",", skiprows=1, usecols=(0, 1), ndmin=2)
    if (n != ref["n"] or baseline_events.shape != (n, 2)
            or not np.array_equal(ids, baseline_events[:, 0])
            or not np.allclose(times, baseline_events[:, 1], rtol=0, atol=1e-9)):
        raise ValueError(f"{dataset}: ETAS and FCNN test events are not aligned")

    joint, temporal, regional = (vector(predictions[k]) for k in ("l_ts", "l_t", "l_t_region"))
    comp = vector(predictions["int_l"])
    if not all(a.size == n and np.isfinite(a).all() for a in (joint, temporal, regional, comp)):
        raise ValueError(f"{dataset}: non-finite or incomplete predictions")
    if min(joint.min(), temporal.min(), regional.min()) <= 0:
        raise ValueError(f"{dataset}: non-positive intensity")
    log_joint, log_time, log_region = map(np.log, (joint, temporal, regional))
    # Preserve official float32 scoring for the first three likelihood columns.
    scores = {
        "LL_ts": vector(model.LL_ts_test),
        "LL_t": vector(model.LL_t_test),
        "LL_s_given_t": vector(model.LL_s_test),
        "LL_t_region": log_region - comp,
        "LL_s_given_t_region": log_joint - log_region,
    }
    row = dict(dataset=dataset, model="etas", seed=1, checkpoint_path=weight.as_posix(),
               n_test=n, mu_loaded=mu, old_LL_ts=ref["LL_ts_total"])
    for name, values in scores.items():
        row[name + "_total"] = float(values.sum())
        row[name + "_mean"] = float(values.mean())
    for name, key in (("LL_ts", "LL_ts"), ("LL_t", "LL_t"), ("LL_s_given_t", "LL_s")):
        delta = row[name + "_mean"] - ref[key]
        if not np.isfinite(delta) or abs(delta) > MEAN_ATOL:
            raise ValueError(f"{dataset}: {name} differs from reference by {delta:.8g} nats/event")
    row["reproduction_difference"] = row["LL_ts_total"] - ref["LL_ts_total"]
    if abs(row["reproduction_difference"]) > n * MEAN_ATOL:
        raise ValueError(f"{dataset}: joint total differs from the archived total")
    for suffix in ("", "_region"):
        residual = scores["LL_ts"] - scores["LL_t" + suffix] - scores["LL_s_given_t" + suffix]
        error = float(np.max(np.abs(residual)))
        if not np.isfinite(error) or error > IDENTITY_ATOL:
            raise ValueError(f"{dataset}: decomposition{suffix} residual {error:.8g}")
        row["decomposition_error" + suffix] = error

    # Keep the original eight columns first; append ready-to-use per-event LL.
    table = {
        "event_id": ids, "time": times, "log_lambda_ts": log_joint,
        "log_lambda_t": log_time, "log_conditional_space": log_joint - log_time,
        "log_lambda_t_region": log_region, "log_conditional_space_region": log_joint - log_region,
        "compensator_increment": comp, **scores,
    }
    with (events_dir / f"{dataset}_etas_seed1_events.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(table)
        writer.writerows(zip(*table.values()))
    print(f"{dataset}: n={n}, LL_ts={row['LL_ts_total']:.6f}, "
          f"reference difference={row['reproduction_difference']:+.6g}", flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--out", type=Path, default=ROOT / "results/etas_decomposition",
                        help="New output directory; existing results are protected")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    dest = args.out.resolve()
    if dest.exists():
        parser.error(f"Output already exists: {dest}. Choose a fresh --out directory.")
    os.chdir(ROOT)  # prepare_run resolves the catalog and polygon relative to the repository.
    records = json.loads(Path("results/etas-h128-aligned-ll.json").read_text(encoding="utf-8"))["datasets"]
    with Path("results/MANIFEST.sha256.csv").open(encoding="utf-8", newline="") as f:
        manifest = {row["path"]: row["sha256"] for row in csv.DictReader(f)}
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent, prefix=".etas-eval-") as tmp:
        output = Path(tmp) / "output"
        events = output / "events"
        events.mkdir(parents=True)
        rows = [evaluate(ds, records[ds], manifest, events, args.batch_size)
                for ds in dict.fromkeys(args.datasets)]
        with (output / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        output.rename(dest)
    print(f"Verified {len(rows)} dataset(s). Results: {dest}", flush=True)


if __name__ == "__main__":
    main()