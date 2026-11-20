#!/usr/bin/env python3
"""Export independent background-rate diagnostics from existing
official LL-decomposition outputs.

Does not train, does not reload weights, and does not recompute
log-likelihoods. Reads μ, area A, the compensator total Λ_total,
the compensator-covered interval ΔT_test, and log λ(t_k, s_k) from
results/decomposition/, then writes:

    P_bg = μ A ΔT_test / Λ_total
    P_trig = 1 - P_bg
    p_bg_mean = (1/N_test) Σ_k μ / λ(t_k, s_k)

ΔT_test is the compensator-covered interval (sum of inter-event
gaps), not official testing_end - testing_start. λ(t_k, s_k) is
recovered as exp(log_lambda_ts) from the event files.

Example:

    python eval_background_rate.py
    python eval_background_rate.py --decomp-dir results/decomposition --out results/background-rate
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent

DATASETS = (
    "SCEDC_30",
    "SCEDC_25",
    "SCEDC_20",
    "SaltonSea",
    "SanJac",
    "WHITE",
    "ComCat",
)

MODEL_ORDER = ("fcnn", "scheme3", "scheme2", "scheme1", "scheme4", "scheme5")

SEED = 1

SUMMARY_FIELDS = [
    "dataset",
    "model",
    "seed",
    "n_test",
    "mu_loaded",
    "area",
    "delta_t_test",
    "Lambda_total",
    "expected_background_count",
    "P_bg",
    "P_trig",
    "p_bg_mean",
    "events_csv",
    "source_meta",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--decomp-dir",
        type=Path,
        default=ROOT / "results" / "decomposition",
        help="Official 42-run decomposition directory (not modified)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "background-rate",
        help="New output directory; official decomposition files are not overwritten",
    )
    p.add_argument("--datasets", nargs="+", default=None, choices=list(DATASETS))
    p.add_argument("--models", nargs="+", default=None, choices=list(MODEL_ORDER))
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def _stem(dataset: str, model_id: str, seed: int) -> str:
    return f"{dataset}_{model_id}_seed{seed}"


def _require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"missing official decomposition file: {path}")
    return path


def _area_from_meta(meta: dict, label: str) -> float:
    notes = meta.get("normalization") or {}
    area_notes = notes.get("area") or {}
    for key in ("used", "checkpoint", "recomputed_from_data"):
        if area_notes.get(key) is not None:
            return float(area_notes[key])
    raise KeyError(f"{label}: meta.json has no normalization.area")


def evaluate_one(decomp_dir: Path, out: Path, dataset: str, model_id: str, seed: int) -> dict:
    events_dir = decomp_dir / "events"
    stem = _stem(dataset, model_id, seed)
    meta_path = _require(events_dir / f"{stem}_meta.json")
    ev_path = _require(events_dir / f"{stem}_events.csv")
    comp_path = _require(events_dir / f"{stem}_compensator.json")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    compensator = json.loads(comp_path.read_text(encoding="utf-8"))

    mu = float(meta["mu_loaded"])
    area = _area_from_meta(meta, f"{dataset}/{model_id}")
    n_meta = int(meta["n_test"])
    delta_t_test = float(compensator["sum_elapsed_interevent"])
    lambda_total = float(compensator["sum_compensator_increments"])

    log_lam_ts = []
    times = []
    event_ids = []
    increments = []
    with ev_path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "log_lambda_ts" not in reader.fieldnames:
            raise KeyError(f"{ev_path} has no log_lambda_ts column")
        for rec in reader:
            event_ids.append(int(rec["event_id"]))
            times.append(float(rec["time"]))
            log_lam_ts.append(float(rec["log_lambda_ts"]))
            increments.append(float(rec["compensator_increment"]))

    log_lam_ts = np.asarray(log_lam_ts, dtype=np.float64)
    increments = np.asarray(increments, dtype=np.float64)
    n = int(log_lam_ts.size)
    if n != n_meta:
        raise RuntimeError(f"{dataset}/{model_id}: events n={n} != meta n_test={n_meta}")
    if int(compensator["n_test"]) != n:
        raise RuntimeError(
            f"{dataset}/{model_id}: compensator n_test={compensator['n_test']} != events n={n}"
        )
    if not np.isfinite(log_lam_ts).all():
        raise FloatingPointError(f"{dataset}/{model_id}: non-finite log_lambda_ts")

    lam_ts = np.exp(log_lam_ts)
    if not np.isfinite(lam_ts).all() or np.any(lam_ts <= 0):
        raise FloatingPointError(
            f"{dataset}/{model_id}: non-positive λ(t,s) from exp(log_lambda_ts)"
        )

    inc_sum = float(increments.sum())
    if not math.isclose(inc_sum, lambda_total, rel_tol=0, abs_tol=1e-6 * max(1.0, abs(lambda_total))):
        raise RuntimeError(
            f"{dataset}/{model_id}: sum(compensator_increment)={inc_sum} "
            f"!= sum_compensator_increments={lambda_total}"
        )

    expected_background_count = mu * area * delta_t_test
    p_bg = (
        expected_background_count / lambda_total
        if lambda_total > 0 and np.isfinite(lambda_total)
        else float("nan")
    )
    p_trig = 1.0 - p_bg
    p_bg_event = mu / lam_ts
    p_bg_mean = float(np.mean(p_bg_event))

    out_events = out / "events"
    out_events.mkdir(parents=True, exist_ok=True)
    out_ev_path = out_events / f"{stem}_background.csv"
    with out_ev_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["event_id", "time", "lambda_ts", "p_bg_posterior"])
        for i in range(n):
            w.writerow(
                [
                    event_ids[i],
                    f"{times[i]:.16g}",
                    f"{lam_ts[i]:.16g}",
                    f"{p_bg_event[i]:.16g}",
                ]
            )

    row = {
        "dataset": dataset,
        "model": model_id,
        "seed": seed,
        "n_test": n,
        "mu_loaded": mu,
        "area": area,
        "delta_t_test": delta_t_test,
        "Lambda_total": lambda_total,
        "expected_background_count": expected_background_count,
        "P_bg": p_bg,
        "P_trig": p_trig,
        "p_bg_mean": p_bg_mean,
        "events_csv": str(out_ev_path),
        "source_meta": str(meta_path),
    }
    print(
        f"{dataset}/{model_id}: n={n} mu={mu:.6e} P_bg={p_bg:.6f} "
        f"p_bg_mean={p_bg_mean:.6f}",
        flush=True,
    )
    return row


def write_summary(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> int:
    args = parse_args()
    decomp_dir = args.decomp_dir.expanduser().resolve()
    out = args.out.expanduser().resolve()
    if not decomp_dir.is_dir():
        raise FileNotFoundError(
            f"official decomposition directory not found: {decomp_dir}"
        )
    out.mkdir(parents=True, exist_ok=True)
    datasets = list(args.datasets) if args.datasets else list(DATASETS)
    models = list(args.models) if args.models else list(MODEL_ORDER)

    rows = []
    missing: list[str] = []
    for dataset in datasets:
        for model_id in models:
            try:
                rows.append(
                    evaluate_one(decomp_dir, out, dataset, model_id, args.seed)
                )
            except Exception as exc:
                msg = f"{dataset}/{model_id}: {type(exc).__name__}: {exc}"
                print("ERROR", msg, flush=True)
                missing.append(msg)

    summary_path = out / "summary.csv"
    write_summary(summary_path, rows)
    (out / "missing.txt").write_text(
        ("\n".join(missing) + "\n") if missing else "",
        encoding="utf-8",
    )
    print(
        f"wrote {summary_path} rows={len(rows)} missing={len(missing)}",
        flush=True,
    )
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())