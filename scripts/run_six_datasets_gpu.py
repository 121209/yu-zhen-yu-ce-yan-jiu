#!/usr/bin/env python3
"""GPU training: four ST-NKF schemes on the six remaining EarthquakeNPP datasets.

Protocol is identical to the SanJac RTX 4090 50-epoch run, applied independently
to each dataset with that dataset's official config.py splits. On a given
dataset the four schemes share the same input data (official jitter catalog,
official auxiliary/train/val/test windows, H=128, batch=16, seed=1) and the
same training method (lr=1e-4, 50 epochs, patience=0, mu rate/area init then
analytic update, 128-angle boundary correction, evaluation on best-val weights).

Datasets (small first, so a limited balance finishes complete catalogs):
  SCEDC_30, SCEDC_25, WHITE, SaltonSea, ComCat, SCEDC_20.
SanJac is skipped (already completed).

In-container shutdown is unreliable on AutoDL (no systemd). This script writes
a DONE marker when it finishes or hits the budget; the operator must 关机
on the AutoDL console.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Import the SanJac GPU trainer for shared scheme definitions and the
# checkpointed training loop. That module locates the repo and chdir's.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from run_four_schemes_gpu import (  # noqa: E402
    BATCH_SIZE,
    BOUNDARY_ANGLES,
    HISTORY,
    LR,
    RUN_ORDER,
    SCHEMES,
    SEED,
    Budget,
    now_label,
    run_scheme,
    torch,
)
from run_models import configure_device, prepare_run  # noqa: E402

# Smallest first so a short balance still yields complete dataset results.
DATASETS = ("SCEDC_30", "SCEDC_25", "WHITE", "SaltonSea", "ComCat", "SCEDC_20")

CATALOGS = {
    "ComCat": "Datasets/ComCat/ComCat_catalog_jitter.csv",
    "SaltonSea": "Datasets/QTM/SaltonSea_catalog_jitter.csv",
    "WHITE": "Datasets/WHITE/WHITE_catalog_jitter.csv",
    "SCEDC_20": "Datasets/SCEDC/SCEDC_catalog_jitter.csv (m ≥ 2.0)",
    "SCEDC_25": "Datasets/SCEDC/SCEDC_catalog_jitter.csv (m ≥ 2.5)",
    "SCEDC_30": "Datasets/SCEDC/SCEDC_catalog_jitter.csv (m ≥ 3.0)",
}

WINDOWS = {
    "ComCat": "aux 1971-01-01, train 1981-01-01, val 1998-01-01, test 2007-01-01–2020-01-17, m0=2.5",
    "SaltonSea": "aux 2008-01-01, train 2009-01-01, val 2014-01-01, test 2016-01-01–2018-01-01, m0=1.0",
    "WHITE": "aux 2008-01-01, train 2009-01-01, val 2014-01-01, test 2017-01-01–2021-01-01, m0=0.6",
    "SCEDC_20": "aux 1981-01-01, train 1985-01-01, val 2005-01-01, test 2014-01-01–2020-01-01, m0=2.0",
    "SCEDC_25": "aux 1981-01-01, train 1985-01-01, val 2005-01-01, test 2014-01-01–2020-01-01, m0=2.5",
    "SCEDC_30": "aux 1981-01-01, train 1985-01-01, val 2005-01-01, test 2014-01-01–2020-01-01, m0=3.0",
}


def split_info(prep: dict) -> dict:
    return {
        "auxiliary_num": int(prep["auxiliary_num"]),
        "training_num": int(prep["training_num"]),
        "validation_num": int(prep["validation_num"]),
        "testing_num": int(prep["testing_num"]),
        "area": float(prep["area"]),
        "global_m0": float(prep["config"]["global_m0"]),
        "history": int(prep["history"]),
        "input_dim_train": int(prep["input_dim_train"]),
        "input_dim_val": int(prep["input_dim_val"]),
        "input_dim_test": int(prep["input_dim_test"]),
        "windows": WINDOWS[prep_dataset_name(prep)],
    }


def prep_dataset_name(prep: dict) -> str:
    path = str(prep["config"]["catalog_path"])
    m0 = float(prep["config"]["global_m0"])
    if "ComCat" in path:
        return "ComCat"
    if "SaltonSea" in path:
        return "SaltonSea"
    if "WHITE" in path:
        return "WHITE"
    if "SCEDC" in path:
        return {2.0: "SCEDC_20", 2.5: "SCEDC_25", 3.0: "SCEDC_30"}[m0]
    raise KeyError(path)


def write_master(master: dict, json_path: Path, md_path: Path) -> None:
    json_path.write_text(json.dumps(master, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Six-dataset GPU training (four schemes each)",
        "",
        f"Updated: {now_label()}",
        "",
        "## Protocol (same on every dataset, matching the SanJac 4090 run)",
        "",
        f"- H={HISTORY}, batch={BATCH_SIZE}, lr={LR}, seed={SEED}, "
        f"boundary_angles={BOUNDARY_ANGLES}, epochs={master['protocol']['epochs']}, "
        f"patience={master['protocol']['patience']} (disabled).",
        "- Official EarthquakeNPP / config.py splits; NKF jitter catalog, no extra train-time jitter.",
        "- μ initialized from train event rate / area, analytic update after each epoch.",
        "- Evaluation always restores the best-validation snapshot.",
        f"- Device: {master['protocol'].get('device', '?')}. "
        f"Scheme order: {' → '.join(master['protocol'].get('scheme_order', RUN_ORDER))}.",
        f"- Dataset order: {' → '.join(master['protocol']['dataset_order'])}.",
        f"- Time budget: {master['protocol'].get('budget_hours', 0)} hours "
        "(0 means no limit). After this process exits, shut down on the AutoDL console.",
        "",
        "| Dataset | Scheme | Status | VAL LL_ts | TEST LL_ts | TEST total LL_ts | Best ep | Seconds |",
        "|---|---|---|---:|---:|---:|---|---:|",
    ]
    for ds in master["protocol"]["dataset_order"]:
        block = master["datasets"].get(ds, {})
        schemes = block.get("schemes", {})
        for sid in master["protocol"].get("scheme_order", RUN_ORDER):
            rec = schemes.get(sid)
            if not rec:
                st = block.get("status", "pending")
                lines.append(f"| {ds} | {sid} | {st} |  |  |  |  |  |")
                continue
            if rec.get("status") in {"ok", "partial"} and "test" in rec:
                n = rec["test"]["n"]
                total = rec["test"]["LL_ts"] * n
                lines.append(
                    "| {ds} | {sid} | {st} | {v:.4f} | {t:.4f} | {tot:.2f} | {ep} | {sec:.1f} |".format(
                        ds=ds, sid=sid, st=rec["status"],
                        v=rec["val"]["LL_ts"], t=rec["test"]["LL_ts"],
                        tot=total, ep=rec["best_epoch"], sec=rec["seconds"],
                    )
                )
            else:
                lines.append(
                    f"| {ds} | {sid} | {rec.get('status')} |  |  |  |  | {rec.get('error', rec.get('stop_reason', ''))} |"
                )
    lines.append("")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train four ST-NKF schemes on the remaining EarthquakeNPP datasets."
    )
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument(
        "--budget-hours",
        type=float,
        default=0,
        help="Wall-clock budget from process start. 0 means no limit. "
             "Set this from remaining AutoDL balance minus a reserve.",
    )
    p.add_argument("--gpu-id", default="0")
    p.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        choices=list(DATASETS),
        help="Subset and order of datasets. Default: all six, smallest first.",
    )
    p.add_argument(
        "--schemes",
        nargs="+",
        default=None,
        choices=list(SCHEMES),
        help="Schemes on each dataset. Default: all four in RUN_ORDER.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = args.out or Path(os.environ.get("STNKF_GPU_OUT", "/root/autodl-tmp/st-nkf-gpu-six"))
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "six-datasets-gpu-ll.json"
    status_path = out / "six-datasets-gpu-status.md"
    log_path = out / "six-datasets-gpu-training.log"
    done_path = out / "DONE"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    dataset_order = list(args.datasets) if args.datasets else list(DATASETS)
    run_list = list(args.schemes) if args.schemes else list(RUN_ORDER)

    configure_device(args.gpu_id)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; refusing to run the GPU training script on CPU.")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda")

    logging.info(
        "Six-dataset GPU training start device=%s name=%s torch=%s cuda=%s out=%s "
        "budget=%.2fh datasets=%s schemes=%s",
        device, torch.cuda.get_device_name(0), torch.__version__, torch.version.cuda,
        out, args.budget_hours, ",".join(dataset_order), ",".join(run_list),
    )

    master = {
        "protocol": {
            "datasets_requested": dataset_order,
            "dataset_order": dataset_order,
            "scheme_order": run_list,
            "history": HISTORY,
            "batch_size": BATCH_SIZE,
            "epochs": args.epochs,
            "patience": args.patience,
            "lr": LR,
            "seed": SEED,
            "boundary_angles": BOUNDARY_ANGLES,
            "mu_init": "mean train event rate / area, then analytic epoch update",
            "evaluated_weights": "best-validation snapshot, always restored",
            "same_controls": (
                "On each dataset the four schemes share the official jitter catalog, "
                "official config.py windows, H, batch, seed, lr, epoch cap, patience, "
                "boundary correction, and mu update rule."
            ),
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": str(torch.version.cuda),
            "budget_hours": args.budget_hours,
            "started": now_label(),
        },
        "datasets": {},
    }
    if json_path.is_file():
        previous = json.loads(json_path.read_text(encoding="utf-8"))
        master["datasets"] = previous.get("datasets", {})
        master["protocol"]["resumed_from"] = previous.get("protocol", {}).get("started")

    budget = Budget(args.budget_hours)
    budget_hit = False
    failed = False

    for ds in dataset_order:
        ds_out = out / ds
        ds_out.mkdir(parents=True, exist_ok=True)
        weights = ds_out / "weights"
        weights.mkdir(exist_ok=True)
        ds_json = ds_out / "four-schemes-gpu-ll.json"

        existing_block = master["datasets"].get(ds, {})
        if existing_block.get("status") == "ok":
            schemes = existing_block.get("schemes", {})
            if all(schemes.get(s, {}).get("status") == "ok" for s in run_list):
                logging.info("skip completed dataset %s", ds)
                continue

        if not budget.allow_another_epoch(30.0):
            logging.info("time budget exhausted before starting %s", ds)
            budget_hit = True
            if ds not in master["datasets"]:
                master["datasets"][ds] = {"status": "skipped_budget", "finished": now_label()}
            write_master(master, json_path, status_path)
            break

        logging.info("===== DATASET %s =====", ds)
        try:
            prep = prepare_run(ds, history=HISTORY)
        except Exception as exc:
            logging.exception("prepare_run(%s) failed", ds)
            master["datasets"][ds] = {
                "status": "failed",
                "error": repr(exc),
                "finished": now_label(),
            }
            failed = True
            write_master(master, json_path, status_path)
            continue

        info = {
            "auxiliary_num": int(prep["auxiliary_num"]),
            "training_num": int(prep["training_num"]),
            "validation_num": int(prep["validation_num"]),
            "testing_num": int(prep["testing_num"]),
            "area": float(prep["area"]),
            "global_m0": float(prep["config"]["global_m0"]),
            "catalog": CATALOGS[ds],
            "windows": WINDOWS[ds],
            "history": HISTORY,
        }
        logging.info(
            "%s aux=%d train=%d val=%d test=%d m0=%s area=%.4f",
            ds, info["auxiliary_num"], info["training_num"],
            info["validation_num"], info["testing_num"],
            info["global_m0"], info["area"],
        )
        if int(prep["input_dim_train"]) != HISTORY:
            raise RuntimeError(f"{ds} input_dim_train={prep['input_dim_train']} != H={HISTORY}")

        data = {
            "protocol": {
                "dataset": ds,
                "catalog": CATALOGS[ds],
                "splits": WINDOWS[ds],
                "counts": info,
                "history": HISTORY,
                "batch_size": BATCH_SIZE,
                "epochs": args.epochs,
                "patience": args.patience,
                "lr": LR,
                "seed": SEED,
                "boundary_angles": BOUNDARY_ANGLES,
                "mu_init": "mean train event rate / area, then analytic epoch update",
                "evaluated_weights": "best-validation snapshot, always restored",
                "run_order": run_list,
                "device": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "cuda": str(torch.version.cuda),
                "started": now_label(),
            },
            "schemes": existing_block.get("schemes", {}),
        }
        if ds_json.is_file() and not data["schemes"]:
            prev = json.loads(ds_json.read_text(encoding="utf-8"))
            data["schemes"] = prev.get("schemes", {})

        for sid in run_list:
            existing = data["schemes"].get(sid)
            if existing and existing.get("status") == "ok":
                logging.info("skip completed %s/%s", ds, sid)
                continue
            if not budget.allow_another_epoch(30.0):
                logging.info("time budget exhausted before starting %s/%s", ds, sid)
                budget_hit = True
                if not existing:
                    data["schemes"][sid] = {
                        "status": "skipped_budget",
                        "name": SCHEMES[sid]["name"],
                        "finished": now_label(),
                    }
                break
            try:
                rec = run_scheme(
                    prep, SCHEMES[sid], device, ds_out, weights, budget,
                    epochs=args.epochs, patience=args.patience,
                )
                if rec.get("stop_reason") == "time_budget":
                    budget_hit = True
            except Exception as exc:
                logging.exception("%s/%s failed", ds, sid)
                rec = {
                    "status": "failed",
                    "name": SCHEMES[sid]["name"],
                    "flags": SCHEMES[sid]["flags"],
                    "error": repr(exc),
                    "finished": now_label(),
                }
                failed = True
            data["schemes"][sid] = rec
            ds_json.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            all_ok = all(data["schemes"].get(s, {}).get("status") == "ok" for s in run_list)
            master["datasets"][ds] = {
                "status": "ok" if all_ok else ("partial" if data["schemes"] else "pending"),
                "counts": info,
                "schemes": data["schemes"],
                "finished": now_label(),
            }
            write_master(master, json_path, status_path)
            if budget_hit:
                break

        data["protocol"]["finished"] = now_label()
        data["protocol"]["elapsed_seconds"] = budget.elapsed()
        ds_json.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        all_ok = all(data["schemes"].get(s, {}).get("status") == "ok" for s in run_list)
        master["datasets"][ds] = {
            "status": "ok" if all_ok else ("partial" if any(
                data["schemes"].get(s, {}).get("status") == "ok" for s in run_list
            ) else data.get("schemes", {}) and list(data["schemes"].values())[-1].get("status", "partial") or "pending"),
            "counts": info,
            "schemes": data["schemes"],
            "finished": now_label(),
        }
        write_master(master, json_path, status_path)
        if budget_hit:
            break

    master["protocol"]["finished"] = now_label()
    master["protocol"]["elapsed_seconds"] = budget.elapsed()
    all_done = all(
        master["datasets"].get(ds, {}).get("status") == "ok" for ds in dataset_order
    )
    outcome = "completed" if all_done else ("budget" if budget_hit else ("failed" if failed else "partial"))
    master["protocol"]["outcome"] = outcome
    write_master(master, json_path, status_path)
    done_path.write_text(
        f"{now_label()}\noutcome={outcome}\nelapsed={budget.elapsed():.1f}s\n",
        encoding="utf-8",
    )
    logging.info(
        "Six-dataset training finished outcome=%s elapsed=%.1fs results=%s",
        outcome, budget.elapsed(), json_path,
    )
    # Do not attempt in-container shutdown. Write the marker and exit so
    # the operator can 关机 on the AutoDL console.


if __name__ == "__main__":
    main()
