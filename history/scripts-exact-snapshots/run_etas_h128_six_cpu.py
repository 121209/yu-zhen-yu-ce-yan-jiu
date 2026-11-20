#!/usr/bin/env python3
"""ETAS (empirical kernels) on official EarthquakeNPP splits, H=128, CPU.

Protocol matches the existing SanJac H=128 full run:
  official jitter catalog and config.py windows, H=128, batch 16, lr=0.005,
  seed 1, epoch cap 1000, patience 20, mu left at 1, no boundary correction.

SanJac is reused from the earlier completed run and is not retrained.
Remaining datasets, smallest first:
  SCEDC_30, SaltonSea, SCEDC_25, SCEDC_20, WHITE, ComCat.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np
import torch

torch.set_num_threads(4)

_REPO = Path("/workspace/ST-NKF-pytorch")
sys.path.insert(0, str(_REPO))
os.chdir(_REPO)

from models import KernelPointProcess
from run_models import prepare_run

SHANGHAI = timezone(timedelta(hours=8))
OUT = Path("/workspace/st-nkf-diagrams/st-nkf-etas-h128-cpu-20261114")
LOG_PATH = OUT / "etas-h128-cpu-training.log"
JSON_PATH = OUT / "etas-h128-cpu-ll.json"
WEIGHTS = OUT / "weights"
SANJAC_PRIOR = Path("/workspace/st-nkf-diagrams/etas-h128-full-result.json")

HISTORY = 128
BATCH_SIZE = 16
EPOCHS = 1000
PATIENCE = 20
LR = 0.005
SEED = 1
MU_BATCH = 256

# Official test counts used in the comparison table; assert after prepare_run.
OFFICIAL_N = {
    "SanJac": {"train": 11770, "val": 3449, "test": 4400},
    "SCEDC_30": {"train": None, "val": None, "test": 1898},
    "SCEDC_25": {"train": None, "val": None, "test": 5000},
    "SCEDC_20": {"train": None, "val": None, "test": 13067},
    "WHITE": {"train": None, "val": None, "test": 24080},
    "SaltonSea": {"train": None, "val": None, "test": 4104},
    "ComCat": {"train": 40701, "val": 14741, "test": 21885},
}

DATASETS = ("SCEDC_30", "SaltonSea", "SCEDC_25", "SCEDC_20", "WHITE", "ComCat")


def now_label() -> str:
    return datetime.now(SHANGHAI).strftime("%Y-%m-%d %H:%M:%S Asia/Shanghai")


def ll_block(model, split: str, n: int) -> dict:
    if split == "val":
        return {
            "LL_ts": float(model.LL_ts_average_val),
            "LL_t": float(model.LL_t_average_val),
            "LL_s": float(model.LL_s_average_val),
            "n": int(n),
            "LL_ts_total": float(model.LL_ts_average_val) * int(n),
        }
    return {
        "LL_ts": float(model.LL_ts_average_test),
        "LL_t": float(model.LL_t_average_test),
        "LL_s": float(model.LL_s_average_test),
        "n": int(n),
        "LL_ts_total": float(model.LL_ts_average_test) * int(n),
    }


def new_master() -> dict:
    return {
        "protocol": {
            "model": "ETAS (empirical kernels), H=128",
            "catalog": "official EarthquakeNPP jitter catalogs, no extra train-time jitter",
            "splits": "official config.py aux/train/val/test windows",
            "history": HISTORY,
            "batch_size": BATCH_SIZE,
            "epoch_cap": EPOCHS,
            "patience": PATIENCE,
            "lr": LR,
            "seed": SEED,
            "mu_batch_size": MU_BATCH,
            "upgrades_applied": False,
            "mu_init": "left at 1 (paper path)",
            "boundary_correction": False,
            "evaluated_weights": "best-validation snapshot from fit_eval",
            "device": "cpu",
            "torch_threads": 4,
            "dataset_order": ["SanJac (reused)", *DATASETS],
            "started": now_label(),
        },
        "datasets": {},
    }


def load_master() -> dict:
    if JSON_PATH.is_file():
        data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
        data.setdefault("protocol", new_master()["protocol"])
        data.setdefault("datasets", {})
        return data
    return new_master()


def save_master(data: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    tmp = JSON_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(JSON_PATH)


def reuse_sanjac(data: dict) -> None:
    if data.get("datasets", {}).get("SanJac", {}).get("status") == "ok":
        return
    prior = json.loads(SANJAC_PRIOR.read_text(encoding="utf-8"))
    n_val = 3449
    n_test = 4400
    val = dict(prior["val"])
    test = dict(prior["test"])
    val["n"] = n_val
    test["n"] = n_test
    val["LL_ts_total"] = float(val["LL_ts"]) * n_val
    test["LL_ts_total"] = float(test["LL_ts"]) * n_test
    data.setdefault("datasets", {})["SanJac"] = {
        "status": "ok",
        "name": "ETAS (empirical kernels)",
        "reused": True,
        "source": str(SANJAC_PRIOR),
        "dataset": "SanJac",
        "official_splits": True,
        "history": 128,
        "batch_size": 16,
        "epoch_cap": 1000,
        "patience": 20,
        "lr": 0.005,
        "seed": 1,
        "mu_batch_size": 256,
        "upgrades_applied": False,
        "best_epoch": prior["best_epoch"],
        "stopped_early": prior["stopped_early"],
        "epochs_ran": prior["epochs_ran"],
        "best_val_loss": prior["best_val_loss"],
        "mu": prior["mu"],
        "n_train": 11770,
        "n_val": n_val,
        "n_test": n_test,
        "val": val,
        "test": test,
        "seconds": prior["seconds"],
        "weights": prior["weights"],
        "finished": prior["finished"],
        "note": "Reused from the earlier SanJac H=128 full ETAS run; not retrained.",
    }
    save_master(data)
    logging.info(
        "Reused SanJac ETAS H=128: test LL_ts=%.6f total=%.2f n=%d best_epoch=%s",
        test["LL_ts"], test["LL_ts_total"], n_test, prior["best_epoch"],
    )


def run_dataset(ds: str) -> dict:
    prep = prepare_run(ds, history=HISTORY)
    official = OFFICIAL_N[ds]
    if official["test"] is not None and int(prep["testing_num"]) != official["test"]:
        raise AssertionError(
            f"{ds} testing_num={prep['testing_num']} != official {official['test']}"
        )
    if official["train"] is not None and int(prep["training_num"]) != official["train"]:
        raise AssertionError(
            f"{ds} training_num={prep['training_num']} != official {official['train']}"
        )
    if official["val"] is not None and int(prep["validation_num"]) != official["val"]:
        raise AssertionError(
            f"{ds} validation_num={prep['validation_num']} != official {official['val']}"
        )
    if int(prep["input_dim_train"]) != HISTORY:
        raise AssertionError(f"{ds} input_dim_train={prep['input_dim_train']} != {HISTORY}")

    logging.info(
        "Loaded %s official splits n_train=%d n_val=%d n_test=%d input_dim=%d area=%s",
        ds, prep["training_num"], prep["validation_num"], prep["testing_num"],
        prep["input_dim_train"], prep["area"],
    )

    t0 = time.perf_counter()
    model = KernelPointProcess(
        prep["input_dim_train"], prep["input_dim_val"], prep["input_dim_test"],
        "empirical", "empirical", "empirical",
        prep["config"]["global_m0"], prep["area"],
        5, 32, seed=SEED, backbone="fcnn",
    )
    model.set_train_data(prep["train"]["t"], prep["train"]["m"], prep["train"]["x"], prep["train"]["y"])
    model.set_val_data(prep["val"]["t"], prep["val"]["m"], prep["val"]["x"], prep["val"]["y"])
    model.set_test_data(prep["test"]["t"], prep["test"]["m"], prep["test"]["x"], prep["test"]["y"])
    model.set_model()
    model.compile(lr=LR)
    model.fit_eval(epochs=EPOCHS, batch_size=BATCH_SIZE, patience=PATIENCE, mu_batch_size=MU_BATCH)
    model.eval_val(batch_size=BATCH_SIZE)
    model.eval_test(batch_size=BATCH_SIZE)
    weights_path = WEIGHTS / f"etas_{ds}_H{HISTORY}_seed{SEED}.pt"
    model.save_weights(str(weights_path))
    seconds = time.perf_counter() - t0

    n_train = int(prep["training_num"])
    n_val = int(prep["validation_num"])
    n_test = int(prep["testing_num"])
    rec = {
        "status": "ok",
        "name": "ETAS (empirical kernels)",
        "reused": False,
        "dataset": ds,
        "official_splits": True,
        "history": HISTORY,
        "batch_size": BATCH_SIZE,
        "epoch_cap": EPOCHS,
        "patience": PATIENCE,
        "lr": LR,
        "seed": SEED,
        "mu_batch_size": MU_BATCH,
        "upgrades_applied": False,
        "best_epoch": int(model.best_epoch),
        "stopped_early": bool(model.stopped_early),
        "epochs_ran": int(model.best_epoch) + PATIENCE if model.stopped_early else EPOCHS,
        "best_val_loss": float(model.best_val_loss),
        "mu": float(model.bac_func.mu.detach()),
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "val": ll_block(model, "val", n_val),
        "test": ll_block(model, "test", n_test),
        "seconds": seconds,
        "weights": str(weights_path),
        "finished": now_label(),
        "device": "cpu",
        "torch_threads": 4,
    }
    del model
    return rec


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8", mode="a"),
            logging.StreamHandler(),
        ],
    )
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    data = load_master()
    reuse_sanjac(data)
    logging.info(
        "ETAS H=128 CPU start datasets=%s cap=%d patience=%d lr=%s seed=%d",
        ",".join(DATASETS), EPOCHS, PATIENCE, LR, SEED,
    )

    for ds in DATASETS:
        existing = data.get("datasets", {}).get(ds, {})
        if existing.get("status") == "ok":
            logging.info("skip completed %s", ds)
            continue
        logging.info("===== START ETAS %s =====", ds)
        t0 = time.perf_counter()
        try:
            rec = run_dataset(ds)
            data["datasets"][ds] = rec
            save_master(data)
            logging.info(
                "===== DONE ETAS %s status=ok best_epoch=%d test LL_ts=%.6f total=%.2f n=%d seconds=%.1f =====",
                ds, rec["best_epoch"], rec["test"]["LL_ts"], rec["test"]["LL_ts_total"],
                rec["n_test"], rec["seconds"],
            )
        except Exception as exc:
            logging.exception("ETAS %s failed", ds)
            data["datasets"][ds] = {
                "status": "failed",
                "dataset": ds,
                "error": f"{exc}\n{traceback.format_exc()}",
                "seconds": time.perf_counter() - t0,
                "finished": now_label(),
            }
            save_master(data)

    data["protocol"]["finished"] = now_label()
    all_ok = all(data["datasets"].get(ds, {}).get("status") == "ok" for ds in ("SanJac", *DATASETS))
    data["protocol"]["outcome"] = "completed" if all_ok else "partial"
    save_master(data)
    logging.info("ETAS H=128 CPU finished outcome=%s results=%s", data["protocol"]["outcome"], JSON_PATH)
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
