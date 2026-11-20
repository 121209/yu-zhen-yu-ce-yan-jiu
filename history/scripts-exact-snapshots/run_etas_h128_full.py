#!/usr/bin/env python3
"""Full ETAS training on official SanJac, H=128, CPU. Default path unchanged."""
from __future__ import annotations

import json
import logging
import os
import sys
import time
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
OUT = Path("/workspace/st-nkf-diagrams")
LOG_PATH = OUT / "etas-h128-full-training.log"
JSON_PATH = OUT / "etas-h128-full-result.json"
WEIGHTS_PATH = OUT / "etas-h128-full.pt"

HISTORY = 128
BATCH_SIZE = 16
EPOCHS = 1000
PATIENCE = 20
LR = 0.005
SEED = 1
MU_BATCH = 256
DATASET = "SanJac"


def now_label() -> str:
    return datetime.now(SHANGHAI).strftime("%Y-%m-%d %H:%M:%S Asia/Shanghai")


def ll_block(model, split: str) -> dict:
    if split == "val":
        return {
            "LL_ts": float(model.LL_ts_average_val),
            "LL_t": float(model.LL_t_average_val),
            "LL_s": float(model.LL_s_average_val),
        }
    return {
        "LL_ts": float(model.LL_ts_average_test),
        "LL_t": float(model.LL_t_average_test),
        "LL_s": float(model.LL_s_average_test),
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8", mode="w"),
            logging.StreamHandler(),
        ],
    )
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    logging.info(
        "Full ETAS training: dataset=%s H=%d batch=%d epochs=%d patience=%d lr=%s seed=%d",
        DATASET, HISTORY, BATCH_SIZE, EPOCHS, PATIENCE, LR, SEED,
    )
    prep = prepare_run(DATASET, history=HISTORY)
    assert prep["training_num"] == 11770
    assert prep["validation_num"] == 3449
    assert prep["testing_num"] == 4400
    assert prep["input_dim_train"] == HISTORY
    logging.info(
        "Loaded official splits n_train=%d n_val=%d n_test=%d input_dim=%d area=%s",
        prep["training_num"], prep["validation_num"], prep["testing_num"],
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
    model.save_weights(str(WEIGHTS_PATH))
    seconds = time.perf_counter() - t0

    epochs_ran = (
        int(model.best_epoch) + PATIENCE if model.stopped_early else EPOCHS
    )
    result = {
        "model": "ETAS (empirical kernels), full training",
        "status": "ok",
        "dataset": DATASET,
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
        "epochs_ran": epochs_ran,
        "best_val_loss": float(model.best_val_loss),
        "mu": float(model.bac_func.mu.detach()),
        "val": ll_block(model, "val"),
        "test": ll_block(model, "test"),
        "seconds": seconds,
        "weights": str(WEIGHTS_PATH),
        "finished": now_label(),
        "torch_threads": 4,
        "note": (
            "Default ETAS path: lr=0.005, mu init unchanged (1), no boundary correction. "
            "Compared with the four-scheme H=128 baseline that early-stopped at epoch 3 "
            "(cap 20, patience 8). Previous complete ETAS at paper window H=1672 is separate."
        ),
    }
    JSON_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logging.info(
        "DONE ETAS H=128 full: best_epoch=%d stopped=%s val LL_ts=%.4f test LL_ts=%.4f "
        "LL_t=%.4f LL_s=%.4f mu=%g seconds=%.1f weights=%s",
        result["best_epoch"], result["stopped_early"],
        result["val"]["LL_ts"], result["test"]["LL_ts"],
        result["test"]["LL_t"], result["test"]["LL_s"],
        result["mu"], seconds, WEIGHTS_PATH,
    )


if __name__ == "__main__":
    main()
