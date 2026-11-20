#!/usr/bin/env python3
"""Sequential official-split trainings for the four ST-NKF schemes (CPU)."""
from __future__ import annotations

import json
import sys
import logging
import os
import resource
import time
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch

torch.set_num_threads(4)

_REPO = Path("/workspace/ST-NKF-pytorch")
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
os.chdir(_REPO)

from models import KernelPointProcess
from run_models import prepare_run

SHANGHAI = timezone(timedelta(hours=8))
OUT = Path("/workspace/st-nkf-diagrams")
WEIGHTS = OUT / "four-schemes-weights"
WEIGHTS.mkdir(parents=True, exist_ok=True)
JSON_PATH = OUT / "four-schemes-ll.json"
STATUS_PATH = OUT / "four-schemes-status.md"
LOG_PATH = OUT / "four-schemes-training.log"

# Fair shared budget for all four neural schemes.
HISTORY = 128
BATCH_SIZE = 16
EPOCHS = 20
PATIENCE = 8
LR = 1e-4
SEED = 1
BOUNDARY_ANGLES = 128
MU_BATCH = 256
DATASET = "SanJac"

SCHEMES = [
    {
        "id": "scheme1",
        "name": "Temporal + Spatial + Kappa Transformer",
        "flags": {
            "temporal_backbone": "transformer",
            "spatial_backbone": "transformer",
            "kappa_backbone": "transformer",
        },
    },
    {
        "id": "scheme2",
        "name": "Temporal + Spatial Transformer, Kappa 5x32 FCNN",
        "flags": {
            "temporal_backbone": "transformer",
            "spatial_backbone": "transformer",
            "kappa_backbone": "fcnn",
        },
    },
    {
        "id": "scheme3",
        "name": "Temporal Transformer only; Spatial and Kappa FCNN",
        "flags": {
            "temporal_backbone": "transformer",
            "spatial_backbone": "fcnn",
            "kappa_backbone": "fcnn",
        },
    },
    {
        "id": "scheme4",
        "name": "Same as scheme 2, Kappa deeper MLP (4x64 GELU)",
        "flags": {
            "temporal_backbone": "transformer",
            "spatial_backbone": "transformer",
            "kappa_backbone": "mlp",
        },
    },
]


def now_label() -> str:
    return datetime.now(SHANGHAI).strftime("%Y-%m-%d %H:%M:%S Asia/Shanghai")


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def load_results() -> dict:
    if JSON_PATH.exists():
        return json.loads(JSON_PATH.read_text())
    return {
        "protocol": {
            "dataset": DATASET,
            "catalog": "NeuralKernelFunction SanJac_catalog_jitter.csv (no extra train-time jitter)",
            "splits": "official config.py train/val/test windows",
            "history": HISTORY,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "patience": PATIENCE,
            "lr": LR,
            "seed": SEED,
            "boundary_angles": BOUNDARY_ANGLES,
            "mu_batch_size": MU_BATCH,
            "mu_init": "mean train event rate / area, then analytic epoch update",
            "loss": "spatiotemporal NLL, same as upstream TensorFlow",
            "evaluated_weights": "fit_eval snapshot (best-val if early-stopped, else last epoch)",
            "started": now_label(),
        },
        "schemes": {},
    }


def save_results(data: dict) -> None:
    data["updated"] = now_label()
    JSON_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def write_status(data: dict) -> None:
    proto = data["protocol"]
    lines = [
        "# Four-scheme ST-NKF status",
        "",
        f"Updated: {data.get('updated', now_label())}",
        "",
        "## Protocol",
        "",
        f"- Dataset: {proto['dataset']} official splits from config.py, jitter catalog from NKF (no extra jitter).",
        f"- H={proto['history']}, batch={proto['batch_size']}, epochs={proto['epochs']}, patience={proto['patience']}, "
        f"lr={proto['lr']}, seed={proto['seed']}, boundary_angles={proto['boundary_angles']}.",
        f"- Loss: spatiotemporal point-process NLL. mu initialized from rate/area, analytic update after each epoch.",
        f"- Default FCNN/ETAS path left bitwise unchanged (not used by these four schemes).",
        "",
        "## Results",
        "",
        "| Scheme | Architecture | Split | LL_ts | LL_t | LL_s | Epoch (best) | Seconds | Notes |",
        "|---|---|---|---:|---:|---:|---|---:|---|",
    ]
    for sch in SCHEMES:
        rec = data["schemes"].get(sch["id"])
        if not rec:
            lines.append(f"| {sch['id']} | {sch['name']} | — | — | — | — | queued | — | |")
            continue
        if rec.get("status") != "ok":
            lines.append(
                f"| {sch['id']} | {sch['name']} | — | — | — | — | {rec.get('status')} | "
                f"{rec.get('seconds', '')} | {rec.get('error', '')[:80]} |"
            )
            continue
        for split in ("val", "test"):
            lines.append(
                "| {id} | {name} | {split} | {llts:.4f} | {llt:.4f} | {lls:.4f} | {ep} | {sec:.1f} | mu={mu:.6e} |".format(
                    id=sch["id"] if split == "val" else "",
                    name=sch["name"] if split == "val" else "",
                    split=split.upper(),
                    llts=rec[split]["LL_ts"],
                    llt=rec[split]["LL_t"],
                    lls=rec[split]["LL_s"],
                    ep=f"{rec['best_epoch']}/{rec['epochs_ran']}" if split == "val" else "",
                    sec=rec["seconds"] if split == "val" else 0.0,
                    mu=rec["mu"] if split == "val" else rec["mu"],
                )
            )
    lines += [
        "",
        f"Max RSS so far: {rss_mb():.0f} MB.",
        "",
        "Logs: `four-schemes-training.log`, `four-schemes-ll.json`. Weights under `four-schemes-weights/`.",
        "Not pushed, no PR, no GPU rental.",
        "",
    ]
    STATUS_PATH.write_text("\n".join(lines))


def ll_block(model, split: str) -> dict:
    return {
        "LL_ts": float(getattr(model, f"LL_ts_average_{split}")),
        "LL_t": float(getattr(model, f"LL_t_average_{split}")),
        "LL_s": float(getattr(model, f"LL_s_average_{split}")),
        "n": int(getattr(model, f"dis_t1_{split}").shape[0]),
    }


def run_scheme(prep: dict, sch: dict) -> dict:
    flags = sch["flags"]
    logging.info("===== START %s: %s flags=%s =====", sch["id"], sch["name"], flags)
    t0 = time.perf_counter()
    model = KernelPointProcess(
        time_step_train=prep["input_dim_train"],
        time_step_val=prep["input_dim_val"],
        time_step_test=prep["input_dim_test"],
        temporal_id="neural",
        spatial_id="neural",
        kappa_id="neural",
        global_m0=prep["config"]["global_m0"],
        area=prep["area"],
        size_layer=5,
        size_nn=32,
        seed=SEED,
        temporal_backbone=flags["temporal_backbone"],
        spatial_backbone=flags["spatial_backbone"],
        kappa_backbone=flags["kappa_backbone"],
        boundary_angles=BOUNDARY_ANGLES,
    )
    model.set_train_data(prep["train"]["t"], prep["train"]["m"], prep["train"]["x"], prep["train"]["y"])
    model.set_val_data(prep["val"]["t"], prep["val"]["m"], prep["val"]["x"], prep["val"]["y"])
    model.set_test_data(prep["test"]["t"], prep["test"]["m"], prep["test"]["x"], prep["test"]["y"])
    model.set_model()
    model.set_region(prep["region_xy"], n_angles=BOUNDARY_ANGLES)
    model.compile(lr=LR)
    logging.info(
        "%s mu0=%.6e max_time=%.6g max_mag=%.6g max_dis=%.6g n_train=%d n_val=%d n_test=%d",
        sch["id"],
        float(model.bac_func.mu.detach()),
        model.max_time,
        model.max_mag,
        model.max_dis,
        model.dis_t1_train.shape[0],
        model.dis_t1_val.shape[0],
        model.dis_t1_test.shape[0],
    )
    model.fit_eval(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        patience=PATIENCE,
        mu_batch_size=MU_BATCH,
    )
    # If training ran the full budget, restore the best-val snapshot for a
    # fair report (fit_eval only auto-restores when early-stopping fires).
    # Re-run a val pass is avoided by loading the in-memory best via a second
    # fit? We stored best only inside fit_eval. If not stopped, last may
    # differ from best. Re-evaluate current first; if not stopped, the last
    # epoch is what fit_eval would keep. We still record best_epoch/best_val.
    model.eval_val(batch_size=BATCH_SIZE)
    model.eval_test(batch_size=BATCH_SIZE)
    weights_path = WEIGHTS / f"{sch['id']}_H{HISTORY}_seed{SEED}.pt"
    model.save_weights(str(weights_path))
    seconds = time.perf_counter() - t0
    rec = {
        "status": "ok",
        "name": sch["name"],
        "flags": flags,
        "cli": (
            f"--temporal-backbone {flags['temporal_backbone']} "
            f"--spatial-backbone {flags['spatial_backbone']} "
            f"--kappa-backbone {flags['kappa_backbone']} "
            f"--history {HISTORY} --lr {LR} --epochs {EPOCHS} "
            f"--batch-size {BATCH_SIZE} --patience {PATIENCE} "
            f"--boundary-angles {BOUNDARY_ANGLES}"
        ),
        "architecture": {
            "temporal": "Omori-mixture Transformer CDF (d_model=32, 8 heads, 1 layer, FFN 128, GELU, dropout 0, pre-norm)"
            if flags["temporal_backbone"] == "transformer"
            else "FCNN 5x32 nonnegative tanh CDF (paper)",
            "spatial": "ETAS-radial-mixture Transformer CDF (same encoder size)"
            if flags["spatial_backbone"] == "transformer"
            else "FCNN 5x32 nonnegative tanh CDF (paper)",
            "kappa": {
                "fcnn": "KappaNetwork 5x32, bias, softplus, NonNeg (paper)",
                "transformer": "history Transformer, d_model=32, 1 layer, softplus per event",
                "mlp": "unconstrained GELU MLP, 4 hidden layers x 64, softplus out",
            }[flags["kappa_backbone"]],
        },
        "history": HISTORY,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "seed": SEED,
        "official_splits": True,
        "n_train": int(model.dis_t1_train.shape[0]),
        "n_val": int(model.dis_t1_val.shape[0]),
        "n_test": int(model.dis_t1_test.shape[0]),
        "best_epoch": int(model.best_epoch),
        "best_val_loss": float(model.best_val_loss),
        "epochs_ran": int(getattr(model, "best_epoch", 0) if model.stopped_early else EPOCHS if not model.stopped_early else model.best_epoch),
        "stopped_early": bool(model.stopped_early),
        "mu": float(model.bac_func.mu.detach()),
        "mu0_rate_init": True,
        "boundary_correction": True,
        "boundary_angles": BOUNDARY_ANGLES,
        "val": ll_block(model, "val"),
        "test": ll_block(model, "test"),
        "seconds": seconds,
        "weights": str(weights_path),
        "rss_mb": rss_mb(),
        "finished": now_label(),
    }
    # epochs actually executed: if stopped, best wait-end; we don't have
    # a direct counter. If early-stopped, patience was hit after some epoch.
    if model.stopped_early:
        rec["epochs_ran"] = int(model.best_epoch) + PATIENCE
    else:
        rec["epochs_ran"] = EPOCHS
    logging.info(
        "===== DONE %s val LL_ts=%.4f test LL_ts=%.4f best_epoch=%d seconds=%.1f =====",
        sch["id"],
        rec["val"]["LL_ts"],
        rec["test"]["LL_ts"],
        rec["best_epoch"],
        seconds,
    )
    del model
    return rec


def maybe_run_baselines(prep: dict, data: dict) -> None:
    """Re-evaluate ETAS and FCNN on the exact same H=128 official targets."""
    baselines = [
        ("etas", "empirical", "empirical", "empirical", 0.005),
        ("fcnn", "neural", "neural", "neural", 0.005),
    ]
    for bid, temporal, spatial, kappa, lr in baselines:
        if data.get("baselines", {}).get(bid, {}).get("status") == "ok":
            logging.info("skip completed baseline %s", bid)
            continue
        logging.info("===== START baseline %s =====", bid)
        t0 = time.perf_counter()
        try:
            model = KernelPointProcess(
                prep["input_dim_train"], prep["input_dim_val"], prep["input_dim_test"],
                temporal, spatial, kappa,
                prep["config"]["global_m0"], prep["area"],
                5, 32, seed=SEED, backbone="fcnn",
            )
            model.set_train_data(prep["train"]["t"], prep["train"]["m"], prep["train"]["x"], prep["train"]["y"])
            model.set_val_data(prep["val"]["t"], prep["val"]["m"], prep["val"]["x"], prep["val"]["y"])
            model.set_test_data(prep["test"]["t"], prep["test"]["m"], prep["test"]["x"], prep["test"]["y"])
            model.set_model()
            model.compile(lr=lr)
            model.fit_eval(epochs=EPOCHS, batch_size=BATCH_SIZE, patience=PATIENCE, mu_batch_size=MU_BATCH)
            model.eval_val(batch_size=BATCH_SIZE)
            model.eval_test(batch_size=BATCH_SIZE)
            path = WEIGHTS / f"baseline_{bid}_H{HISTORY}_seed{SEED}.pt"
            model.save_weights(str(path))
            rec = {
                "status": "ok",
                "name": "ETAS (empirical kernels)" if bid == "etas" else "ST-NKF FCNN 5x32 (paper path, H=128)",
                "lr": lr,
                "seed": SEED,
                "history": HISTORY,
                "official_splits": True,
                "upgrades_applied": False,
                "best_epoch": int(model.best_epoch),
                "stopped_early": bool(model.stopped_early),
                "epochs_ran": int(model.best_epoch) + PATIENCE if model.stopped_early else EPOCHS,
                "best_val_loss": float(model.best_val_loss),
                "mu": float(model.bac_func.mu.detach()),
                "val": ll_block(model, "val"),
                "test": ll_block(model, "test"),
                "seconds": time.perf_counter() - t0,
                "weights": str(path),
                "finished": now_label(),
            }
            logging.info(
                "===== DONE baseline %s test LL_ts=%.4f seconds=%.1f =====",
                bid, rec["test"]["LL_ts"], rec["seconds"],
            )
            del model
        except Exception as exc:
            logging.exception("baseline %s failed", bid)
            rec = {"status": "failed", "error": f"{exc}\n{traceback.format_exc()}", "seconds": time.perf_counter() - t0}
        data.setdefault("baselines", {})[bid] = rec
        save_results(data)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    data = load_results()
    save_results(data)
    write_status(data)
    logging.info("Loading official SanJac splits with H=%d", HISTORY)
    prep = prepare_run(DATASET, history=HISTORY)
    assert prep["training_num"] == 11770
    assert prep["validation_num"] == 3449
    assert prep["testing_num"] == 4400
    # Targets equal official counts when history is taken from the auxiliary window.
    assert len(prep["train"]["t"]) - HISTORY == prep["training_num"]

    for sch in SCHEMES:
        existing = data["schemes"].get(sch["id"])
        if existing and existing.get("status") == "ok":
            logging.info("skip completed %s", sch["id"])
            continue
        try:
            rec = run_scheme(prep, sch)
        except Exception as exc:
            logging.exception("%s failed", sch["id"])
            rec = {
                "status": "failed",
                "name": sch["name"],
                "flags": sch["flags"],
                "error": f"{exc}\n{traceback.format_exc()}",
                "seconds": None,
                "finished": now_label(),
            }
        data["schemes"][sch["id"]] = rec
        save_results(data)
        write_status(data)

    maybe_run_baselines(prep, data)
    write_status(data)
    logging.info("All four schemes (and baselines if reached) finished. JSON at %s", JSON_PATH)


if __name__ == "__main__":
    main()
