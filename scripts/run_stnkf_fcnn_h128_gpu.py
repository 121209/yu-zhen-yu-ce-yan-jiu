#!/usr/bin/env python3
"""ST-NKF original FCNN, official EarthquakeNPP splits, H=128, GPU.

Controlled comparison with the four Transformer schemes:
  same official jitter catalog and config.py windows
    (auxiliary history, train, validation, test / evaluation),
  H=128, batch 16, lr=1e-4, seed 1, 50-epoch cap,
  mu initialized from train event rate / area then analytic epoch update,
  boundary-mass correction with 128 angles,
  evaluation always on the best-validation snapshot.

Difference vs the four schemes: all kernels stay FCNN (paper ST-NKF structure).
Early stopping is driven by the validation loss (patience 20). The 50-epoch
figure is a cap, not a requirement to run every epoch.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

def _early_gpu():
    for i, a in enumerate(sys.argv):
        if a == "--gpu-id" and i + 1 < len(sys.argv):
            gid = sys.argv[i + 1]
            if gid.lower() in {"-1", "cpu", "none"}:
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = gid
            break
    else:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

_early_gpu()

_HERE = Path(__file__).resolve().parent
for candidate in (
    _HERE.parent,  # this archive repo root when the script lives in scripts/
    _HERE,
    _HERE / "ST-NKF-pytorch",
    Path("/hy-tmp/st-nkf-fcnn-h128-pack/ST-NKF-pytorch"),  # historical cloud path only, not a credential
    Path("/workspace/ST-NKF-pytorch"),
):
    if (candidate / "models.py").is_file():
        _REPO = candidate
        break
else:
    _REPO = Path("/workspace/ST-NKF-pytorch")

sys.path.insert(0, str(_REPO))
os.chdir(_REPO)

import numpy as np
import torch

from models import KernelPointProcess
from run_models import prepare_run

SHANGHAI = timezone(timedelta(hours=8))

HISTORY = 128
BATCH_SIZE = 16
EPOCHS = 50
PATIENCE = 20
LR = 1e-4
SEED = 1
MU_BATCH = 256
BOUNDARY_ANGLES = 128

OFFICIAL_N_TEST = {
    "SanJac": 4400,
    "SCEDC_30": 1898,
    "SaltonSea": 4104,
    "SCEDC_25": 5000,
    "SCEDC_20": 13067,
    "WHITE": 24080,
    "ComCat": 21885,
}
# Smallest first, SanJac included so all seven share this protocol.
DATASETS = ("SCEDC_30", "SaltonSea", "SanJac", "SCEDC_25", "SCEDC_20", "WHITE", "ComCat")


def now_label() -> str:
    return datetime.now(SHANGHAI).strftime("%Y-%m-%d %H:%M:%S Asia/Shanghai")


def ll_block(model, split: str, n: int) -> dict:
    if split == "val":
        mean = float(model.LL_ts_average_val)
        return {
            "LL_ts": mean,
            "LL_t": float(model.LL_t_average_val),
            "LL_s": float(model.LL_s_average_val),
            "n": int(n),
            "LL_ts_total": mean * int(n),
        }
    mean = float(model.LL_ts_average_test)
    return {
        "LL_ts": mean,
        "LL_t": float(model.LL_t_average_test),
        "LL_s": float(model.LL_s_average_test),
        "n": int(n),
        "LL_ts_total": mean * int(n),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ST-NKF FCNN H=128 aligned with the four schemes")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--gpu-id", default="0")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--patience", type=int, default=PATIENCE)
    p.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=None)
    return p.parse_args()


def build_model(prep: dict, device: torch.device) -> KernelPointProcess:
    # Explicit upgrade flags: all-FCNN defaults would otherwise leave the
    # paper path (mu=1, no boundary correction). We want the same training
    # method as the four schemes, only the architecture stays FCNN.
    model = KernelPointProcess(
        prep["input_dim_train"], prep["input_dim_val"], prep["input_dim_test"],
        "neural", "neural", "neural",
        prep["config"]["global_m0"], prep["area"],
        5, 32, seed=SEED, backbone="fcnn",
        temporal_backbone="fcnn",
        spatial_backbone="fcnn",
        kappa_backbone="fcnn",
        init_mu_from_rate=True,
        use_boundary_correction=True,
        boundary_angles=BOUNDARY_ANGLES,
    )
    model.set_train_data(prep["train"]["t"], prep["train"]["m"], prep["train"]["x"], prep["train"]["y"])
    model.set_val_data(prep["val"]["t"], prep["val"]["m"], prep["val"]["x"], prep["val"]["y"])
    model.set_test_data(prep["test"]["t"], prep["test"]["m"], prep["test"]["x"], prep["test"]["y"])
    model.set_model()
    model.set_region(prep["region_xy"], n_angles=BOUNDARY_ANGLES)
    model.compile(lr=LR)
    if hasattr(model, "to"):
        model.to(device)
    return model


def fit_with_val_early_stop(model: KernelPointProcess, epochs: int, patience: int, batch_size: int, mu_batch_size: int) -> dict:
    """Run up to ``epochs``, stop when validation loss does not improve for
    ``patience`` epochs, and always restore the best-validation snapshot.
    ``patience`` must be >= 1; it is the validation early-stopping window.
    """
    if patience < 1:
        raise ValueError(f"patience must be >= 1 for validation early stopping, got {patience}")
    n = int(model.dis_t1_train.shape[0])
    optimizer = torch.optim.Adam(model.adam_parameters(), lr=model.lr)
    rng = np.random.default_rng(0 if model.seed is None else int(model.seed))

    best_val = float("inf")
    best_epoch = 0
    best_state = None
    wait = 0
    stopped = False
    stop_reason = "completed"
    epoch_times: list[float] = []

    for epoch in range(int(epochs)):
        t0 = time.perf_counter()
        order = rng.permutation(n)
        running = 0.0
        seen = 0
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss_on("train", idx, create_graph=True)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch + 1}.")
            loss.backward()
            optimizer.step()
            model.project_constraints()
            running += float(loss.detach().cpu()) * len(idx)
            seen += len(idx)
            del loss
        train_loss = running / seen
        val_loss = model._mean_loss("val", batch_size, create_graph=False)
        if not math.isfinite(val_loss):
            raise FloatingPointError(f"Non-finite validation loss at epoch {epoch + 1}.")
        mu_value = model.update_mu(batch_size=mu_batch_size)
        elapsed = time.perf_counter() - t0
        epoch_times.append(elapsed)

        wait += 1
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            best_epoch = epoch
            best_state = model._snapshot()
            wait = 0
        logging.info(
            "Epoch %d/%d loss=%.6f val_loss=%.6f mu=%.6e seconds=%.2f%s",
            epoch + 1, epochs, train_loss, val_loss, mu_value, elapsed,
            " *" if improved else "",
        )
        if wait >= patience and epoch > 0:
            stopped = True
            stop_reason = "early_stop_val"
            logging.info(
                "Validation early stopping at epoch %d. Best val epoch=%d val_loss=%.6f.",
                epoch + 1, best_epoch + 1, best_val,
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        logging.info(
            "Restored best-validation snapshot from epoch %d (val_loss=%.6f).",
            best_epoch + 1, best_val,
        )
    model.stopped_early = stopped
    model.best_epoch = best_epoch + 1
    model.best_val_loss = best_val
    return {
        "stopped_early": stopped,
        "stop_reason": stop_reason,
        "best_epoch": best_epoch + 1,
        "best_val_loss": best_val,
        "epochs_ran": (best_epoch + 1 + patience) if stopped else epochs,
        "mean_epoch_seconds": float(sum(epoch_times) / len(epoch_times)) if epoch_times else None,
    }


def new_master(args: argparse.Namespace, device: torch.device) -> dict:
    return {
        "protocol": {
            "model": "ST-NKF FCNN 5x32, training method aligned with the four schemes",
            "splits": (
                "official config.py windows: auxiliary history, train, "
                "validation (model selection and early stopping), "
                "test (final evaluation only)"
            ),
            "catalog": "official EarthquakeNPP jitter catalogs, no extra train-time jitter",
            "history": HISTORY,
            "batch_size": BATCH_SIZE,
            "epochs": int(args.epochs),
            "patience": int(args.patience),
            "early_stopping": "validation loss, restore best-val snapshot for evaluation",
            "lr": LR,
            "seed": SEED,
            "mu_batch_size": MU_BATCH,
            "mu_init": "mean train event rate / area, then analytic epoch update",
            "upgrades_applied": True,
            "boundary_correction": True,
            "boundary_angles": BOUNDARY_ANGLES,
            "evaluated_weights": "best-validation snapshot, always restored",
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "dataset_order": list(args.datasets or DATASETS),
            "started": now_label(),
        },
        "datasets": {},
    }


def load_master(path: Path, factory) -> dict:
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        data.setdefault("datasets", {})
        return data
    return factory()


def save_master(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def run_dataset(ds: str, out: Path, epochs: int, patience: int, device: torch.device) -> dict:
    prep = prepare_run(ds, history=HISTORY)
    expect = OFFICIAL_N_TEST[ds]
    if int(prep["testing_num"]) != expect:
        raise AssertionError(f"{ds} testing_num={prep['testing_num']} != official {expect}")
    if int(prep["input_dim_train"]) != HISTORY:
        raise AssertionError(f"{ds} input_dim_train={prep['input_dim_train']} != {HISTORY}")

    logging.info(
        "Loaded %s official splits n_aux=%s n_train=%d n_val=%d n_test=%d "
        "history=%d area=%s device=%s windows=%s",
        ds,
        prep.get("auxiliary_num"),
        prep["training_num"], prep["validation_num"], prep["testing_num"],
        prep["input_dim_train"], prep["area"], device,
        prep.get("config"),
    )

    t0 = time.perf_counter()
    model = build_model(prep, device)
    fit = fit_with_val_early_stop(
        model, epochs=epochs, patience=patience,
        batch_size=BATCH_SIZE, mu_batch_size=MU_BATCH,
    )
    model.eval_val(batch_size=BATCH_SIZE)
    model.eval_test(batch_size=BATCH_SIZE)

    weights_dir = out / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights_path = weights_dir / f"stnkf_fcnn_{ds}_H{HISTORY}_seed{SEED}_best.pt"
    model.save_weights(str(weights_path))
    seconds = time.perf_counter() - t0

    n_train = int(prep["training_num"])
    n_val = int(prep["validation_num"])
    n_test = int(prep["testing_num"])
    rec = {
        "status": "ok",
        "name": "ST-NKF FCNN 5x32 (aligned training method, H=128)",
        "dataset": ds,
        "official_splits": True,
        "split_counts": {
            "auxiliary": int(prep.get("auxiliary_num") or 0),
            "train": n_train,
            "val": n_val,
            "test": n_test,
        },
        "history": HISTORY,
        "batch_size": BATCH_SIZE,
        "epochs": epochs,
        "patience": patience,
        "lr": LR,
        "seed": SEED,
        "mu_batch_size": MU_BATCH,
        "upgrades_applied": True,
        "boundary_correction": True,
        "boundary_angles": BOUNDARY_ANGLES,
        "evaluated_weights": "best-validation snapshot",
        "best_epoch": fit["best_epoch"],
        "stopped_early": fit["stopped_early"],
        "stop_reason": fit["stop_reason"],
        "epochs_ran": fit["epochs_ran"],
        "best_val_loss": fit["best_val_loss"],
        "mu": float(model.bac_func.mu.detach()),
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "val": ll_block(model, "val", n_val),
        "test": ll_block(model, "test", n_test),
        "seconds": seconds,
        "mean_epoch_seconds": fit["mean_epoch_seconds"],
        "weights": str(weights_path),
        "finished": now_label(),
        "device": str(device),
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rec


def main() -> None:
    args = parse_args()
    out: Path = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "stnkf-fcnn-h128-training.log"
    json_path = out / "stnkf-fcnn-h128-ll.json"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8", mode="a"),
            logging.StreamHandler(),
        ],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(
        "ST-NKF FCNN H=128 aligned start device=%s name=%s torch=%s cuda=%s "
        "out=%s epochs=%d patience=%d lr=%s boundary_angles=%d",
        device,
        torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        torch.__version__,
        torch.version.cuda,
        out, args.epochs, args.patience, LR, BOUNDARY_ANGLES,
    )
    if device.type != "cuda":
        logging.error("CUDA is not available; aborting so this does not silently fall back to CPU")
        sys.exit(2)
    if args.patience < 1:
        logging.error("patience must be >= 1 so validation early stopping is active")
        sys.exit(2)

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    order = list(args.datasets) if args.datasets else list(DATASETS)
    data = load_master(json_path, lambda: new_master(args, device))
    data.setdefault("protocol", new_master(args, device)["protocol"])
    save_master(json_path, data)

    for ds in order:
        existing = data.get("datasets", {}).get(ds, {})
        if existing.get("status") == "ok":
            logging.info("skip completed %s", ds)
            continue
        logging.info("===== START ST-NKF FCNN %s =====", ds)
        t0 = time.perf_counter()
        try:
            rec = run_dataset(ds, out, args.epochs, args.patience, device)
            data["datasets"][ds] = rec
            save_master(json_path, data)
            logging.info(
                "===== DONE ST-NKF FCNN %s status=ok best_epoch=%d test LL_ts=%.6f total=%.2f n=%d seconds=%.1f =====",
                ds, rec["best_epoch"], rec["test"]["LL_ts"], rec["test"]["LL_ts_total"],
                rec["n_test"], rec["seconds"],
            )
        except Exception as exc:
            logging.exception("ST-NKF FCNN %s failed", ds)
            data["datasets"][ds] = {
                "status": "failed",
                "dataset": ds,
                "error": f"{exc}\n{traceback.format_exc()}",
                "seconds": time.perf_counter() - t0,
                "finished": now_label(),
            }
            save_master(json_path, data)

    data["protocol"]["finished"] = now_label()
    all_ok = all(data["datasets"].get(ds, {}).get("status") == "ok" for ds in order)
    data["protocol"]["outcome"] = "completed" if all_ok else "partial"
    save_master(json_path, data)
    logging.info("ST-NKF FCNN H=128 finished outcome=%s results=%s", data["protocol"]["outcome"], json_path)
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
