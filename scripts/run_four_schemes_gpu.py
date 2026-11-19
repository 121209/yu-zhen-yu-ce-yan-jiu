#!/usr/bin/env python3
"""GPU training for the four ST-NKF schemes (AutoDL A800).

Same data and architectures as the CPU 20-epoch runs, but each scheme
is trained for a full 50 epochs on GPU (no early stopping), with checkpoints,
mid-run resume, a wall-clock budget, and evaluation always on the
best-validation weights.

Protocol:
  dataset           official SanJac splits, NKF jitter catalog (no train-time jitter)
  H, batch, lr, seed, boundary  128, 16, 1e-4, 1, 128 angles
  mu                rate/area init, analytic update after each epoch (not in Adam)
  epoch cap         50 (every scheme runs all 50 epochs)
  patience          0 (disabled: do not early-stop)
  evaluated weights always the best-val snapshot (even if the cap is reached)
  run order         default scheme2, scheme1, scheme4, scheme3; --schemes can run one per container
  baselines         not retrained here (ETAS H=128 already fully trained on CPU)

The default FCNN / ETAS path in the repo is not modified.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pickle
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

SHANGHAI = timezone(timedelta(hours=8))

# Allow running from a packed checkout or from the original box paths.
_HERE = Path(__file__).resolve().parent
for candidate in (
    _HERE.parent,  # this archive repo root when the script lives in scripts/
    _HERE,
    _HERE / "ST-NKF-pytorch",
    Path("/root/autodl-tmp/ST-NKF-pytorch"),  # historical cloud path only, not a credential
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
from run_models import configure_device, prepare_run

HISTORY = 128
BATCH_SIZE = 16
EPOCHS = 50
PATIENCE = 0  # 0 disables early stopping
LR = 1e-4
SEED = 1
BOUNDARY_ANGLES = 128
MU_BATCH = 256
DATASET = "SanJac"

# Fastest useful results first; scheme 3 (FCNN spatial autograd) last.
RUN_ORDER = ("scheme2", "scheme1", "scheme4", "scheme3")

SCHEMES = {
    "scheme1": {
        "id": "scheme1",
        "name": "Temporal + Spatial + Kappa Transformer",
        "flags": {
            "temporal_backbone": "transformer",
            "spatial_backbone": "transformer",
            "kappa_backbone": "transformer",
        },
    },
    "scheme2": {
        "id": "scheme2",
        "name": "Temporal + Spatial Transformer, Kappa 5x32 FCNN",
        "flags": {
            "temporal_backbone": "transformer",
            "spatial_backbone": "transformer",
            "kappa_backbone": "fcnn",
        },
    },
    "scheme3": {
        "id": "scheme3",
        "name": "Temporal Transformer only; Spatial and Kappa FCNN",
        "flags": {
            "temporal_backbone": "transformer",
            "spatial_backbone": "fcnn",
            "kappa_backbone": "fcnn",
        },
    },
    "scheme4": {
        "id": "scheme4",
        "name": "Same as scheme 2, Kappa deeper MLP (4x64 GELU)",
        "flags": {
            "temporal_backbone": "transformer",
            "spatial_backbone": "transformer",
            "kappa_backbone": "mlp",
        },
    },
}


def now_label() -> str:
    return datetime.now(SHANGHAI).strftime("%Y-%m-%d %H:%M:%S Asia/Shanghai")


def ll_block(model, split: str) -> dict:
    suffix = "val" if split == "val" else "test"
    return {
        "LL_ts": float(getattr(model, f"LL_ts_average_{suffix}")),
        "LL_t": float(getattr(model, f"LL_t_average_{suffix}")),
        "LL_s": float(getattr(model, f"LL_s_average_{suffix}")),
        "n": int(getattr(model, f"dis_t1_{split}").shape[0]),
    }


def shutdown_instance() -> None:
    """Stop the AutoDL instance so pay-as-you-go billing ends."""
    logging.info("Requesting instance shutdown to stop billing.")
    for cmd in (
        ["shutdown", "now"],
        ["/usr/sbin/shutdown", "now"],
        ["poweroff"],
    ):
        try:
            subprocess.Popen(cmd)
            logging.info("Issued shutdown command: %s", " ".join(cmd))
            return
        except FileNotFoundError:
            continue
        except OSError as exc:
            logging.warning("shutdown %s failed: %s", cmd, exc)
    logging.warning("Could not issue a shutdown command; please stop the instance in the AutoDL console.")


class Budget:
    def __init__(self, hours: float, reserve_minutes: float = 6.0) -> None:
        self.t0 = time.perf_counter()
        self.limit = max(0.0, float(hours)) * 3600.0
        self.reserve = max(0.0, float(reserve_minutes)) * 60.0

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def remaining(self) -> float:
        if self.limit <= 0:
            return float("inf")
        return self.limit - self.elapsed()

    def allow_another_epoch(self, last_epoch_seconds: float) -> bool:
        if self.limit <= 0:
            return True
        need = max(last_epoch_seconds * 1.4, 30.0) + self.reserve
        return self.remaining() > need


def build_model(prep: dict, sch: dict, device: torch.device) -> KernelPointProcess:
    flags = sch["flags"]
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
    model.to(device)
    return model


def _save_ckpt(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def fit_checkpointed(
    model: KernelPointProcess,
    *,
    epochs: int,
    batch_size: int,
    patience: int,
    mu_batch_size: int,
    ckpt_dir: Path,
    scheme_id: str,
    budget: Budget,
) -> dict:
    """Adam loop matching models.fit_eval, plus checkpoints, resume, budget, always-restore-best."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_path = ckpt_dir / f"{scheme_id}_last.pt"
    best_path = ckpt_dir / f"{scheme_id}_best.pt"
    log_path = ckpt_dir / f"{scheme_id}_epochs.jsonl"

    n = int(model.dis_t1_train.shape[0])
    optimizer = torch.optim.Adam(model.adam_parameters(), lr=model.lr)
    rng = np.random.default_rng(0 if model.seed is None else int(model.seed))

    best_val = float("inf")
    best_epoch = 0
    best_state = None
    wait = 0
    start_epoch = 0
    epoch_times: list[float] = []
    stopped = False
    stop_reason = "completed"

    if last_path.is_file():
        ckpt = torch.load(last_path, map_location=model.bac_func.mu.device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        rng = pickle.loads(ckpt["rng"])
        best_val = float(ckpt["best_val"])
        best_epoch = int(ckpt["best_epoch"])
        wait = int(ckpt["wait"])
        start_epoch = int(ckpt["next_epoch"])
        best_state = ckpt.get("best_state")
        epoch_times = list(ckpt.get("epoch_times", []))
        logging.info(
            "%s resuming at epoch %d (best_epoch=%d best_val=%.6f wait=%d)",
            scheme_id, start_epoch + 1, best_epoch + 1, best_val, wait,
        )

    if start_epoch >= epochs:
        stop_reason = "already_finished"
    else:
        for epoch in range(start_epoch, int(epochs)):
            t_ep = time.perf_counter()
            order = rng.permutation(n)
            running = 0.0
            seen = 0
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss = model.loss_on("train", idx, create_graph=True)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite training loss at epoch {epoch + 1}."
                    )
                loss.backward()
                optimizer.step()
                model.project_constraints()
                running += float(loss.detach().cpu()) * len(idx)
                seen += len(idx)
                del loss
            train_loss = running / seen
            val_loss = model._mean_loss("val", batch_size, create_graph=False)
            if not math.isfinite(val_loss):
                raise FloatingPointError(
                    f"Non-finite validation loss at epoch {epoch + 1}."
                )
            mu_value = model.update_mu(batch_size=mu_batch_size)
            ep_sec = time.perf_counter() - t_ep
            epoch_times.append(ep_sec)

            wait += 1
            improved = val_loss < best_val
            if improved:
                best_val = val_loss
                best_epoch = epoch
                best_state = model._snapshot()
                wait = 0
                _save_ckpt(
                    best_path,
                    {
                        "state_dict": best_state,
                        "best_epoch": best_epoch,
                        "best_val": best_val,
                        "mu": float(model.bac_func.mu.detach()),
                        "epoch_finished": epoch + 1,
                    },
                )

            record = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "mu": float(mu_value),
                "seconds": ep_sec,
                "improved": improved,
                "wait": wait,
                "best_epoch": best_epoch + 1,
                "best_val": best_val,
                "time": now_label(),
            }
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            logging.info(
                "%s Epoch %d/%d loss=%.6f val_loss=%.6f mu=%.6e seconds=%.2f%s",
                scheme_id, epoch + 1, epochs, train_loss, val_loss, mu_value, ep_sec,
                " *" if improved else "",
            )

            payload = {
                "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "optimizer": optimizer.state_dict(),
                "rng": pickle.dumps(rng),
                "best_state": best_state,
                "best_val": best_val,
                "best_epoch": best_epoch,
                "wait": wait,
                "next_epoch": epoch + 1,
                "epoch_times": epoch_times[-50:],
                "mu": float(model.bac_func.mu.detach()),
            }
            _save_ckpt(last_path, payload)

            if patience > 0 and wait >= patience and epoch > 0:
                stopped = True
                stop_reason = "early_stop"
                logging.info(
                    "%s early stopping at epoch %d, best epoch %d val_loss=%.6f",
                    scheme_id, epoch + 1, best_epoch + 1, best_val,
                )
                break
            if not budget.allow_another_epoch(ep_sec):
                stopped = True
                stop_reason = "time_budget"
                logging.info(
                    "%s stopping on time budget at epoch %d (elapsed=%.0fs remaining=%.0fs)",
                    scheme_id, epoch + 1, budget.elapsed(), budget.remaining(),
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        logging.info(
            "%s restored best-val weights from epoch %d (val_loss=%.6f)",
            scheme_id, best_epoch + 1, best_val,
        )
    model.stopped_early = bool(stopped and stop_reason == "early_stop")
    model.best_epoch = best_epoch + 1 if best_state is not None else 0
    model.best_val_loss = best_val
    return {
        "stop_reason": stop_reason,
        "stopped_early": bool(model.stopped_early),
        "best_epoch": int(model.best_epoch),
        "best_val_loss": float(best_val) if math.isfinite(best_val) else None,
        "epochs_ran": start_epoch if start_epoch >= epochs else (
            (best_epoch + 1 + patience) if stop_reason == "early_stop"
            else (int(payload["next_epoch"]) if last_path.is_file() else start_epoch)
        ),
        "mean_epoch_seconds": float(np.mean(epoch_times)) if epoch_times else None,
    }


def run_scheme(
    prep: dict,
    sch: dict,
    device: torch.device,
    out: Path,
    weights: Path,
    budget: Budget,
    epochs: int,
    patience: int,
) -> dict:
    flags = sch["flags"]
    logging.info("===== START %s: %s on %s =====", sch["id"], sch["name"], device)
    t0 = time.perf_counter()
    model = build_model(prep, sch, device)
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
    fit_info = fit_checkpointed(
        model,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        patience=patience,
        mu_batch_size=MU_BATCH,
        ckpt_dir=out / "checkpoints",
        scheme_id=sch["id"],
        budget=budget,
    )
    model.eval_val(batch_size=BATCH_SIZE)
    model.eval_test(batch_size=BATCH_SIZE)
    wpath = weights / f"{sch['id']}_H{HISTORY}_seed{SEED}_best.pt"
    model.save_weights(str(wpath))
    seconds = time.perf_counter() - t0
    rec = {
        "status": "ok" if fit_info["stop_reason"] in {"early_stop", "completed", "already_finished"} else "partial",
        "name": sch["name"],
        "flags": flags,
        "device": str(device),
        "lr": LR,
        "seed": SEED,
        "history": HISTORY,
        "batch_size": BATCH_SIZE,
        "epoch_cap": epochs,
        "patience": patience,
        "official_splits": True,
        "upgrades_applied": True,
        "evaluated_weights": "best-validation snapshot, always restored",
        "cli": (
            f"--temporal-backbone {flags['temporal_backbone']} "
            f"--spatial-backbone {flags['spatial_backbone']} "
            f"--kappa-backbone {flags['kappa_backbone']} "
            f"--history {HISTORY} --lr {LR} --epochs {epochs} "
            f"--batch-size {BATCH_SIZE} --patience {patience} "
            f"--boundary-angles {BOUNDARY_ANGLES}"
        ),
        **fit_info,
        "mu": float(model.bac_func.mu.detach()),
        "val": ll_block(model, "val"),
        "test": ll_block(model, "test"),
        "seconds": seconds,
        "weights": str(wpath),
        "finished": now_label(),
    }
    logging.info(
        "===== DONE %s status=%s val LL_ts=%.4f test LL_ts=%.4f best_epoch=%d reason=%s seconds=%.1f =====",
        sch["id"], rec["status"], rec["val"]["LL_ts"], rec["test"]["LL_ts"],
        rec["best_epoch"], rec["stop_reason"], seconds,
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rec


def write_status(data: dict, path: Path) -> None:
    lines = [
        f"# Four-scheme GPU training",
        "",
        f"Updated: {now_label()}",
        "",
        "## Protocol",
        "",
        f"- Dataset: SanJac official splits, NKF jitter catalog, H={HISTORY}, batch={BATCH_SIZE}, "
        f"lr={LR}, seed={SEED}, boundary_angles={BOUNDARY_ANGLES}.",
        f"- Epoch cap {data['protocol']['epochs']}, patience {data['protocol']['patience']}; "
        "evaluation always uses best-val weights.",
        f"- Device: {data['protocol'].get('device', '?')}. "
        f"Run order: {' → '.join(data['protocol'].get('run_order', RUN_ORDER))}.",
        f"- Time budget: {data['protocol'].get('budget_hours', 0)} hours.",
        "",
        "| Scheme | Status | VAL LL_ts | TEST LL_ts | TEST LL_t | TEST LL_s | Best ep | Reason | Seconds |",
        "|---|---|---:|---:|---:|---:|---|---|---:|",
    ]
    for sid in data['protocol'].get('run_order', RUN_ORDER):
        rec = data.get("schemes", {}).get(sid)
        if not rec:
            lines.append(f"| {sid} | pending |  |  |  |  |  |  |  |")
            continue
        if rec.get("status") in {"ok", "partial"}:
            lines.append(
                "| {id} | {st} | {v:.4f} | {t:.4f} | {tt:.4f} | {ts:.4f} | {ep} | {r} | {sec:.1f} |".format(
                    id=sid, st=rec["status"],
                    v=rec["val"]["LL_ts"], t=rec["test"]["LL_ts"],
                    tt=rec["test"]["LL_t"], ts=rec["test"]["LL_s"],
                    ep=rec["best_epoch"], r=rec["stop_reason"], sec=rec["seconds"],
                )
            )
        else:
            lines.append(f"| {sid} | {rec.get('status')} |  |  |  |  |  | {rec.get('error', '')[:80]} |  |")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the four ST-NKF schemes on GPU.")
    p.add_argument("--out", type=Path, default=None, help="Output directory for logs and weights.")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--patience", type=int, default=PATIENCE)
    p.add_argument("--budget-hours", type=float, default=1.4,
                   help="Wall-clock budget from process start. 0 means no limit. "
                        "Default 1.4h fits ~¥13.12 at 4×¥2.08/时 with a short reserve.")
    p.add_argument("--gpu-id", default="0")
    p.add_argument(
        "--schemes",
        nargs="+",
        default=None,
        choices=list(SCHEMES),
        help="Schemes to run on this machine. Default: all four in RUN_ORDER. "
             "Use a single scheme when each container runs one.",
    )
    p.add_argument("--shutdown-when-done", action="store_true",
                   help="Shut down this machine when training finishes or the budget is hit.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = args.out or Path(os.environ.get("STNKF_GPU_OUT", "/root/autodl-tmp/st-nkf-gpu"))
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    weights = out / "weights"
    weights.mkdir(exist_ok=True)
    json_path = out / "four-schemes-gpu-ll.json"
    status_path = out / "four-schemes-gpu-status.md"
    log_path = out / "four-schemes-gpu-training.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    run_list = list(args.schemes) if args.schemes else list(RUN_ORDER)
    configure_device(args.gpu_id)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; refusing to run the GPU training script on CPU.")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda")
    logging.info(
        "GPU training start device=%s name=%s torch=%s cuda=%s out=%s budget=%.2fh schemes=%s",
        device, torch.cuda.get_device_name(0), torch.__version__, torch.version.cuda, out, args.budget_hours,
        ",".join(run_list),
    )

    data = {
        "protocol": {
            "dataset": DATASET,
            "catalog": "NeuralKernelFunction SanJac_catalog_jitter.csv (no extra train-time jitter)",
            "splits": "official config.py train/val/test windows",
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
            "budget_hours": args.budget_hours,
            "started": now_label(),
        },
        "schemes": {},
    }
    if json_path.is_file():
        previous = json.loads(json_path.read_text(encoding="utf-8"))
        data["schemes"] = previous.get("schemes", {})
        data["protocol"]["resumed_from"] = previous.get("protocol", {}).get("started")

    budget = Budget(args.budget_hours)
    logging.info("Loading official SanJac splits with H=%d", HISTORY)
    prep = prepare_run(DATASET, history=HISTORY)
    assert prep["training_num"] == 11770
    assert prep["validation_num"] == 3449
    assert prep["testing_num"] == 4400
    assert prep["input_dim_train"] == HISTORY

    budget_hit = False
    for sid in run_list:
        existing = data["schemes"].get(sid)
        if existing and existing.get("status") == "ok":
            logging.info("skip completed %s", sid)
            continue
        if not budget.allow_another_epoch(30.0):
            logging.info("time budget exhausted before starting %s", sid)
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
                prep, SCHEMES[sid], device, out, weights, budget,
                epochs=args.epochs, patience=args.patience,
            )
            if rec.get("stop_reason") == "time_budget":
                budget_hit = True
        except Exception as exc:
            logging.exception("%s failed", sid)
            rec = {
                "status": "failed",
                "name": SCHEMES[sid]["name"],
                "flags": SCHEMES[sid]["flags"],
                "error": repr(exc),
                "finished": now_label(),
            }
        data["schemes"][sid] = rec
        json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        write_status(data, status_path)
        if budget_hit:
            break

    data["protocol"]["finished"] = now_label()
    data["protocol"]["elapsed_seconds"] = budget.elapsed()
    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_status(data, status_path)
    logging.info("Training finished. Results at %s elapsed=%.1fs", json_path, budget.elapsed())

    if args.shutdown_when_done:
        # Copy a short done marker that survives a moment after shutdown is issued.
        (out / "SHUTDOWN_REQUESTED").write_text(now_label() + "\n", encoding="utf-8")
        shutdown_instance()


if __name__ == "__main__":
    main()
