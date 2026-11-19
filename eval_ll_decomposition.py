#!/usr/bin/env python3
"""Re-evaluate existing best-validation snapshots and export a
time / conditional-space decomposition of the test log-likelihood.

Does not train, does not modify snapshots, and writes only to a new
output directory. Reuses KernelPointProcess.eval_test so the official
LL_ts / LL_t / LL_s scoring is unchanged.
"""
from __future__ import annotations

import argparse
import os
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent

def resolve_repo_path(user_repo: str | None = None) -> Path:
    """Locate the ST-NKF-pytorch tree (models.py, run_models.py, layers_transformer.py, Datasets/)."""
    candidates = []
    if user_repo:
        candidates.append(Path(user_repo).expanduser().resolve())
    env = os.environ.get("ST_NKF_REPO")
    if env:
        candidates.append(Path(env).expanduser().resolve())
    candidates.extend(
        [
            ROOT,
            ROOT / "ST-NKF-pytorch",
            ROOT.parent / "ST-NKF-pytorch",
            Path.cwd(),
            Path.cwd() / "ST-NKF-pytorch",
        ]
    )
    seen = set()
    for cand in candidates:
        try:
            cand = cand.resolve()
        except OSError:
            continue
        if cand in seen:
            continue
        seen.add(cand)
        if (
            (cand / "models.py").is_file()
            and (cand / "run_models.py").is_file()
            and (cand / "layers_transformer.py").is_file()
            and (cand / "Datasets").is_dir()
        ):
            return cand
    raise FileNotFoundError(
        "未找到 ST-NKF-pytorch 代码仓库（需包含 models.py、run_models.py、layers_transformer.py 及 Datasets/）。"
        "请通过 --repo 指定路径，或设置环境变量 ST_NKF_REPO。"
    )

KernelPointProcess = None  # loaded in setup_repo()
prepare_run = None
REPO = ROOT

def setup_repo(user_repo: Path | None = None) -> Path:
    global KernelPointProcess, prepare_run, REPO
    REPO = resolve_repo_path(str(user_repo) if user_repo else None)
    sys.path.insert(0, str(REPO))
    os.chdir(REPO)
    from models import KernelPointProcess as _KernelPointProcess
    from run_models import prepare_run as _prepare_run
    KernelPointProcess = _KernelPointProcess
    prepare_run = _prepare_run
    return REPO

HISTORY = 128
BATCH_SIZE = 16
SEED = 1
BOUNDARY_ANGLES = 128
LR = 1e-4

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

BACKBONES = {
    "fcnn": {"temporal_backbone": "fcnn", "spatial_backbone": "fcnn", "kappa_backbone": "fcnn"},
    "scheme1": {"temporal_backbone": "transformer", "spatial_backbone": "transformer", "kappa_backbone": "transformer"},
    "scheme2": {"temporal_backbone": "transformer", "spatial_backbone": "transformer", "kappa_backbone": "fcnn"},
    "scheme3": {"temporal_backbone": "transformer", "spatial_backbone": "fcnn", "kappa_backbone": "fcnn"},
    "scheme4": {"temporal_backbone": "transformer", "spatial_backbone": "transformer", "kappa_backbone": "mlp"},
    "scheme5": {"temporal_backbone": "fcnn", "spatial_backbone": "transformer", "kappa_backbone": "fcnn"},
}

FCNN_WEIGHTS = ROOT / "st-nkf-fcnn-h128-5090-20261114/run/weights"
SANJAC_WEIGHTS = ROOT / "st-nkf-gpu-4090/weights"
SIX_ROOT = ROOT / "st-nkf-gpu-six"
COMCAT_WEIGHTS = ROOT / "st-nkf-comcat-5090-20261114/run/ComCat/weights"
SCHEME5_ROOT = ROOT / "st-nkf-scheme5-spatial-tf-20261117"

# Exact totals stored at the original training eval (mean LL_ts * n).
OLD_TEST = {
    ("SanJac", "fcnn"): {"LL_ts": -4.832993507385254, "LL_t": 1.2990045547485352, "LL_s": -6.131998062133789, "n": 4400, "total": -21265.171432495117, "mu": 9.253287862520665e-05},
    ("SanJac", "scheme1"): {"LL_ts": -4.660050392150879, "LL_t": 1.2638661861419678, "LL_s": -5.923916816711426, "n": 4400, "total": -20504.221725463867, "mu": 6.27981498837471e-05},
    ("SanJac", "scheme2"): {"LL_ts": -4.66646671295166, "LL_t": 1.2646692991256714, "LL_s": -5.931136131286621, "n": 4400, "total": -20532.453536987305, "mu": 5.975404928904027e-05},
    ("SanJac", "scheme3"): {"LL_ts": -4.7580885887146, "LL_t": 1.4251023530960083, "LL_s": -6.183190822601318, "n": 4400, "total": -20935.58979034424, "mu": 9.299802331952378e-05},
    ("SanJac", "scheme4"): {"LL_ts": -4.66120719909668, "LL_t": 1.2708419561386108, "LL_s": -5.932048797607422, "n": 4400, "total": -20509.31167602539, "mu": 6.319830572465435e-05},
    ("SCEDC_30", "fcnn"): {"LL_ts": -6.222962856292725, "LL_t": 1.776139736175537, "LL_s": -7.999102592468262, "n": 1898, "total": -11811.183501243591, "mu": 3.784946613905049e-07},
    ("SCEDC_30", "scheme1"): {"LL_ts": -5.861582279205322, "LL_t": 2.072472095489502, "LL_s": -7.934054851531982, "n": 1898, "total": -11125.283165931702, "mu": 6.158618504059632e-08},
    ("SCEDC_30", "scheme2"): {"LL_ts": -5.871125221252441, "LL_t": 2.083517551422119, "LL_s": -7.9546427726745605, "n": 1898, "total": -11143.395669937134, "mu": 6.425275245192097e-08},
    ("SCEDC_30", "scheme3"): {"LL_ts": -5.971762657165527, "LL_t": 1.967586874961853, "LL_s": -7.93934965133667, "n": 1898, "total": -11334.40552330017, "mu": 3.9474883806178696e-07},
    ("SCEDC_30", "scheme4"): {"LL_ts": -5.841712951660156, "LL_t": 2.0409607887268066, "LL_s": -7.882673740386963, "n": 1898, "total": -11087.571182250977, "mu": 7.555797765235184e-08},
    ("SCEDC_25", "fcnn"): {"LL_ts": -6.050898551940918, "LL_t": 2.0316953659057617, "LL_s": -8.08259391784668, "n": 5000, "total": -30254.49275970459, "mu": 9.663220907896175e-07},
    ("SCEDC_25", "scheme1"): {"LL_ts": -5.603105545043945, "LL_t": 2.268159866333008, "LL_s": -7.8712663650512695, "n": 5000, "total": -28015.527725219727, "mu": 1.7064549240330962e-07},
    ("SCEDC_25", "scheme2"): {"LL_ts": -5.6102142333984375, "LL_t": 2.440286636352539, "LL_s": -8.050501823425293, "n": 5000, "total": -28051.071166992188, "mu": 2.145715995993669e-07},
    ("SCEDC_25", "scheme3"): {"LL_ts": -5.777169704437256, "LL_t": 2.281672716140747, "LL_s": -8.058842658996582, "n": 5000, "total": -28885.84852218628, "mu": 1.2408083875925513e-06},
    ("SCEDC_25", "scheme4"): {"LL_ts": -5.610054969787598, "LL_t": 2.2594261169433594, "LL_s": -7.869481086730957, "n": 5000, "total": -28050.27484893799, "mu": 2.2926447229565383e-07},
    ("SCEDC_20", "fcnn"): {"LL_ts": -5.7214131355285645, "LL_t": 2.521883249282837, "LL_s": -8.243295669555664, "n": 13067, "total": -74761.70544195175, "mu": 1.5596505136272754e-06},
    ("SCEDC_20", "scheme1"): {"LL_ts": -5.304586410522461, "LL_t": 2.8161942958831787, "LL_s": -8.120780944824219, "n": 13067, "total": -69315.030626297, "mu": 1.3692377365259745e-07},
    ("SCEDC_20", "scheme2"): {"LL_ts": -5.294692039489746, "LL_t": 2.8842861652374268, "LL_s": -8.178977966308594, "n": 13067, "total": -69185.74088001251, "mu": 2.3024632866963657e-07},
    ("SCEDC_20", "scheme3"): {"LL_ts": -5.495426177978516, "LL_t": 2.730116367340088, "LL_s": -8.225542068481445, "n": 13067, "total": -71808.73386764526, "mu": 2.3751861135679064e-06},
    ("SCEDC_20", "scheme4"): {"LL_ts": -5.2916646003723145, "LL_t": 2.837357997894287, "LL_s": -8.129022598266602, "n": 13067, "total": -69146.18133306503, "mu": 2.4473507664879435e-07},
    ("SaltonSea", "fcnn"): {"LL_ts": -0.5338860154151917, "LL_t": 2.3939316272735596, "LL_s": -2.9278175830841064, "n": 4104, "total": -2191.0682072639465, "mu": 3.2117335649672896e-05},
    ("SaltonSea", "scheme1"): {"LL_ts": -0.16426555812358856, "LL_t": 2.525089740753174, "LL_s": -2.6893556118011475, "n": 4104, "total": -674.1458505392075, "mu": 2.2882388293510303e-05},
    ("SaltonSea", "scheme2"): {"LL_ts": -0.1572764664888382, "LL_t": 2.59037446975708, "LL_s": -2.7476508617401123, "n": 4104, "total": -645.462618470192, "mu": 1.980541492230259e-05},
    ("SaltonSea", "scheme3"): {"LL_ts": -0.38461682200431824, "LL_t": 2.5809519290924072, "LL_s": -2.965568780899048, "n": 4104, "total": -1578.467437505722, "mu": 4.334715777076781e-05},
    ("SaltonSea", "scheme4"): {"LL_ts": -0.17121878266334534, "LL_t": 2.4989564418792725, "LL_s": -2.670175075531006, "n": 4104, "total": -702.6818840503693, "mu": 2.4792383555904962e-05},
    ("WHITE", "fcnn"): {"LL_ts": -2.7567784786224365, "LL_t": 2.2304513454437256, "LL_s": -4.987229824066162, "n": 24080, "total": -66383.22576522827, "mu": 0.0002150695800082758},
    ("WHITE", "scheme1"): {"LL_ts": -2.575941801071167, "LL_t": 2.427493095397949, "LL_s": -5.003435134887695, "n": 24080, "total": -62028.6785697937, "mu": 0.00012076225539203733},
    ("WHITE", "scheme2"): {"LL_ts": -2.5795819759368896, "LL_t": 2.410609483718872, "LL_s": -4.990191459655762, "n": 24080, "total": -62116.3339805603, "mu": 0.00011184185859747231},
    ("WHITE", "scheme3"): {"LL_ts": -2.674067497253418, "LL_t": 2.3776443004608154, "LL_s": -5.051711559295654, "n": 24080, "total": -64391.545333862305, "mu": 0.00023288052761927247},
    ("WHITE", "scheme4"): {"LL_ts": -2.580540180206299, "LL_t": 2.3492772579193115, "LL_s": -4.929817199707031, "n": 24080, "total": -62139.407539367676, "mu": 0.00012111739488318563},
    ("ComCat", "fcnn"): {"LL_ts": -7.435534477233887, "LL_t": 1.3556784391403198, "LL_s": -8.791213035583496, "n": 21885, "total": -162726.6720342636, "mu": 1.5916531992843375e-05},
    ("ComCat", "scheme1"): {"LL_ts": -7.210219383239746, "LL_t": 1.52653968334198, "LL_s": -8.736759185791016, "n": 21885, "total": -157795.65120220184, "mu": 1.54960471263621e-05},
    ("ComCat", "scheme2"): {"LL_ts": -7.212661266326904, "LL_t": 1.4924029111862183, "LL_s": -8.705063819885254, "n": 21885, "total": -157849.0918135643, "mu": 1.5462443116120994e-05},
    ("ComCat", "scheme3"): {"LL_ts": -7.261841773986816, "LL_t": 1.48793363571167, "LL_s": -8.749775886535645, "n": 21885, "total": -158925.40722370148, "mu": 1.565382262924686e-05},
    ("ComCat", "scheme4"): {"LL_ts": -7.2098846435546875, "LL_t": 1.4959665536880493, "LL_s": -8.705850601196289, "n": 21885, "total": -157788.32542419434, "mu": 1.549196895211935e-05},
    ("SCEDC_30", "scheme5"): {"LL_ts": -5.864631175994873, "LL_t": 3.3321115970611572, "LL_s": -9.19674301147461, "n": 1898, "total": -11131.069972038269, "mu": 1.2413360650498362e-07},
    ("SanJac", "scheme5"): {"LL_ts": -4.729705810546875, "LL_t": 2.1568644046783447, "LL_s": -6.886569976806641, "n": 4400, "total": -20810.70556640625, "mu": 4.032108699902892e-05},
    ("WHITE", "scheme5"): {"LL_ts": -2.600156307220459, "LL_t": 3.871072769165039, "LL_s": -6.47122859954834, "n": 24080, "total": -62611.76387786865, "mu": 9.8130498372484e-05},
    ("SCEDC_25", "scheme5"): {"LL_ts": -5.5955657958984375, "LL_t": 3.606903076171875, "LL_s": -9.202468872070312, "n": 5000, "total": -27977.828979492188, "mu": 2.1843101194463088e-07},
    ("SaltonSea", "scheme5"): {"LL_ts": -0.19078439474105835, "LL_t": 4.033719062805176, "LL_s": -4.224503040313721, "n": 4104, "total": -782.9791560173035, "mu": 1.000675001705531e-05},
    ("ComCat", "scheme5"): {"LL_ts": -7.2142815589904785, "LL_t": 2.794207811355591, "LL_s": -10.008489608764648, "n": 21885, "total": -157884.55191850662, "mu": 1.5358409655163996e-05},
    ("SCEDC_20", "scheme5"): {"LL_ts": -5.268044471740723, "LL_t": 4.083512306213379, "LL_s": -9.351556777954102, "n": 13067, "total": -68837.53711223602, "mu": 8.746283697291801e-08},
}

COMPARISONS = (
    ("scheme3_minus_fcnn", "scheme3", "fcnn"),
    ("scheme2_minus_scheme3", "scheme2", "scheme3"),
    ("scheme1_minus_scheme2", "scheme1", "scheme2"),
    ("scheme4_minus_scheme2", "scheme4", "scheme2"),
    ("scheme5_minus_fcnn", "scheme5", "fcnn"),
    ("scheme2_minus_scheme5", "scheme2", "scheme5"),
)


def checkpoint_path(dataset: str, model_id: str) -> Path:
    if model_id == "fcnn":
        return FCNN_WEIGHTS / f"stnkf_fcnn_{dataset}_H128_seed1_best.pt"
    name = f"{model_id}_H128_seed1_best.pt"
    if model_id == "scheme5":
        return SCHEME5_ROOT / dataset / "weights" / name
    if dataset == "SanJac":
        return SANJAC_WEIGHTS / name
    if dataset == "ComCat":
        return COMCAT_WEIGHTS / name
    return SIX_ROOT / dataset / "weights" / name


def load_checkpoint(path: Path, device: torch.device) -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise FileNotFoundError(f"{path} is not a save_weights / state_dict checkpoint")
    missing = [k for k in ("state_dict", "max_time", "max_mag", "max_dis") if k not in ckpt]
    mu = None
    if "bac_func.mu" in ckpt["state_dict"]:
        mu = float(ckpt["state_dict"]["bac_func.mu"])
    elif "mu" in ckpt:
        mu = float(ckpt["mu"])
    else:
        missing.append("mu")
    if missing:
        raise FileNotFoundError(f"{path} missing required fields: {missing}")
    ckpt["_mu"] = mu
    return ckpt


def build_model(prep: dict, model_id: str, device: torch.device) -> KernelPointProcess:
    flags = BACKBONES[model_id]
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
    model.to(device)
    return model


def spatial_params_for(model: KernelPointProcess, history_context):
    if model.spatial_id == "neural" and model.spatial_backbone == "transformer":
        return model.spatial_net.encode_context(history_context)
    return None


def history_context_for(model: KernelPointProcess, time_1, time_2, mags):
    if not model._needs_history_context():
        return None
    lower_t = torch.cat((time_2, torch.zeros_like(mags[:, -1:, :])), dim=1)
    return torch.cat((lower_t / model.max_time, mags / model.max_mag), dim=-1)


def predict_split_with_region(model: KernelPointProcess, split: str, batch_size: int) -> dict:
    """Same loop as KernelPointProcess._predict_split, plus region-consistent
    λ_t = μA + Σ κ h(t) m_i from the identical forward tensors and the same
    _boundary_mass used by the official compensator. Official l_ts / l_t /
    int_l are stored unchanged.
    """
    n = int(model._arrays(split)[0].shape[0])
    keys = (
        "l_ts",
        "l_t",
        "int_l",
        "pdf_temporal",
        "cdf_temporal",
        "pdf_spatial",
        "cdf_spatial",
        "kappa",
    )
    buckets = {key: [] for key in keys}
    region = []
    model.eval()
    for start in range(0, n, batch_size):
        idx = np.arange(start, min(start + batch_size, n))
        if not model.use_boundary_correction:
            raise RuntimeError("region-consistent λ_t requires boundary correction")
        time_1, time_2, mags, dis, hist_x, hist_y = model._batch_tensors(
            split, idx, with_locations=True
        )
        outputs = model.forward(
            time_1, time_2, mags, dis, create_graph=False,
            hist_x=hist_x, hist_y=hist_y,
        )
        ctx = history_context_for(model, time_1, time_2, mags)
        params = spatial_params_for(model, ctx)
        mass = model._boundary_mass(hist_x, hist_y, mags, params)
        lam_t_region = (outputs["kappa"] * outputs["pdf_temporal"] * mass).sum(
            dim=1, keepdim=True
        )
        lam_t_region = lam_t_region + model.bac_func.mu * model.area_scalar
        for key in keys:
            buckets[key].append(outputs[key].detach().cpu())
        region.append(lam_t_region.detach().cpu())
        del outputs
        done = min(start + batch_size, n)
        if start == 0 or done == n or (start // batch_size) % 25 == 0:
            print(f"  {split} {done}/{n}", flush=True)
    out = {key: torch.cat(parts, dim=0).numpy() for key, parts in buckets.items()}
    out["l_t_region"] = torch.cat(region, dim=0).numpy()
    return out


def evaluate_one(dataset: str, model_id: str, out: Path, device: torch.device, batch_size: int) -> dict:
    path = checkpoint_path(dataset, model_id)
    if not path.is_file():
        raise FileNotFoundError(f"missing checkpoint: {path}")

    prep = prepare_run(dataset, history=HISTORY)
    if int(prep["input_dim_train"]) != HISTORY:
        raise RuntimeError(f"{dataset} history became {prep['input_dim_train']}, expected {HISTORY}")

    ckpt = load_checkpoint(path, device)
    model = build_model(prep, model_id, device)
    pre_mu = float(model.bac_func.mu.detach())
    pre_norms = {
        "max_time": float(model.max_time),
        "max_mag": float(model.max_mag),
        "max_dis": float(model.max_dis),
        "area": float(model.area_scalar),
    }
    model.load_weights(str(path))
    model.eval()
    loaded_mu = float(model.bac_func.mu.detach())
    if loaded_mu != ckpt["_mu"] and not math.isclose(loaded_mu, ckpt["_mu"], rel_tol=0, abs_tol=0):
        # Allow exact tensor reconstruction only; otherwise fail rather than guess.
        if not np.isclose(loaded_mu, ckpt["_mu"], rtol=0, atol=1e-15):
            raise RuntimeError(f"mu after load ({loaded_mu}) != checkpoint mu ({ckpt['_mu']})")

    norm_notes = {}
    for key in ("max_time", "max_mag", "max_dis"):
        ck_val = float(ckpt[key])
        rec_val = pre_norms[key]
        model_val = float(getattr(model, key))
        norm_notes[key] = {"checkpoint": ck_val, "recomputed_from_data": rec_val, "used": model_val}
        if not np.isclose(ck_val, rec_val, rtol=1e-6, atol=0):
            norm_notes[key]["warning"] = "checkpoint normalization differs from recomputed data stats; using checkpoint"
    if "area" in ckpt and ckpt["area"] is not None:
        ck_area = float(ckpt["area"])
        norm_notes["area"] = {
            "checkpoint": ck_area,
            "recomputed_from_data": pre_norms["area"],
            "used": float(model.area_scalar),
        }

    # PDFs are input-derivatives of the CDFs, so evaluation must keep
    # autograd enabled. model.eval() plus create_graph=False is the
    # official no-update evaluation path; do not wrap this in torch.no_grad().
    outputs = predict_split_with_region(model, "test", batch_size)
    model._store_eval("test", outputs)

    ll_ts = np.asarray(model.LL_ts_test, dtype=np.float64).reshape(-1)
    ll_t = np.asarray(model.LL_t_test, dtype=np.float64).reshape(-1)
    ll_s = np.asarray(model.LL_s_test, dtype=np.float64).reshape(-1)
    lam_ts = np.asarray(model.lam_ts_test, dtype=np.float64).reshape(-1)
    lam_t = np.asarray(model.lam_t_test, dtype=np.float64).reshape(-1)
    int_l = np.asarray(model.Int_lam_test, dtype=np.float64).reshape(-1)
    times = np.asarray(model.cur_t_test, dtype=np.float64).reshape(-1)
    elapsed = np.asarray(model.elapsed_t_test, dtype=np.float64).reshape(-1)
    n = int(ll_ts.size)

    if not np.isfinite(lam_ts).all() or not np.isfinite(lam_t).all() or not np.isfinite(int_l).all():
        raise FloatingPointError(
            f"{dataset}/{model_id}: non-finite intensity or compensator "
            f"(nan lam_ts={np.isnan(lam_ts).sum()}, inf lam_ts={np.isinf(lam_ts).sum()}, "
            f"nan lam_t={np.isnan(lam_t).sum()}, nan int_l={np.isnan(int_l).sum()})"
        )
    if np.any(lam_ts <= 0) or np.any(lam_t <= 0):
        raise FloatingPointError(
            f"{dataset}/{model_id}: non-positive intensity "
            f"(min lam_ts={lam_ts.min()}, min lam_t={lam_t.min()})"
        )

    log_lam_ts = np.log(lam_ts)
    log_lam_t = np.log(lam_t)
    log_cond = log_lam_ts - log_lam_t
    decomp_err = float(np.max(np.abs((ll_ts - ll_t - ll_s))))
    decomp_err_sum = float(abs(ll_ts.sum() - ll_t.sum() - ll_s.sum()))

    # Region-consistent temporal intensity (same compensator, λ_t = ∫_D λ ds).
    lam_t_region = np.asarray(outputs["l_t_region"], dtype=np.float64).reshape(-1)
    if not np.isfinite(lam_t_region).all() or np.any(lam_t_region <= 0):
        raise FloatingPointError(
            f"{dataset}/{model_id}: non-finite or non-positive region λ_t "
            f"(min={np.nanmin(lam_t_region)}, nan={np.isnan(lam_t_region).sum()})"
        )
    log_lam_t_region = np.log(lam_t_region)
    ll_t_region = log_lam_t_region - int_l
    ll_s_region = log_lam_ts - log_lam_t_region
    decomp_err_region = float(np.max(np.abs(ll_ts - ll_t_region - ll_s_region)))

    old = OLD_TEST.get((dataset, model_id))
    old_total = float(old["total"]) if old else float("nan")
    total_ts = float(ll_ts.sum())
    total_t = float(ll_t.sum())
    total_s = float(ll_s.sum())
    repro = total_ts - old_total if old else float("nan")

    t0_official = float(prep["config"]["testing_start"])
    t1_official = float(prep["config"]["testing_end"])
    first_t = float(times[0])
    last_t = float(times[-1])
    # Inter-event compensator coverage: from the event immediately before the
    # first test target through the last test event. No tail after last event.
    prev_of_first = first_t - float(elapsed[0])

    events_dir = out / "events"
    events_dir.mkdir(parents=True, exist_ok=True)
    ev_path = events_dir / f"{dataset}_{model_id}_seed{SEED}_events.csv"
    with ev_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "event_id",
                "time",
                "log_lambda_ts",
                "log_lambda_t",
                "log_conditional_space",
                "log_lambda_t_region",
                "log_conditional_space_region",
                "compensator_increment",
            ]
        )
        for i in range(n):
            w.writerow(
                [
                    i,
                    f"{times[i]:.16g}",
                    f"{log_lam_ts[i]:.16g}",
                    f"{log_lam_t[i]:.16g}",
                    f"{log_cond[i]:.16g}",
                    f"{log_lam_t_region[i]:.16g}",
                    f"{ll_s_region[i] + 0.0:.16g}",
                    f"{int_l[i]:.16g}",
                ]
            )

    compensator = {
        "dataset": dataset,
        "model": model_id,
        "seed": SEED,
        "n_test": n,
        "official_testing_start": t0_official,
        "official_testing_end": t1_official,
        "first_test_event_time": first_t,
        "last_test_event_time": last_t,
        "compensator_interval_start": prev_of_first,
        "compensator_interval_end": last_t,
        "sum_compensator_increments": float(int_l.sum()),
        "sum_elapsed_interevent": float(elapsed.sum()),
        "tail_after_last_event_to_official_end": t1_official - last_t,
        "head_before_first_event_from_official_start": first_t - t0_official,
        "notes": (
            "Official scoring (eval_test) subtracts the inter-event compensator "
            "increment int_l for each test event and does not add a tail integral "
            "from the last test event to official testing_end. The first increment "
            "starts at the event immediately before the first test target, which "
            "may precede official testing_start. This is the original table metric."
        ),
    }
    comp_path = events_dir / f"{dataset}_{model_id}_seed{SEED}_compensator.json"
    comp_path.write_text(json.dumps(compensator, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    row = {
        "dataset": dataset,
        "model": model_id,
        "seed": SEED,
        "checkpoint_path": str(path),
        "n_test": n,
        "LL_ts_total": total_ts,
        "LL_t_total": total_t,
        "LL_s_given_t_total": total_s,
        "LL_ts_mean": total_ts / n,
        "LL_t_mean": total_t / n,
        "LL_s_given_t_mean": total_s / n,
        "decomposition_error": decomp_err,
        "decomposition_error_sum": decomp_err_sum,
        "old_LL_ts": old_total,
        "reproduction_difference": repro,
        "LL_t_region_total": float(ll_t_region.sum()),
        "LL_s_given_t_region_total": float(ll_s_region.sum()),
        "LL_t_region_mean": float(ll_t_region.mean()),
        "LL_s_given_t_region_mean": float(ll_s_region.mean()),
        "decomposition_error_region": decomp_err_region,
        "mu_checkpoint": ckpt["_mu"],
        "mu_loaded": loaded_mu,
        "mu_before_load": pre_mu,
        "intensity_clipping": False,
        "min_lam_ts": float(lam_ts.min()),
        "min_lam_t": float(lam_t.min()),
        "min_lam_t_region": float(lam_t_region.min()),
        "events_csv": str(ev_path),
        "compensator_json": str(comp_path),
        "normalization": norm_notes,
        "use_boundary_correction": True,
        "batch_size": batch_size,
        "history": HISTORY,
    }
    meta_path = events_dir / f"{dataset}_{model_id}_seed{SEED}_meta.json"
    meta_path.write_text(json.dumps(row, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(
        f"{dataset}/{model_id}: n={n} LL_ts={total_ts:.6f} old={old_total:.6f} "
        f"diff={repro:.4e} decomp={decomp_err:.4e} mu={loaded_mu:.6e}",
        flush=True,
    )
    del model
    return row


SUMMARY_FIELDS = [
    "dataset", "model", "seed", "checkpoint_path", "n_test",
    "LL_ts_total", "LL_t_total", "LL_s_given_t_total",
    "LL_ts_mean", "LL_t_mean", "LL_s_given_t_mean",
    "decomposition_error", "old_LL_ts", "reproduction_difference",
    "LL_t_region_total", "LL_s_given_t_region_total",
    "LL_t_region_mean", "LL_s_given_t_region_mean",
    "decomposition_error_region", "mu_loaded",
]


def write_summary(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_comparisons(path: Path, rows: list[dict]) -> tuple[list[dict], list[str]]:
    by = {(r["dataset"], r["model"]): r for r in rows}
    out_rows = []
    issues = []
    for dataset in DATASETS:
        for name, plus, minus in COMPARISONS:
            a = by.get((dataset, plus))
            b = by.get((dataset, minus))
            if a is None or b is None:
                issues.append(f"{dataset}: skip {name} (missing {plus if a is None else minus})")
                continue
            if a["n_test"] != b["n_test"]:
                issues.append(f"{dataset}: {name} n_test mismatch {a['n_test']} vs {b['n_test']}")
                continue
            n = a["n_test"]
            # Means are already totals / n_test; do not divide by n again.
            def deltas(key_t, key_s, key_ts):
                dt = a[key_t] - b[key_t]
                ds = a[key_s] - b[key_s]
                dts = a[key_ts] - b[key_ts]
                return dt, ds, dts, dts - (dt + ds)
            dt, ds, dts, chk = deltas("LL_t_mean", "LL_s_given_t_mean", "LL_ts_mean")
            dtr, dsr, dtsr, chkr = deltas(
                "LL_t_region_mean", "LL_s_given_t_region_mean", "LL_ts_mean"
            )
            out_rows.append({
                "dataset": dataset,
                "contrast": name,
                "model_plus": plus,
                "model_minus": minus,
                "n_test": n,
                "unit": "nats/event",
                "dLL_t_per_event": dt,
                "dLL_s_given_t_per_event": ds,
                "dLL_ts_per_event": dts,
                "gain_identity_error": chk,
                "dLL_t_region_per_event": dtr,
                "dLL_s_given_t_region_per_event": dsr,
                "dLL_ts_region_per_event": dtsr,
                "gain_identity_error_region": chkr,
            })
    fields = list(out_rows[0].keys()) if out_rows else [
        "dataset", "contrast", "model_plus", "model_minus", "n_test", "unit",
        "dLL_t_per_event", "dLL_s_given_t_per_event", "dLL_ts_per_event",
        "gain_identity_error",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    return out_rows, issues


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, default=None, help="ST-NKF-pytorch 代码与 Datasets 所在目录")
    p.add_argument("--out", type=Path, default=ROOT / "st-nkf-ll-decomposition-20261117")
    p.add_argument("--datasets", nargs="+", default=None, choices=list(DATASETS))
    p.add_argument("--models", nargs="+", default=None, choices=list(MODEL_ORDER))
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    setup_repo(args.repo)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    datasets = list(args.datasets) if args.datasets else list(DATASETS)
    models = list(args.models) if args.models else list(MODEL_ORDER)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    missing_path = out / "missing_checkpoints.txt"
    missing = []
    rows = []
    summary_path = out / "summary.csv"

    # Resume: keep already completed rows if the script is re-run.
    done = set()
    if summary_path.is_file():
        with summary_path.open(encoding="utf-8") as fh:
            for rec in csv.DictReader(fh):
                rows.append(rec)
                # coerce numerics later when writing comparisons
                done.add((rec["dataset"], rec["model"]))
        # convert numeric fields back
        for rec in rows:
            for k, v in list(rec.items()):
                if k in ("dataset", "model", "checkpoint_path"):
                    continue
                if k in ("seed", "n_test"):
                    rec[k] = int(float(v))
                elif k not in ("events_csv", "compensator_json"):
                    try:
                        rec[k] = float(v)
                    except (TypeError, ValueError):
                        pass

    for dataset in datasets:
        for model_id in models:
            if (dataset, model_id) in done:
                print(f"skip completed {dataset}/{model_id}", flush=True)
                continue
            path = checkpoint_path(dataset, model_id)
            if not path.is_file():
                msg = f"{dataset}/{model_id}: missing {path}"
                print("MISSING", msg, flush=True)
                missing.append(msg)
                continue
            try:
                row = evaluate_one(dataset, model_id, out, device, args.batch_size)
                rows.append(row)
                done.add((dataset, model_id))
                write_summary(summary_path, rows)
            except Exception as exc:
                msg = f"{dataset}/{model_id}: {type(exc).__name__}: {exc}"
                print("ERROR", msg, flush=True)
                missing.append(msg)
                (out / "errors.log").write_text(
                    "\n".join(missing) + "\n", encoding="utf-8"
                ) if False else Path(out / "errors.log").open("a", encoding="utf-8").write(msg + "\n")

    write_summary(summary_path, rows)
    missing_path.write_text(("\n".join(missing) + "\n") if missing else "", encoding="utf-8")
    if rows:
        write_comparisons(out / "comparison.csv", rows)
    print(f"wrote {summary_path} rows={len(rows)} missing={len(missing)}", flush=True)
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
