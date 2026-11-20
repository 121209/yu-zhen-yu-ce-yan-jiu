"""Train ETAS or ST-NKF on an EarthquakeNPP catalog.

PyTorch reimplementation of zhancx-edu/NeuralKernelFunction ``run_models.py``.

Command line (README / this port):

    python run_models.py <seed> <gpu_id> <dataset> <kappa> <temporal> <spatial>

``kappa``, ``temporal`` and ``spatial`` are each ``empirical`` or ``neural``.
The upstream script assigns those three positions in the opposite component
order (temporal, spatial, kappa). All-empirical and all-neural commands from
the original README are unchanged. Mixed ablations should follow the order
printed by ``--help``.

``gpu_id`` is a CUDA device index. Use ``-1`` or ``cpu`` when no GPU is
available. Optional flags ``--epochs``, ``--batch-size`` and ``--patience``
default to the paper settings (1000, 128, 20). ``--backbone`` is ``fcnn``
(default, paper path) or ``transformer`` (scheme 2: temporal and spatial
Transformer kernels, kappa stays FCNN, plus Omori CDF / boundary / mu-init
upgrades). Per-kernel flags ``--temporal-backbone``, ``--spatial-backbone``
and ``--kappa-backbone`` select mixed schemes. Default FCNN/ETAS is unchanged.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

from config import DATASET_CONFIGS
from utils import azimuthal_equidistant_projection, quadrilateral_area

logger = logging.getLogger(__name__)

DATASETS = (
    "ComCat",
    "SaltonSea",
    "SanJac",
    "WHITE",
    "SCEDC_20",
    "SCEDC_25",
    "SCEDC_30",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a KernelPointProcess (ETAS, ST-NKF, or a hybrid)."
    )
    parser.add_argument("seed", type=int, help="NumPy / PyTorch / kernel-init seed")
    parser.add_argument(
        "gpu_id",
        type=str,
        help="CUDA device id, or -1/cpu to force CPU",
    )
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("kappa", choices=("empirical", "neural"), help="productivity")
    parser.add_argument("temporal", choices=("empirical", "neural"))
    parser.add_argument("spatial", choices=("empirical", "neural"))
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Adam learning rate. Default 0.005 for FCNN/ETAS, 1e-4 for transformer/MLP schemes.",
    )
    parser.add_argument(
        "--mu-batch-size",
        type=int,
        default=512,
        help="Batch size for the end-of-epoch background-rate update",
    )
    parser.add_argument(
        "--no-patterns",
        action="store_true",
        help="Skip the sorted kernel-curve .npy dumps under results/",
    )
    parser.add_argument(
        "--backbone",
        choices=("fcnn", "transformer"),
        default="fcnn",
        help="Shortcut: fcnn = all FCNN (paper path). transformer = scheme 2 "
        "(temporal+spatial Transformer, kappa FCNN) plus the three upgrades. "
        "Default fcnn preserves the paper reproduction path.",
    )
    parser.add_argument(
        "--temporal-backbone",
        choices=("fcnn", "transformer"),
        default=None,
        help="Override the temporal-kernel family. Default follows --backbone.",
    )
    parser.add_argument(
        "--spatial-backbone",
        choices=("fcnn", "transformer"),
        default=None,
        help="Override the spatial-kernel family. Default follows --backbone.",
    )
    parser.add_argument(
        "--kappa-backbone",
        choices=("fcnn", "transformer", "mlp"),
        default=None,
        help="Productivity network: fcnn (5x32, default), history Transformer, "
        "or a deeper unconstrained MLP.",
    )
    parser.add_argument(
        "--history",
        type=int,
        default=None,
        help="Limit history length H. Default uses the official auxiliary window.",
    )
    parser.add_argument(
        "--boundary-angles",
        type=int,
        default=256,
        help="Polar quadrature angles for the finite-region spatial mass m_i.",
    )
    parser.add_argument(
        "--no-boundary-correction",
        action="store_true",
        help="Disable the finite-region boundary-mass correction (upgraded schemes only).",
    )
    parser.add_argument(
        "--no-mu-rate-init",
        action="store_true",
        help="Leave mu at 1.0 instead of initializing from the mean event rate / area.",
    )
    args = parser.parse_args(argv)
    _finalize_args(args)
    return args


def _upgraded_scheme(args) -> bool:
    return (
        args.temporal_backbone == "transformer"
        or args.spatial_backbone == "transformer"
        or args.kappa_backbone in ("transformer", "mlp")
    )


def _finalize_args(args) -> None:
    """Resolve per-kernel backbones and the default learning rate."""
    if args.backbone == "transformer":
        defaults = ("transformer", "transformer", "fcnn")
    else:
        defaults = ("fcnn", "fcnn", "fcnn")
    if args.temporal_backbone is None:
        args.temporal_backbone = defaults[0]
    if args.spatial_backbone is None:
        args.spatial_backbone = defaults[1]
    if args.kappa_backbone is None:
        args.kappa_backbone = defaults[2]
    if args.lr is None:
        args.lr = 1e-4 if _upgraded_scheme(args) else 0.005


def _run_tag(args) -> str:
    """Catalog / kernel tag used in weights and CSV names.

    The default FCNN path keeps the original filename so reproduction
    outputs are not renamed. Mixed or transformer schemes append the
    per-kernel backbones and, when set, the history length.
    """
    tag = f"{args.dataset}_{args.temporal}_{args.spatial}_{args.kappa}"
    tb = getattr(args, "temporal_backbone", "fcnn")
    sb = getattr(args, "spatial_backbone", "fcnn")
    kb = getattr(args, "kappa_backbone", "fcnn")
    if (tb, sb, kb) != ("fcnn", "fcnn", "fcnn"):
        tag = f"{tag}_t-{tb}_s-{sb}_k-{kb}"
    history = getattr(args, "history", None)
    if history is not None:
        tag = f"{tag}_H{int(history)}"
    return tag


def configure_device(gpu_id: str):
    """Set CUDA_VISIBLE_DEVICES before torch is imported."""
    if gpu_id.lower() in {"-1", "cpu", "none"}:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id


def prepare_run(datasets_id: str, history: int | None = None) -> dict:
    """Load one catalog and cut the same index windows as the TensorFlow script.

    ``history`` limits the number of preceding events used as the kernel
    history (H). The official train/val/test *targets* are unchanged; only
    the left edge of each slice shortens from the full auxiliary window to
    the last ``history`` auxiliary events. ``history=None`` keeps the paper
    window (H = auxiliary_num).
    """
    if datasets_id not in DATASET_CONFIGS:
        raise KeyError(datasets_id)
    config = DATASET_CONFIGS[datasets_id]
    raw_catalog = pd.read_csv(config["catalog_path"])
    cat_shape = np.load(config["shape_path"])
    if datasets_id in ("SCEDC_20", "SCEDC_25", "SCEDC_30"):
        min_magnitude = float(datasets_id.split("_")[-1]) / 10.0
        raw_catalog = raw_catalog[raw_catalog["magnitude"] >= min_magnitude].reset_index(
            drop=True
        )

    times = raw_catalog["time_days"].to_numpy()
    if len(times) > 1 and np.any(times[1:] < times[:-1]):
        raise ValueError(f"{datasets_id} catalog is not sorted by time_days")

    auxiliary_num = int(
        np.sum(
            (times >= config["auxiliary_start"]) & (times < config["training_start"])
        )
    )
    training_num = int(
        np.sum(
            (times >= config["training_start"]) & (times < config["validation_start"])
        )
    )
    validation_num = int(
        np.sum(
            (times >= config["validation_start"]) & (times < config["testing_start"])
        )
    )
    testing_num = int(
        np.sum((times >= config["testing_start"]) & (times <= config["testing_end"]))
    )
    if auxiliary_num < 2 or training_num < 1 or validation_num < 1 or testing_num < 1:
        raise ValueError(
            f"Empty split for {datasets_id}: auxiliary={auxiliary_num}, "
            f"training={training_num}, validation={validation_num}, testing={testing_num}"
        )

    # Count-based slices. Valid when in-window events form a prefix, which
    # holds for these catalogs (SCEDC may have a few rows after testing_end).
    if history is None:
        input_dim = auxiliary_num
    else:
        if int(history) < 2:
            raise ValueError("history must be >= 2")
        input_dim = min(int(history), auxiliary_num)
    input_dim_train = input_dim
    input_dim_val = input_dim
    input_dim_test = input_dim
    train_slice = slice(auxiliary_num - input_dim_train, auxiliary_num + training_num)
    val_slice = slice(
        auxiliary_num + training_num - input_dim_val,
        auxiliary_num + training_num + validation_num,
    )
    test_slice = slice(
        auxiliary_num + training_num + validation_num - input_dim_test,
        auxiliary_num + training_num + validation_num + testing_num,
    )

    def pack(slc: slice) -> dict:
        return {
            "t": raw_catalog["time_days"].to_numpy()[slc],
            "m": raw_catalog["magnitude"].to_numpy()[slc],
            "x": raw_catalog["x"].to_numpy()[slc],
            "y": raw_catalog["y"].to_numpy()[slc],
        }

    center_latitude = float(raw_catalog["latitude"].mean())
    center_longitude = float(raw_catalog["longitude"].mean())
    cat_shape_x, cat_shape_y = azimuthal_equidistant_projection(
        cat_shape[:, 0], cat_shape[:, 1], center_latitude, center_longitude
    )
    cat_shape_xy = np.stack((cat_shape_x, cat_shape_y), axis=1)
    area = quadrilateral_area(cat_shape_xy)

    return {
        "config": config,
        "raw_catalog": raw_catalog,
        "auxiliary_num": auxiliary_num,
        "training_num": training_num,
        "validation_num": validation_num,
        "testing_num": testing_num,
        "input_dim_train": input_dim_train,
        "input_dim_val": input_dim_val,
        "input_dim_test": input_dim_test,
        "train": pack(train_slice),
        "val": pack(val_slice),
        "test": pack(test_slice),
        "area": area,
        "region_xy": cat_shape_xy,
        "history": input_dim,
    }


def _write_log_likelihood(
    datasets_id, temporal_id, spatial_id, kappa_id, model, seed: int, backbone: str = "fcnn"
) -> None:
    line = (
        f"{datasets_id}, {temporal_id}, {spatial_id}, {kappa_id}, "
        f"{model.LL_ts_average_test}, {model.LL_t_average_test}, "
        f"{model.LL_s_average_test}, seed={seed}"
    )
    if backbone != "fcnn":
        line += f", backbone={backbone}"
    line += "\n"
    with open("ll_seismic_model.log", "a", encoding="utf-8") as log_file:
        log_file.write(line)


def _scalar(tensor) -> float:
    return float(tensor.detach().cpu())


def _write_parameters(
    datasets_id, temporal_id, spatial_id, kappa_id, model, seed: int, backbone: str = "fcnn"
) -> None:
    fields = [
        datasets_id,
        f"seed={seed}",
        temporal_id,
        spatial_id,
        kappa_id,
        f"mu={_scalar(model.bac_func.mu):.8g}",
    ]
    if backbone != "fcnn":
        fields.append(f"backbone={backbone}")
    if kappa_id == "empirical":
        fields.append(f"A={_scalar(model.produc_func.mag_A):.8g}")
        fields.append(f"alpha={_scalar(model.produc_func.mag_alpha):.8g}")
    if temporal_id == "empirical":
        fields.append(f"c={_scalar(model.omori_func.omori_c):.8g}")
        fields.append(f"p={_scalar(model.omori_func.omori_p):.8g}")
    if spatial_id == "empirical":
        fields.append(f"q={_scalar(model.spatial_func.spatial_q):.8g}")
        fields.append(f"D2={_scalar(model.spatial_func.spatial_D2):.8g}")
        fields.append(f"gamma={_scalar(model.spatial_func.spatial_gamma):.8g}")
    with open("etas_parameters.log", "a", encoding="utf-8") as log_file:
        log_file.write(", ".join(fields) + "\n")


def _write_catalog_csv(
    prep, model, datasets_id, temporal_id, spatial_id, kappa_id, backbone: str = "fcnn"
) -> None:
    results = prep["raw_catalog"].copy()
    aux = prep["auxiliary_num"]
    n_train = prep["training_num"]
    n_val = prep["validation_num"]
    n_test = prep["testing_num"]
    results.loc[aux : aux + n_train - 1, "step"] = "train"
    results.loc[aux + n_train : aux + n_train + n_val - 1, "step"] = "val"
    results.loc[
        aux + n_train + n_val : aux + n_train + n_val + n_test - 1,
        "step",
    ] = "test"

    blocks = {
        "train": (
            aux,
            n_train,
            model.LL_t_train,
            model.LL_s_train,
            model.LL_ts_train,
            np.log(model.lam_t_train),
            np.log(model.lam_ts_train),
            model.Int_lam_train,
        ),
        "val": (
            aux + n_train,
            n_val,
            model.LL_t_val,
            model.LL_s_val,
            model.LL_ts_val,
            np.log(model.lam_t_val),
            np.log(model.lam_ts_val),
            model.Int_lam_val,
        ),
        "test": (
            aux + n_train + n_val,
            n_test,
            model.LL_t_test,
            model.LL_s_test,
            model.LL_ts_test,
            np.log(model.lam_t_test),
            np.log(model.lam_ts_test),
            model.Int_lam_test,
        ),
    }
    for _name, (start, count, logli_t, logli_s, logli_ts, loglam_t, loglam_ts, intlam) in blocks.items():
        end = start + count - 1
        results.loc[start:end, "logli_t"] = np.asarray(logli_t).reshape(-1)
        results.loc[start:end, "logli_s"] = np.asarray(logli_s).reshape(-1)
        results.loc[start:end, "logli_ts"] = np.asarray(logli_ts).reshape(-1)
        results.loc[start:end, "loglam_t"] = np.asarray(loglam_t).reshape(-1)
        results.loc[start:end, "loglam_ts"] = np.asarray(loglam_ts).reshape(-1)
        results.loc[start:end, "intlam"] = np.asarray(intlam).reshape(-1)

    os.makedirs("csv", exist_ok=True)
    base = f"{datasets_id}_{temporal_id}_{spatial_id}_{kappa_id}"
    if backbone in ("fcnn", base) or backbone is None:
        tag = base
    elif str(backbone).startswith(base):
        tag = str(backbone)
    else:
        tag = f"{base}_{backbone}"
    path = f"csv/{tag}.csv"
    results.to_csv(path, index=False, encoding="utf-8")
    logger.info("Results saved to %s", path)


def _write_patterns(model, tag: str) -> None:
    os.makedirs("results", exist_ok=True)

    def dump(stem: str, values: np.ndarray, reference: np.ndarray) -> None:
        flat_ref = np.asarray(reference[:-1]).reshape(-1)
        order = np.argsort(flat_ref)
        flat_val = np.asarray(values[:-1]).reshape(-1)
        np.save(f"results/{tag}_{stem}.npy", flat_val[order])

    dump("dis_xy_train", model.dis_xy_train, model.dis_xy_train)
    dump("pdf_spatial_train", model.pdf_spatial_train, model.dis_xy_train)
    dump("cdf_spatial_train", model.cdf_spatial_train, model.dis_xy_train)
    dump("dis_t_train", model.dis_t1_train, model.dis_t1_train)
    dump("pdf_temporal_train", model.pdf_temporal_train, model.dis_t1_train)
    dump("cdf_temporal_train", model.cdf_temporal_train, model.dis_t1_train)
    dump("hist_m_train", model.hist_m_train, model.hist_m_train)
    dump("kappa_train", model.kappa_train, model.hist_m_train)
    logger.info("Kernel curves saved under results/%s_*.npy", tag)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = parse_args(argv)
    configure_device(args.gpu_id)

    import torch

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        device = torch.device("cuda")
    else:
        if args.gpu_id.lower() not in {"-1", "cpu", "none"}:
            logger.warning("CUDA is not available; training on CPU.")
        device = torch.device("cpu")

    from models import KernelPointProcess

    prep = prepare_run(args.dataset, history=args.history)
    logger.info(
        "%s: auxiliary=%d, training=%d, validation=%d, testing=%d, area_km2=%.6g, device=%s",
        args.dataset,
        prep["auxiliary_num"],
        prep["training_num"],
        prep["validation_num"],
        prep["testing_num"],
        prep["area"],
        device,
    )

    model = KernelPointProcess(
        time_step_train=prep["input_dim_train"],
        time_step_val=prep["input_dim_val"],
        time_step_test=prep["input_dim_test"],
        temporal_id=args.temporal,
        spatial_id=args.spatial,
        kappa_id=args.kappa,
        global_m0=prep["config"]["global_m0"],
        area=prep["area"],
        size_layer=5,
        size_nn=32,
        seed=args.seed,
        backbone=args.backbone,
        temporal_backbone=args.temporal_backbone,
        spatial_backbone=args.spatial_backbone,
        kappa_backbone=args.kappa_backbone,
        init_mu_from_rate=_upgraded_scheme(args) and not args.no_mu_rate_init,
        use_boundary_correction=_upgraded_scheme(args) and not args.no_boundary_correction,
        boundary_angles=args.boundary_angles,
    )
    model.to(device)
    model.set_train_data(
        prep["train"]["t"], prep["train"]["m"], prep["train"]["x"], prep["train"]["y"]
    )
    model.set_val_data(prep["val"]["t"], prep["val"]["m"], prep["val"]["x"], prep["val"]["y"])
    model.set_test_data(
        prep["test"]["t"], prep["test"]["m"], prep["test"]["x"], prep["test"]["y"]
    )
    model.set_model()
    if model.use_boundary_correction:
        model.set_region(prep["region_xy"], n_angles=args.boundary_angles)
    (
        model.compile(lr=args.lr)
        .fit_eval(
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            mu_batch_size=args.mu_batch_size,
        )
        .eval_train(batch_size=args.batch_size)
        .eval_val(batch_size=args.batch_size)
        .eval_test(batch_size=args.batch_size)
        .save_weights(f"weights/{_run_tag(args)}")
    )

    logger.info(
        "Test results: LL_ts=%s, LL_t=%s, LL_s=%s",
        model.LL_ts_average_test,
        model.LL_t_average_test,
        model.LL_s_average_test,
    )
    scheme_label = _run_tag(args)
    _write_log_likelihood(
        args.dataset, args.temporal, args.spatial, args.kappa, model, args.seed, scheme_label
    )
    _write_parameters(
        args.dataset, args.temporal, args.spatial, args.kappa, model, args.seed, scheme_label
    )
    _write_catalog_csv(
        prep, model, args.dataset, args.temporal, args.spatial, args.kappa, scheme_label
    )
    if not args.no_patterns:
        _write_patterns(model, _run_tag(args))


if __name__ == "__main__":
    main()
