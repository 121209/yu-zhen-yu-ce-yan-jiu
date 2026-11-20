"""Experiment 1C: inspect archived H128/seed1 checkpoints on the CPU.

Run from the project environment:
    python plot_checkpoint_productivity_and_triggering.py
    python plot_checkpoint_productivity_and_triggering.py --check-only
    python plot_checkpoint_productivity_and_triggering.py --datasets SanJac --minutes-after 10

The magnitude sweep varies the mainshock magnitude in a history ending at that
event. The maps use the actual last 128 events strictly before the snapshot.
All four spatial heads use scalar radius. A band in the summed field can arise
from the locations and relative weights of multiple isotropic sources.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from eval_ll_decomposition import BACKBONES, checkpoint_path, history_context_for
from models import KernelPointProcess, _points_in_polygon
from run_models import prepare_run


ROOT = Path(__file__).resolve().parent
DATASETS = ("SCEDC_20", "SanJac")
MAP_MODELS = ("etas", "fcnn", "scheme5", "scheme2")
CURVE_MODELS = (*MAP_MODELS, "scheme4", "scheme1")
LABELS = dict(etas="ETAS", fcnn="ST-NKF (FCNN)", scheme5="Scheme 5",
              scheme2="Scheme 2", scheme4="Scheme 4", scheme1="Scheme 1")
HISTORY = 128


def load_model(dataset, key, prep, manifest):
    path = (ROOT / "weights" / "etas" / f"etas_{dataset}_H128_seed1_aligned_best.pt"
            if key == "etas" else checkpoint_path(dataset, key))
    relative = path.relative_to(ROOT).as_posix()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != manifest[relative]:
        raise ValueError(f"Checkpoint differs from the archive manifest: {relative}")
    mode = "empirical" if key == "etas" else "neural"
    model = KernelPointProcess(
        HISTORY, HISTORY, HISTORY, mode, mode, mode, prep["config"]["global_m0"],
        prep["area"], size_layer=5, size_nn=32, seed=1, init_mu_from_rate=True,
        use_boundary_correction=True, boundary_angles=128, **BACKBONES.get(key, {}),
    )
    model.set_region(prep["region_xy"], n_angles=128)
    model.load_weights(str(path))
    scales = np.asarray([model.max_time, model.max_mag, model.max_dis], dtype=float)
    if not np.all(np.isfinite(scales) & (scales > 0)):
        raise ValueError(f"Invalid checkpoint normalization scales: {relative}")
    return model.eval().requires_grad_(False), dict(path=relative, sha256=digest)


def select_history(prep, minutes, event_id):
    cat, config = prep["raw_catalog"], prep["config"]
    times = cat.time_days.to_numpy()
    test = np.flatnonzero((times >= config["testing_start"]) & (times <= config["testing_end"]))
    if prep["history"] != HISTORY or len(test) != prep["testing_num"]:
        raise ValueError("Unexpected catalog history or test split")
    selection = "user-selected test event"
    if event_id is None:
        event_id = int(np.argmax(cat.magnitude.to_numpy()[test]))  # earliest maximum
        selection = "largest test magnitude, earliest tie"
    if not 0 <= event_id < len(test):
        raise ValueError(f"Test event ID must be in [0, {len(test) - 1}]")
    ref = int(test[event_id])
    query = float(times[ref] + minutes / 1440)
    stop = int(np.searchsorted(times, query, side="left"))
    start = stop - HISTORY
    if ref < HISTORY - 1 or start < 0 or query > config["testing_end"]:
        raise ValueError("The selected event or snapshot is outside the supported test/history window")
    if not start <= ref < stop:
        raise ValueError("The reference event left the 128-event history. Reduce --minutes-after.")
    history = cat.iloc[start:stop]
    focal = ref - start
    meta = dict(test_event_id=event_id, catalog_time=str(cat.time.iloc[ref]),
                catalog_time_days=float(times[ref]), mainshock_magnitude=float(cat.magnitude.iloc[ref]),
                query_time_days=query, minutes_after=minutes, history_size=HISTORY,
                mainshock_history_index=focal, n_history_after_mainshock=stop - ref - 1,
                selection=selection)
    return cat.iloc[ref-HISTORY+1:ref+1], history, focal, meta


def column(values):
    return torch.from_numpy(np.array(values, dtype=np.float32, copy=True)).reshape(1, -1, 1)


def inputs(history, query):
    t = history.time_days.to_numpy()
    return column(query - t), column(t[-1] - t[:-1]), column(history.magnitude)


def forward_at(model, history, query, point):
    xy = history[["x", "y"]].to_numpy()
    return model(*inputs(history, query), column(np.linalg.norm(xy - point, axis=1)),
                 create_graph=False, hist_x=column(xy[:, 0]), hist_y=column(xy[:, 1]))


def productivity_curve(model, history, magnitudes):
    # Only the most recent token's magnitude changes. Earlier events and ages
    # stay fixed, including for the history-dependent Scheme 1 head.
    t1, t2, observed = inputs(history, history.time_days.iloc[-1])
    with torch.no_grad():
        m = column(magnitudes).reshape(-1, 1, 1)
        if model.kappa_id == "empirical":
            values = model.produc_func(m)
        elif model.kappa_backbone == "transformer":
            context = history_context_for(model, t1, t2, observed).expand(len(m), -1, -1).clone()
            context[:, -1, 1] = m.flatten() / model.max_mag
            values = model.kappa_net(context)[:, -1, :]
        else:
            values = model.kappa_net(m / model.max_mag)
    return values.flatten().numpy()


def spatial_fields(model, history, focal, query, offsets):
    """Return single-source plane density and the sum of all triggered rates."""
    xy = history[["x", "y"]].to_numpy()
    centers = xy - xy[focal]
    # This call includes the official finite-region compensator. Only its
    # pointwise components enter the figures. Background mu is excluded.
    native = forward_at(model, history, query, xy[focal])
    amplitude = (native["kappa"] * native["pdf_temporal"]).detach()
    t1, t2, mags = inputs(history, query)
    with torch.no_grad():
        params = (model.spatial_net.encode_context(history_context_for(model, t1, t2, mags))
                  if model.spatial_backbone == "transformer" else None)

    def evaluate(points):
        delta = points[:, None, :] - centers[None, :, :]
        r = torch.as_tensor(np.linalg.norm(delta, axis=-1), dtype=torch.float32).unsqueeze(-1)
        if model.spatial_id == "empirical":
            density, _ = model.spatial_func(r.square(), mags)
        elif params is not None:
            density = model.spatial_net.plane_density_from_parameters(r, params, model.max_dis)
        else:
            # Same radius derivative and origin convention as models.py.
            r.requires_grad_(True)
            cdf = model.spatial_net(r / model.max_dis)
            slope, = torch.autograd.grad(cdf.sum(), r)
            density = slope / (2 * math.pi * (r.detach() + 1e-5))
        rate = (amplitude * density).sum(dim=1)
        return density[:, focal, 0].detach().numpy(), rate[:, 0].detach().numpy()

    # Cross-check the cached spatial computation against full model.forward at
    # the epicenter and off center, including the r=0 convention.
    for point, reference in [(np.zeros(2), native), (np.array([2., -1.]),
                              forward_at(model, history, query, xy[focal] + [2., -1.]))]:
        single, rate = evaluate(point[None, :])
        np.testing.assert_allclose(single[0], reference["pdf_spatial"][0, focal, 0].item(),
                                   rtol=5e-5, atol=1e-12)
        np.testing.assert_allclose(rate[0] + model.bac_func.mu.item(), reference["l_ts"].item(),
                                   rtol=5e-5, atol=1e-12)
    result = [evaluate(offsets[i:i+128]) for i in range(0, len(offsets), 128)]
    single, rate = (np.concatenate(part) for part in zip(*result))
    if not all(np.isfinite(a).all() and np.all(a >= 0) for a in (single, rate)):
        raise ValueError("Non-finite or negative spatial density/rate")
    return single, rate


def plot_results(directory, dataset, curves, single, rate, inside, edges, centers, meta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(7.4, 4.4), layout="constrained")
    colors = ("#333333", "#0072B2", "#009E73", "#D55E00", "#CC79A7", "#7653A4")
    for key, color, style in zip(CURVE_MODELS, colors, ("--", "-", "-.", ":", "--", "-")):
        rows = curves[curves.model == key]
        ax.plot(rows.magnitude, rows.relative_kappa, style, color=color, lw=2,
                label=f"{LABELS[key]} · {rows['head'].iloc[0]}")
    ax.set(title=f"{dataset}  |  Magnitude productivity", xlabel="Reference-event magnitude",
           ylabel=r"Relative productivity  $\kappa(m)/\kappa(M_c)$", yscale="log")
    ax.grid(axis="y", alpha=.22)
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(directory / "magnitude_productivity.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(2, len(MAP_MODELS), figsize=(13.5, 7), sharex=True, sharey=True,
                             layout="constrained")
    extent = (edges[0], edges[-1], edges[0], edges[-1])
    show = np.max(np.abs(centers), axis=1) <= edges[-1]
    for row, (fields, label) in enumerate(((single, "Single-source kernel (km⁻²)"),
                                           (np.where(inside, rate, np.nan),
                                            "Triggered rate (events day⁻¹ km⁻²)"))):
        positive = fields[np.isfinite(fields) & (fields > 0)]
        if not positive.size:
            raise ValueError(f"No positive values to plot: {label}")
        lo, hi = np.quantile(positive, [.01, .995])
        norm = LogNorm(lo, max(hi, lo * 1.001), clip=True)
        for j, key in enumerate(MAP_MODELS):
            ax = axes[row, j]
            ax.set_facecolor("#e6e6e6")
            # Zero density uses the lowest color. NaN (outside region) stays gray.
            im = ax.imshow(np.maximum(fields[j], lo), origin="lower", extent=extent,
                           interpolation="nearest", norm=norm, cmap="viridis", aspect="equal")
            if row == 0:
                ax.set_title(LABELS[key])
            else:
                ax.scatter(*centers[show].T, s=6, facecolors="none", edgecolors="white", lw=.45)
                ax.scatter(0, 0, s=65, marker="*", c="#f9b842", edgecolors="black", lw=.55)
                ax.set_xlabel("Δx (km)")
            ax.set(xlim=extent[:2], ylim=extent[2:])
        axes[row, 0].set_ylabel("Δy (km)")
        fig.colorbar(im, ax=axes[row].tolist(), shrink=.76, extend="both", label=label)
    fig.suptitle(f"{dataset}  |  M{meta['mainshock_magnitude']:g}  ·  {meta['catalog_time']}"
                 f"  ·  +{meta['minutes_after']:g} min  ·  H={HISTORY}")
    fig.legend(handles=[
        Line2D([], [], marker="o", linestyle="", color="gray", markerfacecolor="none", label="History epicenter"),
        Line2D([], [], marker="*", linestyle="", color="#b57800", label="Reference event (origin)")],
        loc="outside lower center", ncol=2, frameon=False)
    fig.savefig(directory / "conditional_spatial_triggering.png", dpi=300)
    plt.close(fig)


def analyze(dataset, args, manifest):
    prep = prepare_run(dataset, history=HISTORY)
    prior, history, focal, meta = select_history(prep, args.minutes_after, args.test_event_id)
    config, cat = prep["config"], prep["raw_catalog"]
    train = cat.time_days.between(config["training_start"], config["validation_start"], inclusive="left")
    mc, mmax = float(config["global_m0"]), float(cat.loc[train, "magnitude"].max())
    if not mc < mmax:
        raise ValueError("Empty magnitude sweep")
    magnitudes = np.linspace(mc, mmax, 121)
    xy = history[["x", "y"]].to_numpy()
    centers = xy - xy[focal]
    # Describe the observed local geometry independently of all model outputs.
    # The ratio of principal-axis standard deviations is 1 for a round cloud.
    local = centers[(history.time_days.to_numpy() > meta["catalog_time_days"]) &
                    (np.linalg.norm(centers, axis=1) <= args.half_width_km)]
    axis_ratio = None
    if len(local) >= 3:
        small, large = np.linalg.eigvalsh(np.cov(local.T))
        if small > 0:
            axis_ratio = float(np.sqrt(large / small))
    meta.update(local_postshock_radius_km=args.half_width_km, n_local_postshock=len(local),
                local_epicenter_major_minor_std_ratio=axis_ratio,
                half_width_km=args.half_width_km, grid_cells_per_axis=args.grid_size)
    edges = np.linspace(-args.half_width_km, args.half_width_km, args.grid_size + 1)
    axis = (edges[:-1] + edges[1:]) / 2
    xx, yy = np.meshgrid(axis, axis)
    offsets = np.column_stack((xx.ravel(), yy.ravel()))
    inside = _points_in_polygon(offsets + xy[focal], prep["region_xy"]).reshape(xx.shape)
    if not inside.any():
        raise ValueError("No grid centers fall inside the catalog study region")
    probes = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
    rows, singles, rates, sources = [], [], [], {}
    for key in CURVE_MODELS:
        model, sources[key] = load_model(dataset, key, prep, manifest)
        values = productivity_curve(model, prior, np.append(magnitudes, meta["mainshock_magnitude"]))
        native = forward_at(model, prior, prior.time_days.iloc[-1], xy[focal])
        np.testing.assert_allclose(values[-1], native["kappa"][0, -1, 0].item(), rtol=5e-5)
        values = values[:-1]
        if not np.all(np.isfinite(values) & (values > 0)):
            raise ValueError(f"{key}: invalid productivity")
        head = "exponential" if key == "etas" else model.kappa_backbone.upper()
        for m, value in zip(magnitudes, values):
            rows.append((key, head, m, value, value / values[0]))
        if key in MAP_MODELS:
            s, r = spatial_fields(model, history, focal, meta["query_time_days"],
                                  probes if args.check_only else np.vstack((probes, offsets)))
            np.testing.assert_allclose(s[:4], s[0], rtol=5e-5)
            if not args.check_only:
                singles.append(s[4:].reshape(xx.shape))
                rates.append(r[4:].reshape(xx.shape))

    print(f"{dataset}: M{meta['mainshock_magnitude']:g}, {meta['catalog_time']}, "
          f"test ID {meta['test_event_id']}, +{args.minutes_after:g} min, "
          f"{meta['n_history_after_mainshock']} later events in H={HISTORY}.", flush=True)
    if args.check_only:
        return
    directory = args.out / dataset
    directory.mkdir(parents=True, exist_ok=True)
    curves = pd.DataFrame(rows, columns=("model", "head", "magnitude", "kappa", "relative_kappa"))
    single, rate = np.stack(singles), np.stack(rates)
    plot_results(directory, dataset, curves, single, rate, inside, edges, centers, meta)
    curves.to_csv(directory / "magnitude_productivity.csv", index=False)
    np.savez_compressed(directory / "spatial_fields.npz", models=np.array(MAP_MODELS),
                        offset_km=axis, edges_km=edges, inside_region=inside,
                        single_source_pdf_per_km2=single, triggered_rate_per_day_km2=rate,
                        history_xy_km=xy, history_magnitude=history.magnitude.to_numpy(),
                        history_time_days=history.time_days.to_numpy())
    meta.update(dataset=dataset, checkpoints=sources, mc_configured=mc, mmax_training=mmax,
                magnitude_context="127 preceding events and the reference event with varied magnitude",
                origin_catalog_xy_km=xy[focal].tolist(), map_axes="model, delta_y, delta_x",
                triggered_field="sum of kappa * temporal PDF * planar spatial PDF over H; excludes mu",
                display="Cell centers; common log colors per row, clipped at pooled positive "
                        "1st/99.5th percentiles. Lower panel masked to study region. NPZ is unclipped.")
    (directory / "case_settings.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(directory, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--minutes-after", type=float, default=20, help="Time from reference event to snapshot")
    parser.add_argument("--test-event-id", type=int, help="Zero-based test event ID; select one dataset")
    parser.add_argument("--half-width-km", type=float, default=20, help="Map half-width in catalog km")
    parser.add_argument("--grid-size", type=int, default=100, help="Number of grid cells per axis")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "checkpoint_triggering_kernels")
    parser.add_argument("--check-only", action="store_true", help="Verify weights and numeric probes without writing output")
    args = parser.parse_args()
    if args.test_event_id is not None and len(args.datasets) != 1:
        parser.error("--test-event-id requires exactly one dataset")
    if (not np.isfinite([args.minutes_after, args.half_width_km]).all() or
            min(args.minutes_after, args.half_width_km) <= 0 or args.grid_size < 2):
        parser.error("Set positive time and extent, and --grid-size >= 2")
    args.out = args.out.resolve()
    os.chdir(ROOT)  # prepare_run reads the archived catalog and polygon relative to the repository.
    torch.set_num_threads(min(4, torch.get_num_threads()))
    with (ROOT / "results" / "MANIFEST.sha256.csv").open(encoding="utf-8-sig", newline="") as f:
        manifest = {row["path"]: row["sha256"] for row in csv.DictReader(f)}
    for dataset in dict.fromkeys(args.datasets):
        analyze(dataset, args, manifest)
    if args.check_only:
        print("All requested checkpoints and numerical checks passed.")


if __name__ == "__main__":
    main()
