"""PyTorch KernelPointProcess used by the ST-NKF / ETAS training script.

The conditional intensity and the compensated log-likelihood follow
``models.py`` in zhancx-edu/NeuralKernelFunction:

    l_ts = mu + sum_i kappa(m_i) * h(t - t_i) * w(x - x_i, y - y_i)
    l_t  = mu * area + sum_i kappa(m_i) * h(t - t_i)
    Int  = mu * area * dt + sum_i kappa(m_i) * (F(t - t_i) - F(t_prev - t_i)) * m_i
    loss = -mean(log(l_ts) - Int)

For the default FCNN / ETAS path, m_i is identically 1 (plane-normalized
kernels, no finite-region correction), matching the upstream formula.
When a transformer or MLP neural kernel is used, m_i is the mass of the
isotropic spatial kernel of source i that falls inside the study polygon R:

    m_i = integral_R w_plane(x - x_i, y - y_i) dx dy
        = (1 / 2 pi) integral_0^{2 pi} [F_r(r_out(theta)) - F_r(r_in(theta))] d theta

for a convex region, where F_r is the radial CDF (plane mass within radius r)
and the rays are shot from the source. Intensity l_ts still uses the
unrestricted plane density; sources are not renormalized onto R.

Neural temporal and spatial densities are input-derivatives of the network
CDF, matching ``tf.keras.backend.gradients``. ``create_graph=True`` keeps the
path into the kernel parameters.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from layers import (
    BackgroundFunction,
    KappaNetwork,
    OmoriFunction,
    ProductionFunction,
    SpatialFunction,
    SpatialKernelNetwork,
    TemporalKernelNetwork,
)
from layers_transformer import (
    KappaMLP,
    KappaTransformer,
    SpatialKernelTransformer,
    TemporalKernelTransformer,
)
from utils import process_data

logger = logging.getLogger(__name__)

_MODES = ("empirical", "neural")
_BACKBONES = ("fcnn", "transformer")
_KAPPA_BACKBONES = ("fcnn", "transformer", "mlp")
_TRANSFORMER_DESIGN = "fixed-history-omori-cdf-mixture-v3"


def _grad_wrt(output: torch.Tensor, inp: torch.Tensor, create_graph: bool) -> torch.Tensor:
    """Elementwise d(output)/d(inp) for a pointwise network or CDF."""
    grad = torch.autograd.grad(
        output.sum(),
        inp,
        create_graph=create_graph,
        retain_graph=True,
    )[0]
    if grad is None:
        raise RuntimeError("CDF does not depend on the differentiated input.")
    return grad


def _resolve_backbones(
    backbone: str,
    temporal_backbone: str | None,
    spatial_backbone: str | None,
    kappa_backbone: str | None,
) -> tuple[str, str, str]:
    if backbone not in _BACKBONES:
        raise ValueError(f"Invalid backbone: {backbone}")
    defaults = (
        ("transformer", "transformer", "fcnn")
        if backbone == "transformer"
        else ("fcnn", "fcnn", "fcnn")
    )
    resolved = (
        defaults[0] if temporal_backbone is None else temporal_backbone,
        defaults[1] if spatial_backbone is None else spatial_backbone,
        defaults[2] if kappa_backbone is None else kappa_backbone,
    )
    if resolved[0] not in _BACKBONES or resolved[1] not in _BACKBONES:
        raise ValueError(f"Invalid temporal/spatial backbone: {resolved}")
    if resolved[2] not in _KAPPA_BACKBONES:
        raise ValueError(f"Invalid kappa_backbone: {resolved[2]}")
    return resolved


def _polygon_edges(vertices: np.ndarray) -> np.ndarray:
    verts = np.asarray(vertices, dtype=np.float64)
    if verts.ndim != 2 or verts.shape[0] < 3 or verts.shape[1] != 2:
        raise ValueError("region polygon must have shape (V, 2) with V >= 3")
    if not np.allclose(verts[0], verts[-1]):
        verts = np.vstack((verts, verts[0]))
    return np.ascontiguousarray(verts)


def _points_in_polygon(xy: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Even-odd ray test. ``vertices`` may be open or closed, shape (V, 2)."""
    pts = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    vs = _polygon_edges(vertices)
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(pts.shape[0], dtype=bool)
    for i in range(vs.shape[0] - 1):
        x1, y1 = vs[i]
        x2, y2 = vs[i + 1]
        straddles = (y1 > y) != (y2 > y)
        denom = (y2 - y1) + np.copysign(1e-300, (y2 - y1) + 1e-300)
        xinters = (x2 - x1) * (y - y1) / denom + x1
        inside ^= straddles & (x < xinters)
    return inside


def _polygon_is_convex(vertices: np.ndarray) -> bool:
    vs = _polygon_edges(vertices)
    n = vs.shape[0] - 1
    signs = []
    for i in range(n):
        dx1 = vs[i + 1, 0] - vs[i, 0]
        dy1 = vs[i + 1, 1] - vs[i, 1]
        dx2 = vs[(i + 2) % n, 0] - vs[i + 1, 0]
        dy2 = vs[(i + 2) % n, 1] - vs[i + 1, 1]
        cross = dx1 * dy2 - dy1 * dx2
        if abs(cross) > 1e-12:
            signs.append(cross)
    if not signs:
        return True
    return all(s > 0 for s in signs) or all(s < 0 for s in signs)


class KernelPointProcess(nn.Module):
    """Limited-history ETAS or ST-NKF, or a mix of empirical and neural kernels."""

    def __init__(
        self,
        time_step_train: int,
        time_step_val: int,
        time_step_test: int,
        temporal_id: str,
        spatial_id: str,
        kappa_id: str,
        global_m0: float,
        area: float,
        size_layer: int,
        size_nn: int,
        seed: int | None = None,
        backbone: str = "fcnn",
        temporal_backbone: str | None = None,
        spatial_backbone: str | None = None,
        kappa_backbone: str | None = None,
        init_mu_from_rate: bool | None = None,
        use_boundary_correction: bool | None = None,
        kappa_mlp_hidden: int = 64,
        kappa_mlp_layers: int = 4,
        boundary_angles: int = 256,
    ) -> None:
        super().__init__()
        for name, value in (
            ("temporal_id", temporal_id),
            ("spatial_id", spatial_id),
            ("kappa_id", kappa_id),
        ):
            if value not in _MODES:
                raise ValueError(f"Invalid {name}: {value}")
        t_bb, s_bb, k_bb = _resolve_backbones(
            backbone, temporal_backbone, spatial_backbone, kappa_backbone
        )
        self.time_step_train = int(time_step_train)
        self.time_step_val = int(time_step_val)
        self.time_step_test = int(time_step_test)
        self.area_scalar = float(area)
        self.size_layer = int(size_layer)
        self.size_nn = int(size_nn)
        self.temporal_id = temporal_id
        self.spatial_id = spatial_id
        self.kappa_id = kappa_id
        self.temporal_backbone = t_bb
        self.spatial_backbone = s_bb
        self.kappa_backbone = k_bb
        # Legacy flag: "transformer" if any upgraded neural module is used,
        # otherwise "fcnn". The default all-FCNN / ETAS path still has
        # backbone == "fcnn" and must stay bitwise unchanged.
        self.upgraded = (
            t_bb == "transformer" or s_bb == "transformer" or k_bb in ("transformer", "mlp")
        )
        self.backbone = "transformer" if self.upgraded else "fcnn"
        self.global_m0 = float(global_m0)
        self.seed = seed
        self.lr = 0.005
        self.max_time: float | None = None
        self.max_mag: float | None = None
        self.max_dis: float | None = None
        self.init_mu_from_rate = (
            bool(self.upgraded) if init_mu_from_rate is None else bool(init_mu_from_rate)
        )
        self.use_boundary_correction = (
            bool(self.upgraded)
            if use_boundary_correction is None
            else bool(use_boundary_correction)
        )
        self.kappa_mlp_hidden = int(kappa_mlp_hidden)
        self.kappa_mlp_layers = int(kappa_mlp_layers)
        self.boundary_angles = int(boundary_angles)
        self._region_vertices: np.ndarray | None = None

        self.bac_func = BackgroundFunction()
        self.produc_func = ProductionFunction(global_m0)
        self.omori_func = OmoriFunction()
        self.spatial_func = SpatialFunction(global_m0)
        if t_bb == "transformer":
            self.temporal_net = TemporalKernelTransformer(d_model=size_nn, seed=seed)
        else:
            self.temporal_net = TemporalKernelNetwork(size_nn, size_layer, seed=seed)
        if s_bb == "transformer":
            self.spatial_net = SpatialKernelTransformer(d_model=size_nn, seed=seed)
        else:
            self.spatial_net = SpatialKernelNetwork(size_nn, size_layer, seed=seed)
        if k_bb == "transformer":
            self.kappa_net = KappaTransformer(d_model=size_nn, seed=seed)
        elif k_bb == "mlp":
            self.kappa_net = KappaMLP(
                hidden_width=self.kappa_mlp_hidden,
                num_hidden=self.kappa_mlp_layers,
                seed=seed,
            )
        else:
            # Default 5x32 KappaNetwork (bias + softplus + NonNeg).
            self.kappa_net = KappaNetwork(size_nn, size_layer, seed=seed)

    def set_train_data(self, times, mags, locs_x, locs_y) -> "KernelPointProcess":
        (
            self.hist_t_train,
            self.hist_m_train,
            self.hist_x_train,
            self.hist_y_train,
            self.cur_t_train,
            self.cur_x_train,
            self.cur_y_train,
            self.dis_xy_train,
            self.dis_t1_train,
            self.dis_t2_train,
            self.elapsed_t_train,
        ) = process_data(self.time_step_train, times, mags, locs_x, locs_y)
        return self

    def set_val_data(self, times, mags, locs_x, locs_y) -> "KernelPointProcess":
        (
            self.hist_t_val,
            self.hist_m_val,
            self.hist_x_val,
            self.hist_y_val,
            self.cur_t_val,
            self.cur_x_val,
            self.cur_y_val,
            self.dis_xy_val,
            self.dis_t1_val,
            self.dis_t2_val,
            self.elapsed_t_val,
        ) = process_data(self.time_step_val, times, mags, locs_x, locs_y)
        return self

    def set_test_data(self, times, mags, locs_x, locs_y) -> "KernelPointProcess":
        (
            self.hist_t_test,
            self.hist_m_test,
            self.hist_x_test,
            self.hist_y_test,
            self.cur_t_test,
            self.cur_x_test,
            self.cur_y_test,
            self.dis_xy_test,
            self.dis_t1_test,
            self.dis_t2_test,
            self.elapsed_t_test,
        ) = process_data(self.time_step_test, times, mags, locs_x, locs_y)
        return self

    def set_model(self) -> "KernelPointProcess":
        """Freeze the training-set scales used to normalize neural inputs."""
        self.max_time = float(np.max(self.dis_t1_train))
        self.max_mag = float(np.max(self.hist_m_train))
        self.max_dis = float(np.max(self.dis_xy_train))
        if self.max_time <= 0 or self.max_mag <= 0 or self.max_dis <= 0:
            raise ValueError(
                "Non-positive normalization scale: "
                f"max_time={self.max_time}, max_mag={self.max_mag}, max_dis={self.max_dis}"
            )
        logger.info(
            "Normalization scales max_time=%.6g max_mag=%.6g max_dis=%.6g area=%.6g",
            self.max_time,
            self.max_mag,
            self.max_dis,
            self.area_scalar,
        )
        if self.init_mu_from_rate:
            n_events = int(self.elapsed_t_train.shape[0])
            total_time = float(np.sum(self.elapsed_t_train, dtype=np.float64))
            mu0 = max(n_events / (total_time * self.area_scalar), 1e-9)
            with torch.no_grad():
                self.bac_func.mu.fill_(mu0)
            logger.info("Initialized mu from mean event rate / area: %.6e", mu0)
        return self

    def set_region(
        self,
        polygon_xy: np.ndarray,
        n_angles: int | None = None,
    ) -> "KernelPointProcess":
        """Store the study polygon used for the finite-region mass m_i.

        The compensator uses Int = mu * A * dt + sum_i kappa_i * Delta F_t(i) * m_i,
        where m_i is the plane-kernel mass inside this convex polygon. Intensity
        lambda_ts is left on the unrestricted plane density. Polar quadrature
        of the radial CDF along rays from each source is used because ETAS-like
        kernels can be extremely peaked; a coarse Cartesian grid would miss
        the near-source mass.
        """
        verts = _polygon_edges(polygon_xy)
        if not _polygon_is_convex(verts):
            raise ValueError(
                "Boundary-mass quadrature assumes a convex study region "
                "(SanJac is a projected rectangle)."
            )
        self._region_vertices = verts
        device = self.bac_func.mu.device
        dtype = self.bac_func.mu.dtype
        n_angles = int(self.boundary_angles if n_angles is None else n_angles)
        if n_angles < 8:
            raise ValueError("n_angles must be >= 8")
        self.boundary_angles = n_angles
        # Open interval [0, 2pi): last edge of a closed polygon is already stored.
        theta = torch.linspace(0.0, 2.0 * math.pi, n_angles + 1, dtype=dtype, device=device)[:-1]
        self.register_buffer("_ray_cos", torch.cos(theta), persistent=False)
        self.register_buffer("_ray_sin", torch.sin(theta), persistent=False)
        edge_origin = torch.as_tensor(verts[:-1], dtype=dtype, device=device)
        edge_vec = torch.as_tensor(verts[1:] - verts[:-1], dtype=dtype, device=device)
        self.register_buffer("_edge_origin", edge_origin, persistent=False)
        self.register_buffer("_edge_vec", edge_vec, persistent=False)
        logger.info(
            "Region set: %d vertices, %d-angle polar quadrature, convex=%s",
            verts.shape[0] - 1,
            n_angles,
            True,
        )
        return self

    def compile(self, lr: float = 0.005) -> "KernelPointProcess":
        self.lr = float(lr)
        return self

    def adam_parameters(self):
        """Trainable tensors except ``mu``, which the epoch callback overwrites."""
        for name, param in self.named_parameters():
            if name == "bac_func.mu":
                continue
            yield param

    def project_constraints(self) -> None:
        self.bac_func.project()
        self.produc_func.project()
        self.omori_func.project()
        self.spatial_func.project()
        self.temporal_net.project()
        self.spatial_net.project()
        self.kappa_net.project()

    def _needs_history_context(self) -> bool:
        return (
            (self.temporal_id == "neural" and self.temporal_backbone == "transformer")
            or (self.spatial_id == "neural" and self.spatial_backbone == "transformer")
            or (self.kappa_id == "neural" and self.kappa_backbone == "transformer")
        )

    def _radial_cdf_at(
        self,
        radius: torch.Tensor,
        mags: torch.Tensor,
        spatial_params: tuple | None,
    ) -> torch.Tensor:
        """Radial enclosed-mass CDF F_r(r) at physical radii, per source."""
        if self.spatial_id == "neural":
            if self.spatial_backbone == "transformer":
                if spatial_params is None:
                    raise RuntimeError("Transformer spatial CDF needs encoded parameters.")
                return self.spatial_net.cdf_from_parameters(
                    radius / self.max_dis, spatial_params
                )
            return self.spatial_net(radius / self.max_dis)
        pdf, cdf = self.spatial_func(radius**2, mags)
        del pdf
        return cdf

    def _boundary_mass(
        self,
        hist_x: torch.Tensor,
        hist_y: torch.Tensor,
        mags: torch.Tensor,
        spatial_params: tuple | None,
    ) -> torch.Tensor:
        """m_i = integral_R w_plane(x-x_i, y-y_i) dA for each history source.

        For a convex polygon this is the angular mean of the radial CDF on
        ray-boundary hits. From an interior source the ray hits once and the
        contribution is F_r(t_hit). From an exterior source the (0 or 2) hits
        contribute F_r(t_out) - F_r(t_in), or 0 if the ray misses R.
        """
        if self._region_vertices is None or not hasattr(self, "_ray_cos"):
            raise RuntimeError("Call set_region() before applying boundary-mass correction.")
        # hist_x, hist_y, mags: (B, H, 1)
        # Rays: origin S, direction (cos theta, sin theta)
        # Intersect S + t d with each edge P + u e, u in [0, 1], t > 0.
        cos_t = self._ray_cos.to(device=hist_x.device, dtype=hist_x.dtype)
        sin_t = self._ray_sin.to(device=hist_x.device, dtype=hist_x.dtype)
        edge_p = self._edge_origin.to(device=hist_x.device, dtype=hist_x.dtype)
        edge_e = self._edge_vec.to(device=hist_x.device, dtype=hist_x.dtype)
        # Shapes: src (B,H,1,1,2), dir (A,2), edge (E,2)
        src = torch.cat((hist_x, hist_y), dim=-1).unsqueeze(2).unsqueeze(3)  # (B,H,1,1,2)
        direction = torch.stack((cos_t, sin_t), dim=-1)  # (A,2)
        # t = cross(P-S, e) / cross(d, e)
        # u = cross(P-S, d) / cross(d, e)
        ps = edge_p.view(1, 1, 1, -1, 2) - src  # (B,H,1,E,2)
        dir_b = direction.view(1, 1, -1, 1, 2)  # (1,1,A,1,2)
        edge_b = edge_e.view(1, 1, 1, -1, 2)
        cross_de = dir_b[..., 0] * edge_b[..., 1] - dir_b[..., 1] * edge_b[..., 0]
        cross_pse = ps[..., 0] * edge_b[..., 1] - ps[..., 1] * edge_b[..., 0]
        cross_psd = ps[..., 0] * dir_b[..., 1] - ps[..., 1] * dir_b[..., 0]
        parallel = cross_de.abs() < 1e-12
        safe_de = torch.where(parallel, torch.ones_like(cross_de), cross_de)
        t_hit = cross_pse / safe_de
        u_hit = cross_psd / safe_de
        valid = (~parallel) & (t_hit > 1e-12) & (u_hit >= 0.0) & (u_hit <= 1.0)
        t_valid = torch.where(valid, t_hit, t_hit.new_full((), math.inf))
        t_sorted, _ = t_valid.sort(dim=-1)  # (B,H,A,E)
        t1 = t_sorted[..., 0]
        t2 = t_sorted[..., 1]
        t1_ok = torch.isfinite(t1)
        t2_ok = torch.isfinite(t2)

        src_xy = torch.cat((hist_x, hist_y), dim=-1).detach().cpu().numpy().reshape(-1, 2)
        inside_np = _points_in_polygon(src_xy, self._region_vertices)
        inside = torch.as_tensor(inside_np, device=hist_x.device).view(hist_x.shape[0], hist_x.shape[1])

        # Evaluate F at t1 and t2. Replace unused (inf) radii with 0 so the
        # CDF is well-defined; those values are masked out below.
        def _f(t_raw: torch.Tensor) -> torch.Tensor:
            t_safe = torch.where(torch.isfinite(t_raw), t_raw, torch.zeros_like(t_raw))
            radius = t_safe.unsqueeze(-1)  # (B,H,A,1)
            # Broadcast mixture params over the angle axis if needed.
            params = spatial_params
            if params is not None:
                params = tuple(p.unsqueeze(2) for p in params)
            mags_b = mags.unsqueeze(2)
            return self._radial_cdf_at(radius, mags_b, params).squeeze(-1)

        f1 = _f(t1)
        f2 = _f(t2)
        interior = torch.where(t1_ok, f1, torch.zeros_like(f1))
        exterior = torch.where(t1_ok & t2_ok, f2 - f1, torch.zeros_like(f1))
        contrib = torch.where(inside.unsqueeze(-1), interior, exterior)
        mass = contrib.mean(dim=-1, keepdim=True)  # (B,H,1)
        return mass.clamp(min=0.0)

    def forward(
        self,
        time_in_1: torch.Tensor,
        time_in_2: torch.Tensor,
        mags_in_1: torch.Tensor,
        dis_in: torch.Tensor,
        create_graph: bool = True,
        hist_x: torch.Tensor | None = None,
        hist_y: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.max_time is None or self.max_mag is None or self.max_dis is None:
            raise RuntimeError("Call set_model() before forward.")

        mags_in_2 = mags_in_1[:, :-1, :]
        # A single history, anchored at the LAST OBSERVED event, is fixed for
        # the whole no-event interval. It does not contain query t, x or y.
        # time_in_2 has H-1 lower-limit ages; the last history event has age 0.
        history_context = None
        lower_t = None
        if self._needs_history_context():
            lower_t = torch.cat((time_in_2, torch.zeros_like(mags_in_1[:, -1:, :])), dim=1)
            history_context = torch.cat(
                (lower_t / self.max_time, mags_in_1 / self.max_mag), dim=-1
            )

        spatial_params = None
        if self.spatial_id == "neural":
            dis = dis_in.detach().clone().requires_grad_(True)
            if self.spatial_backbone == "transformer":
                spatial_params = self.spatial_net.encode_context(history_context)
                cdf_spatial = self.spatial_net(dis / self.max_dis, parameters=spatial_params)
                # Radial CDF has O(r^2) near zero. Use its autograd second
                # derivative to define the exact finite r=0 planar limit;
                # no epsilon Jacobian, no independent PDF head.
                at_zero = dis == 0
                d_v_r = _grad_wrt(cdf_spatial, dis, create_graph or bool(at_zero.any()))
                safe_r = torch.where(at_zero, torch.ones_like(dis), dis)
                pdf_spatial = d_v_r / (2.0 * math.pi * safe_r)
                if bool(at_zero.any()):
                    d2_v_r = _grad_wrt(d_v_r, dis, create_graph)
                    pdf_spatial = torch.where(at_zero, d2_v_r / (2.0 * math.pi), pdf_spatial)
            else:
                cdf_spatial = self.spatial_net(dis / self.max_dis)
                d_v_r = _grad_wrt(cdf_spatial, dis, create_graph)
                pdf_spatial = d_v_r / (2.0 * math.pi * (dis + 1e-5))
        else:
            pdf_spatial, cdf_spatial = self.spatial_func(dis_in ** 2, mags_in_1)

        t1 = time_in_1.detach().clone().requires_grad_(True)
        if self.temporal_id == "neural":
            if self.temporal_backbone == "transformer":
                temporal_params = self.temporal_net.encode_context(history_context)
                cdf_temporal_1 = self.temporal_net(t1 / self.max_time, parameters=temporal_params)
                # H at BOTH endpoints, with identical parameters. The newest
                # source contributes F(0)=0 at the lower endpoint.
                cdf_lower_full = self.temporal_net(lower_t / self.max_time, parameters=temporal_params)
                cdf_temporal_2 = cdf_lower_full[:, :-1, :]
            else:
                cdf_temporal_1 = self.temporal_net(t1 / self.max_time)
                cdf_temporal_2 = self.temporal_net(time_in_2 / self.max_time)
        else:
            cdf_temporal_1 = self.omori_func(t1)
            cdf_temporal_2 = self.omori_func(time_in_2)
        pdf_temporal_1 = _grad_wrt(cdf_temporal_1, t1, create_graph)

        if self.kappa_id == "neural":
            if self.kappa_backbone == "transformer":
                kappa_1 = self.kappa_net(history_context)
                kappa_2 = kappa_1[:, :-1, :]
            else:
                kappa_1 = self.kappa_net(mags_in_1 / self.max_mag)
                kappa_2 = self.kappa_net(mags_in_2 / self.max_mag)
        else:
            kappa_1 = self.produc_func(mags_in_1)
            kappa_2 = self.produc_func(mags_in_2)

        if self.use_boundary_correction:
            if hist_x is None or hist_y is None:
                raise RuntimeError(
                    "Boundary-mass correction requires history coordinates. "
                    "Call set_region() and pass hist_x, hist_y to forward()."
                )
            mass = self._boundary_mass(hist_x, hist_y, mags_in_1, spatial_params)
            int_tri = (cdf_temporal_1 * kappa_1 * mass).sum(dim=1, keepdim=True) - (
                cdf_temporal_2 * kappa_2 * mass[:, :-1, :]
            ).sum(dim=1, keepdim=True)
        else:
            int_tri = (cdf_temporal_1 * kappa_1).sum(dim=1, keepdim=True) - (
                cdf_temporal_2 * kappa_2
            ).sum(dim=1, keepdim=True)
        int_l = int_tri + self.bac_func.mu * self.area_scalar * t1[:, -1:, :]

        lam_tri_ts = (kappa_1 * pdf_temporal_1 * pdf_spatial).sum(dim=1, keepdim=True)
        lam_tri_t = (kappa_1 * pdf_temporal_1).sum(dim=1, keepdim=True)
        l_ts = lam_tri_ts + self.bac_func.mu
        l_t = lam_tri_t + self.bac_func.mu * self.area_scalar
        return {
            "l_ts": l_ts,
            "l_t": l_t,
            "int_l": int_l,
            "pdf_temporal": pdf_temporal_1,
            "cdf_temporal": cdf_temporal_1,
            "pdf_spatial": pdf_spatial,
            "cdf_spatial": cdf_spatial,
            "kappa": kappa_1,
        }

    def _arrays(self, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            getattr(self, f"dis_t1_{split}"),
            getattr(self, f"dis_t2_{split}"),
            getattr(self, f"hist_m_{split}"),
            getattr(self, f"dis_xy_{split}"),
        )

    def _batch_tensors(self, split: str, indices: np.ndarray, with_locations: bool = False):
        dis_t1, dis_t2, hist_m, dis_xy = self._arrays(split)
        device = self.bac_func.mu.device

        def take(array: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.ascontiguousarray(array[indices])).to(device)

        tensors = (take(dis_t1), take(dis_t2), take(hist_m), take(dis_xy))
        if with_locations:
            hist_x = take(getattr(self, f"hist_x_{split}"))
            hist_y = take(getattr(self, f"hist_y_{split}"))
            return (*tensors, hist_x, hist_y)
        return tensors

    @staticmethod
    def nll(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return -(torch.log(outputs["l_ts"]) - outputs["int_l"]).mean()

    def _forward_split(self, split: str, indices: np.ndarray, create_graph: bool):
        if self.use_boundary_correction:
            time_1, time_2, mags, dis, hist_x, hist_y = self._batch_tensors(
                split, indices, with_locations=True
            )
            return self.forward(
                time_1, time_2, mags, dis, create_graph=create_graph,
                hist_x=hist_x, hist_y=hist_y,
            )
        time_1, time_2, mags, dis = self._batch_tensors(split, indices)
        return self.forward(time_1, time_2, mags, dis, create_graph=create_graph)

    def loss_on(self, split: str, indices: np.ndarray, create_graph: bool) -> torch.Tensor:
        return self.nll(self._forward_split(split, indices, create_graph=create_graph))

    def _mean_loss(self, split: str, batch_size: int, create_graph: bool) -> float:
        n = int(self._arrays(split)[0].shape[0])
        total = 0.0
        seen = 0
        for start in range(0, n, batch_size):
            idx = np.arange(start, min(start + batch_size, n))
            loss = self.loss_on(split, idx, create_graph=create_graph)
            total += float(loss.detach().cpu()) * len(idx)
            seen += len(idx)
            del loss
        return total / seen

    def update_mu(self, batch_size: int = 512) -> float:
        """Stochastic-declustering update used by ``UpdateMuCallback``.

        ``mu <- max(sum(mu / l_ts) / (sum(dt) * area), 1e-9)`` on the training
        events. Validation loss for the epoch is computed before this write,
        and early-stopping snapshots are taken after it, matching Keras
        callback order (``UpdateMuCallback`` then ``EarlyStopping``).
        """
        n = int(self.dis_t1_train.shape[0])
        acc = torch.zeros((), dtype=torch.float32, device=self.bac_func.mu.device)
        mu = self.bac_func.mu.detach()
        for start in range(0, n, batch_size):
            idx = np.arange(start, min(start + batch_size, n))
            outputs = self._forward_split("train", idx, create_graph=False)
            lam = outputs["l_ts"].detach()
            acc = acc + (mu / lam).sum()
            del outputs
        background_events = float(acc.detach().cpu())
        total_time = float(np.sum(self.elapsed_t_train, dtype=np.float64))
        mu_new = max(background_events / (total_time * self.area_scalar), 1e-9)
        with torch.no_grad():
            self.bac_func.mu.fill_(mu_new)
        return mu_new

    def _snapshot(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}

    def fit_eval(
        self,
        epochs: int = 1000,
        batch_size: int = 128,
        patience: int = 20,
        mu_batch_size: int = 512,
    ) -> "KernelPointProcess":
        """Adam training with Keras 2.14 early-stopping semantics.

        Best weights are restored only when patience is exceeded. A run that
        reaches ``epochs`` keeps the last epoch, including that epoch's ``mu``
        update. Shuffle is deterministic given ``seed`` but does not reproduce
        the TensorFlow batch order.
        """
        if self.max_time is None:
            raise RuntimeError("Call set_model() before fit_eval().")
        if self.use_boundary_correction and self._region_vertices is None:
            raise RuntimeError("Call set_region() before fit_eval() when boundary correction is on.")
        n = int(self.dis_t1_train.shape[0])
        optimizer = torch.optim.Adam(self.adam_parameters(), lr=self.lr)
        rng = np.random.default_rng(0 if self.seed is None else int(self.seed))

        best_val = float("inf")
        best_epoch = 0
        best_state: dict[str, torch.Tensor] | None = None
        wait = 0
        stopped = False

        for epoch in range(int(epochs)):
            order = rng.permutation(n)
            running = 0.0
            seen = 0
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss = self.loss_on("train", idx, create_graph=True)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite training loss at epoch {epoch + 1}. "
                        "Check kernel constraints and catalog times."
                    )
                loss.backward()
                optimizer.step()
                self.project_constraints()
                running += float(loss.detach().cpu()) * len(idx)
                seen += len(idx)
                del loss
            train_loss = running / seen
            val_loss = self._mean_loss("val", batch_size, create_graph=False)
            if not math.isfinite(val_loss):
                raise FloatingPointError(
                    f"Non-finite validation loss at epoch {epoch + 1}."
                )
            mu_value = self.update_mu(batch_size=mu_batch_size)

            wait += 1
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                best_state = self._snapshot()
                wait = 0
            logger.info(
                "Epoch %d/%d loss=%.6f val_loss=%.6f mu=%.6e",
                epoch + 1,
                epochs,
                train_loss,
                val_loss,
                mu_value,
            )
            if wait >= patience and epoch > 0:
                if best_state is not None:
                    self.load_state_dict(best_state)
                stopped = True
                logger.info(
                    "Early stopping at epoch %d. Restored weights from epoch %d (val_loss=%.6f).",
                    epoch + 1,
                    best_epoch + 1,
                    best_val,
                )
                break

        self.stopped_early = stopped
        self.best_epoch = best_epoch + 1
        self.best_val_loss = best_val
        return self

    def _predict_split(self, split: str, batch_size: int) -> dict[str, np.ndarray]:
        n = int(self._arrays(split)[0].shape[0])
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
        buckets: dict[str, list[torch.Tensor]] = {key: [] for key in keys}
        for start in range(0, n, batch_size):
            idx = np.arange(start, min(start + batch_size, n))
            outputs = self._forward_split(split, idx, create_graph=False)
            for key in keys:
                buckets[key].append(outputs[key].detach().cpu())
            del outputs
        return {
            key: torch.cat(parts, dim=0).numpy()
            for key, parts in buckets.items()
        }

    def _store_eval(self, split: str, outputs: dict[str, np.ndarray]) -> None:
        lam_ts = outputs["l_ts"]
        lam_t = outputs["l_t"]
        int_l = outputs["int_l"]
        ll_ts = np.log(lam_ts) - int_l
        ll_t = np.log(lam_t) - int_l
        ll_s = ll_ts - ll_t
        setattr(self, f"lam_ts_{split}", lam_ts)
        setattr(self, f"lam_t_{split}", lam_t)
        setattr(self, f"Int_lam_{split}", int_l)
        setattr(self, f"pdf_temporal_{split}", outputs["pdf_temporal"])
        setattr(self, f"cdf_temporal_{split}", outputs["cdf_temporal"])
        setattr(self, f"pdf_spatial_{split}", outputs["pdf_spatial"])
        setattr(self, f"cdf_spatial_{split}", outputs["cdf_spatial"])
        setattr(self, f"kappa_{split}", outputs["kappa"])
        setattr(self, f"LL_ts_{split}", ll_ts)
        setattr(self, f"LL_t_{split}", ll_t)
        setattr(self, f"LL_s_{split}", ll_s)
        setattr(self, f"LL_ts_average_{split}", float(np.mean(ll_ts)))
        setattr(self, f"LL_t_average_{split}", float(np.mean(ll_t)))
        setattr(self, f"LL_s_average_{split}", float(np.mean(ll_s)))

    def eval_train(self, batch_size: int = 128) -> "KernelPointProcess":
        self._store_eval("train", self._predict_split("train", batch_size))
        return self

    def eval_val(self, batch_size: int = 128) -> "KernelPointProcess":
        self._store_eval("val", self._predict_split("val", batch_size))
        return self

    def eval_test(self, batch_size: int = 128) -> "KernelPointProcess":
        self._store_eval("test", self._predict_split("test", batch_size))
        return self

    def save_weights(self, path: str) -> "KernelPointProcess":
        destination = Path(path)
        if destination.suffix == "":
            destination = destination.with_suffix(".pt")
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "max_time": self.max_time,
                "max_mag": self.max_mag,
                "max_dis": self.max_dis,
                "area": self.area_scalar,
                "temporal_id": self.temporal_id,
                "spatial_id": self.spatial_id,
                "kappa_id": self.kappa_id,
                "backbone": self.backbone,
                "temporal_backbone": self.temporal_backbone,
                "spatial_backbone": self.spatial_backbone,
                "kappa_backbone": self.kappa_backbone,
                "global_m0": self.global_m0,
                "size_layer": self.size_layer,
                "size_nn": self.size_nn,
                "seed": self.seed,
                "lr": self.lr,
                "init_mu_from_rate": self.init_mu_from_rate,
                "use_boundary_correction": self.use_boundary_correction,
                "transformer_design": (
                    getattr(self.temporal_net, "design_version", None)
                    if self.temporal_backbone == "transformer"
                    else getattr(self.spatial_net, "design_version", None)
                    if self.spatial_backbone == "transformer"
                    else None
                ),
            },
            destination,
        )
        self.weights_path = str(destination)
        logger.info("Weights saved to %s", destination)
        return self

    def load_weights(self, path: str) -> "KernelPointProcess":
        checkpoint = torch.load(path, map_location=self.bac_func.mu.device)
        expected = None
        if self.temporal_backbone == "transformer" or self.spatial_backbone == "transformer":
            expected = _TRANSFORMER_DESIGN
        if expected is not None and checkpoint.get("transformer_design") != expected:
            raise ValueError(
                "Incompatible Transformer checkpoint: expected "
                "fixed-history Omori/ETAS CDF mixture v3"
            )
        self.load_state_dict(checkpoint["state_dict"])
        self.max_time = checkpoint.get("max_time", self.max_time)
        self.max_mag = checkpoint.get("max_mag", self.max_mag)
        self.max_dis = checkpoint.get("max_dis", self.max_dis)
        logger.info("Weights loaded from %s", path)
        return self
