"""History-conditioned, normalized Omori/ETAS-like CDFs; no learned PDF head.

The encoder receives ONLY fixed history features (B,H,2): ages at the last
observed event and magnitudes, both normalized by training scales. Query age
or radius never enters the encoder. Encode once, reuse the SAME parameters
at both compensator limits. No stop-gradient surrogate is used.

Temporal Omori mixture (normalized time u = t / max_time):
    F_t(u) = sum_k w_k (1 - [c_k / (u + c_k)]^{p_k - 1}),  p_k > 1, c_k > 0.

Spatial radial ETAS mixture (normalized radius u = r / max_dis):
    F_r(u) = sum_k w_k (1 - [D_k / (u^2 + D_k)]^{q_k - 1}),  q_k > 1, D_k > 0.

Weights are softmax. F(0)=0, F(inf)=1 and dF/du>=0. Physical PDFs come from
these CDFs with autograd, including scale and radial Jacobian factors in
models.py. The training objective remains NLL, not a PDF regression target.
"""
from __future__ import annotations

import math

from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def _nhead(d_model: int) -> int:
    for heads in (8, 4, 2, 1):
        if d_model % heads == 0:
            return heads
    return 1


def _init_linear_and_attention(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, nn.Linear):
            nn.init.xavier_uniform_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)
        elif isinstance(child, nn.MultiheadAttention):
            nn.init.xavier_uniform_(child.in_proj_weight)
            if child.in_proj_bias is not None:
                nn.init.zeros_(child.in_proj_bias)
            if child.out_proj.bias is not None:
                nn.init.zeros_(child.out_proj.bias)


class _KernelTransformer(nn.Module):
    """Mixture-of-Omori CDF encoder. Subclasses choose the radial vs linear argument."""

    design_version = "fixed-history-omori-cdf-mixture-v3"
    # "linear" -> c/(u+c); "squared" -> D/(u^2+D)
    scale_argument = "linear"
    # Multipliers for softplus scale (c or D) and excess exponent (p-1 or q-1).
    scale_logspace = (-5.0, 0.0)
    exp_scale_value = 0.5

    def __init__(
        self,
        d_model: int = 32,
        nhead: int | None = None,
        num_layers: int = 1,
        dim_feedforward: int | None = None,
        seed: int | None = None,
        num_components: int = 4,
    ) -> None:
        super().__init__()
        if d_model < 1 or num_layers < 1 or num_components < 1:
            raise ValueError("d_model, num_layers and num_components must be >= 1")
        self.d_model = int(d_model)
        self.nhead = int(nhead) if nhead is not None else _nhead(self.d_model)
        if self.nhead < 1 or self.d_model % self.nhead:
            raise ValueError("nhead must be positive and divide d_model")
        self.num_layers = int(num_layers)
        self.num_components = int(num_components)
        self.dim_feedforward = int(dim_feedforward or 4 * self.d_model)
        self.context_features = 2
        # Preserve the caller RNG and seed ALL projections, including attention QKV.
        with torch.random.fork_rng(devices=[]) if seed is not None else nullcontext():
            if seed is not None:
                torch.manual_seed(int(seed))
            layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.in_proj = nn.Linear(self.context_features, self.d_model)
            self.encoder = nn.TransformerEncoder(
                layer, num_layers=self.num_layers, enable_nested_tensor=False
            )
            # K mixture logits, K scale logits, K exponent logits. These are
            # CDF parameters, not PDF values.
            self.out_proj = nn.Linear(self.d_model, 3 * self.num_components)
            _init_linear_and_attention(self)
        lo, hi = self.scale_logspace
        self.register_buffer(
            "scale_scales", torch.logspace(lo, hi, self.num_components)
        )
        self.register_buffer(
            "exp_scales",
            torch.full((self.num_components,), float(self.exp_scale_value)),
        )

    def encode_context(
        self, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode fixed history once; no query argument is accepted here."""
        if context.ndim != 3 or context.shape[-1] != self.context_features:
            raise ValueError(
                "context must have shape (B,H,2): fixed history age, magnitude"
            )
        hidden = self.in_proj(context)
        with sdpa_kernel(SDPBackend.MATH):
            hidden = self.encoder(hidden)
        logits, raw_scale, raw_exp = self.out_proj(hidden).chunk(3, dim=-1)
        weights = torch.softmax(logits, dim=-1)
        scales = (F.softplus(raw_scale) + 1e-4) * self.scale_scales
        # p_k or q_k is strictly greater than 1.
        exponents = 1.0 + (F.softplus(raw_exp) + 1e-3) * self.exp_scales
        return weights, scales, exponents

    def _survival_base(self, query: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """Return c/(u+c) or D/(u^2+D), broadcast against mixture parameters."""
        if self.scale_argument == "linear":
            argument = query
        elif self.scale_argument == "squared":
            argument = query.pow(2)
        else:
            raise ValueError(self.scale_argument)
        return scales / (argument + scales)

    def cdf_from_parameters(
        self,
        query: torch.Tensor,
        parameters: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Evaluate normalized CDF for nonnegative normalized query, context fixed.

        Broadcasting supports quadrature grids. Query must have last dimension 1.
        Negative queries are outside the model domain and are rejected rather
        than silently changing the derivative at zero via a clamp.
        """
        if query.shape[-1] != 1:
            raise ValueError("query must have last dimension 1")
        if bool(torch.any(query < 0)):
            raise ValueError("CDF query must be nonnegative")
        weights, scales, exponents = parameters
        # Use log(c) - log(u+c) rather than clamping c/(u+c) to 1. A max=1
        # clamp would zero the autograd derivative at the origin, where the
        # ratio is exactly 1 and the PDF / second-derivative limit live.
        if self.scale_argument == "linear":
            argument = query
        elif self.scale_argument == "squared":
            argument = query.pow(2)
        else:
            raise ValueError(self.scale_argument)
        log_base = torch.log(scales) - torch.log(argument + scales)
        survival = torch.exp((exponents - 1.0) * log_base)
        component_cdfs = 1.0 - survival
        return (weights * component_cdfs).sum(dim=-1, keepdim=True)

    def plane_density_from_parameters(
        self,
        radius: torch.Tensor,
        parameters: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        max_dis: float,
    ) -> torch.Tensor:
        """Isotropic plane density at a physical radius, for the spatial mixture.

        w(r) = sum_k w_k (q_k-1) D_k^{q_k-1} / (pi * max_dis^2 * (u^2+D_k)^{q_k})
        with u = r / max_dis. At r=0 this is finite: sum_k w_k (q_k-1)/(pi max_dis^2 D_k).
        Only valid when ``scale_argument == 'squared'``.
        """
        if self.scale_argument != "squared":
            raise TypeError("plane_density_from_parameters is only defined for spatial CDFs")
        if radius.shape[-1] != 1:
            raise ValueError("radius must have last dimension 1")
        weights, scales, exponents = parameters
        max_dis = float(max_dis)
        u2 = (radius / max_dis).pow(2)
        denom = u2 + scales
        # (q-1) / (pi * max_dis^2 * (u^2+D)) * [D/(u^2+D)]^{q-1}
        log_base = torch.log(scales) - torch.log(denom)
        dens = (
            weights
            * (exponents - 1.0)
            / (math.pi * (max_dis**2) * denom)
            * torch.exp((exponents - 1.0) * log_base)
        )
        return dens.sum(dim=-1, keepdim=True)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor | None = None,
        *,
        parameters: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """CDF only. A missing context denotes an unconditional zero context.

        The model integration always passes explicit history. Never use query
        values to construct context; context must be fixed during differentiation
        AND numerical integration (not merely detached from the autograd graph).
        """
        if parameters is not None and context is not None:
            raise ValueError("Pass context OR precomputed parameters, not both")
        if parameters is None:
            if context is None:
                context = query.new_zeros((*query.shape[:-1], self.context_features))
            parameters = self.encode_context(context)
        return self.cdf_from_parameters(query, parameters)

    def project(self) -> None:
        """CDF parameter transforms enforce constraints, not weight clipping."""
        return


class TemporalKernelTransformer(_KernelTransformer):
    """Mixture of Omori CDFs on normalized nonnegative elapsed time."""

    scale_argument = "linear"
    scale_logspace = (-5.0, 0.0)


class SpatialKernelTransformer(_KernelTransformer):
    """Mixture of ETAS radial CDFs; isotropic, normalized over the plane."""

    scale_argument = "squared"
    # D is in (r/max_dis)^2 units; physical D^2 can be ~1e-3 km^2, max_dis ~ 1e2 km.
    scale_logspace = (-8.0, -1.0)


class KappaMLP(nn.Module):
    """Deeper unconstrained positive-output MLP on normalized magnitude.

    Unlike KappaNetwork (5x32, nonnegative weights, tanh hidden, softplus out),
    this stack uses GELU hidden layers with bias and no NonNeg constraint.
    Default: 4 hidden layers of width 64.
    """

    def __init__(
        self,
        hidden_width: int = 64,
        num_hidden: int = 4,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if hidden_width < 1 or num_hidden < 1:
            raise ValueError("hidden_width and num_hidden must be >= 1")
        self.hidden_width = int(hidden_width)
        self.num_hidden = int(num_hidden)
        with torch.random.fork_rng(devices=[]) if seed is not None else nullcontext():
            if seed is not None:
                torch.manual_seed(int(seed))
            layers: list[nn.Module] = []
            in_features = 1
            for _ in range(self.num_hidden):
                layers.append(nn.Linear(in_features, self.hidden_width))
                layers.append(nn.GELU())
                in_features = self.hidden_width
            self.hidden = nn.Sequential(*layers)
            self.output = nn.Linear(in_features, 1)
            _init_linear_and_attention(self)

    def forward(self, mag_norm: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.output(self.hidden(mag_norm))) + 1e-8

    def project(self) -> None:
        return


class KappaTransformer(nn.Module):
    """Per-event productivity from a small history Transformer.

    Tokens are the same (age_at_prev, magnitude) features as the kernel
    encoders. Output is a positive kappa per history event (softplus).
    """

    design_version = "kappa-history-transformer-v1"

    def __init__(
        self,
        d_model: int = 32,
        nhead: int | None = None,
        num_layers: int = 1,
        dim_feedforward: int | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if d_model < 1 or num_layers < 1:
            raise ValueError("d_model and num_layers must be >= 1")
        self.d_model = int(d_model)
        self.nhead = int(nhead) if nhead is not None else _nhead(self.d_model)
        if self.nhead < 1 or self.d_model % self.nhead:
            raise ValueError("nhead must be positive and divide d_model")
        self.num_layers = int(num_layers)
        self.dim_feedforward = int(dim_feedforward or 4 * self.d_model)
        self.context_features = 2
        with torch.random.fork_rng(devices=[]) if seed is not None else nullcontext():
            if seed is not None:
                torch.manual_seed(int(seed))
            layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.in_proj = nn.Linear(self.context_features, self.d_model)
            self.encoder = nn.TransformerEncoder(
                layer, num_layers=self.num_layers, enable_nested_tensor=False
            )
            self.out_proj = nn.Linear(self.d_model, 1)
            _init_linear_and_attention(self)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        if context.ndim != 3 or context.shape[-1] != self.context_features:
            raise ValueError("context must have shape (B,H,2)")
        hidden = self.in_proj(context)
        with sdpa_kernel(SDPBackend.MATH):
            hidden = self.encoder(hidden)
        return F.softplus(self.out_proj(hidden)) + 1e-8

    def project(self) -> None:
        return
