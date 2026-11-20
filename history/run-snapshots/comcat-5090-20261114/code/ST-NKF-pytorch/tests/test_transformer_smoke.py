"""CPU smoke tests for the temporal/spatial Transformer backbone.

Kappa must remain the existing FCNN when backbone='transformer'.
The default backbone stays 'fcnn' so the paper reproduction path is unchanged.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from layers import KappaNetwork, SpatialKernelNetwork, TemporalKernelNetwork
from layers_transformer import (
    KappaMLP,
    KappaTransformer,
    SpatialKernelTransformer,
    TemporalKernelTransformer,
)
from models import KernelPointProcess, _points_in_polygon
from run_models import parse_args, prepare_run


def _tiny_catalog(n=24, seed=0):
    rng = np.random.default_rng(seed)
    times = np.cumsum(rng.uniform(0.05, 0.4, size=n)) + 10_000.0
    mags = rng.uniform(1.0, 3.5, size=n)
    x = rng.normal(scale=5.0, size=n)
    y = rng.normal(scale=5.0, size=n)
    return times, mags, x, y


class TransformerKernelTests(unittest.TestCase):
    def test_forward_backward_small_window(self):
        torch.manual_seed(0)
        net = TemporalKernelTransformer(d_model=8, seed=0)
        inputs = torch.rand(2, 8, 1, dtype=torch.float32, requires_grad=True)
        outputs = net(inputs)
        self.assertEqual(tuple(outputs.shape), (2, 8, 1))
        self.assertTrue(torch.isfinite(outputs).all())
        pdf = torch.autograd.grad(outputs.sum(), inputs, create_graph=True)[0]
        self.assertEqual(tuple(pdf.shape), (2, 8, 1))
        self.assertTrue(torch.isfinite(pdf).all())
        loss = (pdf.square().mean() + outputs.mean())
        loss.backward()
        self.assertTrue(math.isfinite(float(loss.detach())))
        grads = [p.grad for p in net.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))

    def test_spatial_same_io_and_finite_pdf(self):
        torch.manual_seed(1)
        net = SpatialKernelTransformer(d_model=8, seed=1)
        inputs = torch.rand(2, 8, 1, dtype=torch.float32, requires_grad=True)
        outputs = net(inputs)
        self.assertEqual(tuple(outputs.shape), (2, 8, 1))
        pdf = torch.autograd.grad(outputs.sum(), inputs, create_graph=True)[0]
        self.assertTrue(torch.isfinite(outputs).all())
        self.assertTrue(torch.isfinite(pdf).all())

    def test_zero_input_gives_zero_cdf(self):
        net = TemporalKernelTransformer(d_model=8, seed=2)
        zeros = torch.zeros(2, 8, 1)
        self.assertTrue(torch.allclose(net(zeros), zeros, atol=1e-6))

    def test_pdf_nonnegative_on_positive_inputs(self):
        torch.manual_seed(4)
        net = TemporalKernelTransformer(d_model=8, seed=4)
        inputs = torch.rand(2, 8, 1, dtype=torch.float32, requires_grad=True) + 0.01
        outputs = net(inputs)
        pdf = torch.autograd.grad(outputs.sum(), inputs, create_graph=True)[0]
        self.assertTrue(torch.isfinite(pdf).all())
        self.assertTrue(torch.all(pdf >= -1e-6))
        self.assertTrue(torch.all(outputs >= -1e-6))
        self.assertTrue(torch.all(outputs < 1.0))

    def test_project_does_not_clamp_attention_weights(self):
        net = TemporalKernelTransformer(d_model=8, seed=0)
        net.in_proj.weight.data.fill_(-0.3)
        # Grab an attention projection if present.
        attn = None
        for module in net.modules():
            if isinstance(module, torch.nn.MultiheadAttention):
                attn = module
                break
        self.assertIsNotNone(attn)
        attn.in_proj_weight.data.fill_(-0.2)
        net.project()
        self.assertTrue(torch.equal(net.in_proj.weight, torch.full_like(net.in_proj.weight, -0.3)))
        self.assertTrue(torch.equal(attn.in_proj_weight, torch.full_like(attn.in_proj_weight, -0.2)))


class TransformerBackboneModelTests(unittest.TestCase):
    def test_kappa_is_still_fcnn_when_backbone_transformer(self):
        model = KernelPointProcess(
            8, 8, 8, "neural", "neural", "neural", 1.0, 250.0, 2, 8, seed=3, backbone="transformer", use_boundary_correction=False, init_mu_from_rate=False
        )
        self.assertIsInstance(model.temporal_net, TemporalKernelTransformer)
        self.assertIsInstance(model.spatial_net, SpatialKernelTransformer)
        self.assertIsInstance(model.kappa_net, KappaNetwork)
        self.assertNotIsInstance(model.temporal_net, TemporalKernelNetwork)
        self.assertNotIsInstance(model.spatial_net, SpatialKernelNetwork)
        # Kappa keeps the FCNN contract: bias + softplus + nonnegative project.
        self.assertIsNotNone(model.kappa_net.hidden[0].bias)
        self.assertEqual(model.kappa_net.output_activation, "softplus")
        model.kappa_net.hidden[0].weight.data.fill_(-0.4)
        model.temporal_net.in_proj.weight.data.fill_(-0.5)
        model.project_constraints()
        self.assertTrue(torch.all(model.kappa_net.hidden[0].weight >= 0))
        self.assertTrue(torch.all(model.temporal_net.in_proj.weight == -0.5))

    def test_default_backbone_is_fcnn(self):
        model = KernelPointProcess(
            8, 8, 8, "neural", "neural", "neural", 1.0, 250.0, 2, 4, seed=1
        )
        self.assertEqual(model.backbone, "fcnn")
        self.assertIsInstance(model.temporal_net, TemporalKernelNetwork)
        self.assertIsInstance(model.spatial_net, SpatialKernelNetwork)
        self.assertIsInstance(model.kappa_net, KappaNetwork)

    def test_model_forward_backward_finite_loss(self):
        times, mags, x, y = _tiny_catalog()
        model = KernelPointProcess(
            time_step_train=8,
            time_step_val=8,
            time_step_test=8,
            temporal_id="neural",
            spatial_id="neural",
            kappa_id="neural",
            global_m0=1.0,
            area=250.0,
            size_layer=2,
            size_nn=8,
            seed=3,
            backbone="transformer",
            use_boundary_correction=False,
            init_mu_from_rate=False,
        )
        model.set_train_data(times, mags, x, y)
        model.set_val_data(times, mags, x, y)
        model.set_test_data(times, mags, x, y)
        model.set_model()
        loss = model.loss_on("train", np.arange(0, 2), create_graph=True)
        self.assertTrue(math.isfinite(float(loss.detach())))
        loss.backward()
        self.assertIsNotNone(model.kappa_net.output.weight.grad)
        self.assertTrue(torch.isfinite(model.kappa_net.output.weight.grad).all())
        self.assertIsNotNone(model.temporal_net.out_proj.weight.grad)
        self.assertTrue(torch.isfinite(model.temporal_net.out_proj.weight.grad).all())
        self.assertIsNotNone(model.spatial_net.out_proj.weight.grad)
        self.assertTrue(torch.isfinite(model.spatial_net.out_proj.weight.grad).all())
        # The history encoder, not just the CDF parameter head, learns via NLL.
        for net in (model.temporal_net, model.spatial_net):
            for param in (net.in_proj.weight, net.encoder.layers[0].self_attn.in_proj_weight):
                self.assertIsNotNone(param.grad)
                self.assertTrue(torch.isfinite(param.grad).all())
                self.assertGreater(float(param.grad.abs().sum()), 0.0)

    def test_cli_backbone_flag_defaults_to_fcnn(self):
        args = parse_args(["1", "-1", "SanJac", "neural", "neural", "neural"])
        self.assertEqual(args.backbone, "fcnn")
        self.assertEqual(args.temporal_backbone, "fcnn")
        self.assertEqual(args.spatial_backbone, "fcnn")
        self.assertEqual(args.kappa_backbone, "fcnn")
        self.assertEqual(args.lr, 0.005)
        args = parse_args(
            ["1", "-1", "SanJac", "neural", "neural", "neural", "--backbone", "transformer"]
        )
        self.assertEqual(args.backbone, "transformer")
        self.assertEqual(args.temporal_backbone, "transformer")
        self.assertEqual(args.spatial_backbone, "transformer")
        self.assertEqual(args.kappa_backbone, "fcnn")
        self.assertEqual(args.lr, 1e-4)
        args = parse_args([
            "1", "-1", "SanJac", "neural", "neural", "neural",
            "--temporal-backbone", "transformer",
            "--spatial-backbone", "fcnn",
            "--kappa-backbone", "mlp",
        ])
        self.assertEqual(args.temporal_backbone, "transformer")
        self.assertEqual(args.spatial_backbone, "fcnn")
        self.assertEqual(args.kappa_backbone, "mlp")
        self.assertEqual(args.lr, 1e-4)


class NormalizedCDFTests(unittest.TestCase):
    """Real derivatives and integration checks with history held fixed."""

    def _kernel_and_context(self, cls):
        torch.manual_seed(12)
        net = cls(d_model=8, seed=12).double().eval()
        context = torch.tensor([[[1.0, .4], [.7, .6], [0.0, .8]]], dtype=torch.float64)
        return net, context

    def test_finite_difference_matches_true_autograd_for_both_cdfs(self):
        for cls in (TemporalKernelTransformer, SpatialKernelTransformer):
            net, context = self._kernel_and_context(cls)
            for values in ([.02, .3, 1.4], [.1, .7, 2.5]):
                u = torch.tensor([[values]], dtype=torch.float64).transpose(1,2).requires_grad_()
                f = net(u, context)
                pdf = torch.autograd.grad(f.sum(), u)[0]
                # Recompute the encoder on identical context at each perturbation.
                eps = 1e-6
                fd = (net(u.detach()+eps, context)-net(u.detach()-eps, context))/(2*eps)
                error = float((fd-pdf).abs().max().detach())
                print(f"CDF_FD {cls.__name__}: max_abs={error:.3e}")
                self.assertTrue(torch.allclose(fd, pdf, atol=2e-8, rtol=2e-7))

    def test_monotonicity_origin_and_unit_asymptote(self):
        for cls in (TemporalKernelTransformer, SpatialKernelTransformer):
            net, context = self._kernel_and_context(cls)
            params = net.encode_context(context)
            u = torch.linspace(0, 20, 2001, dtype=torch.float64).reshape(-1,1,1)
            f = net(u, parameters=params)
            self.assertTrue(torch.equal(f[0], torch.zeros_like(f[0])))
            self.assertGreaterEqual(float(torch.diff(f, dim=0).min().detach()), -1e-14)
            # Heavy tails approach 1 slowly; 1e20 is still a finite-time proxy.
            finf = net(torch.full((1,3,1), 1e80, dtype=torch.float64), parameters=params)
            self.assertTrue(torch.allclose(finf, torch.ones_like(finf), atol=1e-12, rtol=0))
            print(f"CDF_LIMIT {cls.__name__}: max_error={float((finf-1).abs().max().detach()):.3e}")

    def test_numerical_pdf_integral_matches_cdf_difference_with_scale(self):
        for cls in (TemporalKernelTransformer, SpatialKernelTransformer):
            net, context = self._kernel_and_context(cls)
            params = net.encode_context(context)
            scale = 7.3
            q = torch.linspace(.15, 3.0, 20001, dtype=torch.float64).reshape(-1,1,1).expand(-1,3,-1).clone().requires_grad_()
            f = net(q/scale, parameters=params)
            pdf = torch.autograd.grad(f.sum(), q)[0]
            integral = torch.trapezoid(pdf, q[:,0,0], dim=0)
            exact = f[-1]-f[0]
            err = float((integral-exact).abs().max().detach())
            print(f"CDF_INTEGRAL {cls.__name__}: max_abs={err:.3e}")
            self.assertTrue(torch.allclose(integral, exact, atol=2e-7, rtol=2e-6))

    def test_spatial_plane_mass_with_exact_radial_jacobian(self):
        net, context = self._kernel_and_context(SpatialKernelTransformer)
        params = net.encode_context(context)
        scale = 7.3
        # Log-spaced radii so the near-origin spike of a tiny D component is resolved.
        u = torch.logspace(-12, 4, 40001, dtype=torch.float64)
        r = (u * scale).reshape(-1, 1, 1).expand(-1, 3, -1).clone().requires_grad_()
        cdf = net(r / scale, parameters=params)
        radial_pdf = torch.autograd.grad(cdf.sum(), r)[0]
        safe_r = torch.where(r == 0, torch.ones_like(r), r)
        planar_pdf = radial_pdf / (2 * math.pi * safe_r)
        mass = torch.trapezoid(planar_pdf * (2 * math.pi * r), r[:, 0, 0], dim=0)
        # Enclosed mass must match the radial CDF, and a large-u CDF is ~1.
        exact = cdf[-1].reshape(-1)
        err = float((mass - exact).abs().max().detach())
        print(f"SPATIAL_PLANE mass={mass.detach().reshape(-1).tolist()} cdf={exact.detach().tolist()} max_error={err:.3e}")
        self.assertTrue(torch.allclose(mass, exact, atol=2e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(exact, torch.ones_like(exact), atol=2e-3, rtol=0))
        # Closed-form plane density must match the autograd Jacobian form.
        closed = net.plane_density_from_parameters(r.detach(), params, scale)
        self.assertTrue(torch.allclose(closed, planar_pdf, atol=1e-8, rtol=1e-5))

    def _model(self):
        model = KernelPointProcess(4,4,4,"neural","neural","neural",1.,30.,2,8,seed=12,backbone="transformer",use_boundary_correction=False,init_mu_from_rate=False).double()
        model.max_time = 2.0; model.max_mag = 3.0; model.max_dis = 4.0
        return model

    def test_model_compensator_same_history_parameters_and_numeric_integral(self):
        from unittest.mock import patch
        model = self._model()
        lower = torch.tensor([[[1.2],[.5],[.15],[0.0]]], dtype=torch.float64)
        dt = .4
        mags = torch.tensor([[[1.3],[2.0],[1.7],[2.2]]], dtype=torch.float64)
        distance = torch.ones_like(mags)
        with patch.object(model.temporal_net, "encode_context", wraps=model.temporal_net.encode_context) as encode, patch.object(model.temporal_net,"forward",wraps=model.temporal_net.forward) as evaluate:
            out = model(lower+dt, lower[:,:-1], mags, distance)
        self.assertEqual(encode.call_count, 1)
        self.assertEqual(evaluate.call_count, 2)
        self.assertEqual(evaluate.call_args_list[0].args[0].shape[1],4)
        self.assertEqual(evaluate.call_args_list[1].args[0].shape[1],4)
        params = evaluate.call_args_list[0].kwargs["parameters"]
        self.assertIs(params, evaluate.call_args_list[1].kwargs["parameters"])
        # Log-spaced quadrature, including the origin, to resolve sharp Omori peaks.
        s = torch.cat((torch.zeros(1, dtype=torch.float64), torch.logspace(-10, math.log10(dt), 20000, dtype=torch.float64)))
        s = s[s <= dt]
        if s[-1] < dt:
            s = torch.cat((s, torch.tensor([dt], dtype=torch.float64)))
        age = (lower+s[:,None,None]).requires_grad_()
        f = model.temporal_net(age/model.max_time, parameters=params)
        h = torch.autograd.grad(f.sum(),age)[0]
        trigger = (h*out["kappa"]).sum(dim=1)[:,0]
        numeric = torch.trapezoid(trigger,s)
        exact = (out["int_l"] - model.bac_func.mu*model.area_scalar*dt).squeeze()
        err = float((numeric-exact).abs().detach())
        print(f"COMPENSATOR max_abs={err:.3e} analytic={float(exact.detach()):.9f} numeric={float(numeric.detach()):.9f}")
        self.assertTrue(torch.allclose(numeric,exact,atol=2e-5,rtol=2e-5))
        self.assertGreaterEqual(float(exact.detach()),0.)
        # The compensator must equal kappa * Delta F with the same parameters.
        cdf_hi = model.temporal_net((lower+dt)/model.max_time, parameters=params)
        cdf_lo = model.temporal_net(lower/model.max_time, parameters=params)
        closed = ((cdf_hi - cdf_lo) * out["kappa"]).sum()
        self.assertTrue(torch.allclose(closed, exact, atol=1e-12, rtol=1e-10))

    def test_context_independent_of_query_time_and_position(self):
        from unittest.mock import patch
        model = self._model()
        lower = torch.tensor([[[1.2],[.5],[.15],[0.0]]],dtype=torch.float64)
        mags = torch.ones_like(lower)*2
        with patch.object(model.temporal_net,"encode_context",wraps=model.temporal_net.encode_context) as enc:
            model(lower+.2,lower[:,:-1],mags,torch.ones_like(mags))
            model(lower+.7,lower[:,:-1],mags,torch.ones_like(mags)*4)
        self.assertTrue(torch.equal(enc.call_args_list[0].args[0], enc.call_args_list[1].args[0]))

    def test_spatial_zero_radius_limit_finite_and_backward(self):
        model = self._model()
        lower = torch.tensor([[[1.2],[.5],[.15],[0.0]]],dtype=torch.float64)
        mags = torch.ones_like(lower)*2
        out = model(lower+.2,lower[:,:-1],mags,torch.zeros_like(mags),create_graph=True)
        self.assertTrue(torch.isfinite(out["pdf_spatial"]).all())
        self.assertTrue(torch.all(out["pdf_spatial"]>0))
        context = torch.cat((lower/model.max_time,mags/model.max_mag),dim=-1)
        w, scales, exponents = model.spatial_net.encode_context(context)
        expected = (w * (exponents - 1.0) / (math.pi * model.max_dis**2 * scales)).sum(
            dim=-1, keepdim=True
        )
        self.assertTrue(torch.allclose(out["pdf_spatial"], expected, atol=1e-12, rtol=1e-10))
        model.nll(out).backward()
        self.assertTrue(torch.isfinite(model.spatial_net.out_proj.weight.grad).all())
        # The history encoder, not just the CDF parameter head, learns via NLL.
        for net in (model.temporal_net, model.spatial_net):
            for param in (net.in_proj.weight, net.encoder.layers[0].self_attn.in_proj_weight):
                self.assertIsNotNone(param.grad)
                self.assertTrue(torch.isfinite(param.grad).all())
                self.assertGreater(float(param.grad.abs().sum()), 0.0)
        # Evaluation needs a second u derivative at r=0 even if create_graph=False.
        evaluation = model(lower+.2,lower[:,:-1],mags,torch.zeros_like(mags),create_graph=False)
        self.assertTrue(torch.isfinite(evaluation["pdf_spatial"]).all())

    def test_negative_domain_rejected(self):
        net, context = self._kernel_and_context(TemporalKernelTransformer)
        with self.assertRaisesRegex(ValueError,"nonnegative"):
            net(-torch.ones(1,3,1,dtype=torch.float64),context)



class OmoriMixtureAndKappaTests(unittest.TestCase):
    def test_kappa_positive_for_all_backbones(self):
        torch.manual_seed(0)
        mags = torch.rand(3, 5, 1)
        context = torch.rand(3, 5, 2)
        fcnn = KappaNetwork(8, 2, seed=0)
        mlp = KappaMLP(hidden_width=16, num_hidden=3, seed=0)
        trans = KappaTransformer(d_model=8, seed=0)
        for name, out in (
            ("fcnn", fcnn(mags)),
            ("mlp", mlp(mags)),
            ("transformer", trans(context)),
        ):
            self.assertEqual(tuple(out.shape), (3, 5, 1), name)
            self.assertTrue(torch.isfinite(out).all(), name)
            self.assertTrue(torch.all(out > 0), name)

    def test_mixed_backbone_forward_shapes(self):
        times, mags, x, y = _tiny_catalog()
        combos = [
            ("transformer", "transformer", "transformer"),
            ("transformer", "transformer", "fcnn"),
            ("transformer", "fcnn", "fcnn"),
            ("transformer", "transformer", "mlp"),
        ]
        for t_bb, s_bb, k_bb in combos:
            model = KernelPointProcess(
                8, 8, 8, "neural", "neural", "neural", 1.0, 250.0, 2, 8, seed=3,
                temporal_backbone=t_bb, spatial_backbone=s_bb, kappa_backbone=k_bb,
                use_boundary_correction=False, init_mu_from_rate=False,
            )
            model.set_train_data(times, mags, x, y)
            model.set_val_data(times, mags, x, y)
            model.set_test_data(times, mags, x, y)
            model.set_model()
            out = model.forward(*model._batch_tensors("train", np.arange(0, 3)), create_graph=True)
            for key in ("l_ts", "l_t", "int_l"):
                self.assertEqual(tuple(out[key].shape), (3, 1, 1), (t_bb, s_bb, k_bb, key))
                self.assertTrue(torch.isfinite(out[key]).all(), (t_bb, s_bb, k_bb, key))
            self.assertEqual(tuple(out["kappa"].shape), (3, 8, 1))
            self.assertTrue(torch.all(out["kappa"] > 0))
            loss = model.nll(out)
            loss.backward()
            self.assertTrue(math.isfinite(float(loss.detach())))

    def test_scheme2_kappa_remains_fcnn(self):
        model = KernelPointProcess(
            8, 8, 8, "neural", "neural", "neural", 1.0, 250.0, 2, 8, seed=3,
            backbone="transformer", use_boundary_correction=False, init_mu_from_rate=False,
        )
        self.assertIsInstance(model.kappa_net, KappaNetwork)
        self.assertIsInstance(model.temporal_net, TemporalKernelTransformer)
        self.assertIsInstance(model.spatial_net, SpatialKernelTransformer)


class BoundaryMassTests(unittest.TestCase):
    def _square(self, half=5.0):
        return np.array(
            [[-half, -half], [half, -half], [half, half], [-half, half]],
            dtype=np.float64,
        )

    def test_interior_peaked_kernel_has_mass_near_one(self):
        model = KernelPointProcess(
            4, 4, 4, "neural", "neural", "neural", 1.0, 100.0, 2, 8, seed=12,
            backbone="transformer", use_boundary_correction=True, init_mu_from_rate=False,
            boundary_angles=128,
        ).double()
        model.max_time = 2.0
        model.max_mag = 3.0
        model.max_dis = 4.0
        model.set_region(self._square(20.0), n_angles=128)
        # Force a very peaked spatial mixture so almost all mass is near the origin.
        context = torch.tensor([[[0.2, 0.4], [0.1, 0.6]]], dtype=torch.float64)
        with torch.no_grad():
            # Drive the raw scale logits strongly negative so D uses the smallest bin.
            model.spatial_net.out_proj.bias.zero_()
        w, D, q = model.spatial_net.encode_context(context)
        hist_x = torch.tensor([[[0.0], [0.0]]], dtype=torch.float64)
        hist_y = torch.tensor([[[0.0], [0.0]]], dtype=torch.float64)
        mags = torch.tensor([[[1.2], [1.5]]], dtype=torch.float64)
        mass = model._boundary_mass(hist_x, hist_y, mags, (w, D, q))
        self.assertTrue(torch.all(mass > 0.95), mass)
        self.assertTrue(torch.all(mass <= 1.0 + 1e-6), mass)

    def test_mass_matches_monte_carlo_and_finer_polar(self):
        model = KernelPointProcess(
            4, 4, 4, "neural", "neural", "neural", 1.0, 100.0, 2, 8, seed=4,
            backbone="transformer", use_boundary_correction=True, init_mu_from_rate=False,
            boundary_angles=256,
        ).double()
        model.max_time = 2.0
        model.max_mag = 3.0
        model.max_dis = 8.0
        square = self._square(5.0)
        model.set_region(square, n_angles=256)
        context = torch.tensor([[[0.4, 0.5]]], dtype=torch.float64)
        params = model.spatial_net.encode_context(context)
        # Place the source off-center so a non-trivial fraction of mass leaks out.
        hist_x = torch.tensor([[[3.5]]], dtype=torch.float64)
        hist_y = torch.tensor([[[0.0]]], dtype=torch.float64)
        mags = torch.tensor([[[1.0]]], dtype=torch.float64)
        mass = model._boundary_mass(hist_x, hist_y, mags, params)
        analytic = float(mass.squeeze().detach())
        # Independent Monte Carlo: invert the mixture CDF, sample angles.
        w, D, q = [p.detach().reshape(-1) for p in params]
        rng = torch.Generator().manual_seed(0)
        n_mc = 20000
        comp = torch.multinomial(w, n_mc, replacement=True, generator=rng)
        # u01 is the radial CDF value F; invert 1-F = [D/(u^2+D)]^{q-1}.
        u01 = torch.rand(n_mc, dtype=torch.float64, generator=rng).clamp(min=1e-12, max=1-1e-12)
        Dc, qc = D[comp], q[comp]
        u2 = Dc * (torch.pow(1.0 - u01, -1.0 / (qc - 1.0)) - 1.0)
        radius = model.max_dis * torch.sqrt(u2)
        theta = torch.rand(n_mc, dtype=torch.float64, generator=rng) * (2 * math.pi)
        xs = 3.5 + radius * torch.cos(theta)
        ys = 0.0 + radius * torch.sin(theta)
        inside = (xs.abs() <= 5.0) & (ys.abs() <= 5.0)
        mc = float(inside.double().mean())
        # Finer polar quadrature should be close to the 256-angle value.
        model.set_region(square, n_angles=1024)
        mass_fine = float(model._boundary_mass(hist_x, hist_y, mags, params).squeeze().detach())
        print(f"BOUNDARY_MASS analytic256={analytic:.6f} polar1024={mass_fine:.6f} mc={mc:.6f}")
        self.assertTrue(abs(analytic - mass_fine) < 0.01)
        self.assertTrue(abs(analytic - mc) < 0.03)
        self.assertTrue(0.0 < analytic < 1.0)

    def test_compensator_uses_mass_and_matches_space_time_integral(self):
        model = KernelPointProcess(
            4, 4, 4, "neural", "neural", "neural", 1.0, 100.0, 2, 8, seed=12,
            backbone="transformer", use_boundary_correction=True, init_mu_from_rate=False,
            boundary_angles=128,
        ).double()
        model.max_time = 2.0
        model.max_mag = 3.0
        model.max_dis = 8.0
        model.set_region(self._square(5.0), n_angles=128)
        lower = torch.tensor([[[1.2], [0.5], [0.15], [0.0]]], dtype=torch.float64)
        dt = 0.4
        mags = torch.tensor([[[1.3], [2.0], [1.7], [2.2]]], dtype=torch.float64)
        # Put sources near the center; distances to a dummy current event.
        hist_x = torch.tensor([[[0.0], [1.0], [-1.0], [0.5]]], dtype=torch.float64)
        hist_y = torch.tensor([[[0.0], [0.5], [1.0], [-0.5]]], dtype=torch.float64)
        # Current event at (0,0) for the intensity, not used in the compensator test.
        dis = torch.sqrt(hist_x**2 + hist_y**2)
        out = model(
            lower + dt, lower[:, :-1], mags, dis, hist_x=hist_x, hist_y=hist_y,
        )
        context = torch.cat((lower / model.max_time, mags / model.max_mag), dim=-1)
        t_params = model.temporal_net.encode_context(context)
        s_params = model.spatial_net.encode_context(context)
        mass = model._boundary_mass(hist_x, hist_y, mags, s_params)
        s = torch.cat((
            torch.zeros(1, dtype=torch.float64),
            torch.logspace(-10, math.log10(dt), 20000, dtype=torch.float64),
        ))
        s = s[s <= dt]
        if s[-1] < dt:
            s = torch.cat((s, torch.tensor([dt], dtype=torch.float64)))
        age = (lower + s[:, None, None]).requires_grad_()
        f = model.temporal_net(age / model.max_time, parameters=t_params)
        h = torch.autograd.grad(f.sum(), age)[0]
        trigger = (h * out["kappa"] * mass).sum(dim=1)[:, 0]
        numeric = torch.trapezoid(trigger, s)
        exact = (out["int_l"] - model.bac_func.mu * model.area_scalar * dt).squeeze()
        err = float((numeric - exact).abs().detach())
        print(f"BOUNDARY_COMPENSATOR max_abs={err:.3e}")
        self.assertTrue(torch.allclose(numeric, exact, atol=2e-5, rtol=2e-5))
        cdf_hi = model.temporal_net((lower + dt) / model.max_time, parameters=t_params)
        cdf_lo = model.temporal_net(lower / model.max_time, parameters=t_params)
        closed = ((cdf_hi - cdf_lo) * out["kappa"] * mass).sum()
        self.assertTrue(torch.allclose(closed, exact, atol=1e-12, rtol=1e-10))
        self.assertTrue(torch.all(mass > 0))
        self.assertTrue(torch.all(mass <= 1.0 + 1e-5))

    def test_official_sanjac_polygon_is_convex_and_points_inside(self):
        prep = prepare_run("SanJac", history=8)
        poly = prep["region_xy"]
        # A catalog event well inside the 1-degree box should project inside.
        xy = np.stack((prep["train"]["x"][:20], prep["train"]["y"][:20]), axis=1)
        inside = _points_in_polygon(xy, poly)
        self.assertGreater(int(inside.sum()), 15)


if __name__ == "__main__":
    unittest.main()
