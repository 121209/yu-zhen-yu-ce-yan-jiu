"""Signed polar mass for simple nonconvex study polygons (ComCat regression)."""
import unittest
import numpy as np
import torch
from models import KernelPointProcess, _points_in_polygon


class NonconvexBoundaryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.m = KernelPointProcess(
            4, 4, 4, 'neural', 'neural', 'neural', 1., 100., 2, 8,
            seed=4, backbone='transformer', use_boundary_correction=True,
            init_mu_from_rate=False).double()
        self.m.max_dis = 1.
        self.poly = np.array([[0,0],[4,0],[4,4],[3,4],
                              [3,1],[1,1],[1,4],[0,4]], float)
        self.src = np.array([[.5,2],[2,2],[-1,2],[2,.5],[0,0],[1,1]])
        self.x = torch.tensor(self.src[None,:,0:1], dtype=torch.float64)
        self.y = torch.tensor(self.src[None,:,1:2], dtype=torch.float64)
        self.w = torch.ones_like(self.x)
        self.D = torch.full_like(self.x, .49, requires_grad=True)
        self.q = torch.full_like(self.x, 1.8)

    def mass(self, D=None):
        return self.m._boundary_mass(self.x, self.y, self.w,
                                     (self.w, self.D if D is None else D, self.q))

    def test_multiple_intervals_and_boundary_vertices_monte_carlo(self):
        self.m.set_region(self.poly, 128)
        coarse = self.mass().detach().numpy().ravel()
        self.m.set_region(self.poly, 4096)
        fine = self.mass().detach().numpy().ravel()
        rng = np.random.default_rng(20261114)
        r = .7 * np.sqrt((1-rng.random(300000))**(-1/.8)-1)
        theta = rng.random(len(r))*2*np.pi
        delta = np.column_stack((r*np.cos(theta), r*np.sin(theta)))
        mc = np.array([_points_in_polygon(delta+s,self.poly).mean() for s in self.src])
        np.testing.assert_allclose(coarse, mc, atol=.004, rtol=0)
        np.testing.assert_allclose(coarse, fine, atol=.001, rtol=0)
        self.assertTrue(((coarse>=0)&(coarse<=1)).all())
        self.m.set_region(self.poly[::-1], 128)
        np.testing.assert_allclose(self.mass().detach().numpy().ravel(), coarse, atol=1e-12)

    def test_gradient_finite_difference(self):
        self.m.set_region(self.poly, 128)
        grad = torch.autograd.grad(self.mass().sum(), self.D)[0]
        eps = 1e-6
        fd = (self.mass(self.D+eps)-self.mass(self.D-eps))/(2*eps)
        self.assertTrue(torch.isfinite(grad).all())
        torch.testing.assert_close(grad, fd, atol=1e-8, rtol=1e-5)

    def test_convex_original_path_unchanged(self):
        self.m.set_region(np.array([[-5,-5],[5,-5],[5,5],[-5,5]],float),128)
        old = self.mass()
        self.m._region_is_convex = False  # Exercise general formula on same rays.
        new = self.mass()
        torch.testing.assert_close(old, new, atol=1e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
