"""
Module for Inverse problems.
"""

import pickle
import numpy as np
import scipy
from scipy.interpolate import griddata
import torch
import sys

sys.path.insert(0, "..")
sys.path.insert(0, "./shared_modules")
from helper import Timer, cache_x

import problems.boundary_conditions as boundary_conditions
from problems import _Problem

# Import losses - handle both relative and absolute imports
try:
    from training import losses
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from training import losses


class PoissonInv(_Problem):
    """
    Solves the Inverse Poisson problem.
    """

    @property
    def name(self):
        """
        Returns the name of the problem.

        :return: Name of the problem.
        :rtype: str
        """
        return "PoissonInv"

    def __init__(self):
        """
        Initialize the PoissonInv problem.
        """
        self.bbox = [0, 1] * 2
        self.d = (2, 2)
        self.exact_dim_select = slice(1, 2)

    def a_ref(self, xy, gpugrad=False):
        """
        Reference a(x,y) function.

        :param torch.Tensor xy: Input coordinates.
        :param bool gpugrad: Whether to use gradient.
        :return: Value of a.
        :rtype: torch.Tensor or np.ndarray
        """
        if not gpugrad:
            xy = xy.detach().cpu()
        x, y = xy[:, 0:1], xy[:, 1:2]
        return 1 / (1 + x**2 + y**2 + (x - 1)**2 + (y - 1)**2)

    def u_ref(self, xy):
        """
        Reference u(x,y) function.

        :param torch.Tensor xy: Input coordinates.
        :return: Value of u.
        :rtype: np.ndarray
        """
        xy = xy.detach().cpu()
        x, y = xy[:, 0:1], xy[:, 1:2]
        return np.sin(np.pi * x) * np.sin(np.pi * y)

    def f_force(self, xy):
        """
        Forcing term f(x,y).

        :param torch.Tensor xy: Input coordinates.
        :return: Force value.
        :rtype: torch.Tensor
        """
        x, y = xy[:, 0:1], xy[:, 1:2]
        term1 = 2 * np.pi**2 * torch.sin(np.pi * x) * \
            torch.sin(np.pi * y) * self.a_ref(xy, gpugrad=True)
        term2 = 2 * np.pi * (
            (2 * x + 1) * torch.cos(np.pi * x) * torch.sin(np.pi * y) +
            (2 * y + 1) * torch.sin(np.pi * x) * torch.cos(np.pi * y)
        ) * self.a_ref(xy, gpugrad=True)**2
        return term1 + term2

    def get_gradients(self, x, y):
        """
        Calculate gradients for the problem.

        :param torch.Tensor x: Input coordinates.
        :param torch.Tensor y: Output values.
        :return: Tuple containing y and its gradients.
        :rtype: tuple
        """
        u, a = y[:, 0:1], y[:, 1:2]
        ju = torch.autograd.grad(
            u, x, torch.ones_like(u), create_graph=True
        )[0]
        ujx, ujy = ju[:, 0:1], ju[:, 1:2]
        ujxx = torch.autograd.grad(
            ujx, x, torch.ones_like(ujx), create_graph=True
        )[0][:, 0:1]
        ujyy = torch.autograd.grad(
            ujy, x, torch.ones_like(ujy), create_graph=True
        )[0][:, 1:2]
        ja = torch.autograd.grad(
            a, x, torch.ones_like(a), create_graph=True
        )[0]
        ajx, ajy = ja[:, 0:1], ja[:, 1:2]
        return y, ujx, ujy, ujxx, ujyy, ajx, ajy

    def physics_loss(self, x, y, ujx, ujy, ujxx, ujyy, ajx, ajy):
        """
        Calculate the physics loss.

        :param torch.Tensor x: Input coordinates.
        :param torch.Tensor y: Output values.
        :param torch.Tensor ujx: Gradient of u w.r.t x.
        :param torch.Tensor ujy: Gradient of u w.r.t y.
        :param torch.Tensor ujxx: Second gradient of u w.r.t x.
        :param torch.Tensor ujyy: Second gradient of u w.r.t y.
        :param torch.Tensor ajx: Gradient of a w.r.t x.
        :param torch.Tensor ajy: Gradient of a w.r.t y.
        :return: Physics loss value.
        :rtype: torch.Tensor
        """
        u, a = y[:, 0:1], y[:, 1:2]
        phy = a * ujxx + ajx * ujx + a * ujyy + ajy * ujy
        return losses.l2_loss(phy + self.f_force(x), 0)

    def sample_data(self):
        """
        Sample data points for training.

        :return: Data points.
        :rtype: np.ndarray
        """
        bc_x = np.linspace(0, 1, 50)
        bc_y = np.linspace(0, 1, 50)
        bc_x, bc_y = np.meshgrid(bc_x, bc_y)
        bc_xy = np.stack((bc_x.reshape(-1), bc_y.reshape(-1))).T
        return bc_xy

    def sample_bd(self, N_bd):
        """
        Sample boundary points.

        :param int N_bd: Number of boundary points.
        :return: Array of boundary points.
        :rtype: np.ndarray
        """
        N_side = (N_bd + 3) // 4
        pts = torch.zeros(4 * N_side, 2)
        pts[:N_side, 0] = 0
        pts[:N_side, 1] = torch.rand(N_side)
        pts[N_side:2 * N_side, 0] = 1
        pts[N_side:2 * N_side, 1] = torch.rand(N_side)
        pts[2 * N_side:3 * N_side, 1] = 0
        pts[2 * N_side:3 * N_side, 0] = torch.rand(N_side)
        pts[3 * N_side:, 1] = 1
        pts[3 * N_side:, 0] = torch.rand(N_side)
        return pts.numpy()

    def data_loss(self, x, y, ujx, ujy, ujxx, ujyy, ajx, ajy):
        """
        Calculate the data loss.

        :param torch.Tensor x: Input coordinates.
        :param torch.Tensor y: Output values.
        :return: Data loss value.
        :rtype: torch.Tensor
        """
        u, a = y[:, 0:1], y[:, 1:2]
        ur = self.u_ref(x)
        return losses.l2_loss(
            u,
            (ur + np.random.normal(loc=0, scale=0.1, size=ur.shape)).to(u.device)
        )

    def bd_loss(self, x, y, ujx, ujy, ujxx, ujyy, ajx, ajy):
        """
        Calculate the boundary condition loss.

        :param torch.Tensor x: Input coordinates.
        :param torch.Tensor y: Output values.
        :return: Boundary loss value.
        :rtype: torch.Tensor
        """
        u, a = y[:, 0:1], y[:, 1:2]
        return losses.l2_loss(a, self.a_ref(x).to(a.device))

    def exact_solution(self, x, batch_size):
        """
        Calculate the exact solution.

        :param torch.Tensor x: Input coordinates.
        :param tuple batch_size: Batch size.
        :return: Tuple containing exact solution and gradients.
        :rtype: tuple
        """
        return (
            torch.tensor(self.a_ref(x), dtype=torch.float32).to(x.device),
        ) * 7


class HeatInv(_Problem):
    """
    Solves the Inverse Heat problem.
    """

    @property
    def name(self):
        """
        Returns the name of the problem.

        :return: Name of the problem.
        :rtype: str
        """
        return "HeatInv"

    def __init__(self):
        """
        Initialize the HeatInv problem.
        """
        self.bbox = [-1, 1] * 2 + [0, 1]
        self.d = (3, 2)
        self.data_pts = np.loadtxt("../ref/heatinv_points.dat")
        self.exact_dim_select = slice(1, 2)

    def u_ref(self, xyt):
        """
        Reference u(x,y,t) function.

        :param torch.Tensor xyt: Input coordinates.
        :return: Value of u.
        :rtype: np.ndarray
        """
        xyt = xyt.detach().cpu()
        x, y, t = xyt[:, 0:1], xyt[:, 1:2], xyt[:, 2:3]
        return np.exp(-t) * np.sin(np.pi * x) * np.sin(np.pi * y)

    def a_ref(self, xyt):  # irrelevant to time domain
        """
        Reference a(x,y) function.

        :param torch.Tensor xyt: Input coordinates.
        :return: Value of a.
        :rtype: np.ndarray
        """
        xyt = xyt.detach().cpu()
        x, y, t = xyt[:, 0:1], xyt[:, 1:2], xyt[:, 2:3]
        return 2 + np.sin(np.pi * x) * np.sin(np.pi * y)

    def f_force(self, xyt):
        """
        Forcing term f(x,y,t).

        :param torch.Tensor xyt: Input coordinates.
        :return: Force value.
        :rtype: torch.Tensor
        """
        x, y, t = xyt[:, 0:1], xyt[:, 1:2], xyt[:, 2:3]
        s, c, p = torch.sin, torch.cos, np.pi
        return torch.exp(-t) * (
            (4 * p**2 - 1) * s(p * x) * s(p * y) +
            p**2 * (
                2 * s(p * x)**2 * s(p * y)**2 - c(p * x)**2 * s(p * y)**2 -
                s(p * x)**2 * c(p * y)**2
            )
        )

    def get_gradients(self, x, y):
        """
        Calculate gradients for the problem.

        :param torch.Tensor x: Input coordinates.
        :param torch.Tensor y: Output values.
        :return: Tuple containing y and its gradients.
        :rtype: tuple
        """
        u, a = y[:, 0:1], y[:, 1:2]
        ju = torch.autograd.grad(
            u, x, torch.ones_like(u), create_graph=True
        )[0]
        ujx, ujy, ujt = ju[:, 0:1], ju[:, 1:2], ju[:, 2:3]
        ujxx = torch.autograd.grad(
            ujx, x, torch.ones_like(ujx), create_graph=True
        )[0][:, 0:1]
        ujyy = torch.autograd.grad(
            ujy, x, torch.ones_like(ujy), create_graph=True
        )[0][:, 1:2]
        ja = torch.autograd.grad(
            a, x, torch.ones_like(a), create_graph=True
        )[0]
        ajx, ajy = ja[:, 0:1], ja[:, 1:2]
        return y, ujx, ujy, ujt, ujxx, ujyy, ajx, ajy

    def physics_loss(self, x, y, ujx, ujy, ujt, ujxx, ujyy, ajx, ajy):
        """
        Calculate the physics loss.

        :param torch.Tensor x: Input coordinates.
        :param torch.Tensor y: Output values.
        :param torch.Tensor ujx: Gradient of u w.r.t x.
        :param torch.Tensor ujy: Gradient of u w.r.t y.
        :param torch.Tensor ujt: Gradient of u w.r.t t.
        :param torch.Tensor ujxx: Second gradient of u w.r.t x.
        :param torch.Tensor ujyy: Second gradient of u w.r.t y.
        :param torch.Tensor ajx: Gradient of a w.r.t x.
        :param torch.Tensor ajy: Gradient of a w.r.t y.
        :return: Physics loss value.
        :rtype: torch.Tensor
        """
        u, a = y[:, 0:1], y[:, 1:2]
        phy = a * ujxx + ajx * ujx + a * ujyy + ajy * ujy
        return losses.l2_loss(phy + self.f_force(x), ujt)

    def sample_data(self):
        """
        Sample data points.

        :return: Data points.
        :rtype: np.ndarray
        """
        return self.data_pts

    def sample_bd(self, N_bd):
        """
        Sample boundary points.

        :param int N_bd: Number of boundary points.
        :return: Array of boundary points.
        :rtype: np.ndarray
        """
        N_side = (N_bd + 3) // 4
        pts = torch.zeros(4 * N_side, 3)
        pts[:N_side, 0] = -1
        pts[:N_side, 1] = -1 + 2 * torch.rand(N_side)
        pts[N_side:2 * N_side, 0] = 1
        pts[N_side:2 * N_side, 1] = -1 + 2 * torch.rand(N_side)
        pts[2 * N_side:3 * N_side, 1] = -1
        pts[2 * N_side:3 * N_side, 0] = -1 + 2 * torch.rand(N_side)
        pts[3 * N_side:, 1] = 1
        pts[3 * N_side:, 0] = -1 + 2 * torch.rand(N_side)
        pts[:, 2] = torch.rand(4 * N_side)
        return pts.numpy()

    def data_loss(self, x, y, ujx, ujy, ujt, ujxx, ujyy, ajx, ajy):
        """
        Calculate the data loss.

        :param torch.Tensor x: Input coordinates.
        :param torch.Tensor y: Output values.
        :return: Data loss value.
        :rtype: torch.Tensor
        """
        u, a = y[:, 0:1], y[:, 1:2]
        ur = self.u_ref(x)
        return losses.l2_loss(
            u,
            (ur + np.random.normal(loc=0, scale=0.1, size=ur.shape)).to(u.device)
        )

    def bd_loss(self, x, y, ujx, ujy, ujt, ujxx, ujyy, ajx, ajy):
        """
        Calculate the boundary condition loss.

        :param torch.Tensor x: Input coordinates.
        :param torch.Tensor y: Output values.
        :return: Boundary loss value.
        :rtype: torch.Tensor
        """
        u, a = y[:, 0:1], y[:, 1:2]
        return losses.l2_loss(a, self.a_ref(x).to(a.device))

    def exact_solution(self, x, batch_size):
        """
        Calculate the exact solution.

        :param torch.Tensor x: Input coordinates.
        :param tuple batch_size: Batch size.
        :return: Tuple containing exact solution.
        :rtype: tuple
        """
        return (
            torch.tensor(self.a_ref(x), dtype=torch.float32).to(x.device),
        ) * 8
