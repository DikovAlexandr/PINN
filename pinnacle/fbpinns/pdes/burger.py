"""
Module for Burger's equation problems.
"""

import sys
import pickle
import os
import numpy as np
import scipy
from scipy.interpolate import griddata
import torch

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


class CoupledBurgers(_Problem):
    """
    Solves the coupled Burgers equation in 2D.
    """

    @property
    def name(self):
        """
        Returns the name of the problem.

        :return: Name of the problem.
        :rtype: str
        """
        return "CoupledBurgers"

    def __init__(self):
        """
        Initialize the CoupledBurgers problem.
        """
        self.d = (3, 2)  # same with GrayScott
        self.load_ref_data("burgers2d", timepde=(0, 1))  # is not on grid
        self.nu = 0.001
        
        coef_path = "../ref/burgers2d_coef.dat"
        if os.path.exists(coef_path):
            self.ic_coefs = np.loadtxt(coef_path)
        else:
            print(
                f"Warning: {coef_path} not found. "
                "Using deterministic synthetic coefficients (no randomness)."
            )
            # L=4 => 2*L+1 = 9.
            # Shape logic in ic_func:
            # A: 2 * 9*9 = 162
            # B: 2 * 9*9 = 162
            # C: 2
            # Total = 326
            L = 4
            size = 2 * (2 * L + 1)**2 + 2 * (2 * L + 1)**2 + 2
            k = np.arange(size, dtype=np.float64)
            # Deterministic, bounded coefficients (no RNG; stable across runs)
            self.ic_coefs = (
                0.7 * np.sin(0.173 * k + 0.11)
                + 0.3 * np.cos(0.319 * k - 0.07)
            ).astype(np.float32)
            
        self.num_js = 10

    def ic_func(self, x, component):
        """
        Calculate the initial condition function.

        :param x: Input coordinates.
        :type x: np.ndarray
        :param int component: Component index.
        :return: Initial condition values.
        :rtype: np.ndarray
        """
        L, T, M = 4, 1, 1
        A = self.ic_coefs[:2 * (2 * L + 1)**2].reshape(2, 2 * L + 1, 2 * L + 1)
        B = self.ic_coefs[2 * (2 * L + 1)**2: 4 * (2 * L + 1)**2].reshape(
            2, 2 * L + 1, 2 * L + 1
        )
        C = self.ic_coefs[4 * (2 * L + 1)**2:]

        w = np.zeros((x.shape[0], 1))
        for i in range(-L, L + 1):
            for j in range(-L, L + 1):
                term1 = A[component][i][j] * np.sin(
                    2 * np.pi * (i * x[:, 0:1] + j * x[:, 1:2])
                )
                term2 = B[component][i][j] * np.cos(
                    2 * np.pi * (i * x[:, 0:1] + j * x[:, 1:2])
                )
                w += term1 + term2

        return 2 * w / M + C[component]

    def get_gradients(self, x, y):
        """
        Calculate gradients for the problem.

        :param torch.Tensor x: Input coordinates.
        :param torch.Tensor y: Output values.
        :return: Tuple containing y and its gradients.
        :rtype: tuple
        """
        uj = torch.autograd.grad(
            y[:, 0], x, torch.ones_like(y[:, 0]), create_graph=True
        )[0]
        ujx, ujy, ujt = uj[:, 0:1], uj[:, 1:2], uj[:, 2:3]
        
        vj = torch.autograd.grad(
            y[:, 1], x, torch.ones_like(y[:, 1]), create_graph=True
        )[0]
        vjx, vjy, vjt = vj[:, 0:1], vj[:, 1:2], vj[:, 2:3]

        ujxx = torch.autograd.grad(
            ujx, x, torch.ones_like(ujx), create_graph=True
        )[0][:, 0:1]
        ujyy = torch.autograd.grad(
            ujy, x, torch.ones_like(ujy), create_graph=True
        )[0][:, 1:2]
        
        vjxx = torch.autograd.grad(
            vjx, x, torch.ones_like(vjx), create_graph=True
        )[0][:, 0:1]
        vjyy = torch.autograd.grad(
            vjy, x, torch.ones_like(vjy), create_graph=True
        )[0][:, 1:2]
        
        return y, ujx, ujy, ujt, vjx, vjy, vjt, ujxx, ujyy, vjxx, vjyy

    def physics_loss(
        self, x, y, ujx, ujy, ujt, vjx, vjy, vjt, ujxx, ujyy, vjxx, vjyy
    ):
        """
        Calculate the physics loss.

        :param torch.Tensor x: Input coordinates.
        :param torch.Tensor y: Output values.
        :param torch.Tensor ujx: Gradient of u with respect to x.
        :param torch.Tensor ujy: Gradient of u with respect to y.
        :param torch.Tensor ujt: Gradient of u with respect to t.
        :param torch.Tensor vjx: Gradient of v with respect to x.
        :param torch.Tensor vjy: Gradient of v with respect to y.
        :param torch.Tensor vjt: Gradient of v with respect to t.
        :param torch.Tensor ujxx: Second gradient of u with respect to x.
        :param torch.Tensor ujyy: Second gradient of u with respect to y.
        :param torch.Tensor vjxx: Second gradient of v with respect to x.
        :param torch.Tensor vjyy: Second gradient of v with respect to y.
        :return: Physics loss value.
        :rtype: torch.Tensor
        """
        u, v = y[:, 0:1], y[:, 0:2]
        physics_u = ujt + u * ujx + v * ujy - self.nu * (ujxx + ujyy)
        physics_v = vjt + u * vjx + v * vjy - self.nu * (vjxx + vjyy)
        physics = torch.concat((physics_u, physics_v), dim=1)
        return losses.l2_loss(physics, 0)

    def sample_bd(self, N_bd):
        """
        Sample boundary points.

        :param int N_bd: Number of boundary points.
        :return: Array of boundary points.
        :rtype: np.ndarray
        """
        L, T = 4, 1

        def mgrid(x1, x2, nx, y1, y2, ny, d3idx, d3val):
            xl, yl = np.linspace(x1, x2, nx), np.linspace(y1, y2, ny)
            xmesh, ymesh = np.meshgrid(*(xl, yl), indexing='ij')
            zmesh = np.ones_like(xmesh, dtype=xmesh.dtype) * d3val
            meshes = [xmesh, ymesh]
            meshes.insert(d3idx, zmesh)
            return np.stack(meshes, axis=-1).reshape(-1, 3)

        ic_x = mgrid(0, L, 50, 0, L, 50, 2, 0)
        bc_yp = mgrid(0, L, 50, 0, T, 20, 1, 0)
        bc_ypL = mgrid(0, L, 50, 0, T, 20, 1, L)
        bc_xp = mgrid(0, L, 50, 0, T, 20, 0, 0)
        bc_xpL = mgrid(0, L, 50, 0, T, 20, 0, L)
        
        # cache ic
        if not hasattr(self, "ic_cached"):
            self.ic_cached = np.concatenate(
                [self.ic_func(ic_x, 0), self.ic_func(ic_x, 1)], axis=1
            )
        return np.concatenate([ic_x, bc_yp, bc_ypL, bc_xp, bc_xpL], axis=0)

    def bd_loss(
        self, x, y, ujx, ujy, ujt, vjx, vjy, vjt, ujxx, ujyy, vjxx, vjyy
    ):
        """
        Calculate the boundary condition loss.

        :param torch.Tensor x: Input coordinates.
        :param torch.Tensor y: Output values.
        :return: Boundary loss value.
        :rtype: torch.Tensor
        """
        ic_err = y[:2500] - torch.tensor(self.ic_cached, device=x.device)
        yp_err = y[2500:3500] - y[3500:4500]
        xp_err = y[4500:5500] - y[5500:6500]
        total_err = torch.concat([ic_err, yp_err, xp_err], dim=0)
        return losses.l2_loss(total_err, 0)


class Burgers1D(_Problem):
    """
    Solves the time-dependent 1D viscous Burgers equation.

    The equation is:
        .. math::
            \\frac{\\partial u}{\\partial t} + u \\frac{\\partial u}{\\partial x}
            = \\nu \\frac{\\partial^2 u}{\\partial x^2}

    The domain is :math:`-1.0 < x < +1.0`, :math:`0 < t`.

    Boundary conditions:
        - Initial condition:
          :math:`u(x,0) = - \\sin(\\pi x)`
        - Dirichlet boundaries:
          :math:`u(-1,t) = u(+1,t) = 0`
    """

    @property
    def name(self):
        """
        Returns the name of the problem.

        :return: Name of the problem.
        :rtype: str
        """
        return "Burgers1D"

    def __init__(self, nu=0.01 / np.pi):
        """
        Initialize the Burgers1D problem.

        :param float nu: Viscosity coefficient.
        """
        # input params
        self.bbox = [-1, 1, 0, 1]
        self.nu = nu

        # dimensionality of x and y
        self.d = (2, 1)
        self.load_ref_data("burgers1d", timepde=(0, 1))
        self.num_js = 3

    def physics_loss(self, x, y, j0, j1, jj0):
        """
        Calculate the physics loss.

        :param torch.Tensor x: Input coordinates.
        :param torch.Tensor y: Output values.
        :param torch.Tensor j0: Gradient with respect to x.
        :param torch.Tensor j1: Gradient with respect to t.
        :param torch.Tensor jj0: Second gradient with respect to x.
        :return: Physics loss value.
        :rtype: torch.Tensor
        """
        # be careful to slice correctly (transposed calculations otherwise (!))
        physics = (j1[:, 0] + y[:, 0] * j0[:, 0]) - (self.nu * jj0[:, 0])
        return losses.l2_loss(physics, 0)

    def get_gradients(self, x, y):
        """
        Calculate gradients for the problem.

        :param torch.Tensor x: Input coordinates.
        :param torch.Tensor y: Output values.
        :return: Tuple containing y and its gradients.
        :rtype: tuple
        """
        j = torch.autograd.grad(
            y, x, torch.ones_like(y), create_graph=True
        )[0]
        j0, j1 = j[:, 0:1], j[:, 1:2]
        jj = torch.autograd.grad(
            j0, x, torch.ones_like(j0), create_graph=True
        )[0]
        jj0 = jj[:, 0:1]

        return y, j0, j1, jj0

    def boundary_condition(self, x, y, j0, j1, jj0, sd):
        """
        Apply boundary conditions using ansatz.
        
        Ansatz:
        :math:`u = \\tanh((x+1)/sd)\\tanh((x-1)/sd)\\tanh((y-0)/sd)NN - \\sin(\\pi x)`

        :param torch.Tensor x: Input coordinates.
        :param torch.Tensor y: Output values from the neural network.
        :param torch.Tensor j0: Gradient with respect to x.
        :param torch.Tensor j1: Gradient with respect to t.
        :param torch.Tensor jj0: Second gradient with respect to x.
        :param float sd: Scale parameter for tanh functions.
        :return: Transformed y and gradients satisfying boundary conditions.
        :rtype: tuple
        """
        # tanh(x-1)tanh(x+1)
        t0, jt0, jjt0 = boundary_conditions.tanhtanh_2(x[:, 0:1], -1, 1, sd)
        # tanh(y)
        t1, jt1 = boundary_conditions.tanh_1(x[:, 1:2], 0, sd)

        sin = -torch.sin(np.pi * x[:, 0:1])
        cos = -np.pi * torch.cos(np.pi * x[:, 0:1])
        sin2 = (np.pi**2) * torch.sin(np.pi * x[:, 0:1])

        y_new = t0 * t1 * y + sin
        j0_new = jt0 * t1 * y + t0 * t1 * j0 + cos
        j1_new = t0 * jt1 * y + t0 * t1 * j1
        jj0_new = jjt0 * t1 * y + 2 * jt0 * t1 * j0 + t0 * t1 * jj0 + sin2

        return y_new, j0_new, j1_new, jj0_new
