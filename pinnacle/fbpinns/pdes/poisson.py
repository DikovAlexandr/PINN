"""
Module for Poisson equation problems.
"""

import pickle
import os
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


class Poisson2D_1(_Problem):
    """
    Solves the time-independent 2D Poisson equation
    :math:`-\\Delta u = f(x, y)`
    where f is a given complicated term of x,y,c,m,k,n

    for :math:`-1.0 \\le x, y \\le +1.0`

    Boundary conditions:
    :math:`u(x,-1) = 0, u(x,1) = 0`
    :math:`u(-1,y) = \\tanh(-k)\\sin(2\\pi ny), u(1,y) = \\tanh(k)\\sin(2\\pi ny)`
    """

    @property
    def name(self):
        """
        Returns the name of the problem.

        :return: Name of the problem.
        :rtype: str
        """
        return "Poisson2D_1"

    def __init__(self, c=0.1, m=0.5, k=0, n=0.5):
        """
        Initialize the Poisson2D_1 problem.

        :param float c: Parameter c.
        :param float m: Parameter m.
        :param float k: Parameter k.
        :param float n: Parameter n.
        """
        self.bbox = [-1, 1, -1, 1]
        self.c = c
        self.m = m
        self.k = k
        self.n = n
        # dimensionality of x and y
        self.d = (2, 1)

    def comp_u(self, x):
        """
        Compute u components.
        u0 = c*sin(2pi mx)+tanh(kx)
        u1 = sin(2pi ny)

        :param torch.Tensor x: Input coordinates.
        :return: Tuple of u0, u1.
        :rtype: tuple
        """
        u0 = self.c * torch.sin(2 * np.pi * self.m * x[:, 0:1]) + \
            torch.tanh(self.k * x[:, 0:1])
        u1 = torch.sin(2 * np.pi * self.n * x[:, 1:2])
        return u0, u1

    def comp_ujj(self, x):
        """
        Compute second derivatives of u components.

        :param torch.Tensor x: Input coordinates.
        :return: Tuple of ujj0, ujj1.
        :rtype: tuple
        """
        ujj0 = -4 * np.pi**2 * self.m**2 * self.c * \
            torch.sin(2 * np.pi * self.m * x[:, 0:1]) - \
            2 * self.k**2 * torch.tanh(self.k * x[:, 0:1]) * \
            (1 - torch.tanh(self.k * x[:, 0:1])**2)
        ujj1 = -4 * np.pi**2 * self.n**2 * \
            torch.sin(2 * np.pi * self.n * x[:, 1:2])
        return ujj0, ujj1

    def physics_loss(self, x, y, j0, j1, jj0, jj1):
        """
        Calculate the physics loss.
        Note: x[:,0] vs x[:,0:1] leads to different results.

        :param torch.Tensor x: Input coordinates.
        :param torch.Tensor y: Output values.
        :param torch.Tensor j0: Gradient w.r.t x.
        :param torch.Tensor j1: Gradient w.r.t y.
        :param torch.Tensor jj0: Second gradient w.r.t x.
        :param torch.Tensor jj1: Second gradient w.r.t y.
        :return: Physics loss value.
        :rtype: torch.Tensor
        """
        u0_prime = self.c * torch.sin(2 * np.pi * self.m * x[:, 0]) + \
            torch.tanh(self.k * x[:, 0])
        u1_prime = torch.sin(2 * np.pi * self.n * x[:, 1])
        ujj0_prime = -4 * np.pi**2 * self.m**2 * self.c * \
            torch.sin(2 * np.pi * self.m * x[:, 0]) - \
            2 * self.k**2 * torch.tanh(self.k * x[:, 0]) * \
            (1 - torch.tanh(self.k * x[:, 0])**2)
        ujj1_prime = -4 * np.pi**2 * self.n**2 * \
            torch.sin(2 * np.pi * self.n * x[:, 1])
        
        physics_prime = jj0[:, 0] + jj1[:, 0] - \
            ujj0_prime * u1_prime - u0_prime * ujj1_prime
        return losses.l2_loss(physics_prime, 0)

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
        jj0 = torch.autograd.grad(
            j0, x, torch.ones_like(j0), create_graph=True
        )[0][:, 0:1]
        jj1 = torch.autograd.grad(
            j1, x, torch.ones_like(j1), create_graph=True
        )[0][:, 1:2]
        return y, j0, j1, jj0, jj1

    def boundary_condition(self, x, y, j0, j1, jj0, jj1, sd):
        """
        Apply boundary conditions using ansatz.
        'x' is x[:,0:1], 'y' is x[:,1:2] , 'u' is y
        Ansatz:
        :math:`u(x,y) = \\tanh(x-1)\\tanh(x+1)\\tanh(y-1)\\tanh(y+1)NN(x,y)`
        :math:`+ \\sin(2\\pi ny)\\tanh(kx)`

        :param torch.Tensor x: Input coordinates.
        :param torch.Tensor y: Output values.
        :param torch.Tensor j0: Gradient w.r.t x.
        :param torch.Tensor j1: Gradient w.r.t y.
        :param torch.Tensor jj0: Second gradient w.r.t x.
        :param torch.Tensor jj1: Second gradient w.r.t y.
        :param float sd: Scale parameter.
        :return: Transformed y and gradients satisfying boundary conditions.
        :rtype: tuple
        """
        t0, jt0, jjt0 = boundary_conditions.tanhtanh_2(x[:, 0:1], -1, 1, sd)
        t1, jt1, jjt1 = boundary_conditions.tanhtanh_2(x[:, 1:2], -1, 1, sd)
        
        tan = torch.tanh(self.k * x[:, 0:1])
        tan_j = self.k * (1 - torch.tanh(self.k * x[:, 0:1])**2)
        tan_jj = -2 * self.k**2 * torch.tanh(self.k * x[:, 0:1]) * \
            (1 - torch.tanh(self.k * x[:, 0:1])**2)
        
        sin = torch.sin(2 * np.pi * self.n * x[:, 1:2])
        sin_j = 2 * np.pi * self.n * torch.cos(2 * np.pi * self.n * x[:, 1:2])
        sin_jj = -4 * np.pi**2 * self.n**2 * \
            torch.sin(2 * np.pi * self.n * x[:, 1:2])
        
        y_new = t0 * t1 * y + tan * sin
        j0_new = jt0 * t1 * y + t0 * t1 * j0 + tan_j * sin
        j1_new = t0 * jt1 * y + t0 * t1 * j1 + tan * sin_j
        jj0_new = t1 * (jjt0 * y + 2 * jt0 * j0 + t0 * jj0) + tan_jj * sin
        jj1_new = t0 * (jjt1 * y + 2 * jt1 * j1 + t1 * jj1) + tan * sin_jj
        
        return y_new, j0_new, j1_new, jj0_new, jj1_new

    def exact_solution(self, x, batch_size):
        """
        Calculate the exact solution.
        u = u0 * u1 = ( c*sin(2pi mx)+tanh(kx) )*sin(2pi ny)

        :param torch.Tensor x: Input coordinates.
        :param tuple batch_size: Batch size.
        :return: Tuple containing exact solution and ones.
        :rtype: tuple
        """
        u0, u1 = self.comp_u(x)
        uj0 = self.c * 2 * np.pi * self.m * torch.cos(
            2 * np.pi * self.m * x[:, 0:1]
        ) + self.k * (1 - torch.tanh(self.k * x[:, 0:1])**2)
        uj1 = 2 * np.pi * self.n * torch.cos(2 * np.pi * self.n * x[:, 1:2])
        ujj0, ujj1 = self.comp_ujj(x)
        return u0 * u1, uj0 * u1, u0 * uj1, ujj0 * u1, u0 * ujj1


class Poisson2D_Hole(_Problem):
    """
    Solves the Poisson-Boltzmann 2d irregular domain.
    """

    @property
    def name(self):
        """
        Returns the name of the problem.

        :return: Name of the problem.
        :rtype: str
        """
        return "Poisson2D_Hole"

    def __init__(self):
        """
        Initialize the Poisson2D_Hole problem.
        """
        self.bbox = [-1, 1, -1, 1]
        self.d = (2, 1)
        self.filterparams = [
            (0.5, 0.5, 0.2),
            (0.4, -0.4, 0.4),
            (-0.2, -0.7, 0.1),
            (-0.6, 0.5, 0.3)
        ]
        self.mu1 = 1
        self.mu2 = 4
        self.k = 8
        self.A = 10
        self.load_ref_data("poisson_boltzmann2d")

    def mask_x(self, x):
        """
        Input a torch.tensor of shape (b,nd), Output a boolean torch.tensor of shape (nd).

        :param torch.Tensor x: Input coordinates.
        :return: Boolean mask.
        :rtype: torch.Tensor
        """
        masks = [None] * len(self.filterparams)
        for i, param in enumerate(self.filterparams):
            masks[i] = torch.sum(
                (x - torch.tensor(param[:2], device=x.device))**2, dim=1
            ) > param[2]**2
        for i in range(1, len(self.filterparams)):
            masks[i] = masks[i] & masks[i - 1]
        return masks[-1]

    def sample_bd(self, N_bd):
        """
        Sample (approximately) N_bd points on the boundary.

        :param int N_bd: Number of boundary points.
        :return: List of boundary points.
        :rtype: list
        """
        pts = []
        N_bd = (N_bd + 7) // 8
        for i in range(N_bd):
            pts.append([-1, -1 + 2 * i / N_bd])
            pts.append([-1 + 2 * i / N_bd, 1])
            pts.append([1, 1 - 2 * i / N_bd])
            pts.append([1 - 2 * i / N_bd, -1])
        for i in range(N_bd):
            theta = 2 * np.pi * i / N_bd
            for param in self.filterparams:
                pts.append(
                    [
                        param[0] + param[2] * np.cos(theta),
                        param[1] + param[2] * np.sin(theta)
                    ]
                )
        return pts

    def physics_loss(self, x, y, j0, j1, jj0, jj1):
        """
        Calculate the physics loss.

        :param torch.Tensor x: Input coordinates.
        :param torch.Tensor y: Output values.
        :param torch.Tensor j0: Gradient w.r.t x.
        :param torch.Tensor j1: Gradient w.r.t y.
        :param torch.Tensor jj0: Second gradient w.r.t x.
        :param torch.Tensor jj1: Second gradient w.r.t y.
        :return: Physics loss value.
        :rtype: torch.Tensor
        """
        f = self.A * (self.mu1**2 + self.mu2**2 + x[:, 0]**2 + x[:, 1]**2) * \
            torch.sin(self.mu1 * np.pi * x[:, 0]) * \
            torch.sin(self.mu2 * np.pi * x[:, 1])
        physics = jj0[:, 0] + jj1[:, 0] - self.k**2 * y[:, 0] + f
        return losses.l2_loss(physics, 0)

    def bd_loss(self, x, y, j0, j1, jj0, jj1):
        """
        Calculate the boundary condition loss.

        :param torch.Tensor x: Input coordinates.
        :param torch.Tensor y: Output values.
        :return: Boundary loss value.
        :rtype: torch.Tensor
        """
        is4sides = np.logical_or(
            np.isclose(np.abs(x[:, 0].detach().cpu()), 1),
            np.isclose(np.abs(x[:, 1].detach().cpu()), 1)
        )
        is4sides = torch.tensor(is4sides, device=x.device)
        bd_true = torch.where(is4sides, 0.2, 1.)
        return losses.l2_loss(y[:, 0], bd_true)

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
        jj0 = torch.autograd.grad(
            j0, x, torch.ones_like(j0), create_graph=True
        )[0][:, 0:1]
        jj1 = torch.autograd.grad(
            j1, x, torch.ones_like(j1), create_graph=True
        )[0][:, 1:2]
        return y, j0, j1, jj0, jj1

    def boundary_condition(self, x, y, j0, j1, jj0, jj1, sd):
        """
        Apply boundary conditions. No ansatz used here.
        """
        return y, j0, j1, jj0, jj1

    def exact_solution(self, x, batch_size):
        """
        Calculate the exact solution.

        :param torch.Tensor x: Input coordinates.
        :param tuple batch_size: Batch size.
        :return: Tuple containing interpolated solution and ones.
        :rtype: tuple
        """
        # return value is used for visualization only, the nan values produced 
        # by griddata (out of convex hull) leads to blanks in the imshow 
        # (this is expected behavior)
        x_lims = [
            (x[:, in_dim].cpu().min(), x[:, in_dim].cpu().max())
            for in_dim in range(self.d[0])
        ]
        x_mesh = [
            np.linspace(lim[0], lim[1], b)
            for lim, b in zip(x_lims, batch_size)
        ]
        grid_x_tup = np.meshgrid(*x_mesh, indexing="ij")
        grid_interp = griddata(
            self.ref_x, self.ref_y, tuple(grid_x_tup), method="cubic"
        )  # should change code if ref_y is multidimensional
        return (
            torch.tensor(grid_interp.reshape(-1, 1), device=x.device),
        ) + (torch.ones((np.prod(batch_size), 1), device=x.device),) * 4


class Poisson2D_Classic(_Problem):
    """
    Solves the Classic :math:`\\Delta u = 0` equation on irregular domain.
    """

    @property
    def name(self):
        """
        Returns the name of the problem.

        :return: Name of the problem.
        :rtype: str
        """
        return "Poisson2D_Classic"

    def __init__(self, xscale=1):
        """
        Initialize the Poisson2D_Classic problem.

        :param float xscale: Scale factor for coordinates.
        """
        self.bbox = [-0.5, 0.5, -0.5, 0.5]
        self.d = (2, 1)
        self.filterparams = [
            (0.3, 0.3, 0.1), (-0.3, 0.3, 0.1), (0.3, -0.3, 0.1), (-0.3, -0.3, 0.1)
        ]
        self.load_ref_data("poisson1_cg_data")
        # scale the input dimensions, this does not change the solution
        self.bbox = [x * xscale for x in self.bbox]
        self.filterparams = [
            tuple(x * xscale for x in tup) for tup in self.filterparams
        ]
        self.ref_x *= xscale  # self.ref_data changes after this operation.
        # do not tranform twice.
        self.xscale = xscale
        self.num_js = 4

    def mask_x(self, x):
        """
        Input a torch.tensor of shape (b,nd), Output a boolean torch.tensor of shape (nd).

        :param torch.Tensor x: Input coordinates.
        :return: Boolean mask.
        :rtype: torch.Tensor
        """
        masks = [None] * len(self.filterparams)
        for i, param in enumerate(self.filterparams):
            masks[i] = torch.sum(
                (x - torch.tensor(param[:2], device=x.device))**2, dim=1
            ) > param[2]**2
        for i in range(1, len(self.filterparams)):
            masks[i] = masks[i] & masks[i - 1]
        return masks[-1]

    def sample_bd(self, N_bd):
        """
        Sample (approximately) N_bd points on the boundary.

        :param int N_bd: Number of boundary points.
        :return: List of boundary points.
        :rtype: list
        """
        pts = []
        N_bd = (N_bd + 7) // 8
        for i in range(N_bd):
            pts.append(
                [-self.xscale / 2, -self.xscale / 2 + self.xscale * i / N_bd]
            )
            pts.append(
                [-self.xscale / 2 + self.xscale * i / N_bd, self.xscale / 2]
            )
            pts.append(
                [self.xscale / 2, self.xscale / 2 - self.xscale * i / N_bd]
            )
            pts.append(
                [self.xscale / 2 - self.xscale * i / N_bd, -self.xscale / 2]
            )
        for i in range(N_bd):
            theta = 2 * np.pi * i / N_bd
            for param in self.filterparams:
                pts.append(
                    [
                        param[0] + param[2] * np.cos(theta),
                        param[1] + param[2] * np.sin(theta)
                    ]
                )
        return pts

    def physics_loss(self, x, y, j0, j1, jj0, jj1):
        """
        Calculate the physics loss.

        :param torch.Tensor x: Input coordinates.
        :param torch.Tensor y: Output values.
        :param torch.Tensor jj0: Second gradient w.r.t x.
        :param torch.Tensor jj1: Second gradient w.r.t y.
        :return: Physics loss value.
        :rtype: torch.Tensor
        """
        physics = jj0[:, 0] + jj1[:, 0]
        return losses.l2_loss(physics, 0)

    def bd_loss(self, x, y, j0, j1, jj0, jj1):
        """
        Calculate the boundary condition loss.

        :param torch.Tensor x: Input coordinates.
        :param torch.Tensor y: Output values.
        :return: Boundary loss value.
        :rtype: torch.Tensor
        """
        is4sides = np.logical_or(
            np.isclose(np.abs(x[:, 0].detach().cpu()), 0.5 * self.xscale),
            np.isclose(np.abs(x[:, 1].detach().cpu()), 0.5 * self.xscale)
        )
        is4sides = torch.tensor(is4sides, device=x.device)
        bd_true = torch.where(is4sides, 1., 0.)
        return losses.l2_loss(y[:, 0], bd_true)

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
        jj0 = torch.autograd.grad(
            j0, x, torch.ones_like(j0), create_graph=True
        )[0][:, 0:1]
        jj1 = torch.autograd.grad(
            j1, x, torch.ones_like(j1), create_graph=True
        )[0][:, 1:2]
        return y, j0, j1, jj0, jj1

    def boundary_condition(self, x, y, j0, j1, jj0, jj1, sd):
        """
        Apply boundary conditions. No ansatz used here.
        """
        return y, j0, j1, jj0, jj1


class Poisson2DManyArea(_Problem):
    """
    Solves the Poisson equation on many areas.
    """

    @property
    def name(self):
        """
        Returns the name of the problem.

        :return: Name of the problem.
        :rtype: str
        """
        return "Poisson2DManyArea"

    def __init__(self, bbox=[-10, 10, -10, 10], split=(5, 5)):
        """
        Initialize the Poisson2DManyArea problem.

        :param list bbox: Bounding box coordinates.
        :param tuple split: Split parameters.
        """
        self.bbox = bbox
        self.d = (2, 1)
        self.mbbox, self.msplit = bbox, split
        
        coef_a_path = "../ref/poisson_a_coef.dat"
        coef_f_path = "../ref/poisson_f_coef.dat"
        
        if os.path.exists(coef_a_path) and os.path.exists(coef_f_path):
            self.a_cof = np.loadtxt(coef_a_path)
            self.f_cof = np.loadtxt(coef_f_path).reshape(5, 5, 2, 2)
        else:
            print(
                "Warning: Coefficients files not found. "
                "Using deterministic synthetic coefficients (no randomness)."
            )
            # bbox size: 20x20. split 5x5.
            # a_cof size: 5x5 (strictly positive).
            # f_cof size: 5x5x2x2.
            i = np.arange(5).reshape(5, 1).astype(np.float64)
            j = np.arange(5).reshape(1, 5).astype(np.float64)

            # Positive diffusion-like coefficient per region.
            a = 1.0 + 0.25 * np.sin(0.9 * i + 0.4 * j) + 0.15 * np.cos(0.7 * i - 0.2 * j)
            self.a_cof = a.astype(np.float32)

            # Deterministic forcing coefficients per region.
            f = np.zeros((5, 5, 2, 2), dtype=np.float64)
            for ii in range(5):
                for jj in range(5):
                    base = 0.2 * np.sin(0.6 * ii + 0.3 * jj) - 0.15 * np.cos(0.5 * ii - 0.4 * jj)
                    f[ii, jj, 0, 0] = base
                    f[ii, jj, 0, 1] = 0.1 * np.sin(base + 0.7)
                    f[ii, jj, 1, 0] = 0.1 * np.cos(base - 0.5)
                    f[ii, jj, 1, 1] = -0.05 * np.sin(base * 1.3)
            self.f_cof = f.astype(np.float32)

        self.load_ref_data("poisson_manyarea")
        # prepare interpn
        self.num_js = 4

    def prepare_a_f(self):
        """
        Prepare coefficient functions a and f.
        """
        bbox, split = self.mbbox, self.msplit
        block_size = np.array([
            (bbox[1] - bbox[0] + 2e-5) / split[0],
            (bbox[3] - bbox[2] + 2e-5) / split[1]
        ])

        def domain(x):
            reduced_x = (x - np.array(bbox[::2]) + 1e-5)
            dom = np.floor(reduced_x / block_size).astype("int32")
            return dom, reduced_x - dom * block_size

        def a(x):
            dom, res = domain(x)
            return self.a_cof[dom[0], dom[1]]

        self.a_vct = np.vectorize(a, signature="(2)->()")

        def f(x):
            dom, res = domain(x)

            def f_fn(coef):
                ans = coef[0, 0]
                for i in range(coef.shape[0]):
                    for j in range(coef.shape[1]):
                        tmp = np.sin(
                            np.pi * np.array((i, j)) * (res / block_size)
                        )
                        ans += coef[i, j] * tmp[0] * tmp[1]
                return ans

            return f_fn(self.f_cof[dom[0], dom[1]])

        self.f_vct = np.vectorize(f, signature="(2)->()")

    @cache_x(maxsize=200)
    def get_coef(self, x):
        """
        Get coefficients at x.

        :param torch.Tensor x: Input coordinates.
        :return: Tuple of coefficients.
        :rtype: tuple
        """
        x_cpu = x.detach().cpu()
        if not hasattr(self, "a_vct"):
            self.prepare_a_f()
        return (
            torch.tensor(self.a_vct(x_cpu), device=x.device),
            torch.tensor(self.f_vct(x_cpu), device=x.device)
        )

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
        jx, jy = j[:, 0:1], j[:, 1:2]
        jjx = torch.autograd.grad(
            jx, x, torch.ones_like(jx), create_graph=True
        )[0][:, 0:1]
        jjy = torch.autograd.grad(
            jy, x, torch.ones_like(jy), create_graph=True
        )[0][:, 1:2]
        return y, jx, jy, jjx, jjy

    def physics_loss(self, x, y, jx, jy, jjx, jjy):
        """
        Calculate the physics loss.

        :param torch.Tensor x: Input coordinates.
        :param torch.Tensor y: Output values.
        :param torch.Tensor jjx: Second gradient w.r.t x.
        :param torch.Tensor jjy: Second gradient w.r.t y.
        :return: Physics loss value.
        :rtype: torch.Tensor
        """
        a, f = self.get_coef(x)
        physics = a * (jjx + jjy) + f
        return losses.l2_loss(physics, 0)

    def sample_bd(self, N_bd):
        """
        Sample boundary points.

        :param int N_bd: Number of boundary points.
        :return: List of boundary points.
        :rtype: list
        """
        pts = []
        N_bd = (N_bd + 3) // 4
        for i in range(N_bd):
            pts.append([-10, -10 + 20 * (i + 0.5) / N_bd])
            pts.append([-10 + 20 * (i + 0.5) / N_bd, 10])
            pts.append([10, 10 - 20 * (i + 0.5) / N_bd])
            pts.append([10 - 20 * (i + 0.5) / N_bd, -10])
        return pts

    def bd_loss(self, x, y, jx, jy, jjx, jjy):
        """
        Calculate the boundary condition loss.

        :param torch.Tensor x: Input coordinates.
        :param torch.Tensor y: Output values.
        :param torch.Tensor jx: Gradient w.r.t x.
        :param torch.Tensor jy: Gradient w.r.t y.
        :return: Boundary loss value.
        :rtype: torch.Tensor
        """
        isl = np.isclose(x[:, 0].detach().cpu(), -10)
        isr = np.isclose(x[:, 0].detach().cpu(), 10)
        isb = np.isclose(x[:, 1].detach().cpu(), -10)
        ist = np.isclose(x[:, 1].detach().cpu(), 10)
        isl, isr, isb, ist = (
            torch.tensor(i, device=x.device) for i in (isl, isr, isb, ist)
        )
        normal_deri = torch.sum(
            torch.stack([
                torch.where(cond, j[:, 0], 0.)
                for cond, j in [
                    (isl, -jx), (isr, jx), (isb, -jy), (ist, jy)
                ]
            ]),
            dim=0
        )
        return losses.l2_loss(normal_deri + y[:, 0], 0)


class Poisson3D(_Problem):
    """
    Solves the 3D Poisson equation.
    """

    @property
    def name(self):
        """
        Returns the name of the problem.

        :return: Name of the problem.
        :rtype: str
        """
        return "Poisson3D"

    def __init__(self):
        """
        Initialize the Poisson3D problem.
        """
        self.d = (3, 1)
        self.bbox = [0, 1, 0, 1, 0, 1]
        self.interface_z = 0.5
        self.circs = [
            (0.4, 0.3, 0.6, 0.2), (0.6, 0.7, 0.6, 0.2),
            (0.2, 0.8, 0.7, 0.1), (0.6, 0.2, 0.3, 0.1)
        ]
        self.A_param = (20, 100)
        self.m_param = (1, 10, 5)
        self.k_param = (8, 10)
        self.mu_param = (1, 1)
        self.load_ref_data("poisson_3d")
        self.num_js = 6

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
        jx, jy, jz = j[:, 0:1], j[:, 1:2], j[:, 2:3]
        jxx = torch.autograd.grad(
            jx, x, torch.ones_like(jx), create_graph=True
        )[0][:, 0:1]
        jyy = torch.autograd.grad(
            jy, x, torch.ones_like(jy), create_graph=True
        )[0][:, 1:2]
        jzz = torch.autograd.grad(
            jz, x, torch.ones_like(jz), create_graph=True
        )[0][:, 2:3]
        return y, jx, jy, jz, jxx, jyy, jzz

    def physics_loss(self, x, y, jx, jy, jz, jxx, jyy, jzz):
        """
        Calculate the physics loss.

        :param torch.Tensor x: Input coordinates.
        :param torch.Tensor y: Output values.
        :param torch.Tensor jxx: Second gradient w.r.t x.
        :param torch.Tensor jyy: Second gradient w.r.t y.
        :param torch.Tensor jzz: Second gradient w.r.t z.
        :return: Physics loss value.
        :rtype: torch.Tensor
        """
        def f(xyz):
            x, y, z = xyz[:, 0:1], xyz[:, 1:2], xyz[:, 2:3]
            xlen2 = x**2 + y**2 + z**2
            part1 = torch.exp(
                torch.sin(self.m_param[0] * x) +
                torch.sin(self.m_param[1] * y) +
                torch.sin(self.m_param[2] * z)
            ) * (xlen2 - 1) / (xlen2 + 1)
            part2 = torch.sin(self.m_param[0] * torch.pi * x) + \
                torch.sin(self.m_param[1] * torch.pi * y) + \
                torch.sin(self.m_param[2] * torch.pi * z)
            return self.A_param[0] * part1 + self.A_param[1] * part2

        mus = torch.where(
            x[:, 2] < self.interface_z, self.mu_param[0], self.mu_param[1]
        ).unsqueeze(dim=-1)
        ks = torch.where(
            x[:, 2] < self.interface_z, self.k_param[0]**2, self.k_param[1]**2
        ).unsqueeze(dim=-1)
        
        physics = -mus * (jxx + jyy + jzz) + ks * y - f(x)
        return losses.l2_loss(physics, 0)

    def mask_x(self, x):
        """
        Create a mask for the input coordinates.

        :param torch.Tensor x: Input coordinates.
        :return: Boolean mask.
        :rtype: torch.Tensor
        """
        x_cpu = x.detach().cpu()
        masks_all = torch.ones_like(x_cpu[:, 0], dtype=torch.bool)
        for circ in self.circs:
            masks_all = masks_all & (
                torch.sum(
                    (x_cpu - torch.tensor(circ[:3]))**2, dim=1
                ) >= circ[3]**2
            )
        return masks_all.to(x.device)

    def sample_bd(self, N_bd):
        """
        Sample boundary points.

        :param int N_bd: Number of boundary points.
        :return: Array of boundary points.
        :rtype: np.ndarray
        """
        nside = int(np.sqrt(N_bd // 6))

        def mgrid(x1, x2, nx, y1, y2, ny, d3idx, d3val):
            xl, yl = np.linspace(x1, x2, nx), np.linspace(y1, y2, ny)
            xmesh, ymesh = np.meshgrid(*(xl, yl), indexing='ij')
            zmesh = np.ones_like(xmesh, dtype=xmesh.dtype) * d3val
            meshes = [xmesh, ymesh]
            meshes.insert(d3idx, zmesh)
            return np.stack(meshes, axis=-1).reshape(-1, 3)

        bc_z0 = mgrid(0, 1, nside, 0, 1, nside, 2, 0)
        bc_z1 = mgrid(0, 1, nside, 0, 1, nside, 2, 1)
        bc_y0 = mgrid(0, 1, nside, 0, 1, nside, 1, 0)
        bc_y1 = mgrid(0, 1, nside, 0, 1, nside, 1, 1)
        bc_x0 = mgrid(0, 1, nside, 0, 1, nside, 0, 0)
        bc_x1 = mgrid(0, 1, nside, 0, 1, nside, 0, 1)
        bd_pts = np.concatenate(
            [bc_z0, bc_z1, bc_y0, bc_y1, bc_x0, bc_x1], axis=0
        )
        return bd_pts

    def bd_loss(self, x, y, jx, jy, jz, jxx, jyy, jzz):
        """
        Calculate the boundary condition loss.

        :param torch.Tensor x: Input coordinates.
        :param torch.Tensor y: Output values.
        :param torch.Tensor jx: Gradient w.r.t x.
        :param torch.Tensor jy: Gradient w.r.t y.
        :param torch.Tensor jz: Gradient w.r.t z.
        :return: Boundary loss value.
        :rtype: torch.Tensor
        """
        x_cpu = x.detach().cpu()
        isx, isy, isz = (
            torch.tensor(
                np.isclose(x_cpu[:, idim], 1) | np.isclose(x_cpu[:, idim], 0)
            )
            for idim in (0, 1, 2)
        )
        xloss, yloss, zloss = (
            torch.where(isd.to(x.device), jd, 0.)
            for isd, jd in zip((isx, isy, isz), (jx, jy, jz))
        )
        return losses.l2_loss(
            torch.concat((xloss, yloss, zloss), dim=1), 0
        )


class PoissonND(_Problem):
    """
    Solves the N-dimensional Poisson equation.
    """

    @property
    def name(self):
        """
        Returns the name of the problem.

        :return: Name of the problem.
        :rtype: str
        """
        return "PoissonND"

    def __init__(self, dim=5):
        """
        Initialize the PoissonND problem.

        :param int dim: Dimensionality.
        """
        self.bbox = [0, 1] * dim
        self.d = (dim, 1)
        self.xdim = dim

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
        jjsum = torch.zeros_like(y)
        for idim in range(self.xdim):
            ji = j[:, idim:idim + 1]
            jjsum += torch.autograd.grad(
                ji, x, torch.ones_like(ji), create_graph=True
            )[0][:, idim:idim + 1]
        return y, jjsum

    def f(self, x):
        """
        Source function f.

        :param torch.Tensor x: Input coordinates.
        :return: Source values.
        :rtype: torch.Tensor
        """
        return torch.sin(torch.pi / 2 * x).sum(axis=1).reshape(-1, 1)

    def physics_loss(self, x, y, jjsum):
        """
        Calculate the physics loss.

        :param torch.Tensor x: Input coordinates.
        :param torch.Tensor y: Output values.
        :param torch.Tensor jjsum: Sum of second gradients.
        :return: Physics loss value.
        :rtype: torch.Tensor
        """
        physics = jjsum + (torch.pi**2) / 4 * self.f(x)
        return losses.l2_loss(physics, 0)

    def exact_solution(self, x, batch_size):
        """
        Calculate the exact solution.

        :param torch.Tensor x: Input coordinates.
        :param tuple batch_size: Batch size.
        :return: Tuple containing exact solution and gradients (partial).
        :rtype: tuple
        """
        y_exact = self.f(x)
        jjsum_exact = -(torch.pi**2) / 4 * self.f(x)
        return y_exact, jjsum_exact

    def sample_bd(self, N_bds):
        """
        Sample boundary points.

        :param int N_bds: Number of boundary points.
        :return: Array of boundary points.
        :rtype: np.ndarray
        """
        def hyperplane(keepdim):
            vardims = [np.linspace(0, 1, 8) for _ in range(self.xdim - 1)]
            varmeshes = [item for item in np.meshgrid(*vardims)]
            keepmesh_0 = np.zeros_like(varmeshes[0], dtype=varmeshes[0].dtype)
            keepmesh_1 = np.ones_like(varmeshes[0], dtype=varmeshes[0].dtype)
            varmeshes.insert(keepdim, keepmesh_0)
            ret_0 = np.stack(varmeshes, axis=-1).reshape(-1, self.xdim)
            varmeshes[keepdim] = keepmesh_1
            ret_1 = np.stack(varmeshes, axis=-1).reshape(-1, self.xdim)
            return np.stack([ret_0, ret_1], axis=0).reshape(-1, self.xdim)

        def hyperplane_random(keepdim):
            ret = torch.rand((N_bds, self.xdim))
            ret[:N_bds // 2, keepdim] = 0.
            ret[N_bds // 2:, keepdim] = 1.
            return ret

        retlist = list()
        for idim in range(self.xdim):
            retlist.append(hyperplane_random(idim))
        return np.stack(retlist, axis=0).reshape(-1, self.xdim)

    def bd_loss(self, x, y, jjsum):
        """
        Calculate the boundary condition loss.

        :param torch.Tensor x: Input coordinates.
        :param torch.Tensor y: Output values.
        :return: Boundary loss value.
        :rtype: torch.Tensor
        """
        return losses.l2_loss(self.f(x), y)
