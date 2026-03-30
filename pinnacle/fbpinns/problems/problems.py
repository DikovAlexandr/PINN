"""
Module for PDE problem definitions.

This module defines a set of PDE problems to solve. Each problem is defined
by a problem class, which must inherit from the :class:`_Problem` base class.
Each problem class must define the NotImplemented methods which compute the
PINN physics loss, the gradients required to evaluate the PINN physics loss,
the hard boundary conditions applied to the ansatz, and the exact solution
(if it exists).

Problem classes are used by :mod:`config.constants` when defining FBPINN / PINN
problems (and subsequently :mod:`main`).
"""

from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import scipy
from scipy.interpolate import griddata
import torch

sys.path.insert(0, "./shared_modules")
from helper import Timer, cache_x

from problems.boundary_conditions import (
    A_1D_1,
    AB_1D_2,
    tanh_1,
    tanh_2,
    tanhtanh_2,
)

# Import losses - handle both relative and absolute imports
try:
    from training import losses
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from training import losses



class _Problem:
    """
    Base problem class to be inherited by different problem classes.

    Each problem class must implement:
    - :meth:`name`: Property returning a name string (used for labelling automated training runs)
    - :meth:`__init__`: Initialize the problem with parameters
    - :meth:`physics_loss`: Compute the PINN physics loss to train the NN
    - :meth:`get_gradients`: Return the gradients yj required for this problem
    - :meth:`boundary_condition`: Define the hard boundary condition to be applied to the NN ansatz
    - :meth:`exact_solution`: Define exact solution if it exists (default: use ref solution to interpolate)
    """

    @property
    def name(self):
        """
        Return a name string for this problem.

        Used for labelling automated training runs.

        :return: Problem name.
        :rtype: str
        """
        raise NotImplementedError

    def __init__(self):
        """Initialize the problem."""
        raise NotImplementedError

    def physics_loss(self, x, *yj):
        """
        Define the PINN physics loss to train the NN.

        :param x: Input coordinates.
        :type x: torch.Tensor
        :param yj: Gradients and values required for the physics loss.
        :type yj: tuple[torch.Tensor, ...]
        :return: Physics loss value.
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    def get_gradients(self, x, y):
        """
        Return the gradients yj required for this problem.

        :param x: Input coordinates.
        :type x: torch.Tensor
        :param y: Output values.
        :type y: torch.Tensor
        :return: Tuple containing y and its gradients.
        :rtype: tuple
        """
        raise NotImplementedError

    def boundary_condition(self, x, *yj_and_sd):
        """
        Define the hard boundary condition to be applied to the NN ansatz.

        Default: does nothing (returns yj without sd).

        :param x: Input coordinates.
        :type x: torch.Tensor
        :param yj_and_sd: Gradients, values, and scale parameter.
        :type yj_and_sd: tuple
        :return: Transformed yj satisfying boundary conditions.
        :rtype: tuple
        """
        return yj_and_sd[:-1]

    def exact_solution(self, x, batch_size):
        """
        Define exact solution if it exists.

        Default: use ref solution to interpolate.

        :param x: Input coordinates.
        :type x: torch.Tensor
        :param batch_size: Batch size tuple.
        :type batch_size: tuple[int, ...]
        :return: Tuple containing exact solution values and gradients.
        :rtype: tuple
        """
        
        # Check if reference data is available
        has_ref = hasattr(self, "ref_in_coords") or (
            hasattr(self, "ref_x") and hasattr(self, "ref_y")
        )
        if not has_ref:
            # Return dummy exact solution (zeros)
            # We assume output dimension matches self.d[1] if available, otherwise 1
            out_dim = self.d[1] if hasattr(self, "d") and len(self.d) > 1 else 1
            vals = torch.zeros((np.prod(batch_size), out_dim), device=x.device)
            num_js = getattr(self, "num_js", 1)  # Default to 1 if not set
            return (vals,) + (
                torch.ones((np.prod(batch_size), 1), device=x.device),
            ) * num_js

        if hasattr(self, "ref_in_coords"):  # reference solution is on regular grid
            vals = list()
            ref_pts = tuple(xs.astype(np.float64) for xs in self.ref_in_coords)
            for ref_val in self.ref_values:
                intp_result = scipy.interpolate.interpn(
                    ref_pts,
                    ref_val.astype(np.float64),
                    x.cpu().numpy().astype(np.float64),
                )
                vals.append(torch.tensor(intp_result.astype(np.float32), device=x.device))
            vals = torch.stack(vals, dim=-1)  # (-1, out_dims)
        else:
            param_str = self.param2str() if hasattr(self, "param2str") else ""
            cache_dir = "interpolate_cache"
            os.makedirs(cache_dir, exist_ok=True)
            cache_str = (
                cache_dir
                + "/"
                + self.name
                + "_"
                + param_str
                + "_"
                + "_".join([str(n) for n in batch_size])
                + ".pkl"
            )
            try:
                with open(cache_str, "rb") as f:
                    vals = pickle.load(f)
            except FileNotFoundError:
                with Timer("interpolate"):
                    vals = list()
                    for i_od in range(self.d[1]):
                        intp_result = scipy.interpolate.griddata(
                            self.ref_x,
                            self.ref_y[:, i_od],
                            x.cpu().numpy(),
                            fill_value=0,
                        )
                        vals.append(
                            torch.tensor(intp_result.astype(np.float32), device=x.device)
                        )
                    vals = torch.stack(vals, dim=-1)
                with open(cache_str, "wb") as f:
                    pickle.dump(vals, f)
        num_js = getattr(self, "num_js", 1)  # Default to 1 if not set
        return (vals.to(x.device),) + (
            torch.ones((np.prod(batch_size), 1), device=x.device),
        ) * num_js

    def load_ref_data(self, name, timepde=None):
        """
        Load reference data from file.

        If PDE is a time-dependent PDE, provide timepde=(t_start, t_end).

        :param name: Name of the reference data file (without .dat extension).
        :type name: str
        :param timepde: Optional tuple (t_start, t_end) for time-dependent PDEs.
        :type timepde: tuple[float, float] | None
        """
        datapath = "../ref/" + name + ".dat"
        if not os.path.exists(datapath):
            print(
                f"Warning: Reference data file {datapath} not found. "
                "Proceeding without reference data."
            )
            # Create dummy ref_data so that downstream code in load_ref_data
            # doesn't crash immediately. We need self.d to be defined before
            # calling this.
            if not hasattr(self, "d"):
                return

            # Construct dummy ref_data to satisfy shape requirements if possible.
            # But usually we just want to skip ref_data dependent logic.
            # For now, let's just return and let exact_solution handle the
            # missing attributes.
            return

        with open(datapath, "r", encoding="utf-8") as f:
            self.ref_data = np.loadtxt(f, comments="%").astype(np.float32)

        if timepde is not None:  # transform ref_data
            time_starts, time_ends = timepde
            data = self.ref_data
            num_tsample = (data.shape[1] - (self.d[0] - 1)) // self.d[1]
            assert num_tsample * self.d[1] == data.shape[1] - (self.d[0] - 1)
            t = np.linspace(time_starts, time_ends, num_tsample)
            t, x0 = np.meshgrid(
                t, data[:, 0]
            )  # add the first input dimension that is not time
            list_x = [
                x0.reshape(-1)
            ]  # x0.reshape(-1) gives [e1,e1,...,e1, e2,e2,...,e2, ...]
            # each element repeats num_tsample times (adjacent)
            for i in range(1, self.d[0] - 1):  # add other input dimensions that is not time
                list_x.append(
                    np.stack([data[:, i] for _ in range(num_tsample)]).T.reshape(-1)
                )  # each element repeats num_tsample times (adjacent)
            list_x.append(
                t.reshape(-1)
            )  # t is the last input dimension. (Other) input dimension order
            # should be in accordance with .dat file
            for i in range(self.d[1]):
                list_x.append(data[:, self.d[0] - 1 + i :: self.d[1]].reshape(-1))
            self.ref_data = np.stack(list_x).T.astype(np.float32)

        self.ref_x = self.ref_data[:, : self.d[0]]
        self.ref_y = self.ref_data[:, self.d[0] :]

    def downsample_ref_data(self, factor):
        """
        Downsample reference data by a given factor.

        :param factor: Downsampling factor.
        :type factor: int
        """
        if not hasattr(self, "ref_data"):
            return
        ndat = self.ref_data.shape[0]
        ia = np.random.choice(np.arange(ndat), ndat // factor)
        self.ref_data = self.ref_data[ia, :]
        self.ref_x = self.ref_x[ia, :]
        self.ref_y = self.ref_y[ia, :]


# 1D problems

class Cos1D_1(_Problem):
    """
    Solves the 1D ODE: du/dx = cos(wx).

    Boundary conditions:
        u(0) = A

    :param w: Frequency parameter.
    :type w: float
    :param A: Boundary value at x=0 (default: 0).
    :type A: float
    """

    @property
    def name(self):
        """Return problem name."""
        return f"Cos1D_1_w{self.w}"

    def __init__(self, w, A=0):
        # Input params
        self.w = w

        # Boundary params
        self.A = A

        # Dimensionality of x and y
        self.d = (1, 1)
        self.num_js = 1
        self.num_js = 1

    def physics_loss(self, x, y, j):
        """
        Calculate the physics loss.

        :param x: Input coordinates.
        :type x: torch.Tensor
        :param y: Output values.
        :type y: torch.Tensor
        :param j: Gradient with respect to x.
        :type j: torch.Tensor
        :return: Physics loss value.
        :rtype: torch.Tensor
        """
        physics = j - torch.cos(self.w * x)
        return losses.l2_loss(physics, 0)

    def get_gradients(self, x, y):
        """
        Calculate gradients for the problem.

        :param x: Input coordinates.
        :type x: torch.Tensor
        :param y: Output values.
        :type y: torch.Tensor
        :return: Tuple containing y and its gradient.
        :rtype: tuple
        """
        j = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
        return y, j

    def boundary_condition(self, x, y, j, sd):
        """
        Apply boundary conditions using ansatz.

        :param x: Input coordinates.
        :type x: torch.Tensor
        :param y: Output values from the neural network.
        :type y: torch.Tensor
        :param j: Gradient with respect to x.
        :type j: torch.Tensor
        :param sd: Scale parameter.
        :type sd: float
        :return: Transformed y and gradient satisfying boundary conditions.
        :rtype: tuple
        """
        y, j = A_1D_1(x, y, j, self.A, 0, sd)
        return y, j

    def exact_solution(self, x, batch_size):
        """
        Calculate exact solution.

        :param x: Input coordinates.
        :type x: torch.Tensor
        :param batch_size: Batch size tuple.
        :type batch_size: tuple[int, ...]
        :return: Tuple containing exact solution values and gradients.
        :rtype: tuple
        """
        Ap = self.A
        y = (1 / self.w) * torch.sin(self.w * x) + Ap
        j = torch.cos(self.w * x)
        return y, j
    

class Cos_multi1D_1(_Problem):
    """
    Solves the 1D ODE: du/dx = w1*cos(w1*x) + w2*cos(w2*x).

    Boundary conditions:
        u(0) = A

    :param w1: First frequency parameter.
    :type w1: float
    :param w2: Second frequency parameter.
    :type w2: float
    :param A: Boundary value at x=0 (default: 0).
    :type A: float
    """

    @property
    def name(self):
        """Return problem name."""
        return f"Cos_multi1D_1_w{self.w1}w{self.w2}"

    def __init__(self, w1, w2, A=0):
        # Input params
        self.w1, self.w2 = w1, w2

        # Boundary params
        self.A = A

        # Dimensionality of x and y
        self.d = (1, 1)
        self.num_js = 1

    def physics_loss(self, x, y, j):
        """
        Calculate the physics loss.

        :param x: Input coordinates.
        :type x: torch.Tensor
        :param y: Output values.
        :type y: torch.Tensor
        :param j: Gradient with respect to x.
        :type j: torch.Tensor
        :return: Physics loss value.
        :rtype: torch.Tensor
        """
        physics = j - (self.w1 * torch.cos(self.w1 * x) + self.w2 * torch.cos(self.w2 * x))
        return losses.l2_loss(physics, 0)

    def get_gradients(self, x, y):
        """
        Calculate gradients for the problem.

        :param x: Input coordinates.
        :type x: torch.Tensor
        :param y: Output values.
        :type y: torch.Tensor
        :return: Tuple containing y and its gradient.
        :rtype: tuple
        """
        j = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
        return y, j
    
    def boundary_condition(self, x, y, j, sd):
        """
        Apply boundary conditions using ansatz.

        :param x: Input coordinates.
        :type x: torch.Tensor
        :param y: Output values from the neural network.
        :type y: torch.Tensor
        :param j: Gradient with respect to x.
        :type j: torch.Tensor
        :param sd: Scale parameter.
        :type sd: float
        :return: Transformed y and gradient satisfying boundary conditions.
        :rtype: tuple
        """
        y, j = A_1D_1(x, y, j, self.A, 0, sd)
        return y, j

    def exact_solution(self, x, batch_size):
        """
        Calculate exact solution.

        :param x: Input coordinates.
        :type x: torch.Tensor
        :param batch_size: Batch size tuple.
        :type batch_size: tuple[int, ...]
        :return: Tuple containing exact solution values and gradients.
        :rtype: tuple
        """
        Ap = self.A
        y = torch.sin(self.w1 * x) + torch.sin(self.w2 * x) + Ap
        j = self.w1 * torch.cos(self.w1 * x) + self.w2 * torch.cos(self.w2 * x)
        return y, j

    
class Sin1D_2(_Problem):
    """
    Solves the 1D ODE: d^2u/dx^2 = sin(w*x).

    Boundary conditions:
        u(0) = A
        u'(0) = B

    :param w: Frequency parameter.
    :type w: float
    :param A: Boundary value at x=0 (default: 0).
    :type A: float
    :param B: Boundary value for derivative at x=0 (default: 0).
    :type B: float
    """

    @property
    def name(self):
        """Return problem name."""
        return f"Sin1D_2_w{self.w}"

    def __init__(self, w, A=0, B=0):
        # Input params
        self.w = w

        # Boundary params
        self.A = A
        self.B = B

        # Dimensionality of x and y
        self.d = (1, 1)
        self.num_js = 2

    def physics_loss(self, x, y, j, jj):
        """
        Calculate the physics loss.

        :param x: Input coordinates.
        :type x: torch.Tensor
        :param y: Output values.
        :type y: torch.Tensor
        :param j: First gradient with respect to x.
        :type j: torch.Tensor
        :param jj: Second gradient with respect to x.
        :type jj: torch.Tensor
        :return: Physics loss value.
        :rtype: torch.Tensor
        """
        physics = jj - torch.sin(self.w * x)
        return losses.l2_loss(physics, 0)

    def get_gradients(self, x, y):
        """
        Calculate gradients for the problem.

        :param x: Input coordinates.
        :type x: torch.Tensor
        :param y: Output values.
        :type y: torch.Tensor
        :return: Tuple containing y and its gradients.
        :rtype: tuple
        """
        j = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
        jj = torch.autograd.grad(j, x, torch.ones_like(j), create_graph=True)[0]
        return y, j, jj

    def boundary_condition(self, x, y, j, jj, sd):
        """
        Apply boundary conditions using ansatz.

        :param x: Input coordinates.
        :type x: torch.Tensor
        :param y: Output values from the neural network.
        :type y: torch.Tensor
        :param j: First gradient with respect to x.
        :type j: torch.Tensor
        :param jj: Second gradient with respect to x.
        :type jj: torch.Tensor
        :param sd: Scale parameter.
        :type sd: float
        :return: Transformed y and gradients satisfying boundary conditions.
        :rtype: tuple
        """
        y, j, jj = AB_1D_2(x, y, j, jj, self.A, self.B, 0, sd)
        return y, j, jj

    def exact_solution(self, x, batch_size):
        """
        Calculate exact solution.

        :param x: Input coordinates.
        :type x: torch.Tensor
        :param batch_size: Batch size tuple.
        :type batch_size: tuple[int, ...]
        :return: Tuple containing exact solution values and gradients.
        :rtype: tuple
        """
        Ap = self.A
        Bp = self.B + (1 / self.w)
        y = -(1 / self.w**2) * torch.sin(self.w * x) + Bp * x + Ap
        j = -(1 / self.w) * torch.cos(self.w * x) + Bp
        jj = torch.sin(self.w * x)
        return y, j, jj


# 2D problems

class Cos_Cos2D_1(_Problem):
    """
    Solves the 2D PDE: du/dx + du/dy = cos(w*x) + cos(w*y).

    Not an ODE, because u is multivariate.

    Boundary conditions:
        u(0, y) = (1/w) * sin(w*y) + A

    Note: The solution is unique. Consider two solutions u1, u2, let v = u1 - u2.
    Then dv/dx + dv/dy = 0, v(0, y) = 0. Consider f(t) = v(0+t, y+t),
    f'(t) = dv/dx + dv/dy = 0, f(0) = 0, thus for each t, f(t) = 0.
    Thus for each x, y, v(x, y) = 0, thus u1 equals u2.

    :param w: Frequency parameter.
    :type w: float
    :param A: Boundary value constant (default: 0).
    :type A: float
    """

    @property
    def name(self):
        """Return problem name."""
        return f"Cos_Cos2D_1_w{self.w}"

    def __init__(self, w, A=0):
        # Input params
        self.w = w

        # Boundary params
        self.A = A

        # Dimensionality of x and y
        self.d = (2, 1)
        self.num_js = 2

    def physics_loss(self, x, y, j0, j1):
        """
        Calculate the physics loss.

        :param x: Input coordinates.
        :type x: torch.Tensor
        :param y: Output values.
        :type y: torch.Tensor
        :param j0: Gradient with respect to x (first dimension).
        :type j0: torch.Tensor
        :param j1: Gradient with respect to y (second dimension).
        :type j1: torch.Tensor
        :return: Physics loss value.
        :rtype: torch.Tensor
        """
        # Be careful to slice correctly (transposed calculations otherwise (!))
        physics = (j0[:, 0] + j1[:, 0]) - (
            torch.cos(self.w * x[:, 0]) + torch.cos(self.w * x[:, 1])
        )
        return losses.l2_loss(physics, 0)

    def get_gradients(self, x, y):
        """
        Calculate gradients for the problem.

        :param x: Input coordinates.
        :type x: torch.Tensor
        :param y: Output values.
        :type y: torch.Tensor
        :return: Tuple containing y and its gradients.
        :rtype: tuple
        """
        j = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
        j0, j1 = j[:, 0:1], j[:, 1:2]

        return y, j0, j1

    def boundary_condition(self, x, y, j0, j1, sd):
        """
        Apply boundary conditions using ansatz.

        Apply u = tanh((x-0)/sd) * NN + A + (1/w) * sin(w*y) ansatz.

        :param x: Input coordinates.
        :type x: torch.Tensor
        :param y: Output values from the neural network.
        :type y: torch.Tensor
        :param j0: Gradient with respect to x (first dimension).
        :type j0: torch.Tensor
        :param j1: Gradient with respect to y (second dimension).
        :type j1: torch.Tensor
        :param sd: Scale parameter.
        :type sd: float
        :return: Transformed y and gradients satisfying boundary conditions.
        :rtype: tuple
        """
        A, w = self.A, self.w

        t0, jt0 = tanh_1(x[:, 0:1], 0, sd)  # tanh(w*x_1), d/dx_1 tanh(w*x_1)

        sin = (1 / w) * torch.sin(w * x[:, 1:2])
        cos = torch.cos(w * x[:, 1:2])

        y_new = t0 * y + A + sin
        j0_new = jt0 * y + t0 * j0  # du/dx_1
        j1_new = t0 * j1 + cos  # du/dx_2

        return y_new, j0_new, j1_new

    def exact_solution(self, x, batch_size):
        """
        Calculate exact solution.

        :param x: Input coordinates.
        :type x: torch.Tensor
        :param batch_size: Batch size tuple.
        :type batch_size: tuple[int, ...]
        :return: Tuple containing exact solution values and gradients.
        :rtype: tuple
        """
        Ap = self.A
        y = (
            (1 / self.w) * torch.sin(self.w * x[:, 0:1])
            + (1 / self.w) * torch.sin(self.w * x[:, 1:2])
            + Ap
        )
        j0 = torch.cos(self.w * x[:, 0:1])
        j1 = torch.cos(self.w * x[:, 1:2])
        return y, j0, j1
    
    
class Sin2D_1(_Problem):
    """
    Solves the 2D PDE: du/dx + du/dy = -sin(w*(x+y)).

    Boundary conditions:
        u(x, x) = (1/w) * cos^2(w*x) + A

    :param w: Frequency parameter.
    :type w: float
    :param A: Boundary value constant (default: 0).
    :type A: float
    """

    @property
    def name(self):
        """Return problem name."""
        return f"Sin2D_1_w{self.w}"

    def __init__(self, w, A=0):
        # Input params
        self.w = w

        # Boundary params
        self.A = A

        # Dimensionality of x and y
        self.d = (2, 1)
        self.num_js = 2

    def physics_loss(self, x, y, j0, j1):
        """
        Calculate the physics loss.

        :param x: Input coordinates.
        :type x: torch.Tensor
        :param y: Output values.
        :type y: torch.Tensor
        :param j0: Gradient with respect to x (first dimension).
        :type j0: torch.Tensor
        :param j1: Gradient with respect to y (second dimension).
        :type j1: torch.Tensor
        :return: Physics loss value.
        :rtype: torch.Tensor
        """
        # Be careful to slice correctly (transposed calculations otherwise (!))
        physics = (j0[:, 0] + j1[:, 0]) + torch.sin(self.w * (x[:, 0] + x[:, 1]))
        return losses.l2_loss(physics, 0)

    def get_gradients(self, x, y):
        """
        Calculate gradients for the problem.

        :param x: Input coordinates.
        :type x: torch.Tensor
        :param y: Output values.
        :type y: torch.Tensor
        :return: Tuple containing y and its gradients.
        :rtype: tuple
        """
        j = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
        j0, j1 = j[:, 0:1], j[:, 1:2]

        return y, j0, j1

    def boundary_condition(self, x, y, j0, j1, sd):
        """
        Apply boundary conditions using ansatz.

        Apply u = tanh((x+y)/sd) * NN + A + (1/w) * cos^2(w*x) ansatz.

        :param x: Input coordinates.
        :type x: torch.Tensor
        :param y: Output values from the neural network.
        :type y: torch.Tensor
        :param j0: Gradient with respect to x (first dimension).
        :type j0: torch.Tensor
        :param j1: Gradient with respect to y (second dimension).
        :type j1: torch.Tensor
        :param sd: Scale parameter.
        :type sd: float
        :return: Transformed y and gradients satisfying boundary conditions.
        :rtype: tuple
        """
        A, w = self.A, self.w

        t, jt = tanh_1(x[:, 0:1] + x[:, 1:2], 0, sd)

        cos2 = (1 / w) * torch.cos(w * x[:, 0:1]) ** 2
        sin2 = -2 * torch.sin(w * x[:, 0:1]) * torch.cos(w * x[:, 0:1])

        y_new = t * y + A + cos2
        j0_new = jt * y + t * j0 + sin2
        j1_new = jt * y + t * j1

        return y_new, j0_new, j1_new

    def exact_solution(self, x, batch_size):
        """
        Calculate exact solution.

        :param x: Input coordinates.
        :type x: torch.Tensor
        :param batch_size: Batch size tuple.
        :type batch_size: tuple[int, ...]
        :return: Tuple containing exact solution values and gradients.
        :rtype: tuple
        """
        Ap = self.A
        y = (
            (1 / self.w)
            * torch.cos(self.w * x[:, 0:1])
            * torch.cos(self.w * x[:, 1:2])
            + Ap
        )
        j0 = -torch.sin(self.w * x[:, 0:1]) * torch.cos(self.w * x[:, 1:2])
        j1 = -torch.cos(self.w * x[:, 0:1]) * torch.sin(self.w * x[:, 1:2])
        return y, j0, j1


# 2x2 D problems

class Sin2x2D(_Problem):
    """
    Solves the 2x2D Problem.

    Domain: [0, 2*pi] x [0, 2*pi]

    Exact solution:
        u(x, y) = sin(x)
        v(x, y) = sin(y)

    Equations:
        du/dx = cos(x)
        dv/dy = cos(y)

    Boundary conditions:
        u(0, y) = 0
        v(x, 0) = 0

    Ansatz:
        u = tanh((x-mu)/sd) * NN(x, y)
        v = tanh((y-mu)/sd) * NN(x, y)
    """

    @property
    def name(self):
        """Return problem name."""
        return "Sin2x2D"

    def __init__(self):
        # Dimensionality of x and y
        self.d = (2, 2)
        self.num_js = 2

    def physics_loss(self, x, y, ju_0, jv_1):
        """
        Calculate the physics loss.

        :param x: Input coordinates.
        :type x: torch.Tensor
        :param y: Output values.
        :type y: torch.Tensor
        :param ju_0: Gradient of u with respect to x.
        :type ju_0: torch.Tensor
        :param jv_1: Gradient of v with respect to y.
        :type jv_1: torch.Tensor
        :return: Physics loss value.
        :rtype: torch.Tensor
        """
        physics_one = ju_0 - torch.cos(x[:, 0:1])
        physics_two = jv_1 - torch.cos(x[:, 1:2])
        physics = torch.concat((physics_one, physics_two), dim=1)
        return losses.l2_loss(physics, 0)

    def get_gradients(self, x, y):
        """
        Calculate gradients for the problem.

        :param x: Input coordinates.
        :type x: torch.Tensor
        :param y: Output values.
        :type y: torch.Tensor
        :return: Tuple containing y and its gradients.
        :rtype: tuple
        """
        j_u = torch.autograd.grad(
            y[:, 0], x, torch.ones_like(y[:, 0]), create_graph=True
        )[0]
        j_v = torch.autograd.grad(
            y[:, 1], x, torch.ones_like(y[:, 1]), create_graph=True
        )[0]
        ju_0 = j_u[:, 0:1]
        jv_1 = j_v[:, 1:2]
        return y, ju_0, jv_1

    def boundary_condition(self, x, y, ju_0, jv_1, sd):
        """
        Apply boundary conditions using ansatz.

        :param x: Input coordinates.
        :type x: torch.Tensor
        :param y: Output values from the neural network.
        :type y: torch.Tensor
        :param ju_0: Gradient of u with respect to x.
        :type ju_0: torch.Tensor
        :param jv_1: Gradient of v with respect to y.
        :type jv_1: torch.Tensor
        :param sd: Scale parameter.
        :type sd: float
        :return: Transformed y and gradients satisfying boundary conditions.
        :rtype: tuple
        """
        u, v = y[:, 0:1], y[:, 1:2]
        tu, jtu0 = tanh_1(x[:, 0:1], 0, sd)
        tv, jtv1 = tanh_1(x[:, 1:2], 0, sd)
        u_new = tu * u
        v_new = tv * v
        ju_0_new = jtu0 * u + tu * ju_0
        jv_1_new = jtv1 * v + tv * jv_1
        return torch.concat((u_new, v_new), dim=1), ju_0_new, jv_1_new

    def exact_solution(self, x, batch_size):
        """
        Calculate exact solution.

        :param x: Input coordinates.
        :type x: torch.Tensor
        :param batch_size: Batch size tuple.
        :type batch_size: tuple[int, ...]
        :return: Tuple containing exact solution values and gradients.
        :rtype: tuple
        """
        y = torch.sin(x)  # equal to torch.concat((torch.sin(x[:,0:1]), torch.sin(x[:,1:2])),dim=1)
        ju_0 = torch.cos(x[:, 0:1])
        jv_1 = torch.cos(x[:, 1:2])
        return y, ju_0, jv_1


class CavityFlow(_Problem):
    """
    Solves the 2x3D problem (Navier-Stokes cavity flow).

    Domain: [0, 1] x [0, 1]
    Unknowns: Velocity vector field {u(x, y), v(x, y)}, pressure p(x, y)

    Equations:
        (u · ∇)u + ∇p = (1/Re) * Δu
        ∇ · u = 0

    Boundary conditions:
        u(x, 1) = 1, v(x, 1) = 0
        u, v = 0 on other boundary sides

    Ansatz:
        u(x, y) = tanh(x) * tanh(x-1) * tanh(y) * tanh(y-1) * NN(x, y)[0]
                  + 0.5 * (1 + tanh(20*(y-0.9)))  # an approximation
        v(x, y) = tanh(x) * tanh(x-1) * tanh(y) * tanh(y-1) * NN(x, y)[1]

    Note: u, v, p = y[:, 0], y[:, 1], y[:, 2]
    """

    @property
    def name(self):
        """Return problem name."""
        return "Cavityflow2x3D"

    def __init__(self):
        # Dimensionality of x and y
        # u, v, p = y[:,0], y[:,1], y[:,2]
        self.d = (2, 3)
        self.re = 100
        self.num_js = 10

    def physics_loss(
        self, x, y, u_x, u_y, u_xx, u_yy, v_x, v_y, v_xx, v_yy, p_x, p_y
    ):
        """
        Calculate the physics loss.

        :param x: Input coordinates.
        :type x: torch.Tensor
        :param y: Output values.
        :type y: torch.Tensor
        :param u_x: Gradient of u with respect to x.
        :type u_x: torch.Tensor
        :param u_y: Gradient of u with respect to y.
        :type u_y: torch.Tensor
        :param u_xx: Second derivative of u with respect to x.
        :type u_xx: torch.Tensor
        :param u_yy: Second derivative of u with respect to y.
        :type u_yy: torch.Tensor
        :param v_x: Gradient of v with respect to x.
        :type v_x: torch.Tensor
        :param v_y: Gradient of v with respect to y.
        :type v_y: torch.Tensor
        :param v_xx: Second derivative of v with respect to x.
        :type v_xx: torch.Tensor
        :param v_yy: Second derivative of v with respect to y.
        :type v_yy: torch.Tensor
        :param p_x: Gradient of p with respect to x.
        :type p_x: torch.Tensor
        :param p_y: Gradient of p with respect to y.
        :type p_y: torch.Tensor
        :return: Physics loss value.
        :rtype: torch.Tensor
        """
        u, v, p = y[:, 0:1], y[:, 1:2], y[:, 2:3]
        momentum_u = u * u_x + v * u_y + p_x - (1 / self.re) * (u_xx + u_yy)
        momentum_v = u * v_x + v * v_y + p_y - (1 / self.re) * (v_xx + v_yy)
        continuity = u_x + v_y
        physics = torch.concat((momentum_u, momentum_v, continuity), dim=1)
        return losses.l2_loss(physics, 0)

    def get_gradients(self, x, y):
        """
        Calculate gradients for the problem.

        :param x: Input coordinates.
        :type x: torch.Tensor
        :param y: Output values.
        :type y: torch.Tensor
        :return: Tuple containing y and its gradients.
        :rtype: tuple
        """
        j_u = torch.autograd.grad(
            y[:, 0], x, torch.ones_like(y[:, 0]), create_graph=True
        )[0]
        u_x, u_y = j_u[:, 0:1], j_u[:, 1:2]
        j_v = torch.autograd.grad(
            y[:, 1], x, torch.ones_like(y[:, 1]), create_graph=True
        )[0]
        v_x, v_y = j_v[:, 0:1], j_v[:, 1:2]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0][
            :, 0:1
        ]
        u_yy = torch.autograd.grad(u_y, x, torch.ones_like(u_y), create_graph=True)[0][
            :, 1:2
        ]
        v_xx = torch.autograd.grad(v_x, x, torch.ones_like(v_x), create_graph=True)[0][
            :, 0:1
        ]
        v_yy = torch.autograd.grad(v_y, x, torch.ones_like(v_y), create_graph=True)[0][
            :, 1:2
        ]
        j_p = torch.autograd.grad(
            y[:, 2], x, torch.ones_like(y[:, 2]), create_graph=True
        )[0]
        p_x, p_y = j_p[:, 0:1], j_p[:, 1:2]
        return y, u_x, u_y, u_xx, u_yy, v_x, v_y, v_xx, v_yy, p_x, p_y

    def boundary_condition(
        self, x, y, u_x, u_y, u_xx, u_yy, v_x, v_y, v_xx, v_yy, p_x, p_y, sd
    ):
        """
        Apply boundary conditions using ansatz.

        :param x: Input coordinates.
        :type x: torch.Tensor
        :param y: Output values from the neural network.
        :type y: torch.Tensor
        :param u_x: Gradient of u with respect to x.
        :type u_x: torch.Tensor
        :param u_y: Gradient of u with respect to y.
        :type u_y: torch.Tensor
        :param u_xx: Second derivative of u with respect to x.
        :type u_xx: torch.Tensor
        :param u_yy: Second derivative of u with respect to y.
        :type u_yy: torch.Tensor
        :param v_x: Gradient of v with respect to x.
        :type v_x: torch.Tensor
        :param v_y: Gradient of v with respect to y.
        :type v_y: torch.Tensor
        :param v_xx: Second derivative of v with respect to x.
        :type v_xx: torch.Tensor
        :param v_yy: Second derivative of v with respect to y.
        :type v_yy: torch.Tensor
        :param p_x: Gradient of p with respect to x.
        :type p_x: torch.Tensor
        :param p_y: Gradient of p with respect to y.
        :type p_y: torch.Tensor
        :param sd: Scale parameter.
        :type sd: float
        :return: Transformed y and gradients satisfying boundary conditions.
        :rtype: tuple
        """
        u, v, p = y[:, 0:1], y[:, 1:2], y[:, 2:3]
        tx, jtx, jjtx = tanhtanh_2(x[:, 0:1], 0, 1, sd)
        ty, jty, jjty = tanhtanh_2(x[:, 1:2], 0, 1, sd)
        tbd, jtbd, jjtbd = tanh_2(x[:, 1:2], 0.9, 0.05)
        u_new = tx * ty * u + 0.5 * (tbd + 1)
        u_new_x = ty * (jtx * u + tx * u_x)
        u_new_y = tx * (jty * u + ty * u_y) + 0.5 * jtbd
        u_new_xx = ty * (jjtx * u + 2 * jtx * u_x + tx * u_xx)
        u_new_yy = tx * (jjty * u + 2 * jty * u_y + ty * u_yy) + 0.5 * jjtbd
        v_new = tx * ty * v
        v_new_x = ty * (jtx * v + tx * v_x)
        v_new_y = tx * (jty * v + ty * v_y)
        v_new_xx = ty * (jjtx * v + 2 * jtx * v_x + tx * v_xx)
        v_new_yy = tx * (jjty * v + 2 * jty * v_y + ty * v_yy)
        y_new = torch.concat((u_new, v_new, p), dim=1)
        return (
            y_new,
            u_new_x,
            u_new_y,
            u_new_xx,
            u_new_yy,
            v_new_x,
            v_new_y,
            v_new_xx,
            v_new_yy,
            p_x,
            p_y,
        )

    def exact_solution(self, x, batch_size):
        """
        Calculate exact solution.

        Note: No exact solution available, returns zeros.

        :param x: Input coordinates.
        :type x: torch.Tensor
        :param batch_size: Batch size tuple.
        :type batch_size: tuple[int, ...]
        :return: Tuple containing dummy solution values and gradients (zeros).
        :rtype: tuple
        """
        # Shallow copy, but yj_true is read only
        return (torch.zeros((np.prod(batch_size),) + (3,)),) + (
            torch.zeros((np.prod(batch_size),) + (1,)),
        ) * 10



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    import problems
    from main import _x_mesh

    # Check velocity models for WaveEquation3D
    P = problems.WaveEquation3D(c=1, source_sd=0.2)
    subdomain_xs = [
        np.array([-10, -5, 0, 5, 10]),
        np.array([-10, -5, 0, 5, 10]),
        np.array([0, 5, 10]),
    ]
    batch_size_test = (50, 50, 15)
    x = _x_mesh(subdomain_xs, batch_size_test, "cpu")

    for f in P._gaussian_c, P._constant_c:
        y = f(x)
        print(y.shape)
        y = y[:, 0].numpy().reshape(batch_size_test)

        plt.figure()
        plt.imshow(y[:, :, 0].T, origin="lower")
        plt.colorbar()
        plt.figure()
        plt.imshow(y[:, :, -1].T, origin="lower")
        plt.colorbar()
        plt.show()
