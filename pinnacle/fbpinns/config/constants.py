"""FBPINNs constants (vendored).

This module defines the :class:`Constants` class, which stores the full problem
setup and all hyperparameters for both FBPINN and standard PINN experiments.
An instantiated :class:`Constants` object is passed to the trainer classes in
``main.py``.
"""

from __future__ import annotations

import os
import socket
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from models import models
import problems
from training import active_schedulers
from config.constantsBase import ConstantsBase


def get_subdomain_xs(ds: Sequence[np.ndarray], scales: Sequence[float]) -> List[np.ndarray]:
    """Compute (scaled) subdomain coordinates from segment lengths.

    Each entry in ``ds`` is treated as a sequence of segment lengths along a
    given dimension. The cumulative sum is normalised to :math:`[-1, 1]` and
    then scaled by the corresponding value in ``scales``.

    :param ds: List of 1D arrays describing segment lengths per dimension.
    :param scales: Scaling factors per dimension.
    :return: List of 1D arrays of subdomain edge coordinates.
    """
    xs: List[np.ndarray] = []
    for d, scale in zip(ds, scales):
        x = np.cumsum(np.pad(d, (1, 0)))
        x = 2.0 * (x - x.min()) / (x.max() - x.min()) - 1.0  # normalise to [-1, 1]
        xs.append(scale * x)
    return xs


def get_subdomain_ws(subdomain_xs: Sequence[np.ndarray], width: float) -> List[np.ndarray]:
    """Compute overlap widths for each subdomain edge array.

    ``width`` is interpreted as the (maximum) fraction of the smallest segment
    length in a given dimension. This value is broadcast across all subdomains
    in that dimension.

    Examples
    --------
    - ``get_subdomain_ws([np.array([-1.0, -0.5, 0, 0.5, 1])], 0.4)``
      returns ``[array([0.2, 0.2, 0.2, 0.2, 0.2])]``.
    - ``get_subdomain_ws([np.array([-1.0, -0.75, 0, 0.5, 1])], 0.4)``
      returns ``[array([0.1, 0.1, 0.1, 0.1, 0.1])]``.

    :param subdomain_xs: List of 1D arrays of subdomain edge coordinates.
    :param float width: Maximum overlap fraction per dimension.
    :return: List of 1D arrays of overlap widths per dimension.
    """
    ws: List[np.ndarray] = []
    for x in subdomain_xs:
        dx_min = float(np.min(np.diff(x)))
        ws.append(width * dx_min * np.ones_like(x))
    return ws


class Constants(ConstantsBase):
    """Configuration container for FBPINN and PINN runs."""

    def __init__(self, **kwargs) -> None:
        """Initialise the configuration with sensible defaults.

        All core hyperparameters and problem/domain definitions are set here.
        Any keyword arguments passed to the constructor override the
        corresponding default values (via :meth:`ConstantsBase.__setitem__`).

        :param kwargs: Optional overrides for default configuration values.
        """
        # Run identifier
        self.RUN = "test"

        # Problem definition
        w = 1e-10
        # self.P = problems.Cos1D_1(w=w, A=0)
        self.P = problems.Sin1D_2(w=w, A=0, B=-1 / w)

        # Domain definition (see domainsBase.py for details)
        self.SUBDOMAIN_XS = get_subdomain_xs(
            [np.array([2, 3, 2, 4, 3])],
            [2 * np.pi / self.P.w],
        )  # list of rectangle edges along each dimension

        self.SUBDOMAIN_WS = get_subdomain_ws(
            self.SUBDOMAIN_XS,
            0.7,
        )  # list of overlap widths along each dimension

        # Normalisation parameters
        self.BOUNDARY_N = (1 / self.P.w,)  # sd
        # self.Y_N = (0,1/self.P.w)       # mu, sd
        self.Y_N = (0, 1 / self.P.w**2)  # mu, sd

        # Active scheduler
        self.ACTIVE_SCHEDULER = active_schedulers.PointActiveSchedulerND
        self.ACTIVE_SCHEDULER_ARGS = (np.array([0]),)

        # Device / GPU parameters
        self.DEVICE = 0  # CUDA device index

        # Model architecture parameters
        self.MODEL = models.FCN  # specific parameters are passed later
        self.N_HIDDEN = 16
        self.N_LAYERS = 2

        # Optimisation parameters
        self.BATCH_SIZE = (500,)
        self.RANDOM = False
        self.LRATE = 1e-3
        # loss = 1 * loss_physics + BOUNDARY_WEIGHT * loss_boundary
        self.BOUNDARY_WEIGHT = 100
        self.BOUNDARY_BATCH_SIZE = 50
        self.DATALOSS_WEIGHT = 10
        self.N_STEPS = 50000

        # Random seed
        self.SEED = 123

        # Evaluation / plotting parameters
        self.BATCH_SIZE_TEST = (5000,)
        self.BOUNDARY_BATCH_SIZE_TEST = 100
        self.PLOT_LIMS = (1, False)

        # Summary/output frequencies
        self.SUMMARY_FREQ = 250
        self.TEST_FREQ = 5000
        self.MODEL_SAVE_FREQ = 10000
        self.SHOW_FIGURES = False  # whether to show figures
        self.SAVE_FIGURES = True  # whether to save figures
        self.CLEAR_OUTPUT = False  # whether to clear output periodically

        # Apply user overrides
        for key, value in kwargs.items():
            # Invokes __setitem__ in ConstantsBase (validates keys).
            self[key] = value

        # Derived output paths and environment-dependent configuration
        # Allow redirecting results outside fbpinns/ when embedded
        # in a larger benchmark.
        #
        # Examples:
        #   export FBPINNS_RESULTS_ROOT=".../pinnacle/runs/<exp>/<case>/fbpinns_results/"
        #   export FBPINNS_BENCHMARK_RESULTS_ROOT=".../pinnacle/runs/<exp>/<case>/fbpinns_benchmark_results/"
        self.RESULTS_ROOT = os.environ.get("FBPINNS_RESULTS_ROOT", "results/")
        self.BENCHMARK_RESULTS_ROOT = os.environ.get(
            "FBPINNS_BENCHMARK_RESULTS_ROOT",
            "benchmark_results/",
        )

        # Normalise trailing separators (important for string concatenations).
        if not self.RESULTS_ROOT.endswith(("/", "\\")):
            self.RESULTS_ROOT += "/"
        if not self.BENCHMARK_RESULTS_ROOT.endswith(("/", "\\")):
            self.BENCHMARK_RESULTS_ROOT += "/"

        self.SUMMARY_OUT_DIR = os.path.join(
            self.RESULTS_ROOT,
            "summaries",
            f"{self.RUN}",
        ) + os.sep

        self.MODEL_OUT_DIR = os.path.join(
            self.RESULTS_ROOT,
            "models",
            f"{self.RUN}",
        ) + os.sep

        self.HOSTNAME = socket.gethostname().lower()
