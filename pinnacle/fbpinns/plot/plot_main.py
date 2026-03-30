"""Plot dispatchers and style helpers for the vendored FBPINNs code.

This module:
- selects the appropriate plotting backend (`plot_main_1D/2D/3D/...`) based on
  the problem dimensionality,
- defines a small IEEE-friendly matplotlib style used by all plots.

The training loop calls these functions from `fbpinns/main.py`.
"""

from __future__ import annotations

from typing import Tuple

import matplotlib as mpl

from . import plot_main_1D
from . import plot_main_2D
from . import plot_main_2x2D
from . import plot_main_3D
from . import plot_main_3x2D


IEEE_SINGLE_COL_IN = 3.5
IEEE_DOUBLE_COL_IN = 7.16


def apply_ieee_style() -> None:
    """Apply a conservative IEEE-style matplotlib theme (idempotent)."""
    # Keep it minimal and robust across environments (fonts may vary).
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "lines.linewidth": 1.2,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.25,
            "axes.grid": False,
            "pdf.fonttype": 42,  # TrueType
            "ps.fonttype": 42,
        }
    )


def ieee_figsize(kind: str = "single", aspect: float = 0.75) -> Tuple[float, float]:
    """Return a (width, height) in inches suitable for IEEE figures."""
    if kind not in {"single", "double"}:
        raise ValueError(f"Unknown kind {kind!r}. Use 'single' or 'double'.")
    w = IEEE_SINGLE_COL_IN if kind == "single" else IEEE_DOUBLE_COL_IN
    return (w, w * aspect)


def plot_FBPINN(*args):
    """Generate FBPINN plots during training (best-effort)."""
    apply_ieee_style()
    
    # figure out dimensionality of problem, use appropriate plotting function
    c = args[9]
    nd = c.P.d[0]
    ndy = len(range(c.P.d[1])[c.P.exact_dim_select]) if hasattr(c.P, "exact_dim_select") else c.P.d[1]
    if ndy == 1:
        if nd == 1:
            return plot_main_1D.plot_1D_FBPINN(*args)
        elif nd == 2:
            return plot_main_2D.plot_2D_FBPINN(*args)
        elif nd == 3:
            return plot_main_3D.plot_3D_FBPINN(*args)
        else:
            return None
            # TODO: implement higher dimension plotting
    else: # ndy >= 1
        if nd == 2:
            return plot_main_2x2D.plot_2D_FBPINN(*args)
        elif nd == 3:
            return plot_main_3x2D.plot_3D_FBPINN(*args)
        else:
            return None


def plot_PINN(*args):
    """Generate PINN plots during training (best-effort)."""
    apply_ieee_style()
    
    # figure out dimensionality of problem, use appropriate plotting function
    c = args[7]
    nd = c.P.d[0]
    ndy = c.P.d[1]
    if ndy == 1:
        if nd == 1:
            return plot_main_1D.plot_1D_PINN(*args)
        elif nd == 2:
            return plot_main_2D.plot_2D_PINN(*args)
        elif nd == 3:
            return plot_main_3D.plot_3D_PINN(*args)
        else:
            return None
            # TODO: implement higher dimension plotting
    else: # ndy >= 1
        if nd == 2:
            return plot_main_2x2D.plot_2x2D_PINN(*args)
        elif nd == 3:
            return plot_main_3D.plot_3D_PINN(*args)
        else:
            return None
