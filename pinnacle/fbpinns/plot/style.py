"""Lightweight plotting style helpers shared outside the FBPINNs runtime.

This module mirrors the IEEE-style helpers from `plot_main.py` but avoids
importing the heavier plotting backends that require additional deps.
"""

from __future__ import annotations

from typing import Tuple

import matplotlib as mpl

IEEE_SINGLE_COL_IN = 3.5
IEEE_DOUBLE_COL_IN = 7.16


def apply_ieee_style() -> None:
    """Apply a conservative IEEE-style matplotlib theme (idempotent)."""
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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def ieee_figsize(kind: str = "single", aspect: float = 0.75) -> Tuple[float, float]:
    """Return a (width, height) in inches suitable for IEEE figures."""
    if kind not in {"single", "double"}:
        raise ValueError(f"Unknown kind {kind!r}. Use 'single' or 'double'.")
    w = IEEE_SINGLE_COL_IN if kind == "single" else IEEE_DOUBLE_COL_IN
    return (w, w * aspect)
