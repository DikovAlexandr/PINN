"""Domain plotting helpers for the vendored FBPINNs code.

This module provides lightweight plotting utilities used by `plot_main_*.py`.
The functions are designed to be:
- dependency-light (matplotlib + numpy only),
- robust across 1D/2D/3D problems,
- easy to embed into IEEE-style figures by optionally plotting into an Axes.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt


def _make_colors(n: int) -> List:
    """Return a color-blind friendly list of matplotlib colors."""
    if n <= 0:
        return []
    cmap = plt.get_cmap("tab20")
    return [cmap(i % cmap.N) for i in range(n)]


# `plot_main_*.py` expects `plot_domain.colors[im]`.
colors: List = []


def _get_ax(ax: Optional[plt.Axes]) -> Tuple[plt.Figure, plt.Axes]:
    """Return (fig, ax), creating them if needed."""
    if ax is not None:
        return ax.figure, ax
    fig, ax = plt.subplots(1, 1)
    return fig, ax


def plot_1D(
    subdomain_xs: Sequence[np.ndarray],
    D,
    create_fig: bool = True,
    ax: Optional[plt.Axes] = None,
) -> None:
    """Plot a 1D domain decomposition (interval subdomains).

    Args:
        subdomain_xs: List with one array of subdomain edges.
        D: Domain object (used for `N_MODELS` if present).
        create_fig: If True and `ax` is None, create a new figure.
        ax: Optional axes to draw into.
    """
    global colors
    n_models = getattr(D, "N_MODELS", len(subdomain_xs[0]) - 1)
    colors = _make_colors(int(n_models))

    if ax is None and create_fig:
        _, ax = _get_ax(None)

    x_edges = np.asarray(subdomain_xs[0])
    y0, y1 = 0.0, 1.0
    for i in range(len(x_edges) - 1):
        x0, x1 = x_edges[i], x_edges[i + 1]
        ax.fill_between(
            [x0, x1], [y0, y0], [y1, y1],
            color=colors[i], alpha=0.15,
        )
        ax.plot(
            [x0, x0], [y0, y1],
            color="k", linewidth=0.5, alpha=0.4,
        )
    ax.plot(
        [x_edges[-1], x_edges[-1]], [y0, y1],
        color="k", linewidth=0.5, alpha=0.4,
    )
    ax.set_ylim(y0, y1)
    ax.set_yticks([])
    ax.set_title("Domain decomposition (1D)")
    ax.set_xlabel("x")


def plot_2D(
    subdomain_xs: Sequence[np.ndarray],
    D,
    create_fig: bool = True,
    ax: Optional[plt.Axes] = None,
) -> None:
    """Plot a 2D domain decomposition (grid of rectangles).

    Args:
        subdomain_xs: Two arrays (x edges, y edges).
        D: Domain object (used for `N_MODELS` if present).
        create_fig: If True and `ax` is None, create a new figure.
        ax: Optional axes to draw into.
    """
    global colors
    # Fallback: product of segments in x/y
    n_models = getattr(
        D,
        "N_MODELS",
        (len(subdomain_xs[0]) - 1) * (len(subdomain_xs[1]) - 1),
    )
    colors = _make_colors(int(n_models))

    if ax is None and create_fig:
        _, ax = _get_ax(None)

    x_edges = np.asarray(subdomain_xs[0])
    y_edges = np.asarray(subdomain_xs[1])
    im = 0
    for ix in range(len(x_edges) - 1):
        for iy in range(len(y_edges) - 1):
            x0, x1 = x_edges[ix], x_edges[ix + 1]
            y0, y1 = y_edges[iy], y_edges[iy + 1]
            ax.fill(
                [x0, x1, x1, x0],
                [y0, y0, y1, y1],
                color=colors[im],
                alpha=0.15,
                linewidth=0,
            )
            ax.plot(
                [x0, x1, x1, x0, x0],
                [y0, y0, y1, y1, y0],
                color="k",
                linewidth=0.5,
                alpha=0.3,
            )
            im += 1
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Domain decomposition (2D)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def plot_2D_cross_section(
    subdomain_xs: Sequence[np.ndarray],
    D,
    dims: Sequence[int],
    create_fig: bool = True,
    ax: Optional[plt.Axes] = None,
) -> None:
    """Plot a 2D cross-section decomposition for higher-dimensional domains.

    This is a thin wrapper used by `plot_main_3D.py`/`plot_main_3x2D.py`.
    The plot is schematic: it draws the rectangle grid for the selected dims.
    """
    if len(dims) != 2:
        raise ValueError(f"dims must have length 2, got {dims!r}")
    xs2 = [subdomain_xs[dims[0]], subdomain_xs[dims[1]]]
    plot_2D(xs2, D, create_fig=create_fig, ax=ax)
