"""Plotting helpers for 2D vector-valued FBPINN / PINN problems (2x2D).

Typical use case: Navier–Stokes (u, v, p) on a 2D grid.
This module generates separate IEEE-friendly figures (PDF) per plot.
"""

from __future__ import annotations

from typing import List, Tuple

import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib import cm

from . import plot_domain
from . import plot_main
from .palette import field_cmap
from .plot_main_1D import _plot_setup, _to_numpy

import sys

sys.path.insert(0, "./shared_modules/")
from helper import Timer


SCALAR_CMAP = field_cmap()


def _imshow_field(ax, fig, y, xlim, c, title: str, vmin=None, vmax=None) -> None:
    im = ax.imshow(
        y.reshape(c.BATCH_SIZE_TEST).T,
        origin="lower",
        extent=(xlim[0][0], xlim[1][0], xlim[0][1], xlim[1][1]),
        cmap=SCALAR_CMAP,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def _plot_quiver(x_test, yj, bsize):
    x = x_test.reshape(bsize + (2,))
    y = yj.reshape(bsize + (-1,))
    if yj.shape[1] == 3:
        # pressure: counter plot (color)
        plt.contourf(x[:, :, 0], x[:, :, 1], y[:, :, 2], alpha=0.5, cmap=SCALAR_CMAP)
        plt.colorbar()
        plt.contour(x[:, :, 0], x[:, :, 1], y[:, :, 2], cmap=SCALAR_CMAP)
    x_gap, y_gap = bsize[0] // 15, bsize[1] // 15
    plt.quiver(
        x[::x_gap, ::y_gap, 0],
        x[::x_gap, ::y_gap, 1],
        y[::x_gap, ::y_gap, 0],
        y[::x_gap, ::y_gap, 1],
    )


def _fix_plot(xlim):
    plt.colorbar()
    plt.xlim(xlim[0][0], xlim[1][0])
    plt.ylim(xlim[0][1], xlim[1][1])
    plt.gca().set_aspect("equal")


@_to_numpy
def plot_2D_FBPINN(x_test, yj_true,   xs, yjs, yjs_sum,   yj_full, yjs_full, ys_full_raw,   yj_test_losses,   c, D, i):
    xlim, yjlims, n_yj, boundary, yj_test_losses, has_ref = _plot_setup(
        x_test, yj_true, yj_full, yj_test_losses, c
    )

    figs: List[Tuple[str, plt.Figure]] = []

    # Domain
    fig, ax = plt.subplots(1, 1, figsize=plot_main.ieee_figsize("single", 0.85), constrained_layout=True)
    plot_domain.plot_2D(c.SUBDOMAIN_XS, D, create_fig=False, ax=ax)
    figs.append(("domain", fig))

    # Vector field (u,v) + optional pressure contour from yj0
    fig, ax = plt.subplots(1, 1, figsize=plot_main.ieee_figsize("single", 0.85), constrained_layout=True)
    _plot_quiver(x_test, yj_full[0], c.BATCH_SIZE_TEST)
    plt.title("Full solution (vector field)")
    figs.append(("full_solution_yj0_quiver", fig))

    if has_ref:
        fig, ax = plt.subplots(1, 1, figsize=plot_main.ieee_figsize("single", 0.85), constrained_layout=True)
        _plot_quiver(x_test, yj_true[0], c.BATCH_SIZE_TEST)
        plt.title("Reference (vector field)")
        figs.append(("reference_yj0_quiver", fig))

        fig, ax = plt.subplots(1, 1, figsize=plot_main.ieee_figsize("single", 0.85), constrained_layout=True)
        _plot_quiver(x_test, yj_full[0] - yj_true[0], c.BATCH_SIZE_TEST)
        plt.title("Difference (vector field)")
        figs.append(("difference_yj0_quiver", fig))

    # Boundary response (show first component as scalar field)
    fig, ax = plt.subplots(1, 1, figsize=plot_main.ieee_figsize("single", 0.85), constrained_layout=True)
    _imshow_field(ax, fig, boundary[0][:, 0], xlim, c, title="Boundary condition response")
    figs.append(("boundary", fig))

    return tuple(figs)


@_to_numpy
def plot_2x2D_PINN(x_test, yj_true,   x, yj,   yj_full, y_full_raw,   yj_test_losses,   c, i):
    xlim, yjlims, n_yj, boundary, yj_test_losses, has_ref = _plot_setup(
        x_test, yj_true, yj_full, yj_test_losses, c
    )

    figs: List[Tuple[str, plt.Figure]] = []

    fig, ax = plt.subplots(1, 1, figsize=plot_main.ieee_figsize("single", 0.85), constrained_layout=True)
    _plot_quiver(x_test, yj_full[0], c.BATCH_SIZE_TEST)
    plt.title("Full solution (vector field)")
    figs.append(("full_solution_yj0_quiver", fig))

    fig, ax = plt.subplots(1, 1, figsize=plot_main.ieee_figsize("single", 0.85), constrained_layout=True)
    _imshow_field(ax, fig, boundary[0][:, 0], xlim, c, title="Boundary condition response")
    figs.append(("boundary", fig))

    return tuple(figs)
