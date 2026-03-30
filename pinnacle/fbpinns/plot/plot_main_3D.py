"""Plotting helpers for 3D scalar-output FBPINN / PINN problems.

We avoid a single dashboard figure and instead export a small set of
publication-friendly 2D slices (PDF).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from . import plot_domain
from . import plot_main
from .palette import diff_cmap, field_cmap
from .plot_main_1D import _plot_setup, _to_numpy


FIELD_CMAP = field_cmap()
DIFF_CMAP = diff_cmap()


def _robust_absmax(arr: np.ndarray) -> float:
    if np.ma.isMaskedArray(arr):
        arr = np.ma.filled(arr, np.nan)
    vmax = float(np.nanmax(np.abs(np.asarray(arr))))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    return vmax


def _imshow_slice(
    ax,
    fig,
    it: int,
    y,
    xlim,
    c,
    title: str,
    *,
    cmap=None,
    vmin=None,
    vmax=None,
    add_colorbar: bool = True,
    cbar_fraction: float = 0.046,
    cbar_pad: float = 0.04,
) -> None:
    im = ax.imshow(
        y.reshape(c.BATCH_SIZE_TEST)[:, :, it].T,
        origin="lower",
        extent=(xlim[0][0], xlim[1][0], xlim[0][1], xlim[1][1]),
        cmap=FIELD_CMAP if cmap is None else cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )
    if add_colorbar:
        fig.colorbar(im, ax=ax, fraction=cbar_fraction, pad=cbar_pad)
    ax.set_xlim(xlim[0][0], xlim[1][0])
    ax.set_ylim(xlim[0][1], xlim[1][1])
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


@_to_numpy
def plot_3D_FBPINN(x_test, yj_true,   xs, yjs, yjs_sum,   yj_full, yjs_full, ys_full_raw,   yj_test_losses,   c, D, i):
    xlim, yjlims, n_yj, boundary, yj_test_losses, has_ref = _plot_setup(
        x_test, yj_true, yj_full, yj_test_losses, c
    )

    figs: List[Tuple[str, plt.Figure]] = []

    n_t = c.BATCH_SIZE_TEST[-1]
    t_ids = [0, n_t // 2, n_t - 1] if n_t >= 3 else list(range(n_t))

    # Cross-section domain plots (schematic)
    fig, ax = plt.subplots(1, 1, figsize=plot_main.ieee_figsize("single", 0.85), constrained_layout=True)
    plot_domain.plot_2D_cross_section(c.SUBDOMAIN_XS, D, [0, 1], create_fig=False, ax=ax)
    figs.append(("domain_xy", fig))

    # Slices (yj0 only by default)
    for it in t_ids:
        fig, ax = plt.subplots(1, 1, figsize=plot_main.ieee_figsize("single", 0.85), constrained_layout=True)
        _imshow_slice(ax, fig, it, yj_full[0][:, 0], xlim, c, title=f"Full solution slice t[{it}]", cmap=FIELD_CMAP)
        figs.append((f"full_solution_t{it}", fig))

        fig, ax = plt.subplots(1, 1, figsize=plot_main.ieee_figsize("single", 0.85), constrained_layout=True)
        _imshow_slice(
            ax, fig, it, boundary[0][:, 0], xlim, c, title=f"Boundary response slice t[{it}]", cmap=FIELD_CMAP
        )
        figs.append((f"boundary_t{it}", fig))

        if has_ref:
            fig, ax = plt.subplots(1, 1, figsize=plot_main.ieee_figsize("single", 0.85), constrained_layout=True)
            _imshow_slice(ax, fig, it, yj_true[0][:, 0], xlim, c, title=f"Reference slice t[{it}]", cmap=FIELD_CMAP)
            figs.append((f"reference_t{it}", fig))

            diff = (yj_full[0][:, 0] - yj_true[0][:, 0])
            dv = _robust_absmax(diff)

            fig, ax = plt.subplots(1, 1, figsize=plot_main.ieee_figsize("single", 0.85), constrained_layout=True)
            _imshow_slice(
                ax,
                fig,
                it,
                diff,
                xlim,
                c,
                title=f"Difference slice t[{it}]",
                cmap=DIFF_CMAP,
                vmin=-dv,
                vmax=dv,
            )
            figs.append((f"difference_t{it}", fig))

            # Triptych: prediction / reference / difference (three panels in a row)
            fig, axes = plt.subplots(
                1,
                3,
                figsize=plot_main.ieee_figsize("double", 0.40),
                constrained_layout=True,
            )
            _imshow_slice(
                axes[0],
                fig,
                it,
                yj_full[0][:, 0],
                xlim,
                c,
                title="Prediction",
                cmap=FIELD_CMAP,
                cbar_fraction=0.04,
                cbar_pad=0.02,
            )
            _imshow_slice(
                axes[1],
                fig,
                it,
                yj_true[0][:, 0],
                xlim,
                c,
                title="Reference",
                cmap=FIELD_CMAP,
                cbar_fraction=0.04,
                cbar_pad=0.02,
            )
            _imshow_slice(
                axes[2],
                fig,
                it,
                diff,
                xlim,
                c,
                title="Difference",
                cmap=DIFF_CMAP,
                vmin=-dv,
                vmax=dv,
                cbar_fraction=0.04,
                cbar_pad=0.02,
            )
            for k in (1, 2):
                axes[k].set_ylabel("")
            figs.append((f"solution_reference_difference_t{it}", fig))

    return tuple(figs)


@_to_numpy
def plot_3D_PINN(x_test, yj_true,   x, yj,   yj_full, y_full_raw,   yj_test_losses,   c, i):
    # Minimal PINN plotting: reuse the FBPINN slice layout.
    return plot_3D_FBPINN(
        x_test,
        yj_true,
        None,
        None,
        None,
        yj_full,
        None,
        None,
        yj_test_losses,
        c,
        None,
        i,
    )
