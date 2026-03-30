"""Plotting helpers for 2D FBPINN / PINN problems.

We generate separate IEEE-friendly figures (one plot per PDF), instead of a
single dashboard figure.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from . import plot_domain
from . import plot_main
from .palette import diff_cmap, field_cmap
from .plot_main_1D import _plot_setup, _to_numpy

import sys

sys.path.insert(0, "./shared_modules/")
from helper import Timer


FIELD_CMAP = field_cmap()
DIFF_CMAP = diff_cmap()


def _robust_absmax(arr: np.ndarray) -> float:
    if np.ma.isMaskedArray(arr):
        arr = np.ma.filled(arr, np.nan)
    vmax = float(np.nanmax(np.abs(np.asarray(arr))))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    return vmax


def _imshow_field(
    ax,
    fig,
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
        y.reshape(c.BATCH_SIZE_TEST).T,
        origin="lower",
        extent=(xlim[0][0], xlim[1][0], xlim[0][1], xlim[1][1]),
        cmap=FIELD_CMAP if cmap is None else cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )
    if add_colorbar:
        fig.colorbar(im, ax=ax, fraction=cbar_fraction, pad=cbar_pad)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


@_to_numpy
def plot_2D_FBPINN(
    x_test,
    yj_true,
    xs,
    yjs,
    yjs_sum,
    yj_full,
    yjs_full,
    ys_full_raw,
    yj_test_losses,
    c,
    D,
    i,
):
    """Generate IEEE-friendly plots for 2D FBPINN runs."""
    xlim, yjlims, n_yj, boundary, yj_test_losses, has_ref = _plot_setup(
        x_test, yj_true, yj_full, yj_test_losses, c
    )

    figs: List[Tuple[str, plt.Figure]] = []
    plot_js = list(range(n_yj)) if getattr(c, "PLOT_ALL_YJ", False) else [0]

    # Domain decomposition
    fig, ax = plt.subplots(1, 1, figsize=plot_main.ieee_figsize("single", 0.85), constrained_layout=True)
    plot_domain.plot_2D(c.SUBDOMAIN_XS, D, create_fig=False, ax=ax)
    figs.append(("domain", fig))

    # Full solution per yj
    for j in plot_js:
        fig, ax = plt.subplots(1, 1, figsize=plot_main.ieee_figsize("single", 0.85), constrained_layout=True)
        _imshow_field(ax, fig, yj_full[j][:, 0], xlim, c, title=f"Full solution: yj{j}", cmap=FIELD_CMAP)
        figs.append((f"full_solution_yj{j}", fig))

        if has_ref:
            fig, ax = plt.subplots(1, 1, figsize=plot_main.ieee_figsize("single", 0.85), constrained_layout=True)
            _imshow_field(ax, fig, yj_true[j][:, 0], xlim, c, title=f"Reference: yj{j}", cmap=FIELD_CMAP)
            figs.append((f"reference_yj{j}", fig))

            diff = (yj_full[j][:, 0] - yj_true[j][:, 0])
            dv = _robust_absmax(diff)
            fig, ax = plt.subplots(1, 1, figsize=plot_main.ieee_figsize("single", 0.85), constrained_layout=True)
            _imshow_field(
                ax,
                fig,
                diff,
                xlim,
                c,
                title=f"Difference: yj{j}",
                cmap=DIFF_CMAP,
                vmin=-dv,
                vmax=dv,
            )
            figs.append((f"difference_yj{j}", fig))

            # Triptych: prediction / reference / difference (three panels in a row)
            fig, axes = plt.subplots(
                1,
                3,
                figsize=plot_main.ieee_figsize("double", 0.40),
                constrained_layout=True,
            )
            _imshow_field(
                axes[0],
                fig,
                yj_full[j][:, 0],
                xlim,
                c,
                title="Prediction",
                cmap=FIELD_CMAP,
                cbar_fraction=0.04,
                cbar_pad=0.02,
            )
            _imshow_field(
                axes[1],
                fig,
                yj_true[j][:, 0],
                xlim,
                c,
                title="Reference",
                cmap=FIELD_CMAP,
                cbar_fraction=0.04,
                cbar_pad=0.02,
            )
            _imshow_field(
                axes[2],
                fig,
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
            figs.append((f"solution_reference_difference_yj{j}", fig))

    # Boundary response
    fig, ax = plt.subplots(1, 1, figsize=plot_main.ieee_figsize("single", 0.85), constrained_layout=True)
    _imshow_field(ax, fig, boundary[0][:, 0], xlim, c, title="Boundary condition response", cmap=FIELD_CMAP)
    figs.append(("boundary", fig))

    # Raw NN histogram
    fig, ax = plt.subplots(1, 1, figsize=plot_main.ieee_figsize("single", 0.55), constrained_layout=True)
    for j, im in enumerate(D.active_fixed_ims):
        ax.hist(ys_full_raw[j][:, 0], bins=80, color=plot_domain.colors[im], alpha=0.6)
    ax.set_title("Raw model outputs (histogram)")
    ax.set_xlabel("value")
    ax.set_yticks([])
    figs.append(("raw_hist", fig))

    # Loss curves (per yj)
    if yj_test_losses.size > 0:
        for j in plot_js:
            fig, ax = plt.subplots(1, 1, figsize=plot_main.ieee_figsize("single", 0.55), constrained_layout=True)
            x = yj_test_losses[:, 0]
            y_test = yj_test_losses[:, 3 + j]

            need_bd = hasattr(c.P, "sample_bd")
            train_idx_expected = (3 + n_yj + 1 + (1 if need_bd else 0))  # last column index if present
            has_train = yj_test_losses.ndim == 2 and yj_test_losses.shape[1] >= (train_idx_expected + 1)
            y_train = yj_test_losses[:, -1] if has_train else None

            ax.plot(x, y_test, color="C0", marker="o", markersize=3.5, label="test")
            if y_train is not None:
                ax.plot(x, y_train, color="C1", marker="s", markersize=3.0, label="train")

            ax.set_title(f"Loss (train+test): yj{j}")
            ax.set_xlabel("step")
            ax.set_ylabel("loss")
            ax.set_yscale("log")
            ax.legend(frameon=False, loc="best")
            figs.append((f"test_loss_yj{j}", fig))

    return tuple(figs)


@_to_numpy
def plot_2D_PINN(x_test, yj_true, x, yj, yj_full, y_full_raw, yj_test_losses, c, i):
    """Generate IEEE-friendly plots for 2D PINN runs."""
    xlim, yjlims, n_yj, boundary, yj_test_losses, has_ref = _plot_setup(
        x_test, yj_true, yj_full, yj_test_losses, c
    )
    figs: List[Tuple[str, plt.Figure]] = []

    for j in range(n_yj):
        fig, ax = plt.subplots(1, 1, figsize=plot_main.ieee_figsize("single", 0.85), constrained_layout=True)
        _imshow_field(ax, fig, yj_full[j][:, 0], xlim, c, title=f"Full solution: yj{j}", cmap=FIELD_CMAP)
        figs.append((f"full_solution_yj{j}", fig))

        if has_ref:
            fig, ax = plt.subplots(1, 1, figsize=plot_main.ieee_figsize("single", 0.85), constrained_layout=True)
            _imshow_field(ax, fig, yj_true[j][:, 0], xlim, c, title=f"Reference: yj{j}", cmap=FIELD_CMAP)
            figs.append((f"reference_yj{j}", fig))

            diff = (yj_full[j][:, 0] - yj_true[j][:, 0])
            dv = _robust_absmax(diff)
            fig, ax = plt.subplots(1, 1, figsize=plot_main.ieee_figsize("single", 0.85), constrained_layout=True)
            _imshow_field(
                ax,
                fig,
                diff,
                xlim,
                c,
                title=f"Difference: yj{j}",
                cmap=DIFF_CMAP,
                vmin=-dv,
                vmax=dv,
            )
            figs.append((f"difference_yj{j}", fig))

            fig, axes = plt.subplots(
                1,
                3,
                figsize=plot_main.ieee_figsize("double", 0.40),
                constrained_layout=True,
            )
            _imshow_field(
                axes[0],
                fig,
                yj_full[j][:, 0],
                xlim,
                c,
                title="Prediction",
                cmap=FIELD_CMAP,
                cbar_fraction=0.04,
                cbar_pad=0.02,
            )
            _imshow_field(
                axes[1],
                fig,
                yj_true[j][:, 0],
                xlim,
                c,
                title="Reference",
                cmap=FIELD_CMAP,
                cbar_fraction=0.04,
                cbar_pad=0.02,
            )
            _imshow_field(
                axes[2],
                fig,
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
            figs.append((f"solution_reference_difference_yj{j}", fig))

    fig, ax = plt.subplots(1, 1, figsize=plot_main.ieee_figsize("single", 0.85), constrained_layout=True)
    _imshow_field(ax, fig, boundary[0][:, 0], xlim, c, title="Boundary condition response", cmap=FIELD_CMAP)
    figs.append(("boundary", fig))

    return tuple(figs)
