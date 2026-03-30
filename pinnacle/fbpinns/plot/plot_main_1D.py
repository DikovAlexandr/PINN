"""Plotting helpers for 1D FBPINN / PINN problems.

The upstream FBPINNs repository produced a single large "dashboard" figure with
many subplots. For the PINN benchmark we instead generate multiple separate
IEEE-friendly figures (one plot per PDF).

This module is used by `plot_main.py` (and subsequently `main.py`).
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from . import plot_domain


LINE_CMAP = plt.get_cmap("tab10")


def lim(
    mi: float, ma: float, factor: float = 1.0, zero_center: bool = False
) -> Tuple[float, float]:
    """Compute symmetric-ish limits with optional re-centering."""
    c = 0.0 if zero_center else (mi + ma) / 2.0
    w = factor * (ma - mi) / 2.0
    if not np.isfinite(w) or w <= 0:
        w = 1.0
    return (c - w, c + w)


def _to_numpy(f: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator converting torch.Tensor inputs into numpy arrays."""

    def recurse(obj: Any) -> Any:
        if isinstance(obj, (list, tuple)):
            return [recurse(o) for o in obj]
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().copy()
        return obj

    def wrapper(*args: Any) -> Any:
        return f(*recurse(args))

    return wrapper


def _has_reference_solution(c) -> bool:
    """Best-effort check whether the problem has usable reference data.

    We treat the solution as available only if the corresponding arrays exist
    and are non-empty. This prevents plotting "reference" curves when the
    code falls back to dummy zeros/ones.
    """
    if hasattr(c.P, "ref_in_coords") and hasattr(c.P, "ref_values"):
        try:
            return len(c.P.ref_in_coords) > 0 and len(c.P.ref_values) > 0
        except Exception:
            return False
    if hasattr(c.P, "ref_x") and hasattr(c.P, "ref_y"):
        try:
            return c.P.ref_x is not None and c.P.ref_y is not None and len(c.P.ref_x) > 0
        except Exception:
            return False
    return False


def _plot_setup(
    x_test: np.ndarray,
    yj_true: Sequence[np.ndarray],
    yj_full: Sequence[np.ndarray],
    yj_test_losses: Sequence[Sequence[float]],
    c,
) -> Tuple[Tuple[np.ndarray, np.ndarray], List[Tuple[float, float]], int, Sequence[np.ndarray], np.ndarray, bool]:
    """Shared computations for plotters."""
    has_ref = _has_reference_solution(c)
    xlim = (np.min(x_test, axis=0), np.max(x_test, axis=0))
    n_yj = len(yj_true)

    # Loss history
    yj_test_losses = np.array(yj_test_losses)

    # Boundary condition response (applied to constant NN output)
    boundary = c.P.boundary_condition(
        torch.from_numpy(x_test),
        *[torch.ones(t.shape) * c.Y_N[1] + c.Y_N[0] for t in yj_full],
        *c.BOUNDARY_N,
    )

    # Robust y-limits: prefer reference if available and non-degenerate, else use prediction.
    yjlims: List[Tuple[float, float]] = []
    for j in range(n_yj):
        arr_ref = yj_true[j]
        arr_pred = yj_full[j]
        rng = float(np.nanmax(arr_ref) - np.nanmin(arr_ref))
        if (not has_ref) or (not np.isfinite(rng)) or rng < 1e-10:
            mi, ma = float(np.nanmin(arr_pred)), float(np.nanmax(arr_pred))
        else:
            mi, ma = float(np.nanmin(arr_ref)), float(np.nanmax(arr_ref))
        yjlims.append(lim(mi, ma, *c.PLOT_LIMS))

    return xlim, yjlims, n_yj, boundary, yj_test_losses, has_ref


@_to_numpy
def plot_1D_FBPINN(
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
    """Generate IEEE-friendly plots for 1D FBPINN runs.

    Returns:
        Tuple of (name, figure) pairs.
    """
    _, yjlims, n_yj, boundary, yj_test_losses, has_ref = _plot_setup(
        x_test, yj_true, yj_full, yj_test_losses, c
    )

    from . import plot_main
    
    figs: List[Tuple[str, plt.Figure]] = []
    plot_js = list(range(n_yj)) if getattr(c, "PLOT_ALL_YJ", False) else [0]

    # Domain decomposition
    fig, ax = plt.subplots(
        1, 1,
        figsize=plot_main.ieee_figsize("single", 0.55),
        constrained_layout=True
    )
    plot_domain.plot_1D(c.SUBDOMAIN_XS, D, create_fig=False, ax=ax)
    figs.append(("domain", fig))

    # Individual models before sum/BC
    fig, ax = plt.subplots(
        1, 1,
        figsize=plot_main.ieee_figsize("single", 0.75),
        constrained_layout=True
    )
    if has_ref:
        ax.plot(x_test[:, 0], yj_true[0][:, 0], color="k", label="reference")
    for im, i1 in D.active_fixed_neighbours_ims:
        ax.scatter(
            xs[i1][:, 0],
            yjs[i1][0][:, 0],
            s=10,
            color=plot_domain.colors[im],
            alpha=0.6,
            edgecolors="none",
        )
    ax.set_title("Individual models (before sum/BC)")
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    if has_ref:
        ax.legend(frameon=False)
    figs.append(("individual_before_sum_bc", fig))

    # Individual models after sum/BC (per yj)
    for j in plot_js:
        fig, ax = plt.subplots(
            1, 1,
            figsize=plot_main.ieee_figsize("single", 0.75),
            constrained_layout=True
        )
        if has_ref:
            ax.plot(
                x_test[:, 0], yj_true[j][:, 0],
                color="k",
                label="reference"
            )
        for im, i1 in D.active_ims:
            ax.scatter(
                xs[i1][:, 0],
                yjs_sum[i1][j][:, 0],
                s=10,
                color=plot_domain.colors[im],
                alpha=0.6,
                edgecolors="none",
            )
        ax.set_ylim(*yjlims[j])
        ax.set_title(f"Individual models (after sum/BC): yj{j}")
        ax.set_xlabel("x")
        ax.set_ylabel(f"yj{j}")
        if has_ref:
            ax.legend(frameon=False)
        figs.append((f"individual_after_sum_bc_yj{j}", fig))

    # Full solution (per yj)
    for j in plot_js:
        fig, ax = plt.subplots(
            1, 1,
            figsize=plot_main.ieee_figsize("single", 0.75),
            constrained_layout=True
        )
        ax.plot(
            x_test[:, 0], yj_full[j][:, 0],
            color=LINE_CMAP(0),
            label="prediction"
        )
        if has_ref:
            ax.plot(
                x_test[:, 0], yj_true[j][:, 0],
                color="k",
                linestyle="--",
                label="reference"
            )
        ax.set_ylim(*yjlims[j])
        ax.set_title(f"Full solution: yj{j}")
        ax.set_xlabel("x")
        ax.set_ylabel(f"yj{j}")
        ax.legend(frameon=False)
        figs.append((f"full_solution_yj{j}", fig))

    # Boundary response
    fig, ax = plt.subplots(1, 1, figsize=plot_main.ieee_figsize("single", 0.55), constrained_layout=True)
    ax.plot(x_test[:, 0], boundary[0][:, 0], color=LINE_CMAP(1))
    ax.set_title("Boundary condition response")
    ax.set_xlabel("x")
    ax.set_ylabel("BC(x)")
    figs.append(("boundary", fig))

    # Loss curves (per yj)
    if yj_test_losses.size > 0:
        for j in plot_js:
            fig, ax = plt.subplots(
                1, 1, figsize=plot_main.ieee_figsize("single", 0.55), constrained_layout=True
            )
            x = yj_test_losses[:, 0]
            y_test = yj_test_losses[:, 3 + j]

            # Train loss is appended by the trainer (see `fbpinns/main.py`).
            need_bd = hasattr(c.P, "sample_bd")
            train_idx_expected = (3 + n_yj + 1 + (1 if need_bd else 0))  # last column index if present
            has_train = yj_test_losses.ndim == 2 and yj_test_losses.shape[1] >= (train_idx_expected + 1)
            y_train = yj_test_losses[:, -1] if has_train else None

            # Make sure a single-point history is still visible.
            ax.plot(x, y_test, color=LINE_CMAP(2), marker="o", markersize=3.5, label="test")
            if y_train is not None:
                ax.plot(x, y_train, color=LINE_CMAP(3), marker="s", markersize=3.0, label="train")

            ax.set_title(f"Loss (train+test): yj{j}")
            ax.set_xlabel("step")
            ax.set_ylabel("loss")
            ax.set_yscale("log")
            ax.legend(frameon=False, loc="best")
            figs.append((f"test_loss_yj{j}", fig))

    return tuple(figs)


@_to_numpy
def plot_1D_PINN(
    x_test,
    yj_true,
    x,
    yj,
    yj_full,
    y_full_raw,
    yj_test_losses,
    c,
    i,
):
    """Generate IEEE-friendly plots for 1D PINN runs."""
    from . import plot_main
    
    _, yjlims, n_yj, boundary, yj_test_losses, has_ref = _plot_setup(
        x_test, yj_true, yj_full, yj_test_losses, c
    )

    figs: List[Tuple[str, plt.Figure]] = []

    # Full solution (per yj)
    for j in range(n_yj):
        fig, ax = plt.subplots(
            1, 1, figsize=plot_main.ieee_figsize("single", 0.75), constrained_layout=True
        )
        ax.plot(x_test[:, 0], yj_full[j][:, 0], color=LINE_CMAP(0), label="prediction")
        if has_ref:
            ax.plot(x_test[:, 0], yj_true[j][:, 0], color="k", linestyle="--", label="reference")
        ax.set_ylim(*yjlims[j])
        ax.set_title(f"Full solution: yj{j}")
        ax.set_xlabel("x")
        ax.set_ylabel(f"yj{j}")
        ax.legend(frameon=False)
        figs.append((f"full_solution_yj{j}", fig))

    # Boundary response
    fig, ax = plt.subplots(1, 1, figsize=plot_main.ieee_figsize("single", 0.55), constrained_layout=True)
    ax.plot(x_test[:, 0], boundary[0][:, 0], color=LINE_CMAP(1))
    ax.set_title("Boundary condition response")
    ax.set_xlabel("x")
    ax.set_ylabel("BC(x)")
    figs.append(("boundary", fig))

    return tuple(figs)
