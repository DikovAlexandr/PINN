"""Plotting helpers for 3D vector-valued problems (3x2D).

Typical use case: 2D vector field evolving in time (x, y, t) with (u, v, p).
We export a small set of time slices as separate IEEE-friendly PDFs.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from . import plot_domain
from . import plot_main
from .palette import field_cmap
from .plot_main_1D import _plot_setup, _to_numpy


FIELD_CMAP = field_cmap()


def _plot_test_im(it, y, xlim, ylim, c):
    plt.imshow(y.reshape(c.BATCH_SIZE_TEST)[:,:,it].T, # need to transpose because torch.meshgrid uses np indexing="ij"
               origin="lower", extent=(xlim[0][0], xlim[1][0], xlim[0][1], xlim[1][1]),
               cmap=FIELD_CMAP, vmin=ylim[0], vmax=ylim[1])

def _plot_quiver(it, x_test, yj, bsize):
    x = x_test.reshape(bsize+(3,))[:,:,it,:]
    y = yj.reshape(bsize+(-1,))[:,:,it,:]
    if yj.shape[1] == 3:
        # pressure: counter plot (color)
        plt.contourf(x[:,:,0], x[:,:,1], y[:,:,2], alpha=0.5, cmap=FIELD_CMAP)
        plt.colorbar()
        plt.contour(x[:,:,0], x[:,:,1], y[:,:,2], cmap=FIELD_CMAP)
    x_gap, y_gap = bsize[0] // 15, bsize[1] // 15
    plt.quiver(x[::x_gap,::y_gap,0], x[::x_gap,::y_gap,1], y[::x_gap,::y_gap,0], y[::x_gap,::y_gap,1])

def _fix_plot(xlim):
    plt.colorbar()
    plt.xlim(xlim[0][0], xlim[1][0])
    plt.ylim(xlim[0][1], xlim[1][1])
    plt.gca().set_aspect("equal")


@_to_numpy
def plot_3D_FBPINN(x_test, yj_true,   xs, yjs, yjs_sum,   yj_full, yjs_full, ys_full_raw,   yj_test_losses,   c, D, i):
    xlim, yjlims, n_yj, boundary, yj_test_losses, has_ref = _plot_setup(
        x_test, yj_true, yj_full, yj_test_losses, c
    )

    figs: List[Tuple[str, plt.Figure]] = []

    n_t = c.BATCH_SIZE_TEST[-1]
    t_ids = [0, n_t // 2, n_t - 1] if n_t >= 3 else list(range(n_t))

    # Domain (xy cross-section)
    fig, ax = plt.subplots(1, 1, figsize=plot_main.ieee_figsize("single", 0.85), constrained_layout=True)
    plot_domain.plot_2D_cross_section(c.SUBDOMAIN_XS, D, [0, 1], create_fig=False, ax=ax)
    figs.append(("domain_xy", fig))

    for it in t_ids:
        fig = plt.figure(figsize=plot_main.ieee_figsize("single", 0.85), constrained_layout=True)
        _plot_quiver(it, x_test, yj_full[0], c.BATCH_SIZE_TEST)
        plt.title(f"Full solution (quiver) t[{it}]")
        figs.append((f"full_solution_quiver_t{it}", fig))

        fig = plt.figure(figsize=plot_main.ieee_figsize("single", 0.85), constrained_layout=True)
        _plot_test_im(it, boundary[0][:, 0], xlim, (None, None), c)
        _fix_plot(xlim)
        plt.title(f"Boundary response t[{it}]")
        figs.append((f"boundary_t{it}", fig))

        if has_ref:
            fig = plt.figure(figsize=plot_main.ieee_figsize("single", 0.85), constrained_layout=True)
            _plot_quiver(it, x_test, yj_true[0], c.BATCH_SIZE_TEST)
            plt.title(f"Reference (quiver) t[{it}]")
            figs.append((f"reference_quiver_t{it}", fig))

            fig = plt.figure(figsize=plot_main.ieee_figsize("single", 0.85), constrained_layout=True)
            _plot_quiver(it, x_test, yj_full[0] - yj_true[0], c.BATCH_SIZE_TEST)
            plt.title(f"Difference (quiver) t[{it}]")
            figs.append((f"difference_quiver_t{it}", fig))

    return tuple(figs)


@_to_numpy
def plot_3D_PINN(x_test, yj_true,   x, yj,   yj_full, y_full_raw,   yj_test_losses,   c, i):
    
    xlim, yjlims, n_yj, boundary, yj_test_losses = _plot_setup(x_test, yj_true, yj_full, yj_test_losses, c)
    
    n_t = c.BATCH_SIZE_TEST[-1]
    shape = (2+n_t, max(4,n_yj))# nrows, ncols
    f1 = plt.figure(figsize=(4*shape[1],3*shape[0]))
    
    # TEST PLOT
    
    j = 0
    for it in range(n_t):
        
        # Boundary response
        plt.subplot2grid(shape,(1+it,0))
        
        _plot_test_im(it, boundary[0][:,0], xlim, (None, None), c)
        
        _fix_plot(xlim)
        plt.title("[%i] Boundary condition"%(i,))
        
        # full model after sum and BC
        plt.subplot2grid(shape,(1+it,1))
        
        _plot_test_im(it, yj_full[j][:,0], xlim, yjlims[j], c)
        
        _fix_plot(xlim)
        plt.title("[%i] Full solution - $yj_{%i}$"%(i,j))
        
        # ground truth
        plt.subplot2grid(shape,(1+it,2))
        
        _plot_test_im(it, yj_true[j][:,0], xlim, yjlims[j], c)
        
        _fix_plot(xlim)
        plt.title("[%i] Ground truth - $yj_{%i}$"%(i,j))
        
        # difference
        plt.subplot2grid(shape,(1+it,3))
        
        _plot_test_im(it, yj_full[j][:,0]-yj_true[j][:,0], xlim, yjlims[j], c)
        
        _fix_plot(xlim)
        plt.title("[%i] Difference - $yj_{%i}$"%(i,j))
        
    # raw NN plot
    plt.subplot2grid(shape, (0,3))
    
    plt.hist(y_full_raw[:,0], bins=100)
    
    plt.yticks([])
    plt.title("[%i] Individual model - Raw"%(i,))
    
    # loss plot
    for j in range(n_yj):# plot all yjs
        plt.subplot2grid(shape,(1+n_t,j))
        
        plt.plot(yj_test_losses[:,0], yj_test_losses[:,3+j])
        
        plt.title("[%i] Test loss $yj_{%i}$"%(i,j))
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    
    return (("train-test",f1),)
