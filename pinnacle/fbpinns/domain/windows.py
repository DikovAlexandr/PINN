"""
Module for window functions.

This module defines window functions which are applied to the output of each
subdomain neural network. The windows are defined using torch such that we can
autodiff through them during training.

This module is used by :mod:`domains`.
"""

from __future__ import annotations

import torch


def _create_kernel(xmin, xmax, smin, smax):
    """
    Create a 1D kernel function.

    :param xmin: Left endpoint (None for unbounded).
    :type xmin: float | None
    :param xmax: Right endpoint (None for unbounded).
    :type xmax: float | None
    :param smin: Scale parameter for left side.
    :type smin: float
    :param smax: Scale parameter for right side.
    :type smax: float
    :return: Kernel function.
    :rtype: callable
    """
    tol = 1e-10  # for numerical stability when evaluating gradients
    clamp = lambda x: torch.clamp(x, min=tol)  # every element e <- max(e, tol)

    # w at left endpoint of overlap: w_l = sigmoid(-0.5*overlap / 0.05*overlap)
    # = sigmoid(-10) = 0.00005
    # w at right endpoint of overlap: w_r = sigmoid(10) = 1 - sigmoid(-10)
    # = 0.99995
    if xmax is None and xmin is None:
        kernel = lambda x: torch.ones_like(x)
    elif xmax is None:
        if smin <= 0:
            raise Exception(f"ERROR smin <= 0 ({smin})!")
        kernel = lambda x: clamp(torch.sigmoid((x - xmin) / smin))
    elif xmin is None:
        if smax <= 0:
            raise Exception(f"ERROR smax <= 0 ({smax})!")
        kernel = lambda x: clamp(torch.sigmoid((xmax - x) / smax))
    else:
        if xmin > xmax:
            raise Exception(f"ERROR: xmin ({xmin}) > xmax ({xmax})!")
        if smin <= 0:
            raise Exception(f"ERROR smin <= 0 ({smin})!")
        if smax <= 0:
            raise Exception(f"ERROR smax <= 0 ({smax})!")
        kernel = lambda x: clamp(
            clamp(torch.sigmoid((x - xmin) / smin))
            * clamp(torch.sigmoid((xmax - x) / smax))
        )

    return kernel


def construct_window_function_ND(xs_min, xs_max, scales_min, scales_max):
    """
    Construct an N-dimensional window function.

    :param xs_min: Left endpoints for each dimension (None for unbounded).
    :type xs_min: list[float | None]
    :param xs_max: Right endpoints for each dimension (None for unbounded).
    :type xs_max: list[float | None]
    :param scales_min: Scale parameters for left side (scale * wmin).
    :type scales_min: list[float]
    :param scales_max: Scale parameters for right side (scale * wmax).
    :type scales_max: list[float]
    :return: Window function.
    :rtype: callable
    """
    if not (len(xs_min) == len(xs_max) == len(scales_min) == len(scales_max)):
        raise Exception("ERROR input lengths do not match!")

    kernels = [
        _create_kernel(*args) for args in zip(xs_min, xs_max, scales_min, scales_max)
    ]
    nd = len(xs_min)

    def window_function(x):
        if x.ndim != 2:
            raise Exception(f"ERROR!: x.ndim ({x.shape}) != 2!")
        if x.shape[-1] != nd:
            raise Exception(f"ERROR!: x.shape[1] ({x.shape[1]}) != nd ({nd})")

        xs = x.unbind(-1)  # separate out dims
        ws = [kernels[i](x) for i, x in enumerate(xs)]
        w = torch.stack(ws, -1)
        w = torch.prod(w, keepdim=True, dim=-1)  # get product of windows over each dimension

        return w

    return window_function


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    # 1D test
    x = np.expand_dims(np.arange(-10, 10, 0.1), -1).astype(np.float32)
    x = torch.from_numpy(x)

    window_function = construct_window_function_ND([-1], [6], [1.2], [0.5])
    w1 = window_function(x)

    window_function = construct_window_function_ND([None], [-1], [0.5], [0.5])
    w2 = window_function(x)

    window_function = construct_window_function_ND([6], [None], [0.1], [0.1])
    w3 = window_function(x)

    plt.figure()
    plt.plot(x, w1)
    plt.plot(x, w2)
    plt.plot(x, w3)
    plt.plot(x, w1 + w2 + w3, color="k", alpha=0.4)
    plt.show()

    # 2D test
    x = np.linspace(-20, 20, 220)
    y = np.linspace(-15, 18, 200)
    xx = np.stack(np.meshgrid(x, y, indexing="ij"), -1)
    x = xx.reshape((220 * 200, -1))

    window_function = construct_window_function_ND([0, -10], [15, -5], [4, 1], [0.2, 0.4])
    w1 = window_function(torch.from_numpy(x))
    w1 = w1.reshape((220, 200))

    plt.figure()
    plt.imshow(w1.T, origin="lower", extent=(x.min(), x.max(), y.min(), y.max()))
    plt.colorbar()
    plt.show()
