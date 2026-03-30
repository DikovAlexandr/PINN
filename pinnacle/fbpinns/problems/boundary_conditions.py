"""
Helper functions for applying hard boundary conditions to FBPINN / PINN ansatz.

This module contains analytical functions for computing boundary condition
transformations and their derivatives. These functions are used by problem
classes in :mod:`problems` to enforce boundary conditions.

This module is used by :mod:`problems`.
"""

from __future__ import annotations

import torch


# Helper analytical functions


def tanh_1(x, mu, sd):
    """
    Compute solution and gradient of y = tanh((x - mu) / sd).

    :param x: Input tensor.
    :type x: torch.Tensor
    :param mu: Mean parameter.
    :type mu: float
    :param sd: Scale parameter.
    :type sd: float
    :return: Tuple of (t, jt) where t is the value and jt is the gradient.
    :rtype: tuple[torch.Tensor, torch.Tensor]
    """
    xn = (x - mu) / sd

    tanh = torch.tanh(xn)
    sech2 = 1 - tanh**2

    t = tanh
    jt = (1 / sd) * sech2

    return t, jt


def tanh_2(x, mu, sd):
    """
    Compute solution and gradients (j, jj) of y = tanh((x - mu) / sd).

    :param x: Input tensor.
    :type x: torch.Tensor
    :param mu: Mean parameter.
    :type mu: float
    :param sd: Scale parameter.
    :type sd: float
    :return: Tuple of (t, jt, jtt) where t is the value, jt is the first
        derivative, and jtt is the second derivative.
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    xn = (x - mu) / sd

    tanh = torch.tanh(xn)
    sech2 = 1 - tanh**2

    t = tanh
    jt = (1 / sd) * sech2
    jtt = (1 / sd**2) * (-2 * tanh * sech2)

    return t, jt, jtt


def tanh2_2(x, mu, sd):
    """
    Compute solution and gradients of y = tanh^2((x - mu) / sd).

    :param x: Input tensor.
    :type x: torch.Tensor
    :param mu: Mean parameter.
    :type mu: float
    :param sd: Scale parameter.
    :type sd: float
    :return: Tuple of (t2, jt2, jjt2) where t2 is the value, jt2 is the
        first derivative, and jjt2 is the second derivative.
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    xn = (x - mu) / sd

    tanh = torch.tanh(xn)
    sech2 = 1 - tanh**2

    t2 = tanh**2
    jt2 = (1 / sd) * (2 * tanh * sech2)
    jjt2 = (1 / sd**2) * (2 * (sech2**2) - 4 * (tanh**2) * sech2)

    return t2, jt2, jjt2


def tanhtanh_2(x, mu1, mu2, sd):
    """
    Compute solution and gradients of y = tanh((x - mu1) / sd) * tanh((x - mu2) / sd).

    :param x: Input tensor.
    :type x: torch.Tensor
    :param mu1: First mean parameter.
    :type mu1: float
    :param mu2: Second mean parameter.
    :type mu2: float
    :param sd: Scale parameter.
    :type sd: float
    :return: Tuple of (t, jt, jjt) where t is the value, jt is the first
        derivative, and jjt is the second derivative.
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    xn_1 = (x - mu1) / sd
    xn_2 = (x - mu2) / sd

    tanh_1 = torch.tanh(xn_1)
    tanh_2 = torch.tanh(xn_2)
    sech2_1 = 1 - tanh_1**2
    sech2_2 = 1 - tanh_2**2

    t = tanh_1 * tanh_2
    jt = (1 / sd) * (tanh_1 * sech2_2 + sech2_1 * tanh_2)
    jjt = (1 / sd**2) * (
        2 * sech2_1 * sech2_2 - 2 * tanh_1 * tanh_2 * (sech2_1 + sech2_2)
    )

    return t, jt, jjt


def sigmoid_2(x, mu, sd):
    """
    Compute solution and gradients of y = sigmoid((x - mu) / sd).

    :param x: Input tensor.
    :type x: torch.Tensor
    :param mu: Mean parameter.
    :type mu: float
    :param sd: Scale parameter.
    :type sd: float
    :return: Tuple of (s, js, jjs) where s is the value, js is the first
        derivative, and jjs is the second derivative.
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    xn = (x - mu) / sd

    sig = torch.sigmoid(xn)

    s = sig
    js = (1 / sd) * sig * (1 - sig)
    jjs = (1 / sd**2) * sig * (1 - sig) * (1 - 2 * sig)

    return s, js, jjs


# Helper analytical functions (fused)


def tanh_tanh2_2(x, mu, sd):
    """
    Compute solution and gradients of both y = tanh((x - mu) / sd) and
    y = tanh^2((x - mu) / sd).

    :param x: Input tensor.
    :type x: torch.Tensor
    :param mu: Mean parameter.
    :type mu: float
    :param sd: Scale parameter.
    :type sd: float
    :return: Tuple of (t, jt, jjt, t2, jt2, jjt2) where t, jt, jjt are for
        tanh and t2, jt2, jjt2 are for tanh^2.
    :rtype: tuple[torch.Tensor, ...]
    """
    xn = (x - mu) / sd

    tanh = torch.tanh(xn)
    sech2 = 1 - tanh**2

    t = tanh
    jt = (1 / sd) * sech2
    jjt = (1 / sd**2) * (-2 * tanh * sech2)

    t2 = tanh**2
    jt2 = (1 / sd) * (2 * tanh * sech2)
    jjt2 = (1 / sd**2) * (2 * (sech2**2) - 4 * (tanh**2) * sech2)

    return t, jt, jjt, t2, jt2, jjt2


# Helper apply functions


def A_1D_1(x, y, j, A, mu, sd):
    """
    Apply ansatz: y = tanh((x - mu) / sd) * NN + A.

    Let t = tanh((x - mu) / sd), jt = d/dx (tanh((x - mu) / sd)), y = NN.
    Then:
    - y <- t * y + A
    - d/dx y <- (d/dx t) * y + t * (d/dx y) = jt * y + t * j

    :param x: Input coordinates.
    :type x: torch.Tensor
    :param y: Neural network output.
    :type y: torch.Tensor
    :param j: Gradient of y with respect to x.
    :type j: torch.Tensor
    :param A: Constant offset.
    :type A: float
    :param mu: Mean parameter for tanh.
    :type mu: float
    :param sd: Scale parameter for tanh.
    :type sd: float
    :return: Tuple of (y_new, j_new).
    :rtype: tuple[torch.Tensor, torch.Tensor]
    """
    t, jt = tanh_1(x, mu, sd)

    y_new = t * y + A
    j_new = jt * y + t * j

    return y_new, j_new


def AB_1D_2(x, y, j, jj, A, B, mu, sd):
    """
    Apply ansatz: y = tanh^2((x - mu) / sd) * NN + B * sd * tanh((x - mu) / sd) + A.

    :param x: Input coordinates.
    :type x: torch.Tensor
    :param y: Neural network output.
    :type y: torch.Tensor
    :param j: First gradient of y with respect to x.
    :type j: torch.Tensor
    :param jj: Second gradient of y with respect to x.
    :type jj: torch.Tensor
    :param A: Constant offset.
    :type A: float
    :param B: Coefficient for tanh term.
    :type B: float
    :param mu: Mean parameter for tanh.
    :type mu: float
    :param sd: Scale parameter for tanh.
    :type sd: float
    :return: Tuple of (y_new, j_new, jj_new).
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    t, jt, jjt, t2, jt2, jjt2 = tanh_tanh2_2(x, mu, sd)
    B = B * sd

    y_new = t2 * y + B * t + A
    j_new = jt2 * y + t2 * j + B * jt
    jj_new = jjt2 * y + 2 * jt2 * j + t2 * jj + B * jjt

    return y_new, j_new, jj_new
