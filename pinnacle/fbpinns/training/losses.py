"""Loss functions for FBPINN / PINN training.

This module defines common loss and error metrics used throughout the training
code (see :mod:`main`, :mod:`problems`, and various PDE modules).
"""

from __future__ import annotations

import torch


def l2_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Standard L2 (mean squared) loss.

    :param a: Predicted tensor.
    :param b: Target tensor.
    :return: Mean squared error between ``a`` and ``b``.
    """
    return torch.mean((a - b) ** 2)


def l1_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Standard L1 (mean absolute) loss.

    :param a: Predicted tensor.
    :param b: Target tensor.
    :return: Mean absolute error between ``a`` and ``b``.
    """
    return torch.mean(torch.abs(a - b))


def l2_rel_err(y_true: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
    r"""Compute L2 relative error.

    .. math::

        \mathrm{L2RE}(y_{\mathrm{true}}, y_{\mathrm{hat}})
        = \frac{\lVert y_{\mathrm{true}} - y_{\mathrm{hat}} \rVert_2}
               {\lVert y_{\mathrm{true}} \rVert_2}

    :param y_true: Ground truth tensor.
    :param y_hat: Predicted tensor.
    :return: L2 relative error.
    """
    top = torch.sqrt(torch.mean((y_true - y_hat) ** 2))
    bottom = torch.sqrt(torch.mean(y_true ** 2))
    return top / bottom


def l1_rel_err(y_true: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
    """Compute L1 relative error.

    :param y_true: Ground truth tensor.
    :param y_hat: Predicted tensor.
    :return: L1 relative error.
    """
    top = torch.mean(torch.abs(y_true - y_hat))
    bottom = torch.mean(torch.abs(y_true))
    return top / bottom


def max_err(y_true: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
    """Compute maximum absolute error.

    :param y_true: Ground truth tensor.
    :param y_hat: Predicted tensor.
    :return: Maximum absolute deviation.
    """
    return torch.max(torch.abs(y_true - y_hat))


def err_csv(y_true: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
    """Error metric used for CSV-like aggregated outputs.

    Historically this was the absolute mean difference between the two
    tensors, which is preserved here.

    :param y_true: Ground truth tensor.
    :param y_hat: Predicted tensor.
    :return: Aggregated absolute mean error.
    """
    return torch.abs(torch.mean(y_true - y_hat))
