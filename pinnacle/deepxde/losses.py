from . import backend as bkd
from . import config
from .backend import torch


def mean_absolute_error(y_true, y_pred):
    """Mean Absolute Error (MAE) - PyTorch implementation."""
    return bkd.reduce_mean(bkd.abs(y_true - y_pred))


def mean_absolute_percentage_error(y_true, y_pred):
    """Mean Absolute Percentage Error (MAPE) - PyTorch implementation."""
    epsilon = 1e-10  # Small value to avoid division by zero
    return bkd.reduce_mean(bkd.abs((y_true - y_pred) / (bkd.abs(y_true) + epsilon)))


def mean_squared_error(y_true, y_pred):
    """Mean Squared Error (MSE) - PyTorch implementation."""
    return bkd.reduce_mean(bkd.square(y_true - y_pred))


def mean_l2_relative_error(y_true, y_pred):
    """Mean L2 Relative Error - PyTorch implementation."""
    return bkd.reduce_mean(bkd.norm(y_true - y_pred, axis=1) / bkd.norm(y_true, axis=1))


def softmax_cross_entropy(y_true, y_pred):
    """Softmax Cross Entropy - PyTorch implementation."""
    # PyTorch expects logits as input for cross entropy
    return torch.nn.functional.cross_entropy(y_pred, y_true)


def zero(*_):
    """Zero loss - PyTorch implementation."""
    return torch.tensor(0.0, dtype=config.real(torch))


LOSS_DICT = {
    "mean absolute error": mean_absolute_error,
    "MAE": mean_absolute_error,
    "mae": mean_absolute_error,
    "mean squared error": mean_squared_error,
    "MSE": mean_squared_error,
    "mse": mean_squared_error,
    "mean absolute percentage error": mean_absolute_percentage_error,
    "MAPE": mean_absolute_percentage_error,
    "mape": mean_absolute_percentage_error,
    "mean l2 relative error": mean_l2_relative_error,
    "softmax cross entropy": softmax_cross_entropy,
    "zero": zero,
}


def get(identifier):
    """Retrieves a loss function.

    Args:
        identifier: A loss identifier. String name of a loss function, or a loss function.

    Returns:
        A loss function.
    """
    if isinstance(identifier, (list, tuple)):
        return list(map(get, identifier))

    if isinstance(identifier, str):
        return LOSS_DICT[identifier]
    if callable(identifier):
        return identifier
    raise ValueError("Could not interpret loss function identifier:", identifier)
