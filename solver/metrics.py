import torch


def accuracy(y_true, y_pred):
    """
    Calculate the accuracy between the true labels and the predicted labels.

    Parameters:
        y_true (torch.Tensor): The true labels.
        y_pred (torch.Tensor): The predicted labels.

    Returns:
        float: The accuracy of the predictions.
    """
    true_labels = torch.argmax(y_true, dim=-1)
    pred_labels = torch.argmax(y_pred, dim=-1)
    correct_predictions = torch.sum(true_labels == pred_labels).item()
    total_samples = y_true.size(0)
    return correct_predictions / total_samples


def l2_norm(y_true, y_pred):
    """
    Calculate the L2 norm (Euclidean norm) between the true values and the predicted values.

    Parameters:
        y_true (torch.Tensor): The true values.
        y_pred (torch.Tensor): The predicted values.

    Returns:
        float: The L2 norm.
    """
    l2_norm_value = torch.norm(y_true - y_pred)
    return l2_norm_value.item()


def l2_relative_error(y_true, y_pred):
    """
    Calculate the L2 relative error between the true values and the predicted values.

    Parameters:
        y_true (torch.Tensor): The true values.
        y_pred (torch.Tensor): The predicted values.

    Returns:
        float: The L2 relative error.
    """
    l2_norm_true = torch.norm(y_true)
    l2_norm_diff = torch.norm(y_true - y_pred)
    relative_error = l2_norm_diff / l2_norm_true
    return relative_error.item()


def max_value_error(y_true, y_pred):
    """
    Calculate the error between the maximum true value and the corresponding predicted value.

    Args:
        y_true (torch.Tensor): The true values.
        y_pred (torch.Tensor): The predicted values.

    Returns:
        float: The error between the maximum true value and the corresponding predicted value.
    """
    idx_max = torch.argmax(y_true)
    max_true_value = y_true[idx_max]
    pred_value_at_max = y_pred[idx_max]
    error = max_true_value - pred_value_at_max
    return error.item()


def calculate_error(y_true, y_pred, is_abs=False):
    """
    Calculate the difference between the true values and the predicted values.

    Parameters:
        y_true (torch.Tensor): The true values.
        y_pred (torch.Tensor): The predicted values.
        is_abs (bool): Whether to calculate the absolute error.

    Returns:
        torch.Tensor: The difference between the true and predicted values.
    """
    if is_abs:
        return torch.abs(y_true - y_pred)
    else:
        return y_true - y_pred