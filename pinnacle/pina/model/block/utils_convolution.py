"""Module for utility functions for the convolutional layer."""

import torch


def check_point(x, current_stride, dim):
    """
    Check if the point is in the current stride.

    :param torch.Tensor x: The input data.
    :param int current_stride: The current stride.
    :param int dim: The shape of the filter.
    :return: The indeces of the points in the current stride.
    :rtype: torch.Tensor
    """
    max_stride = current_stride + dim
    indeces = torch.logical_and(
        x[..., :-1] < max_stride, x[..., :-1] >= current_stride
    ).all(dim=-1)
    return indeces


def map_points_(x, filter_position):
    """
    The mapping function for n-dimensional case.

    :param torch.Tensor x: The two-dimensional input data.
    :param list[int] filter_position: The position of the filter.
    :return: The data mapped in-place.
    :rtype: torch.tensor
    """
    x.add_(-filter_position)

    return x


def optimizing(f):
    """
    Decorator to call the function only once.

    :param f: python function
    :type f: Callable
    """

    def wrapper(*args, **kwargs):
        """
        Wrapper function.

        :param args: The arguments of the function.
        :param kwargs: The keyword arguments of the function.
        """
        if kwargs["type_"] == "forward":
            if not wrapper.has_run_inverse:
                wrapper.has_run_inverse = True
                return f(*args, **kwargs)

        if kwargs["type_"] == "inverse":
            if not wrapper.has_run:
                wrapper.has_run = True
                return f(*args, **kwargs)

        return f(*args, **kwargs)

    wrapper.has_run_inverse = False
    wrapper.has_run = False

    return wrapper
