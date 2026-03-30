"""Module for utility functions."""

import types
from functools import reduce
import torch

from .label_tensor import LabelTensor


# Codacy error unused parameters
def custom_warning_format(
    message, category, filename, lineno, file=None, line=None
):
    """
    Custom warning formatting function.

    :param str message: The warning message.
    :param Warning category: The warning category.
    :param str filename: The filename where the warning is raised.
    :param int lineno: The line number where the warning is raised.
    :param str file: The file object where the warning is raised.
        Default is None.
    :param int line: The line where the warning is raised.
    :return: The formatted warning message.
    :rtype: str
    """
    return f"{filename}: {category.__name__}: {message}\n"


def check_consistency(object_, object_instance, subclass=False):
    """
    Check if an object maintains inheritance consistency.

    This function checks whether a given object is an instance of a specified
    class or, if ``subclass=True``, whether it is a subclass of the specified
    class.

    :param object: The object to check.
    :type object: Iterable | Object
    :param Object object_instance: The expected parent class.
    :param bool subclass: If True, checks whether ``object_`` is a subclass
        of ``object_instance`` instead of an instance. Default is ``False``.
    :raises ValueError: If ``object_`` does not inherit from ``object_instance``
        as expected.
    """
    if not isinstance(object_, (list, set, tuple)):
        object_ = [object_]

    for obj in object_:
        is_class = isinstance(obj, type)
        expected_type_name = (
            object_instance.__name__
            if isinstance(object_instance, type)
            else str(object_instance)
        )

        if subclass:
            if not is_class:
                raise ValueError(
                    f"You passed {repr(obj)} "
                    f"(an instance of {type(obj).__name__}), "
                    f"but a {expected_type_name} class was expected. "
                    f"Please pass a {expected_type_name} class or a "
                    "derived one."
                )
            if not issubclass(obj, object_instance):
                raise ValueError(
                    f"You passed {obj.__name__} class, but a "
                    f"{expected_type_name} class was expected. "
                    f"Please pass a {expected_type_name} class or a "
                    "derived one."
                )
        else:
            if is_class:
                raise ValueError(
                    f"You passed {obj.__name__} class, but a "
                    f"{expected_type_name} instance was expected. "
                    f"Please pass a {expected_type_name} instance."
                )
            if not isinstance(obj, object_instance):
                raise ValueError(
                    f"You passed {repr(obj)} "
                    f"(an instance of {type(obj).__name__}), "
                    f"but a {expected_type_name} instance was expected. "
                    f"Please pass a {expected_type_name} instance."
                )


def labelize_forward(forward, input_variables, output_variables):
    """
    Decorator to enable or disable the use of
    :class:`~pina.label_tensor.LabelTensor` during the forward pass.

    :param Callable forward: The forward function of a :class:`torch.nn.Module`.
    :param list[str] input_variables: The names of the input variables of a
        :class:`~pina.problem.abstract_problem.AbstractProblem`.
    :param list[str] output_variables: The names of the output variables of a
        :class:`~pina.problem.abstract_problem.AbstractProblem`.
    :return: The decorated forward function.
    :rtype: Callable
    """

    def wrapper(x, *args, **kwargs):
        """
        Decorated forward function.

        :param LabelTensor x: The labelized input of the forward pass of an
            instance of :class:`torch.nn.Module`.
        :param Iterable args: Additional positional arguments passed to
            ``forward`` method.
        :param dict kwargs: Additional keyword arguments passed to
            ``forward`` method.
        :return: The labelized output of the forward pass of an instance of
            :class:`torch.nn.Module`.
        :rtype: LabelTensor
        """
        x = x.extract(input_variables)
        output = forward(x, *args, **kwargs)
        # keep it like this, directly using LabelTensor(...) raises errors
        # when compiling the code
        output = output.as_subclass(LabelTensor)
        output.labels = output_variables
        return output

    return wrapper


def merge_tensors(tensors):
    """
    Merge a list of :class:`~pina.label_tensor.LabelTensor` instances into a
    single :class:`~pina.label_tensor.LabelTensor` tensor, by applying
    iteratively the cartesian product.

    :param list[LabelTensor] tensors: The list of tensors to merge.
    :raises ValueError: If the list of tensors is empty.
    :return: The merged tensor.
    :rtype: LabelTensor
    """
    if tensors:
        return reduce(merge_two_tensors, tensors[1:], tensors[0])
    raise ValueError("Expected at least one tensor")


def merge_two_tensors(tensor1, tensor2):
    """
    Merge two :class:`~pina.label_tensor.LabelTensor` instances into a single
    :class:`~pina.label_tensor.LabelTensor` tensor, by applying the cartesian
    product.

    :param LabelTensor tensor1: The first tensor to merge.
    :param LabelTensor tensor2: The second tensor to merge.
    :return: The merged tensor.
    :rtype: LabelTensor
    """
    n1 = tensor1.shape[0]
    n2 = tensor2.shape[0]

    tensor1 = LabelTensor(tensor1.repeat(n2, 1), labels=tensor1.labels)
    tensor2 = LabelTensor(
        tensor2.repeat_interleave(n1, dim=0), labels=tensor2.labels
    )
    return tensor1.append(tensor2)


def torch_lhs(n, dim):
    """
    The Latin Hypercube Sampling torch routine, sampling in :math:`[0, 1)`$.

    :param int n: The number of points to sample.
    :param int dim: The number of dimensions of the sampling space.
    :raises TypeError: If `n` or `dim` are not integers.
    :raises ValueError: If `dim` is less than 1.
    :return: The sampled points.
    :rtype: torch.tensor
    """

    if not isinstance(n, int):
        raise TypeError("number of point n must be int")

    if not isinstance(dim, int):
        raise TypeError("dim must be int")

    if dim < 1:
        raise ValueError("dim must be greater than one")

    samples = torch.rand(size=(n, dim))

    perms = torch.tile(torch.arange(1, n + 1), (dim, 1))

    for row in range(dim):
        idx_perm = torch.randperm(perms.shape[-1])
        perms[row, :] = perms[row, idx_perm]

    perms = perms.T

    samples = (perms - samples) / n

    return samples


def is_function(f):
    """
    Check if the given object is a function or a lambda.

    :param Object f: The object to be checked.
    :return: ``True`` if ``f`` is a function, ``False`` otherwise.
    :rtype: bool
    """
    return callable(f)


def chebyshev_roots(n):
    """
    Compute the roots of the Chebyshev polynomial of degree ``n``.

    :param int n: The number of roots to return.
    :return: The roots of the Chebyshev polynomials.
    :rtype: torch.Tensor
    """
    pi = torch.acos(torch.zeros(1)).item() * 2
    k = torch.arange(n)
    nodes = torch.sort(torch.cos(pi * (k + 0.5) / n))[0]
    return nodes


def check_positive_integer(value, strict=True):
    """
    Check if the value is a positive integer.

    :param int value: The value to check.
    :param bool strict: If True, the value must be strictly positive.
        Default is True.
    :raises AssertionError: If the value is not a positive integer.
    """
    if strict:
        assert (
            isinstance(value, int) and value > 0
        ), f"Expected a strictly positive integer, got {value}."
    else:
        assert (
            isinstance(value, int) and value >= 0
        ), f"Expected a non-negative integer, got {value}."


def in_range(value, range_vals, strict=True):
    """
    Check if a value is within a specified range.

    :param int value: The integer value to check.
    :param list[int] range_vals: A list of two integers representing the range
        limits. The first element specifies the lower bound, and the second
        specifies the upper bound.
    :param bool strict: If True, the value must be strictly positive.
        Default is True.
    :return: True if the value satisfies the range condition, False otherwise.
    :rtype: bool
    """
    # Validate inputs
    check_consistency(value, (float, int))
    check_consistency(range_vals, (float, int))
    assert (
        isinstance(range_vals, list) and len(range_vals) == 2
    ), "range_vals must be a list of two integers [lower, upper]"
    lower, upper = range_vals

    # Check the range
    if strict:
        return lower < value < upper

    return lower <= value <= upper
