"""Module for the System of Equation."""

import torch
from .equation_interface import EquationInterface
from .equation import Equation
from ..utils import check_consistency


class SystemEquation(EquationInterface):
    """
    Implementation of the System of Equations, to be passed to a
    :class:`~pina.condition.condition.Condition` object.

    Unlike the :class:`~pina.equation.equation.Equation` class, which represents
    a single equation, the :class:`SystemEquation` class allows multiple
    equations to be grouped together into a system. This is particularly useful
    when dealing with multi-component outputs or coupled physical models, where
    the residual must be computed collectively across several constraints.

    Each equation in the system must be either:
    - An instance of :class:`~pina.equation.equation.Equation`;
    - A callable function.

    The residuals from each equation are computed independently and then
    aggregated using an optional reduction strategy (e.g., ``mean``, ``sum``).
    The resulting residual is returned as a single :class:`~pina.LabelTensor`.

    :Example:

    >>> from pina.equation import SystemEquation, FixedValue, FixedGradient
    >>> from pina import LabelTensor
    >>> import torch
    >>> pts = LabelTensor(torch.rand(10, 2), labels=["x", "y"])
    >>> pts.requires_grad = True
    >>> output_ = torch.pow(pts, 2)
    >>> output_.labels = ["u", "v"]
    >>> system_equation = SystemEquation(
    ...     [
    ...         FixedValue(value=1.0, components=["u"]),
    ...         FixedGradient(value=0.0, components=["v"],d=["y"]),
    ...     ],
    ...     reduction="mean",
    ... )
    >>> residual = system_equation.residual(pts, output_)

    """

    def __init__(self, list_equation, reduction=None):
        """
        Initialization of the :class:`SystemEquation` class.

        :param list_equation: A list containing either callable functions or
            instances of :class:`~pina.equation.equation.Equation`, used to
            compute the residuals of mathematical equations.
        :type list_equation: list[Callable] | list[Equation]
        :param str reduction: The reduction method to aggregate the residuals of
            each equation. Available options are: ``None``, ``mean``, ``sum``,
            ``callable``.
            If ``None``, no reduction is applied. If ``mean``, the output sum is
            divided by the number of elements in the output. If ``sum``, the
            output is summed. ``callable`` is a user-defined callable function
            to perform reduction, no checks guaranteed. Default is ``None``.
        :raises NotImplementedError: If the reduction is not implemented.
        """
        check_consistency([list_equation], list)

        # equations definition
        self.equations = [
            equation if isinstance(equation, Equation) else Equation(equation)
            for equation in list_equation
        ]

        # possible reduction
        if reduction == "mean":
            self.reduction = torch.mean
        elif reduction == "sum":
            self.reduction = torch.sum
        elif (reduction is None) or callable(reduction):
            self.reduction = reduction
        else:
            raise NotImplementedError(
                "Only mean and sum reductions are currenly supported."
            )

    def residual(self, input_, output_, params_=None):
        """
        Compute the residual for each equation in the system of equations and
        aggregate it according to the ``reduction`` specified in the
        ``__init__`` method.

        :param LabelTensor input_: Input points where each equation of the
            system is evaluated.
        :param LabelTensor output_: Output tensor, eventually produced by a
            :class:`torch.nn.Module` instance.
        :param dict params_: Dictionary of unknown parameters, associated with a
            :class:`~pina.problem.inverse_problem.InverseProblem` instance.
            If the equation is not related to a
            :class:`~pina.problem.inverse_problem.InverseProblem` instance, the
            parameters must be initialized to ``None``. Default is ``None``.

        :return: The aggregated residuals of the system of equations.
        :rtype: LabelTensor
        """
        # Move the equation to the input_ device
        self.to(input_.device)

        # Compute the residual for each equation
        residual = torch.hstack(
            [
                equation.residual(input_, output_, params_)
                for equation in self.equations
            ]
        )

        # Skip reduction if not specified
        if self.reduction is None:
            return residual

        return self.reduction(residual, dim=-1)
