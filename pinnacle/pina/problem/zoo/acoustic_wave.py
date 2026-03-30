"""Formulation of the acoustic wave problem."""

import torch
from ... import Condition
from ...problem import SpatialProblem, TimeDependentProblem
from ...utils import check_consistency
from ...domain import CartesianDomain
from ...equation import (
    Equation,
    SystemEquation,
    FixedValue,
    FixedGradient,
    AcousticWave,
)


def initial_condition(input_, output_):
    """
    Definition of the initial condition of the acoustic wave problem.

    :param LabelTensor input_: The input data of the problem.
    :param LabelTensor output_: The output data of the problem.
    :return: The residual of the initial condition.
    :rtype: LabelTensor
    """
    arg = torch.pi * input_["x"]
    return output_ - torch.sin(arg) - 0.5 * torch.sin(4 * arg)


class AcousticWaveProblem(TimeDependentProblem, SpatialProblem):
    r"""
    Implementation of the acoustic wave problem in the spatial interval
    :math:`[0, 1]` and temporal interval :math:`[0, 1]`.

    .. seealso::

        **Original reference**: Wang, Sifan, Xinling Yu, and
        Paris Perdikaris. *When and why PINNs fail to train:
        A neural tangent kernel perspective*. Journal of
        Computational Physics 449 (2022): 110768.
        DOI: `10.1016 <https://doi.org/10.1016/j.jcp.2021.110768>`_.

    :Example:

        >>> problem = AcousticWaveProblem(c=2.0)
    """

    output_variables = ["u"]
    spatial_domain = CartesianDomain({"x": [0, 1]})
    temporal_domain = CartesianDomain({"t": [0, 1]})

    domains = {
        "D": spatial_domain.update(temporal_domain),
        "t0": spatial_domain.update(CartesianDomain({"t": 0})),
        "boundary": spatial_domain.partial().update(temporal_domain),
    }

    conditions = {
        "boundary": Condition(domain="boundary", equation=FixedValue(0.0)),
        "t0": Condition(
            domain="t0",
            equation=SystemEquation(
                [Equation(initial_condition), FixedGradient(0.0, d="t")]
            ),
        ),
    }

    def __init__(self, c=2.0):
        """
        Initialization of the :class:`AcousticWaveProblem` class.

        :param c: The wave propagation speed. Default is 2.0.
        :type c: float | int
        """
        super().__init__()
        check_consistency(c, (float, int))
        self.c = c

        self.conditions["D"] = Condition(
            domain="D", equation=AcousticWave(self.c)
        )

    def solution(self, pts):
        """
        Implementation of the analytical solution of the acoustic wave problem.

        :param LabelTensor pts: Points where the solution is evaluated.
        :return: The analytical solution of the acoustic wave problem.
        :rtype: LabelTensor
        """
        arg_x = torch.pi * pts["x"]
        arg_t = self.c * torch.pi * pts["t"]
        term1 = torch.sin(arg_x) * torch.cos(arg_t)
        term2 = 0.5 * torch.sin(4 * arg_x) * torch.cos(4 * arg_t)
        return term1 + term2
