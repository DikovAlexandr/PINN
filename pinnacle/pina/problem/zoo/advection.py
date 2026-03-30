"""Formulation of the advection problem."""

import torch
from ... import Condition
from ...problem import SpatialProblem, TimeDependentProblem
from ...equation import Equation, Advection
from ...utils import check_consistency
from ...domain import CartesianDomain


def initial_condition(input_, output_):
    """
    Implementation of the initial condition.

    :param LabelTensor input_: Input data of the problem.
    :param LabelTensor output_: Output data of the problem.
    :return: The residual of the initial condition.
    :rtype: LabelTensor
    """
    return output_ - torch.sin(input_.extract("x"))


class AdvectionProblem(SpatialProblem, TimeDependentProblem):
    r"""
    Implementation of the advection problem in the spatial interval
    :math:`[0, 2 \pi]` and temporal interval :math:`[0, 1]`.

    .. seealso::

        **Original reference**: Wang, Sifan, et al. *An expert's guide to
        training physics-informed neural networks*.
        arXiv preprint arXiv:2308.08468 (2023).
        DOI: `arXiv:2308.08468  <https://arxiv.org/abs/2308.08468>`_.

    :Example:

        >>> problem = AdvectionProblem()
    """

    output_variables = ["u"]
    spatial_domain = CartesianDomain({"x": [0, 2 * torch.pi]})
    temporal_domain = CartesianDomain({"t": [0, 1]})

    domains = {
        "D": spatial_domain.update(temporal_domain),
        "t0": spatial_domain.update(CartesianDomain({"t": 0})),
    }

    conditions = {
        "t0": Condition(domain="t0", equation=Equation(initial_condition)),
    }

    def __init__(self, c=1.0):
        """
        Initialization of the :class:`AdvectionProblem`.

        :param c: The advection velocity parameter. Default is 1.0.
        :type c: float | int
        """
        super().__init__()
        check_consistency(c, (float, int))
        self.c = c

        self.conditions["D"] = Condition(domain="D", equation=Advection(self.c))

    def solution(self, pts):
        """
        Implementation of the analytical solution of the advection problem.

        :param LabelTensor pts: Points where the solution is evaluated.
        :return: The analytical solution of the advection problem.
        :rtype: LabelTensor
        """
        sol = torch.sin(pts.extract("x") - self.c * pts.extract("t"))
        sol.labels = self.output_variables
        return sol
