"""Formulation of the Allen Cahn problem."""

import torch
from ... import Condition
from ...problem import SpatialProblem, TimeDependentProblem
from ...equation import Equation, AllenCahn
from ...utils import check_consistency
from ...domain import CartesianDomain


def initial_condition(input_, output_):
    """
    Definition of the initial condition of the Allen Cahn problem.

    :param LabelTensor input_: The input data of the problem.
    :param LabelTensor output_: The output data of the problem.
    :return: The residual of the initial condition.
    :rtype: LabelTensor
    """
    x = input_.extract("x")
    u_0 = x**2 * torch.cos(torch.pi * x)
    return output_ - u_0


class AllenCahnProblem(TimeDependentProblem, SpatialProblem):
    r"""
    Implementation of the Allen Cahn problem in the spatial interval
    :math:`[-1, 1]` and temporal interval :math:`[0, 1]`.

    .. seealso::

        **Original reference**: Sokratis J. Anagnostopoulos, Juan D. Toscano,
        Nikolaos Stergiopulos, and George E. Karniadakis.
        *Residual-based attention and connection to information
        bottleneck theory in PINNs*.
        Computer Methods in Applied Mechanics and Engineering 421 (2024): 116805
        DOI: `10.1016/
        j.cma.2024.116805 <https://doi.org/10.1016/j.cma.2024.116805>`_.

    :Example:

        >>> problem = AllenCahnProblem()
    """

    output_variables = ["u"]
    spatial_domain = CartesianDomain({"x": [-1, 1]})
    temporal_domain = CartesianDomain({"t": [0, 1]})

    domains = {
        "D": spatial_domain.update(temporal_domain),
        "t0": spatial_domain.update(CartesianDomain({"t": 0})),
    }

    conditions = {
        "t0": Condition(domain="t0", equation=Equation(initial_condition)),
    }

    def __init__(self, alpha=1e-4, beta=5):
        """
        Initialization of the :class:`AllenCahnProblem`.

        :param alpha: The diffusion coefficient. Default is 1e-4.
        :type alpha: float | int
        :param beta: The reaction coefficient. Default is 5.0.
        :type beta: float | int
        """
        super().__init__()
        check_consistency(alpha, (float, int))
        check_consistency(beta, (float, int))
        self.alpha = alpha
        self.beta = beta

        self.conditions["D"] = Condition(
            domain="D",
            equation=AllenCahn(alpha=self.alpha, beta=self.beta),
        )
