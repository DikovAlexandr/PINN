"""Formulation of the diffusion-reaction problem."""

import torch
from ... import Condition
from ...equation import Equation, FixedValue, DiffusionReaction
from ...problem import SpatialProblem, TimeDependentProblem
from ...utils import check_consistency
from ...domain import CartesianDomain


def initial_condition(input_, output_):
    """
    Definition of the initial condition of the diffusion-reaction problem.

    :param LabelTensor input_: The input data of the problem.
    :param LabelTensor output_: The output data of the problem.
    :return: The residual of the initial condition.
    :rtype: LabelTensor
    """
    x = input_.extract("x")
    u_0 = (
        torch.sin(x)
        + (1 / 2) * torch.sin(2 * x)
        + (1 / 3) * torch.sin(3 * x)
        + (1 / 4) * torch.sin(4 * x)
        + (1 / 8) * torch.sin(8 * x)
    )
    return output_ - u_0


class DiffusionReactionProblem(TimeDependentProblem, SpatialProblem):
    r"""
    Implementation of the diffusion-reaction problem in the spatial interval
    :math:`[-\pi, \pi]` and temporal interval :math:`[0, 1]`.

    .. seealso::

        **Original reference**: Si, Chenhao, et al. *Complex Physics-Informed
        Neural Network.* arXiv preprint arXiv:2502.04917 (2025).
        DOI: `arXiv:2502.04917 <https://arxiv.org/abs/2502.04917>`_.

    :Example:

        >>> problem = DiffusionReactionProblem()
    """

    output_variables = ["u"]
    spatial_domain = CartesianDomain({"x": [-torch.pi, torch.pi]})
    temporal_domain = CartesianDomain({"t": [0, 1]})

    domains = {
        "D": spatial_domain.update(temporal_domain),
        "boundary": spatial_domain.partial().update(temporal_domain),
        "t0": spatial_domain.update(CartesianDomain({"t": 0})),
    }

    conditions = {
        "boundary": Condition(domain="boundary", equation=FixedValue(0.0)),
        "t0": Condition(domain="t0", equation=Equation(initial_condition)),
    }

    def __init__(self, alpha=1e-4):
        """
        Initialization of the :class:`DiffusionReactionProblem`.

        :param alpha: The diffusion coefficient. Default is 1e-4.
        :type alpha: float | int
        """
        super().__init__()
        check_consistency(alpha, (float, int))
        self.alpha = alpha

        def forcing_term(input_):
            """
            Implementation of the forcing term.
            """
            # Extract spatial and temporal variables
            spatial_d = [di for di in input_.labels if di != "t"]
            x = input_.extract(spatial_d)
            t = input_.extract("t")

            return torch.exp(-t) * (
                1.5 * torch.sin(2 * x)
                + (8 / 3) * torch.sin(3 * x)
                + (15 / 4) * torch.sin(4 * x)
                + (63 / 8) * torch.sin(8 * x)
            )

        self.conditions["D"] = Condition(
            domain="D",
            equation=DiffusionReaction(self.alpha, forcing_term),
        )

    def solution(self, pts):
        """
        Implementation of the analytical solution of the diffusion-reaction
        problem.

        :param LabelTensor pts: Points where the solution is evaluated.
        :return: The analytical solution of the diffusion-reaction problem.
        :rtype: LabelTensor
        """
        t = pts.extract("t")
        x = pts.extract("x")
        sol = torch.exp(-t) * (
            torch.sin(x)
            + (1 / 2) * torch.sin(2 * x)
            + (1 / 3) * torch.sin(3 * x)
            + (1 / 4) * torch.sin(4 * x)
            + (1 / 8) * torch.sin(8 * x)
        )
        sol.labels = self.output_variables
        return sol
