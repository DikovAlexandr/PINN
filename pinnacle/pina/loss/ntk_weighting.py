"""Module for Neural Tangent Kernel Class"""

import torch
from .weighting_interface import WeightingInterface
from ..utils import check_consistency, in_range


class NeuralTangentKernelWeighting(WeightingInterface):
    """
    A neural tangent kernel scheme for weighting different losses to
    boost the convergence.

    .. seealso::

        **Original reference**: Wang, Sifan, Xinling Yu, and
        Paris Perdikaris. *When and why PINNs fail to train:
        A neural tangent kernel perspective*. Journal of
        Computational Physics 449 (2022): 110768.
        DOI: `10.1016 <https://doi.org/10.1016/j.jcp.2021.110768>`_.

    """

    def __init__(self, update_every_n_epochs=1, alpha=0.5):
        """
        Initialization of the :class:`NeuralTangentKernelWeighting` class.

        :param int update_every_n_epochs: The number of training epochs between
            weight updates. If set to 1, the weights are updated at every epoch.
            Default is 1.
        :param float alpha: The alpha parameter. Default is 0.5.
        :raises ValueError: If ``alpha`` is not between 0 and 1 (inclusive).
        """
        super().__init__(update_every_n_epochs=update_every_n_epochs)

        # Check consistency
        check_consistency(alpha, float)
        if not in_range(alpha, [0, 1], strict=False):
            raise ValueError("alpha must be in range (0, 1).")

        # Initialize parameters
        self.alpha = alpha
        self.weights = {}

    def weights_update(self, losses):
        """
        Update the weighting scheme based on the given losses.

        :param dict losses: The dictionary of losses.
        :return: The updated weights.
        :rtype: dict
        """
        # Get model parameters and define a dictionary to store the norms
        params = [p for p in self.solver.model.parameters() if p.requires_grad]
        norms = {}

        # Iterate over conditions
        for condition, loss in losses.items():

            # Compute gradients
            grads = torch.autograd.grad(
                loss,
                params,
                retain_graph=True,
                allow_unused=True,
            )

            # Compute norms
            norms[condition] = torch.cat(
                [g.flatten() for g in grads if g is not None]
            ).norm()

        return {
            condition: self.alpha * self.last_saved_weights().get(condition, 1)
            + (1 - self.alpha) * norms[condition] / sum(norms.values())
            for condition in losses
        }
