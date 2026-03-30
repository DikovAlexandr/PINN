"""Module for Self-Adaptive Weighting class."""

import torch
from .weighting_interface import WeightingInterface


class SelfAdaptiveWeighting(WeightingInterface):
    """
    A self-adaptive weighting scheme to tackle the imbalance among the loss
    components. This formulation equalizes the gradient norms of the losses,
    preventing bias toward any particular term during training.

    .. seealso::

        **Original reference**:
        Wang, S., Sankaran, S., Stinis., P., Perdikaris, P. (2025).
        *Simulating Three-dimensional Turbulence with Physics-informed Neural
        Networks*.
        DOI: `arXiv preprint arXiv:2507.08972.
        <https://arxiv.org/abs/2507.08972>`_

    """

    def __init__(self, update_every_n_epochs=1):
        """
        Initialization of the :class:`SelfAdaptiveWeighting` class.

        :param int update_every_n_epochs: The number of training epochs between
            weight updates. If set to 1, the weights are updated at every epoch.
            Default is 1.
        """
        super().__init__(update_every_n_epochs=update_every_n_epochs)

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

        # Update the weights
        return {
            condition: sum(norms.values()) / norms[condition]
            for condition in losses
        }
