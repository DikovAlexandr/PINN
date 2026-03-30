"""Module for the Scalar Weighting."""

from .weighting_interface import WeightingInterface
from ..utils import check_consistency


class ScalarWeighting(WeightingInterface):
    """
    Weighting scheme that assigns a scalar weight to each loss term.
    """

    def __init__(self, weights):
        """
        Initialization of the :class:`ScalarWeighting` class.

        :param weights: The weights to be assigned to each loss term.
            If a single scalar value is provided, it is assigned to all loss
            terms. If a dictionary is provided, the keys are the conditions and
            the values are the weights. If a condition is not present in the
            dictionary, the default value (1) is used.
        :type weights: float | int | dict
        """
        super().__init__(update_every_n_epochs=1, aggregator="sum")

        # Check consistency
        check_consistency([weights], (float, dict, int))

        # Initialization
        if isinstance(weights, dict):
            self.values = weights
            self.default_value_weights = 1
        else:
            self.values = {}
            self.default_value_weights = weights

    def weights_update(self, losses):
        """
        Update the weighting scheme based on the given losses.

        :param dict losses: The dictionary of losses.
        :return: The updated weights.
        :rtype: dict
        """
        return {
            condition: self.values.get(condition, self.default_value_weights)
            for condition in losses.keys()
        }


class _NoWeighting(ScalarWeighting):
    """
    Weighting scheme that does not apply any weighting to the losses.
    """

    def __init__(self):
        """
        Initialization of the :class:`_NoWeighting` class.
        """
        super().__init__(weights=1)
