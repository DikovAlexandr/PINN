"""Module for the LinearWeighting class."""

from ..loss import WeightingInterface
from ..utils import check_consistency, check_positive_integer


class LinearWeighting(WeightingInterface):
    """
    A weighting scheme that linearly scales weights from initial values to final
    values over a specified number of epochs.
    """

    def __init__(self, initial_weights, final_weights, target_epoch):
        """
        :param dict initial_weights: The weights to be assigned to each loss
            term at the beginning of training. The keys are the conditions and
            the values are the corresponding weights. If a condition is not
            present in the dictionary, the default value (1) is used.
        :param dict final_weights: The weights to be assigned to each loss term
            once the target epoch is reached. The keys are the conditions and
            the values are the corresponding weights. If a condition is not
            present in the dictionary, the default value (1) is used.
        :param int target_epoch: The epoch at which the weights reach their
            final values.
        :raises ValueError: If the keys of the two dictionaries are not
            consistent.
        """
        super().__init__(update_every_n_epochs=1, aggregator="sum")

        # Check consistency
        check_consistency([initial_weights, final_weights], dict)
        check_positive_integer(value=target_epoch, strict=True)

        # Check that the keys of the two dictionaries are the same
        if initial_weights.keys() != final_weights.keys():
            raise ValueError(
                "The keys of the initial_weights and final_weights "
                "dictionaries must be the same."
            )

        # Initialization
        self.initial_weights = initial_weights
        self.final_weights = final_weights
        self.target_epoch = target_epoch

    def weights_update(self, losses):
        """
        Update the weighting scheme based on the given losses.

        :param dict losses: The dictionary of losses.
        :return: The updated weights.
        :rtype: dict
        """
        return {
            condition: self.last_saved_weights().get(
                condition, self.initial_weights.get(condition, 1)
            )
            + (
                self.final_weights.get(condition, 1)
                - self.initial_weights.get(condition, 1)
            )
            / (self.target_epoch)
            for condition in losses.keys()
        }
