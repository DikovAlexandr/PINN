"""Module for the Weighting Interface."""

from abc import ABCMeta, abstractmethod
from typing import final
from ..utils import check_positive_integer, is_function

_AGGREGATE_METHODS = {"sum": sum, "mean": lambda x: sum(x) / len(x)}


class WeightingInterface(metaclass=ABCMeta):
    """
    Abstract base class for all loss weighting schemas. All weighting schemas
    should inherit from this class.
    """

    def __init__(self, update_every_n_epochs=1, aggregator="sum"):
        """
        Initialization of the :class:`WeightingInterface` class.

        :param int update_every_n_epochs: The number of training epochs between
            weight updates. If set to 1, the weights are updated at every epoch.
            This parameter is ignored by static weighting schemes. Default is 1.
        :param aggregator: The aggregation method. Either:
            - 'sum' → torch.sum
            - 'mean' → torch.mean
            - callable → custom aggregation function
        :type aggregator: str | Callable
        """
        # Check consistency
        check_positive_integer(value=update_every_n_epochs, strict=True)

        # Aggregation
        if isinstance(aggregator, str):
            if aggregator not in _AGGREGATE_METHODS:
                raise ValueError(
                    f"Invalid aggregator '{aggregator}'. Must be one of "
                    f"{list(_AGGREGATE_METHODS.keys())}."
                )
            aggregator = _AGGREGATE_METHODS[aggregator]

        elif not is_function(aggregator):
            raise TypeError(
                f"Aggregator must be either a string or a callable, "
                f"got {type(aggregator).__name__}."
            )

        # Initialization
        self._solver = None
        self.update_every_n_epochs = update_every_n_epochs
        self.aggregator_fn = aggregator
        self._saved_weights = {}

    @abstractmethod
    def weights_update(self, losses):
        """
        Update the weighting scheme based on the given losses.

        This method must be implemented by subclasses. Its role is to update the
        values of the weights. The updated weights will then be used by
        :meth:`aggregate` to compute the final aggregated loss.

        :param dict losses: The dictionary of losses.
        :return: The updated weights.
        :rtype: dict
        """

    @final
    def aggregate(self, losses):
        """
        Update the weights (if needed) and aggregate the given losses.

        This method first checks whether the loss weights need to be updated
        based on the current epoch and the ``update_every_n_epochs`` setting.
        If an update is required, it calls :meth:`weights_update` to refresh the
        weights. Afterwards, it aggregates the (weighted) losses into a single
        scalar tensor using the configured aggregator function. This method must
        not be overridden.

        :param dict losses: The dictionary of losses.
        :return: The aggregated loss tensor.
        :rtype: torch.Tensor
        """
        # Update weights
        if self.solver.trainer.current_epoch % self.update_every_n_epochs == 0:
            self._saved_weights = self.weights_update(losses)

        # Aggregate. Using direct indexing instead of .get() ensures that a
        # KeyError is raised if the expected condition is missing from the dict.
        return self.aggregator_fn(
            self._saved_weights[condition] * loss
            for condition, loss in losses.items()
        )

    def last_saved_weights(self):
        """
        Get the last saved weights.

        :return: The last saved weights.
        :rtype: dict
        """
        return self._saved_weights

    @property
    def solver(self):
        """
        The solver employing this weighting schema.

        :return: The solver.
        :rtype: :class:`~pina.solver.SolverInterface`
        """
        return self._solver
