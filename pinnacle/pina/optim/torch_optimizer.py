"""Module for the PINA Torch Optimizer"""

import torch

from ..utils import check_consistency
from .optimizer_interface import Optimizer


class TorchOptimizer(Optimizer):
    """
    A wrapper class for using PyTorch optimizers.
    """

    def __init__(self, optimizer_class, **kwargs):
        """
        Initialization of the :class:`TorchOptimizer` class.

        :param torch.optim.Optimizer optimizer_class: A
            :class:`torch.optim.Optimizer` class.
        :param dict kwargs: Additional parameters passed to ``optimizer_class``,
            see more
            `here <https://pytorch.org/docs/stable/optim.html#algorithms>`_.
        """
        check_consistency(optimizer_class, torch.optim.Optimizer, subclass=True)

        self.optimizer_class = optimizer_class
        self.kwargs = kwargs
        self._optimizer_instance = None

    def hook(self, parameters):
        """
        Initialize the optimizer instance with the given parameters.

        :param dict parameters: The parameters of the model to be optimized.
        """
        self._optimizer_instance = self.optimizer_class(
            parameters, **self.kwargs
        )

    @property
    def instance(self):
        """
        Get the optimizer instance.

        :return: The optimizer instance.
        :rtype: torch.optim.Optimizer
        """
        return self._optimizer_instance
