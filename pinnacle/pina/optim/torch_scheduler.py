"""Module for the PINA Torch Optimizer"""

try:
    from torch.optim.lr_scheduler import LRScheduler  # torch >= 2.0
except ImportError:
    from torch.optim.lr_scheduler import (
        _LRScheduler as LRScheduler,
    )  # torch < 2.0

from ..utils import check_consistency
from .optimizer_interface import Optimizer
from .scheduler_interface import Scheduler


class TorchScheduler(Scheduler):
    """
    A wrapper class for using PyTorch schedulers.
    """

    def __init__(self, scheduler_class, **kwargs):
        """
        Initialization of the :class:`TorchScheduler` class.

        :param torch.optim.LRScheduler scheduler_class: A
            :class:`torch.optim.LRScheduler` class.
        :param dict kwargs: Additional parameters passed to ``scheduler_class``,
            see more
            `here <https://pytorch.org/docs/stable/optim.html#algorithms>_`.
        """
        check_consistency(scheduler_class, LRScheduler, subclass=True)

        self.scheduler_class = scheduler_class
        self.kwargs = kwargs
        self._scheduler_instance = None

    def hook(self, optimizer):
        """
        Initialize the scheduler instance with the given parameters.

        :param dict parameters: The parameters of the optimizer.
        """
        check_consistency(optimizer, Optimizer)
        self._scheduler_instance = self.scheduler_class(
            optimizer.instance, **self.kwargs
        )

    @property
    def instance(self):
        """
        Get the scheduler instance.

        :return: The scheduelr instance.
        :rtype: torch.optim.LRScheduler
        """
        return self._scheduler_instance
