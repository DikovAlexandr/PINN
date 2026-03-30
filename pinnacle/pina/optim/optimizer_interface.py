"""Module for the PINA Optimizer."""

from abc import ABCMeta, abstractmethod


class Optimizer(metaclass=ABCMeta):
    """
    Abstract base class for defining an optimizer. All specific optimizers
    should inherit form this class and implement the required methods.
    """

    @property
    @abstractmethod
    def instance(self):
        """
        Abstract property to retrieve the optimizer instance.
        """

    @abstractmethod
    def hook(self):
        """
        Abstract method to define the hook logic for the optimizer.
        """
