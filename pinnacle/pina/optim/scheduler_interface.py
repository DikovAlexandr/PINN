"""Module for the PINA Scheduler."""

from abc import ABCMeta, abstractmethod


class Scheduler(metaclass=ABCMeta):
    """
    Abstract base class for defining a scheduler. All specific schedulers should
    inherit form this class and implement the required methods.
    """

    @property
    @abstractmethod
    def instance(self):
        """
        Abstract property to retrieve the scheduler instance.
        """

    @abstractmethod
    def hook(self):
        """
        Abstract method to define the hook logic for the scheduler.
        """
