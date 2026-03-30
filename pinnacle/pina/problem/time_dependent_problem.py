"""Module for the TimeDependentProblem class."""

from abc import abstractmethod

from .abstract_problem import AbstractProblem


class TimeDependentProblem(AbstractProblem):
    """
    Class for defining time-dependent problems, where the system's behavior
    changes with respect to time.
    """

    @abstractmethod
    def temporal_domain(self):
        """
        The temporal domain of the problem.
        """

    @property
    def temporal_variable(self):
        """
        Get the time variable of the problem.

        :return: The time variable of the problem.
        :rtype: list[str]
        """
        return self.temporal_domain.variables
