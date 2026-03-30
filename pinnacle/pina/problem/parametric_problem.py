"""Module for the ParametricProblem class."""

from abc import abstractmethod

from .abstract_problem import AbstractProblem


class ParametricProblem(AbstractProblem):
    """
    Class for defining parametric problems, where certain input variables are
    treated as parameters that can vary, allowing the model to adapt to
    different scenarios based on the chosen parameters.
    """

    @abstractmethod
    def parameter_domain(self):
        """
        The domain of the parameters of the problem.
        """

    @property
    def parameters(self):
        """
        Get the parameters of the problem.

        :return: The parameters of the problem.
        :rtype: list[str]
        """
        return self.parameter_domain.variables
