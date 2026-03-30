"""Module for the InverseProblem class."""

from abc import abstractmethod
import torch
from .abstract_problem import AbstractProblem


class InverseProblem(AbstractProblem):
    """
    Class for defining inverse problems, where the objective is to determine
    unknown parameters through training, based on given data.
    """

    def __init__(self):
        """
        Initialization of the :class:`InverseProblem` class.
        """
        super().__init__()
        # storing unknown_parameters for optimization
        self.unknown_parameters = {}
        for var in self.unknown_variables:
            range_var = self.unknown_parameter_domain._range[var]
            tensor_var = (
                torch.rand(1, requires_grad=True) * range_var[1] + range_var[0]
            )
            self.unknown_parameters[var] = torch.nn.Parameter(tensor_var)

    @abstractmethod
    def unknown_parameter_domain(self):
        """
        The domain of the unknown parameters of the problem.
        """

    @property
    def unknown_variables(self):
        """
        Get the unknown variables of the problem.

        :return: The unknown variables of the problem.
        :rtype: list[str]
        """
        return self.unknown_parameter_domain.variables

    @property
    def unknown_parameters(self):
        """
        Get the unknown parameters of the problem.

        :return: The unknown parameters of the problem.
        :rtype: torch.nn.Parameter
        """
        return self.__unknown_parameters

    @unknown_parameters.setter
    def unknown_parameters(self, value):
        """
        Set the unknown parameters of the problem.

        :param torch.nn.Parameter value: The unknown parameters of the problem.
        """
        self.__unknown_parameters = value
