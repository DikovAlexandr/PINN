"""Module for the SpatialProblem class."""

from abc import abstractmethod

from .abstract_problem import AbstractProblem


class SpatialProblem(AbstractProblem):
    """
    Class for defining spatial problems, where the problem domain is defined in
    terms of spatial variables.
    """

    @abstractmethod
    def spatial_domain(self):
        """
        The spatial domain of the problem.
        """

    @property
    def spatial_variables(self):
        """
        Get the spatial input variables of the problem.

        :return: The spatial input variables of the problem.
        :rtype: list[str]
        """
        return self.spatial_domain.variables
