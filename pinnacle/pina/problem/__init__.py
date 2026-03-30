"""Module for the Problems."""

__all__ = [
    "AbstractProblem",
    "SpatialProblem",
    "TimeDependentProblem",
    "ParametricProblem",
    "InverseProblem",
]

from .abstract_problem import AbstractProblem
from .spatial_problem import SpatialProblem
from .time_dependent_problem import TimeDependentProblem
from .parametric_problem import ParametricProblem
from .inverse_problem import InverseProblem
