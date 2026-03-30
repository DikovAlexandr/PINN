"""Module to create and handle domains."""

__all__ = [
    "DomainInterface",
    "BaseDomain",
    "CartesianDomain",
    "EllipsoidDomain",
    "SimplexDomain",
    "OperationInterface",
    "Union",
    "Intersection",
    "Difference",
    "Exclusion",
]

from .domain_interface import DomainInterface
from .base_domain import BaseDomain
from .cartesian_domain import CartesianDomain
from .ellipsoid_domain import EllipsoidDomain
from .simplex_domain import SimplexDomain
from .operation_interface import OperationInterface
from .union import Union
from .intersection import Intersection
from .difference import Difference
from .exclusion import Exclusion
