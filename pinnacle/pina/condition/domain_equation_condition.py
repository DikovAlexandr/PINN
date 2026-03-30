"""Module for the DomainEquationCondition class."""

from .condition_interface import ConditionInterface
from ..utils import check_consistency
from ..domain import DomainInterface
from ..equation.equation_interface import EquationInterface


class DomainEquationCondition(ConditionInterface):
    """
    The class :class:`DomainEquationCondition` defines a condition based on a
    ``domain`` and an ``equation``. This condition is typically used in
    physics-informed problems, where the model is trained to satisfy a given
    ``equation`` over a specified ``domain``. The ``domain`` is used to sample
    points where the ``equation`` residual is evaluated and minimized during
    training.

    :Example:

    >>> from pina.domain import CartesianDomain
    >>> from pina.equation import Equation
    >>> from pina import Condition

    >>> # Equation to be satisfied over the domain: # x^2 + y^2 - 1 = 0
    >>> def dummy_equation(pts):
    ...     return pts["x"]**2 + pts["y"]**2 - 1

    >>> domain = CartesianDomain({"x": [0, 1], "y": [0, 1]})
    >>> condition = Condition(domain=domain, equation=Equation(dummy_equation))
    """

    # Available slots
    __slots__ = ["domain", "equation"]

    def __init__(self, domain, equation):
        """
        Initialization of the :class:`DomainEquationCondition` class.

        :param DomainInterface domain: The domain over which the equation is
            defined.
        :param EquationInterface equation: The equation to be satisfied over the
            specified domain.
        """
        super().__init__()
        self.domain = domain
        self.equation = equation

    def __setattr__(self, key, value):
        """
        Set the attribute value with type checking.

        :param str key: The attribute name.
        :param any value: The value to set for the attribute.
        """
        if key == "domain":
            check_consistency(value, (DomainInterface, str))
            DomainEquationCondition.__dict__[key].__set__(self, value)

        elif key == "equation":
            check_consistency(value, (EquationInterface))
            DomainEquationCondition.__dict__[key].__set__(self, value)

        elif key in ("_problem"):
            super().__setattr__(key, value)
