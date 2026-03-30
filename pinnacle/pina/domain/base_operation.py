"""Module for all set-based operations Base class."""

from copy import deepcopy
from abc import ABCMeta
from .operation_interface import OperationInterface
from .base_domain import BaseDomain
from ..utils import check_consistency


class BaseOperation(OperationInterface, BaseDomain, metaclass=ABCMeta):
    """
    Base class for all set operation defined on geometric domains, implementing
    common functionality.

    All specific operation types should inherit from this class and implement
    the abstract methods defined in both the following interfaces:
    :class:`~pina.domain.operation_interface.OperationInterface`, and
    :class:`~pina.domain.domain_interface.DomainInterface`.

    This class is not meant to be instantiated directly.
    """

    def __init__(self, geometries):
        """
        Initialization of the :class:`OperationInterface` class.

        :param geometries: The list of domains on which to perform the set
            operation.
        :type geometries: list[BaseDomain] | tuple[BaseDomain]
        :raises TypeError: If geometries is neither a list nor a tuple.
        :raises ValueError: If geometries elements are not instances of
            :class:`~pina.domain.base_domain.BaseDomain`.
        :raises NotImplementedError: If the dimensions of the geometries are not
            consistent.
        """
        super().__init__()
        self.geometries = geometries

    def update(self, domain):
        """
        Update the domain resulting from the operation.

        :param DomainInterface domain: The domain whose labels are to be merged
            into the current one.
        :raises NotImplementedError: If the geometries involved in the operation
            are of different types.
        :raises TypeError: If the passed domain is not of the same type of all
            the geometries involved in the operation.
        :return: A new domain instance with the merged labels.
        :rtype: BaseOperation
        """
        # Check all geometries are of the same type
        domain_type = type(self.geometries[0])
        if not all(isinstance(g, domain_type) for g in self.geometries):
            raise NotImplementedError(
                f"The {self.__class__.__name__} of geometries of different"
                " types does not support the update operation. All geometries"
                " must be of the same type."
            )

        # Check domain type consistency
        if not isinstance(domain, domain_type):
            raise TypeError(
                f"Cannot update the {self.__class__.__name__} of domains of"
                f" type {domain_type} with domain of type {type(domain)}."
            )

        # Update each geometry
        updated = deepcopy(self)
        updated.geometries = [geom.update(domain) for geom in self.geometries]

        return updated

    @property
    def sample_modes(self):
        """
        The list of available sampling modes.

        :return: The list of available sampling modes.
        :rtype: list[str]
        """
        return list(
            set.intersection(
                *map(set, [g.sample_modes for g in self.geometries])
            )
        )

    @property
    def variables(self):
        """
        The list of variables of the domain.

        :return: The list of variables of the domain.
        :rtype: list[str]
        """
        return sorted({v for g in self.geometries for v in g.variables})

    @property
    def domain_dict(self):
        """
        Returns a dictionary representation of the operation domain.

        :return: The dictionary representation of the operation domain.
        :rtype: dict
        """
        return {
            "type": self.__class__.__name__,
            "geometries": [geom.domain_dict for geom in self.geometries],
        }

    @property
    def geometries(self):
        """
        The domains on which to perform the set operation.

        :return: The domains on which to perform the set operation.
        :rtype: list[BaseDomain]
        """
        return self._geometries

    @property
    def range(self):
        """
        The range variables of each geometry.

        :return: The range variables of each geometry.
        :rtype: dict
        """
        return {f"geometry_{i}": g.range for i, g in enumerate(self.geometries)}

    @property
    def fixed(self):
        """
        The fixed variables of each geometry.

        :return: The fixed variables of each geometry.
        :rtype: dict
        """
        return {f"geometry_{i}": g.fixed for i, g in enumerate(self.geometries)}

    @geometries.setter
    def geometries(self, values):
        """
        Setter for the ``geometries`` property.

        :param values: The geometries to be set.
        :type values: list[BaseDomain] | tuple[BaseDomain]
        :raises TypeError: If values is neither a list nor a tuple.
        :raises ValueError: If values elements are not instances of
            :class:`~pina.domain.base_domain.BaseDomain`.
        :raises NotImplementedError: If the dimensions of the geometries are not
            consistent.
        """
        # Check geometries are list or tuple
        if not isinstance(values, (list, tuple)):
            raise TypeError(
                "geometries must be either a list or a tuple of BaseDomain."
            )

        # Check consistency
        check_consistency(values, (BaseDomain, BaseOperation))

        # Check geometries
        for v in values:
            if v.variables != values[0].variables:
                raise NotImplementedError(
                    f"The {self.__class__.__name__} of geometries living in "
                    "different ambient spaces is not well-defined. "
                    "All geometries must share the same dimensions and labels."
                )

        self._geometries = values
