"""Module for the Base class for domains."""

from copy import deepcopy
from abc import ABCMeta
from .domain_interface import DomainInterface
from ..utils import check_consistency, check_positive_integer


class BaseDomain(DomainInterface, metaclass=ABCMeta):
    """
    Base class for all geometric domains, implementing common functionality.

    All specific domain types should inherit from this class and implement the
    abstract methods of :class:`~pina.domain.domain_interface.DomainInterface`.

    This class is not meant to be instantiated directly.
    """

    def __init__(self, variables_dict=None):
        """
        Initialization of the :class:`BaseDomain` class.

        :param variables_dict: A dictionary where the keys are the variable
            names and the values are the domain extrema. The domain extrema can
            be either a list or tuple with two elements or a single number. If
            the domain extrema is a single number, the variable is fixed to that
            value.
        :type variables_dict: dict | None
        :raises TypeError: If the domain dictionary is not a dictionary.
        :raises ValueError: If the domain dictionary is empty.
        :raises ValueError: If the domain dictionary contains variables with
            invalid ranges.
        :raises ValueError: If the domain dictionary contains values that are
            neither numbers nor lists/tuples of numbers of length 2.
        """
        # Initialize fixed and ranged variables
        self._fixed = {}
        self._range = {}
        invalid = []

        # Skip checks if variables_dict is None -- SimplexDomain case
        if variables_dict is None:
            return

        # Check variables_dict is a dictionary
        if not isinstance(variables_dict, dict):
            raise TypeError(
                "variables_dict must be dict: {name: number | (low, high)}"
            )

        # Check variables_dict is not empty
        if not variables_dict:
            raise ValueError(
                "The dictionary defining the domain cannot be empty."
            )

        # Check consistency
        for v in variables_dict.values():
            check_consistency(v, (int, float))

        # Iterate over variables_dict items
        for k, v in variables_dict.items():

            # Fixed variables
            if isinstance(v, (int, float)):
                self._fixed[k] = v

            # Ranged variables
            elif isinstance(v, (list, tuple)) and len(v) == 2:
                low, high = v
                if low >= high:
                    raise ValueError(
                        f"Invalid range for variable '{k}': "
                        f"low ({low}) >= high ({high})"
                    )
                self._range[k] = (low, high)

            # Save invalid keys
            else:
                invalid.append(k)

        # Raise an error if there are invalid keys
        if invalid:
            raise ValueError(f"Invalid value(s) for key(s): {invalid}")

    def update(self, domain):
        """
        Update the current domain by adding the labels contained in ``domain``.
        Each new label introduces a new dimension. Only domains of the same type
        can be used for update.

        :param BaseDomain domain: The domain whose labels are to be merged
            into the current one.
        :raises TypeError: If the provided domain is not of the same type as
            the current one.
        :return: A new domain instance with the merged labels.
        :rtype: BaseDomain
        """
        # Raise an error if the domain types do not match
        if not isinstance(domain, type(self)):
            raise TypeError(
                f"Cannot update domain of type {type(self)} "
                f"with domain of type {type(domain)}."
            )

        # Update fixed and ranged variables
        updated = deepcopy(self)
        updated.fixed.update(domain.fixed)
        updated.range.update(domain.range)

        return updated

    def _validate_sampling(self, n, mode, variables):
        """
        Validate the sampling settings.

        :param int n: The number of samples to generate.
        :param str mode: The sampling method.
        :param variables: The list of variables to sample. If ``all``, all
            variables are sampled.
        :raises AssertionError: If ``n`` is not a positive integer.
        :raises ValueError: If the sampling mode is invalid.
        :raises ValueError: If ``variables`` is neither ``all``, a string, nor a
            list/tuple of strings.
        :raises ValueError: If any of the specified variables is unknown.
        :return: The validated list of variables to sample.
        :rtype: list[str]
        """
        # Validate n
        check_positive_integer(value=n, strict=True)

        # Validate mode
        if mode not in self.sample_modes:
            raise ValueError(
                f"Invalid sampling mode: {mode}. Available: {self.sample_modes}"
            )

        # Validate variables
        check_consistency(variables, str)
        if variables == "all":
            variables = self.variables
        elif isinstance(variables, str):
            variables = [variables]
        else:
            variables = list(dict.fromkeys(variables))

        # Check for unknown variables
        unknown = [v for v in variables if v not in self.variables]
        if unknown:
            raise ValueError(
                f"Unknown variable(s): {unknown}. Available: {self.variables}"
            )

        return sorted(variables)

    @property
    def sample_modes(self):
        """
        The list of available sampling modes.

        :return: The list of available sampling modes.
        :rtype: list[str]
        """
        return list(self._sample_modes)

    @property
    def variables(self):
        """
        The list of variables of the domain.

        :return: The list of variables of the domain.
        :rtype: list[str]
        """
        return sorted(list(self._fixed.keys()) + list(self._range.keys()))

    @property
    def domain_dict(self):
        """
        The dictionary representing the domain.

        :return: The dictionary representing the domain.
        :rtype: dict
        """
        return {**self._fixed, **self._range}

    @property
    def range(self):
        """
        The range variables of the domain.

        :return: The range variables of the domain.
        :rtype: dict
        """
        return self._range

    @property
    def fixed(self):
        """
        The fixed variables of the domain.

        :return: The fixed variables of the domain.
        :rtype: dict
        """
        return self._fixed
