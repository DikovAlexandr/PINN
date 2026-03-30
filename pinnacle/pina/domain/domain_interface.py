"""Module for the Domain Interface."""

from abc import ABCMeta, abstractmethod


class DomainInterface(metaclass=ABCMeta):
    """
    Abstract interface for all geometric domains.
    """

    @abstractmethod
    def is_inside(self, point, check_border):
        """
        Check if a point is inside the domain.

        :param LabelTensor point: The point to check.
        :param bool check_border: If ``True``, the boundary is considered inside
            the domain.
        :return: Whether the point is inside the domain or not.
        :rtype: bool
        """

    @abstractmethod
    def update(self, domain):
        """
        Update the current domain by adding the labels contained in ``domain``.
        Each new label introduces a new dimension. Only domains of the same type
        can be used for update.

        :param BaseDomain domain: The domain whose labels are to be merged into
            the current one.
        :return: A new domain instance with the merged labels.
        :rtype: DomainInterface
        """

    @abstractmethod
    def sample(self, n, mode, variables):
        """
        The sampling routine.

        :param int n: The number of samples to generate.
        :param str mode: The sampling method.
        :param list[str] variables: The list of variables to sample.
        :return: The sampled points.
        :rtype: LabelTensor
        """

    @abstractmethod
    def partial(self):
        """
        Return the boundary of the domain as a new domain object.

        :return: The boundary of the domain.
        :rtype: DomainInterface
        """

    @property
    @abstractmethod
    def sample_modes(self):
        """
        The list of available sampling modes.

        :return: The list of available sampling modes.
        :rtype: list[str]
        """

    @property
    @abstractmethod
    def variables(self):
        """
        The list of variables of the domain.

        :return: The list of variables of the domain.
        :rtype: list[str]
        """

    @property
    @abstractmethod
    def domain_dict(self):
        """
        The dictionary representing the domain.

        :return: The dictionary representing the domain.
        :rtype: dict
        """

    @property
    @abstractmethod
    def range(self):
        """
        The range variables of the domain.

        :return: The range variables of the domain.
        :rtype: dict
        """

    @property
    @abstractmethod
    def fixed(self):
        """
        The fixed variables of the domain.

        :return: The fixed variables of the domain.
        :rtype: dict
        """
