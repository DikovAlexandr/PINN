"""Module for the AbstractProblem class."""

from abc import ABCMeta, abstractmethod
import warnings
from copy import deepcopy
from ..utils import check_consistency
from ..domain import DomainInterface, CartesianDomain
from ..condition.domain_equation_condition import DomainEquationCondition
from ..label_tensor import LabelTensor
from ..utils import merge_tensors, custom_warning_format


class AbstractProblem(metaclass=ABCMeta):
    """
    Abstract base class for PINA problems. All specific problem types should
    inherit from this class.

    A PINA problem is defined by key components, which typically include output
    variables, conditions, and domains over which the conditions are applied.
    """

    def __init__(self):
        """
        Initialization of the :class:`AbstractProblem` class.
        """
        self._discretised_domains = {}

        # create hook conditions <-> problems
        for condition_name in self.conditions:
            self.conditions[condition_name].problem = self

        # Store in domains dict all the domains object directly passed to
        # ConditionInterface. Done for back compatibility with PINA <0.2
        if not hasattr(self, "domains"):
            self.domains = {}
        for cond_name, cond in self.conditions.items():
            if isinstance(cond, DomainEquationCondition):
                if isinstance(cond.domain, DomainInterface):
                    self.domains[cond_name] = cond.domain
                    cond.domain = cond_name

        self._collected_data = {}

    @property
    def collected_data(self):
        """
        Return the collected data from the problem's conditions. If some domains
        are not sampled, they will not be returned by collected data.

        :return: The collected data. Keys are condition names, and values are
            dictionaries containing the input points and the corresponding
            equations or target points.
        :rtype: dict
        """
        # collect data so far
        self.collect_data()
        # raise warning if some sample data are missing
        if not self.are_all_domains_discretised:
            warnings.formatwarning = custom_warning_format
            warnings.filterwarnings("always", category=RuntimeWarning)
            warning_message = "\n".join(
                [
                    f"""{" " * 13} ---> Domain {key} {
                    "sampled" if key in self.discretised_domains 
                    else
                    "not sampled"}"""
                    for key in self.domains
                ]
            )
            warnings.warn(
                "Some of the domains are still not sampled. Consider calling "
                "problem.discretise_domain function for all domains before "
                "accessing the collected data:\n"
                f"{warning_message}",
                RuntimeWarning,
            )
        return self._collected_data

    #  back compatibility 0.1
    @property
    def input_pts(self):
        """
        Return a dictionary mapping condition names to their corresponding
        input points. If some domains are not sampled, they will not be returned
        and the corresponding condition will be empty.

        :return: The input points of the problem.
        :rtype: dict
        """
        to_return = {}
        for cond_name, data in self.collected_data.items():
            to_return[cond_name] = data["input"]
        return to_return

    @property
    def discretised_domains(self):
        """
        Return a dictionary mapping domains to their corresponding sampled
        points.

        :return: The discretised domains.
        :rtype: dict
        """
        return self._discretised_domains

    def __deepcopy__(self, memo):
        """
        Perform a deep copy of the :class:`AbstractProblem` instance.

        :param dict memo: A dictionary used to track objects already copied
            during the deep copy process to prevent redundant copies.
        :return: A deep copy of the :class:`AbstractProblem` instance.
        :rtype: AbstractProblem
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    @property
    def are_all_domains_discretised(self):
        """
        Check if all the domains are discretised.

        :return: ``True`` if all domains are discretised, ``False`` otherwise.
        :rtype: bool
        """
        return all(
            domain in self.discretised_domains for domain in self.domains
        )

    @property
    def input_variables(self):
        """
        Get the input variables of the problem.

        :return: The input variables of the problem.
        :rtype: list[str]
        """
        variables = []

        if hasattr(self, "spatial_variables"):
            variables += self.spatial_variables
        if hasattr(self, "temporal_variable"):
            variables += self.temporal_variable
        if hasattr(self, "parameters"):
            variables += self.parameters

        return variables

    @input_variables.setter
    def input_variables(self, variables):
        """
        Set the input variables of the AbstractProblem.

        :param list[str] variables: The input variables of the problem.
        :raises RuntimeError: Not implemented.
        """
        raise RuntimeError

    @property
    @abstractmethod
    def output_variables(self):
        """
        Get the output variables of the problem.
        """

    @property
    @abstractmethod
    def conditions(self):
        """
        Get the conditions of the problem.

        :return: The conditions of the problem.
        :rtype: dict
        """
        return self.conditions

    def discretise_domain(
        self, n=None, mode="random", domains="all", sample_rules=None
    ):
        """
        Discretize the problem's domains by sampling a specified number of
        points according to the selected sampling mode.

        :param int n: The number of points to sample.
        :param mode: The sampling method. Default is ``random``.
            Available modes include: random sampling, ``random``;
            latin hypercube sampling, ``latin`` or ``lh``;
            chebyshev sampling, ``chebyshev``; grid sampling ``grid``.
        :param domains: The domains from which to sample. Default is ``all``.
        :type domains: str | list[str]
        :param dict sample_rules: A dictionary defining custom sampling rules
            for input variables. If provided, it must contain a dictionary
            specifying the sampling rule for each variable, overriding the
            ``n`` and ``mode`` arguments. Each key must correspond to the
            input variables from
            :meth:~pina.problem.AbstractProblem.input_variables, and its value
            should be another dictionary with
            two keys: ``n`` (number of points to sample) and ``mode``
            (sampling method). Defaults to None.
        :raises RuntimeError: If both ``n`` and ``sample_rules`` are specified.
        :raises RuntimeError: If neither ``n`` nor ``sample_rules`` are set.

        :Example:
            >>> problem.discretise_domain(n=10, mode='grid')
            >>> problem.discretise_domain(n=10, mode='grid', domains=['gamma1'])
            >>> problem.discretise_domain(
            ...     sample_rules={
            ...         'x': {'n': 10, 'mode': 'grid'},
            ...         'y': {'n': 100, 'mode': 'grid'}
            ...     },
            ...     domains=['D']
            ... )

        .. warning::
            ``random`` is currently the only implemented ``mode`` for all
            geometries, i.e. :class:`~pina.domain.ellipsoid.EllipsoidDomain`,
            :class:`~pina.domain.cartesian.CartesianDomain`,
            :class:`~pina.domain.simplex.SimplexDomain`, and geometry
            compositions :class:`~pina.domain.union_domain.Union`,
            :class:`~pina.domain.difference_domain.Difference`,
            :class:`~pina.domain.exclusion_domain.Exclusion`, and
            :class:`~pina.domain.intersection_domain.Intersection`.
            The modes ``latin`` or ``lh``,  ``chebyshev``, ``grid`` are only
            implemented for :class:`~pina.domain.cartesian.CartesianDomain`.

        .. warning::
            If custom discretisation is applied by setting ``sample_rules`` not
            to ``None``, then the discretised domain must be of class
            :class:`~pina.domain.cartesian.CartesianDomain`
        """

        # check consistecy n, mode, variables, locations
        if sample_rules is not None:
            check_consistency(sample_rules, dict)
        if mode is not None:
            check_consistency(mode, str)
        check_consistency(domains, (list, str))

        # check correct location
        if domains == "all":
            domains = self.domains.keys()
        elif not isinstance(domains, (list)):
            domains = [domains]
        if n is not None and sample_rules is None:
            self._apply_default_discretization(n, mode, domains)
        if n is None and sample_rules is not None:
            self._apply_custom_discretization(sample_rules, domains)
        elif n is not None and sample_rules is not None:
            raise RuntimeError(
                "You can't specify both n and sample_rules at the same time."
            )
        elif n is None and sample_rules is None:
            raise RuntimeError("You have to specify either n or sample_rules.")

    def _apply_default_discretization(self, n, mode, domains):
        """
        Apply default discretization to the problem's domains.

        :param int n: The number of points to sample.
        :param mode: The sampling method.
        :param domains: The domains from which to sample.
        :type domains: str | list[str]
        """
        for domain in domains:
            self.discretised_domains[domain] = (
                self.domains[domain].sample(n, mode).sort_labels()
            )

    def _apply_custom_discretization(self, sample_rules, domains):
        """
        Apply custom discretization to the problem's domains.

        :param dict sample_rules: A dictionary of custom sampling rules.
        :param domains: The domains from which to sample.
        :type domains: str | list[str]
        :raises RuntimeError: If the keys of the sample_rules dictionary are not
            the same as the input variables.
        :raises RuntimeError: If custom discretisation is applied on a domain
            that is not a CartesianDomain.
        """
        if sorted(list(sample_rules.keys())) != sorted(self.input_variables):
            raise RuntimeError(
                "The keys of the sample_rules dictionary must be the same as "
                "the input variables."
            )
        for domain in domains:
            if not isinstance(self.domains[domain], CartesianDomain):
                raise RuntimeError(
                    "Custom discretisation can be applied only on Cartesian "
                    "domains"
                )
            discretised_tensor = []
            for var, rules in sample_rules.items():
                n, mode = rules["n"], rules["mode"]
                points = self.domains[domain].sample(n, mode, var)
                discretised_tensor.append(points)

            self.discretised_domains[domain] = merge_tensors(
                discretised_tensor
            ).sort_labels()

    def add_points(self, new_points_dict):
        """
        Add new points to an already sampled domain.

        :param dict new_points_dict: The dictionary mapping new points to their
            corresponding domain.
        """
        for k, v in new_points_dict.items():
            self.discretised_domains[k] = LabelTensor.vstack(
                [self.discretised_domains[k], v]
            )

    def collect_data(self):
        """
        Aggregate data from the problem's conditions into a single dictionary.
        """
        data = {}
        # Iterate over the conditions and collect data
        for condition_name in self.conditions:
            condition = self.conditions[condition_name]
            # Check if the condition has an domain attribute
            if hasattr(condition, "domain"):
                # Only store the discretisation points if the domain is
                # in the dictionary
                if condition.domain in self.discretised_domains:
                    samples = self.discretised_domains[condition.domain][
                        self.input_variables
                    ]
                    data[condition_name] = {
                        "input": samples,
                        "equation": condition.equation,
                    }
            else:
                # If the condition does not have a domain attribute, store
                # the input and target points
                keys = condition.__slots__
                values = [
                    getattr(condition, name)
                    for name in keys
                    if getattr(condition, name) is not None
                ]
                data[condition_name] = dict(zip(keys, values))
        self._collected_data = data
