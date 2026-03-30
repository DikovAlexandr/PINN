"""Module for the Condition class."""

from .data_condition import DataCondition
from .domain_equation_condition import DomainEquationCondition
from .input_equation_condition import InputEquationCondition
from .input_target_condition import InputTargetCondition


class Condition:
    """
    The :class:`Condition` class is a core component of the PINA framework that
    provides a unified interface to define heterogeneous constraints that must
    be satisfied by a :class:`~pina.problem.abstract_problem.AbstractProblem`.

    It encapsulates all types of constraints - physical, boundary, initial, or
    data-driven - that the solver must satisfy during training. The specific
    behavior is inferred from the arguments passed to the constructor.

    Multiple types of conditions can be used within the same problem, allowing
    for a high degree of flexibility in defining complex problems.

    The :class:`Condition` class behavior specializes internally based on the
    arguments provided during instantiation. Depending on the specified keyword
    arguments, the class automatically selects the appropriate internal
    implementation.


    Available `Condition` types:

    - :class:`~pina.condition.input_target_condition.InputTargetCondition`:
      represents a supervised condition defined by both ``input`` and ``target``
      data. The model is trained to reproduce the ``target`` values given the
      ``input``. Supported data types include :class:`torch.Tensor`,
      :class:`~pina.label_tensor.LabelTensor`, :class:`~pina.graph.Graph`, or
      :class:`~torch_geometric.data.Data`.
      The class automatically selects the appropriate implementation based on
      the types of ``input`` and ``target``.

    - :class:`~pina.condition.domain_equation_condition.DomainEquationCondition`
      : represents a general physics-informed condition defined by a ``domain``
      and an ``equation``. The model learns to minimize the equation residual
      through evaluations performed at points sampled from the specified domain.

    - :class:`~pina.condition.input_equation_condition.InputEquationCondition`:
      represents a general physics-informed condition defined by ``input``
      points and an ``equation``. The model learns to minimize the equation
      residual through evaluations performed at the provided ``input``.
      Supported data types for the ``input`` include
      :class:`~pina.label_tensor.LabelTensor` or :class:`~pina.graph.Graph`.
      The class automatically selects the appropriate implementation based on
      the types of the ``input``.

    - :class:`~pina.condition.data_condition.DataCondition`: represents an
      unsupervised, data-driven condition defined by the ``input`` only.
      The model is trained using a custom unsupervised loss determined by the
      chosen :class:`~pina.solver.solver.SolverInterface`, while leveraging the
      provided data during training. Optional ``conditional_variables`` can be
      specified when the model depends on additional parameters.
      Supported data types include :class:`torch.Tensor`,
      :class:`~pina.label_tensor.LabelTensor`, :class:`~pina.graph.Graph`, or
      :class:`~torch_geometric.data.Data`.
      The class automatically selects the appropriate implementation based on
      the type of the ``input``.

    .. note::

        The user should always instantiate :class:`Condition` directly, without
        manually creating subclass instances. Please refer to the specific
        :class:`Condition` classes for implementation details.

    :Example:

    >>> from pina import Condition

    >>> # Example of InputTargetCondition signature
    >>> condition = Condition(input=input, target=target)

    >>> # Example of DomainEquationCondition signature
    >>> condition = Condition(domain=domain, equation=equation)

    >>> # Example of InputEquationCondition signature
    >>> condition = Condition(input=input, equation=equation)

    >>> # Example of DataCondition signature
    >>> condition = Condition(input=data, conditional_variables=cond_vars)
    """

    # Combine all possible keyword arguments from the different Condition types
    __slots__ = list(
        set(
            InputTargetCondition.__slots__
            + InputEquationCondition.__slots__
            + DomainEquationCondition.__slots__
            + DataCondition.__slots__
        )
    )

    def __new__(cls, *args, **kwargs):
        """
        Instantiate the appropriate :class:`Condition` object based on the
        keyword arguments passed.

        :param tuple args: The positional arguments (should be empty).
        :param dict kwargs: The keyword arguments corresponding to the
            parameters of the specific :class:`Condition` type to instantiate.
        :raises ValueError: If unexpected positional arguments are provided.
        :raises ValueError: If the keyword arguments are invalid.
        :return: The appropriate :class:`Condition` object.
        :rtype: ConditionInterface
        """
        # Check keyword arguments
        if len(args) != 0:
            raise ValueError(
                "Condition takes only the following keyword "
                f"arguments: {Condition.__slots__}."
            )

        # Class specialization based on keyword arguments
        sorted_keys = sorted(kwargs.keys())

        # Input - Target Condition
        if sorted_keys == sorted(InputTargetCondition.__slots__):
            return InputTargetCondition(**kwargs)

        # Input - Equation Condition
        if sorted_keys == sorted(InputEquationCondition.__slots__):
            return InputEquationCondition(**kwargs)

        # Domain - Equation Condition
        if sorted_keys == sorted(DomainEquationCondition.__slots__):
            return DomainEquationCondition(**kwargs)

        # Data Condition
        if (
            sorted_keys == sorted(DataCondition.__slots__)
            or sorted_keys[0] == DataCondition.__slots__[0]
        ):
            return DataCondition(**kwargs)

        # Invalid keyword arguments
        raise ValueError(f"Invalid keyword arguments {kwargs.keys()}.")
