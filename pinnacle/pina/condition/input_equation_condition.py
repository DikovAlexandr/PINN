"""Module for the InputEquationCondition class and its subclasses."""

from .condition_interface import ConditionInterface
from ..label_tensor import LabelTensor
from ..graph import Graph
from ..utils import check_consistency
from ..equation.equation_interface import EquationInterface


class InputEquationCondition(ConditionInterface):
    """
    The class :class:`InputEquationCondition` defines a condition based on
    ``input`` data and an ``equation``. This condition is typically used in
    physics-informed problems, where the model is trained to satisfy a given
    ``equation`` through the evaluation of the residual performed at the
    provided ``input``.

    The class automatically selects the appropriate implementation based on
    the type of the ``input`` data. Depending on whether the ``input`` is a
    tensor or graph-based data, one of the following specialized subclasses is
    instantiated:

    - :class:`InputTensorEquationCondition`: For cases where the ``input``
      data is a :class:`~pina.label_tensor.LabelTensor` object.

    - :class:`InputGraphEquationCondition`: For cases where the ``input`` data
      is a :class:`~pina.graph.Graph` object.

    :Example:

    >>> from pina import Condition, LabelTensor
    >>> from pina.equation import Equation
    >>> import torch

    >>> # Equation to be satisfied over the input points: # x^2 + y^2 - 1 = 0
    >>> def dummy_equation(pts):
    ...     return pts["x"]**2 + pts["y"]**2 - 1

    >>> pts = LabelTensor(torch.randn(100, 2), labels=["x", "y"])
    >>> condition = Condition(input=pts, equation=Equation(dummy_equation))
    """

    # Available input data types
    __slots__ = ["input", "equation"]
    _avail_input_cls = (LabelTensor, Graph, list, tuple)
    _avail_equation_cls = EquationInterface

    def __new__(cls, input, equation):
        """
        Instantiate the appropriate subclass of :class:`InputEquationCondition`
        based on the type of ``input`` data.

        :param input: The input data for the condition.
        :type input: LabelTensor | Graph |  list[Graph] | tuple[Graph]
        :param EquationInterface equation: The equation to be satisfied over the
            specified ``input`` data.
        :return: The subclass of InputEquationCondition.
        :rtype: pina.condition.input_equation_condition.
            InputTensorEquationCondition |
            pina.condition.input_equation_condition.InputGraphEquationCondition

        :raises ValueError: If input is not of type :class:`~pina.graph.Graph`
            or :class:`~pina.label_tensor.LabelTensor`.
        """
        if cls != InputEquationCondition:
            return super().__new__(cls)

        # If the input is a Graph object
        if isinstance(input, (Graph, list, tuple)):
            subclass = InputGraphEquationCondition
            cls._check_graph_list_consistency(input)
            subclass._check_label_tensor(input)
            return subclass.__new__(subclass, input, equation)

        # If the input is a LabelTensor
        if isinstance(input, LabelTensor):
            subclass = InputTensorEquationCondition
            return subclass.__new__(subclass, input, equation)

        # If the input is not a LabelTensor or a Graph object raise an error
        raise ValueError(
            "The input data object must be a LabelTensor or a Graph object."
        )

    def __init__(self, input, equation):
        """
        Initialization of the :class:`InputEquationCondition` class.

        :param input: The input data for the condition.
        :type input: LabelTensor | Graph | list[Graph] | tuple[Graph]
        :param EquationInterface equation: The equation to be satisfied over the
            specified input points.

        .. note::

            If ``input`` is a list of :class:`~pina.graph.Graph` all elements in
            the list must share the same structure, with matching keys and
            consistent data types.
        """
        super().__init__()
        self.input = input
        self.equation = equation

    def __setattr__(self, key, value):
        """
        Set the attribute value with type checking.

        :param str key: The attribute name.
        :param any value: The value to set for the attribute.
        """
        if key == "input":
            check_consistency(value, self._avail_input_cls)
            InputEquationCondition.__dict__[key].__set__(self, value)

        elif key == "equation":
            check_consistency(value, self._avail_equation_cls)
            InputEquationCondition.__dict__[key].__set__(self, value)

        elif key in ("_problem"):
            super().__setattr__(key, value)


class InputTensorEquationCondition(InputEquationCondition):
    """
    Specialization of the :class:`InputEquationCondition` class for the case
    where ``input`` is a :class:`~pina.label_tensor.LabelTensor` object.
    """


class InputGraphEquationCondition(InputEquationCondition):
    """
    Specialization of the :class:`InputEquationCondition` class for the case
    where ``input`` is a :class:`~pina.graph.Graph` object.
    """

    @staticmethod
    def _check_label_tensor(input):
        """
        Check if at least one :class:`~pina.label_tensor.LabelTensor` is present
        in the ``input`` object.

        :param input: The input data.
        :type input: torch.Tensor | Graph | list[Graph] | tuple[Graph]
        :raises ValueError: If the input data object does not contain at least
            one LabelTensor.
        """

        # Store the first element: it is sufficient to check this since all
        # elements must have the same type and structure (already checked).
        data = input[0] if isinstance(input, (list, tuple)) else input

        # Check if the input data contains at least one LabelTensor
        for v in data.values():
            if isinstance(v, LabelTensor):
                return

        raise ValueError("The input must contain at least one LabelTensor.")
