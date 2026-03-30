"""Module for the DataCondition class."""

import torch
from torch_geometric.data import Data
from .condition_interface import ConditionInterface
from ..label_tensor import LabelTensor
from ..graph import Graph


class DataCondition(ConditionInterface):
    """
    The class :class:`DataCondition` defines an unsupervised condition based on
    ``input`` data. This condition is typically used in data-driven problems,
    where the model is trained using a custom unsupervised loss determined by
    the chosen :class:`~pina.solver.solver.SolverInterface`, while leveraging
    the provided data during training. Optional ``conditional_variables`` can be
    specified when the model depends on additional parameters.

    The class automatically selects the appropriate implementation based on the
    type of the ``input`` data. Depending on whether the ``input`` is a tensor
    or graph-based data, one of the following specialized subclasses is
    instantiated:

    - :class:`TensorDataCondition`: For cases where the ``input`` is either a
      :class:`torch.Tensor` or a :class:`~pina.label_tensor.LabelTensor` object.

    - :class:`GraphDataCondition`: For cases where the ``input`` is either a
      :class:`~pina.graph.Graph` or :class:`~torch_geometric.data.Data` object.

    :Example:

    >>> from pina import Condition, LabelTensor
    >>> import torch

    >>> pts = LabelTensor(torch.randn(100, 2), labels=["x", "y"])
    >>> cond_vars = LabelTensor(torch.randn(100, 1), labels=["w"])
    >>> condition = Condition(input=pts, conditional_variables=cond_vars)
    """

    # Available input data types
    __slots__ = ["input", "conditional_variables"]
    _avail_input_cls = (torch.Tensor, LabelTensor, Data, Graph, list, tuple)
    _avail_conditional_variables_cls = (torch.Tensor, LabelTensor)

    def __new__(cls, input, conditional_variables=None):
        """
        Instantiate the appropriate subclass of :class:`DataCondition` based on
        the type of the ``input``.

        :param input: The input data for the condition.
        :type input: torch.Tensor | LabelTensor | Graph |
            Data | list[Graph] | list[Data] | tuple[Graph] | tuple[Data]
        :param conditional_variables: The conditional variables for the
            condition. Default is ``None``.
        :type conditional_variables: torch.Tensor | LabelTensor
        :return: The subclass of DataCondition.
        :rtype: pina.condition.data_condition.TensorDataCondition |
            pina.condition.data_condition.GraphDataCondition
        :raises ValueError: If ``input`` is not of type :class:`torch.Tensor`,
            :class:`~pina.label_tensor.LabelTensor`, :class:`~pina.graph.Graph`,
            or :class:`~torch_geometric.data.Data`.
        """
        if cls != DataCondition:
            return super().__new__(cls)

        # If the input is a tensor
        if isinstance(input, (torch.Tensor, LabelTensor)):
            subclass = TensorDataCondition
            return subclass.__new__(subclass, input, conditional_variables)

        # If the input is a graph
        if isinstance(input, (Graph, Data, list, tuple)):
            cls._check_graph_list_consistency(input)
            subclass = GraphDataCondition
            return subclass.__new__(subclass, input, conditional_variables)

        # If the input is not of the correct type raise an error
        raise ValueError(
            "Invalid input type. Expected one of the following: "
            "torch.Tensor, LabelTensor, Graph, Data or "
            "an iterable of the previous types."
        )

    def __init__(self, input, conditional_variables=None):
        """
        Initialization of the :class:`DataCondition` class.

        :param input: The input data for the condition.
        :type input: torch.Tensor | LabelTensor | Graph | Data | list[Graph] |
            list[Data] | tuple[Graph] | tuple[Data]
        :param conditional_variables: The conditional variables for the
            condition. Default is ``None``.
        :type conditional_variables: torch.Tensor | LabelTensor

        .. note::

            If ``input`` is a list of :class:`~pina.graph.Graph` or
            :class:`~torch_geometric.data.Data`, all elements in
            the list must share the same structure, with matching keys and
            consistent data types.
        """
        super().__init__()
        self.input = input
        self.conditional_variables = conditional_variables


class TensorDataCondition(DataCondition):
    """
    Specialization of the :class:`DataCondition` class for the case where
    ``input`` is either a :class:`~pina.label_tensor.LabelTensor` object or a
    :class:`torch.Tensor` object.
    """


class GraphDataCondition(DataCondition):
    """
    Specialization of the :class:`DataCondition` class for the case where
    ``input`` is either a :class:`~pina.graph.Graph` object or a
    :class:`~torch_geometric.data.Data` object.
    """
