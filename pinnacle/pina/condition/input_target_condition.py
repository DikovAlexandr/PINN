"""
This module contains condition classes for supervised learning tasks.
"""

import torch
from torch_geometric.data import Data
from ..label_tensor import LabelTensor
from ..graph import Graph
from .condition_interface import ConditionInterface


class InputTargetCondition(ConditionInterface):
    """
    The :class:`InputTargetCondition` class represents a supervised condition
    defined by both ``input`` and ``target`` data. The model is trained to
    reproduce the ``target`` values given the ``input``. Supported data types
    include :class:`torch.Tensor`, :class:`~pina.label_tensor.LabelTensor`,
    :class:`~pina.graph.Graph`, or :class:`~torch_geometric.data.Data`.

    The class automatically selects the appropriate implementation based on
    the types of ``input`` and ``target``. Depending on whether the ``input``
    and ``target`` are tensors or graph-based data, one of the following
    specialized subclasses is instantiated:

    - :class:`TensorInputTensorTargetCondition`: For cases where both ``input``
      and ``target`` data are either :class:`torch.Tensor` or
      :class:`~pina.label_tensor.LabelTensor`.

    - :class:`TensorInputGraphTargetCondition`: For cases where ``input`` is
      either a :class:`torch.Tensor` or :class:`~pina.label_tensor.LabelTensor`
      and ``target`` is either a :class:`~pina.graph.Graph` or a
      :class:`torch_geometric.data.Data`.

    - :class:`GraphInputTensorTargetCondition`: For cases where ``input`` is
      either a :class:`~pina.graph.Graph` or :class:`torch_geometric.data.Data`
      and ``target`` is either a :class:`torch.Tensor` or a
      :class:`~pina.label_tensor.LabelTensor`.

    - :class:`GraphInputGraphTargetCondition`: For cases where both ``input``
      and ``target`` are either :class:`~pina.graph.Graph` or
      :class:`torch_geometric.data.Data`.

    :Example:

    >>> from pina import Condition, LabelTensor
    >>> from pina.graph import Graph
    >>> import torch

    >>> pos = LabelTensor(torch.randn(100, 2), labels=["x", "y"])
    >>> edge_index = torch.randint(0, 100, (2, 300))
    >>> graph = Graph(pos=pos, edge_index=edge_index)

    >>> input = LabelTensor(torch.randn(100, 2), labels=["x", "y"])
    >>> condition = Condition(input=input, target=graph)
    """

    # Available input and target data types
    __slots__ = ["input", "target"]
    _avail_input_cls = (torch.Tensor, LabelTensor, Data, Graph, list, tuple)
    _avail_output_cls = (torch.Tensor, LabelTensor, Data, Graph, list, tuple)

    def __new__(cls, input, target):
        """
        Instantiate the appropriate subclass of :class:`InputTargetCondition`
        based on the types of both ``input`` and ``target`` data.

        :param input: The input data for the condition.
        :type input: torch.Tensor | LabelTensor | Graph | Data | list[Graph] |
            list[Data] | tuple[Graph] | tuple[Data]
        :param target: The target data for the condition.
        :type target: torch.Tensor | LabelTensor | Graph | Data | list[Graph] |
            list[Data] | tuple[Graph] | tuple[Data]
        :return: The subclass of InputTargetCondition.
        :rtype: pina.condition.input_target_condition.
            TensorInputTensorTargetCondition |
            pina.condition.input_target_condition.
            TensorInputGraphTargetCondition |
            pina.condition.input_target_condition.
            GraphInputTensorTargetCondition |
            pina.condition.input_target_condition.GraphInputGraphTargetCondition

        :raises ValueError: If ``input`` and/or ``target`` are not of type
            :class:`torch.Tensor`, :class:`~pina.label_tensor.LabelTensor`,
            :class:`~pina.graph.Graph`, or :class:`~torch_geometric.data.Data`.
        """
        if cls != InputTargetCondition:
            return super().__new__(cls)

        # Tensor - Tensor
        if isinstance(input, (torch.Tensor, LabelTensor)) and isinstance(
            target, (torch.Tensor, LabelTensor)
        ):
            subclass = TensorInputTensorTargetCondition
            return subclass.__new__(subclass, input, target)

        # Tensor - Graph
        if isinstance(input, (torch.Tensor, LabelTensor)) and isinstance(
            target, (Graph, Data, list, tuple)
        ):
            cls._check_graph_list_consistency(target)
            subclass = TensorInputGraphTargetCondition
            return subclass.__new__(subclass, input, target)

        # Graph - Tensor
        if isinstance(input, (Graph, Data, list, tuple)) and isinstance(
            target, (torch.Tensor, LabelTensor)
        ):
            cls._check_graph_list_consistency(input)
            subclass = GraphInputTensorTargetCondition
            return subclass.__new__(subclass, input, target)

        # Graph - Graph
        if isinstance(input, (Graph, Data, list, tuple)) and isinstance(
            target, (Graph, Data, list, tuple)
        ):
            cls._check_graph_list_consistency(input)
            cls._check_graph_list_consistency(target)
            subclass = GraphInputGraphTargetCondition
            return subclass.__new__(subclass, input, target)

        # If the input and/or target are not of the correct type raise an error
        raise ValueError(
            "Invalid input | target types."
            "Please provide either torch_geometric.data.Data, Graph, "
            "LabelTensor or torch.Tensor objects."
        )

    def __init__(self, input, target):
        """
        Initialization of the :class:`InputTargetCondition` class.

        :param input: The input data for the condition.
        :type input: torch.Tensor | LabelTensor | Graph | Data | list[Graph] |
            list[Data] | tuple[Graph] | tuple[Data]
        :param target: The target data for the condition.
        :type target: torch.Tensor | LabelTensor | Graph | Data | list[Graph] |
            list[Data] | tuple[Graph] | tuple[Data]

        .. note::

            If either ``input`` or ``target`` is a list of
            :class:`~pina.graph.Graph` or :class:`~torch_geometric.data.Data`
            objects, all elements in the list must share the same structure,
            with matching keys and consistent data types.
        """
        super().__init__()
        self._check_input_target_len(input, target)
        self.input = input
        self.target = target

    @staticmethod
    def _check_input_target_len(input, target):
        """
        Check that the length of the input and target lists are the same.

        :param input: The input data.
        :type input: torch.Tensor | LabelTensor | Graph | Data | list[Graph] |
            list[Data] | tuple[Graph] | tuple[Data]
        :param target: The target data.
        :type target: torch.Tensor | LabelTensor | Graph | Data | list[Graph] |
            list[Data] | tuple[Graph] | tuple[Data]
        :raises ValueError: If the lengths of the input and target lists do not
            match.
        """
        if isinstance(input, (Graph, Data)) or isinstance(
            target, (Graph, Data)
        ):
            return

        # Raise an error if the lengths of the input and target do not match
        if len(input) != len(target):
            raise ValueError(
                "The input and target lists must have the same length."
            )


class TensorInputTensorTargetCondition(InputTargetCondition):
    """
    Specialization of the :class:`InputTargetCondition` class for the case where
    both ``input`` and ``target`` are :class:`torch.Tensor` or
    :class:`~pina.label_tensor.LabelTensor` objects.
    """


class TensorInputGraphTargetCondition(InputTargetCondition):
    """
    Specialization of the :class:`InputTargetCondition` class for the case where
    ``input`` is either a :class:`torch.Tensor` or a
    :class:`~pina.label_tensor.LabelTensor` object and ``target`` is either a
    :class:`~pina.graph.Graph` or a :class:`torch_geometric.data.Data` object.
    """


class GraphInputTensorTargetCondition(InputTargetCondition):
    """
    Specialization of the :class:`InputTargetCondition` class for the case where
    ``input`` is either a :class:`~pina.graph.Graph` or
    :class:`torch_geometric.data.Data` object and ``target`` is either a
    :class:`torch.Tensor` or a :class:`~pina.label_tensor.LabelTensor` object.
    """


class GraphInputGraphTargetCondition(InputTargetCondition):
    """
    Specialization of the :class:`InputTargetCondition` class for the case where
    both ``input`` and ``target`` are either :class:`~pina.graph.Graph` or
    :class:`torch_geometric.data.Data` objects.
    """
