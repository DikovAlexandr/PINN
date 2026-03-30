"""Module for the Condition interface."""

from abc import ABCMeta
from torch_geometric.data import Data
from ..label_tensor import LabelTensor
from ..graph import Graph


class ConditionInterface(metaclass=ABCMeta):
    """
    Abstract base class for PINA conditions. All specific conditions must
    inherit from this interface.

    Refer to :class:`pina.condition.condition.Condition` for a thorough
    description of all available conditions and how to instantiate them.
    """

    def __init__(self):
        """
        Initialization of the :class:`ConditionInterface` class.
        """
        self._problem = None

    @property
    def problem(self):
        """
        Return the problem associated with this condition.

        :return: Problem associated with this condition.
        :rtype: ~pina.problem.abstract_problem.AbstractProblem
        """
        return self._problem

    @problem.setter
    def problem(self, value):
        """
        Set the problem associated with this condition.

        :param pina.problem.abstract_problem.AbstractProblem value: The problem
            to associate with this condition
        """
        self._problem = value

    @staticmethod
    def _check_graph_list_consistency(data_list):
        """
        Check the consistency of the list of Data | Graph objects.
        The following checks are performed:

        - All elements in the list must be of the same type (either
          :class:`~torch_geometric.data.Data` or :class:`~pina.graph.Graph`).

        - All elements in the list must have the same keys.

        - The data type of each tensor must be consistent across all elements.

        - If a tensor is a :class:`~pina.label_tensor.LabelTensor`, its labels
          must also be consistent across all elements.

        :param data_list: The list of Data | Graph objects to check.
        :type data_list: list[Data] | list[Graph] | tuple[Data] | tuple[Graph]
        :raises ValueError: If the input types are invalid.
        :raises ValueError: If all elements in the list do not have the same
            keys.
        :raises ValueError: If the type of each tensor is not consistent across
            all elements in the list.
        :raises ValueError: If the labels of the LabelTensors are not consistent
            across all elements in the list.
        """
        # If the data is a Graph or Data object, perform no checks
        if isinstance(data_list, (Graph, Data)):
            return

        # Check all elements in the list are of the same type
        if not all(isinstance(i, (Graph, Data)) for i in data_list):
            raise ValueError(
                "Invalid input. Please, provide either Data or Graph objects."
            )

        # Store the keys, data types and labels of the first element
        data = data_list[0]
        keys = sorted(list(data.keys()))
        data_types = {name: tensor.__class__ for name, tensor in data.items()}
        labels = {
            name: tensor.labels
            for name, tensor in data.items()
            if isinstance(tensor, LabelTensor)
        }

        # Iterate over the list of Data | Graph objects
        for data in data_list[1:]:

            # Check that all elements in the list have the same keys
            if sorted(list(data.keys())) != keys:
                raise ValueError(
                    "All elements in the list must have the same keys."
                )

            # Iterate over the tensors in the current element
            for name, tensor in data.items():
                # Check that the type of each tensor is consistent
                if tensor.__class__ is not data_types[name]:
                    raise ValueError(
                        f"Data {name} must be a {data_types[name]}, got "
                        f"{tensor.__class__}"
                    )

                # Check that the labels of each LabelTensor are consistent
                if isinstance(tensor, LabelTensor):
                    if tensor.labels != labels[name]:
                        raise ValueError(
                            "LabelTensor must have the same labels"
                        )

    def __getattribute__(self, name):
        """
        Get an attribute from the object.

        :param str name: The name of the attribute to get.
        :return: The requested attribute.
        :rtype: Any
        """
        to_return = super().__getattribute__(name)
        if isinstance(to_return, (Graph, Data)):
            to_return = [to_return]
        return to_return
