"""Module for the PINA dataset classes."""

from abc import abstractmethod, ABC
from torch.utils.data import Dataset
from torch_geometric.data import Data
from ..graph import Graph, LabelBatch


class PinaDatasetFactory:
    """
    Factory class for the PINA dataset.

    Depending on the data type inside the conditions, it instanciate an object
    belonging to the appropriate subclass of 
    :class:`~pina.data.dataset.PinaDataset`. The possible subclasses are:

    - :class:`~pina.data.dataset.PinaTensorDataset`, for handling \
        :class:`torch.Tensor` and :class:`~pina.label_tensor.LabelTensor` data.
    - :class:`~pina.data.dataset.PinaGraphDataset`, for handling \
        :class:`~pina.graph.Graph` and :class:`~torch_geometric.data.Data` data.
    """

    def __new__(cls, conditions_dict, **kwargs):
        """
        Instantiate the appropriate subclass of
        :class:`~pina.data.dataset.PinaDataset`.

        If a graph is present in the conditions, returns a
        :class:`~pina.data.dataset.PinaGraphDataset`, otherwise returns a
        :class:`~pina.data.dataset.PinaTensorDataset`.

        :param dict conditions_dict: Dictionary containing all the conditions
            to be included in the dataset instance.
        :return: A subclass of :class:`~pina.data.dataset.PinaDataset`.
        :rtype: PinaTensorDataset | PinaGraphDataset

        :raises ValueError: If an empty dictionary is provided.
        """

        # Check if conditions_dict is empty
        if len(conditions_dict) == 0:
            raise ValueError("No conditions provided")

        # Check is a Graph is present in the conditions
        is_graph = cls._is_graph_dataset(conditions_dict)
        if is_graph:
            # If a Graph is present, return a PinaGraphDataset
            return PinaGraphDataset(conditions_dict, **kwargs)
        # If no Graph is present, return a PinaTensorDataset
        return PinaTensorDataset(conditions_dict, **kwargs)

    @staticmethod
    def _is_graph_dataset(conditions_dict):
        """
        Check if a graph is present in the conditions (at least one time).

        :param conditions_dict: Dictionary containing the conditions.
        :type conditions_dict: dict
        :return: True if a graph is present in the conditions, False otherwise.
        :rtype: bool
        """

        # Iterate over the conditions dictionary
        for v in conditions_dict.values():
            # Iterate over the values of the current condition
            for cond in v.values():
                # Check if the current value is a list of Data objects
                if isinstance(cond, (Data, Graph, list, tuple)):
                    return True
        return False


class PinaDataset(Dataset, ABC):
    """
    Abstract class for the PINA dataset which extends the PyTorch
    :class:`~torch.utils.data.Dataset` class. It defines the common interface
    for :class:`~pina.data.dataset.PinaTensorDataset` and
    :class:`~pina.data.dataset.PinaGraphDataset` classes.
    """

    def __init__(
        self, conditions_dict, max_conditions_lengths, automatic_batching
    ):
        """
        Initialize the instance by storing the conditions dictionary, the
        maximum number of items per conditions to consider, and the automatic
        batching flag.

        :param dict conditions_dict: A dictionary mapping condition names to
            their respective data. Each key represents a condition name, and the
            corresponding value is a dictionary containing the associated data.
        :param dict max_conditions_lengths: Maximum number of data points that
            can be included in a single batch per condition.
        :param bool automatic_batching: Indicates whether PyTorch automatic
            batching is enabled in
            :class:`~pina.data.data_module.PinaDataModule`.
        """

        # Store the conditions dictionary
        self.conditions_dict = conditions_dict
        # Store the maximum number of conditions to consider
        self.max_conditions_lengths = max_conditions_lengths
        # Store length of each condition
        self.conditions_length = {
            k: len(v["input"]) for k, v in self.conditions_dict.items()
        }
        # Store the maximum length of the dataset
        self.length = max(self.conditions_length.values())
        # Dynamically set the getitem function based on automatic batching
        if automatic_batching:
            self._getitem_func = self._getitem_int
        else:
            self._getitem_func = self._getitem_dummy

    def _get_max_len(self):
        """
        Returns the length of the longest condition in the dataset.

        :return: Length of the longest condition in the dataset.
        :rtype: int
        """

        max_len = 0
        for condition in self.conditions_dict.values():
            max_len = max(max_len, len(condition["input"]))
        return max_len

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self._getitem_func(idx)

    def _getitem_dummy(self, idx):
        """
        Return the index itself. This is used when automatic batching is
        disabled to postpone the data retrieval to the dataloader.

        :param int idx: Index.
        :return: Index.
        :rtype: int
        """

        # If automatic batching is disabled, return the data at the given index
        return idx

    def _getitem_int(self, idx):
        """
        Return the data at the given index in the dataset. This is used when
        automatic batching is enabled.

        :param int idx: Index.
        :return: A dictionary containing the data at the given index.
        :rtype: dict
        """

        # If automatic batching is enabled, return the data at the given index
        return {
            k: {k_data: v[k_data][idx % len(v["input"])] for k_data in v.keys()}
            for k, v in self.conditions_dict.items()
        }

    def get_all_data(self):
        """
        Return all data in the dataset.

        :return: A dictionary containing all the data in the dataset.
        :rtype: dict
        """
        to_return_dict = {}
        for condition, data in self.conditions_dict.items():
            len_condition = len(
                data["input"]
            )  # Length of the current condition
            to_return_dict[condition] = self._retrive_data(
                data, list(range(len_condition))
            )  # Retrieve the data from the current condition
        return to_return_dict

    def fetch_from_idx_list(self, idx):
        """
        Return data from the dataset given a list of indices.

        :param list[int] idx: List of indices.
        :return: A dictionary containing the data at the given indices.
        :rtype: dict
        """

        to_return_dict = {}
        for condition, data in self.conditions_dict.items():
            # Get the indices for the current condition
            cond_idx = idx[: self.max_conditions_lengths[condition]]
            # Get the length of the current condition
            condition_len = self.conditions_length[condition]
            # If the length of the dataset is greater than the length of the
            # current condition, repeat the indices
            if self.length > condition_len:
                cond_idx = [idx % condition_len for idx in cond_idx]
            # Retrieve the data from the current condition
            to_return_dict[condition] = self._retrive_data(data, cond_idx)
        return to_return_dict

    @abstractmethod
    def _retrive_data(self, data, idx_list):
        """
        Abstract method to retrieve data from the dataset given a list of
        indices.
        """


class PinaTensorDataset(PinaDataset):
    """
    Dataset class for the PINA dataset with :class:`torch.Tensor` and
    :class:`~pina.label_tensor.LabelTensor` data.
    """

    # Override _retrive_data method for torch.Tensor data
    def _retrive_data(self, data, idx_list):
        """
        Retrieve data from the dataset given a list of indices.

        :param dict data: Dictionary containing the data
            (only :class:`torch.Tensor` or
            :class:`~pina.label_tensor.LabelTensor`).
        :param list[int] idx_list: indices to retrieve.
        :return: Dictionary containing the data at the given indices.
        :rtype: dict
        """

        return {k: v[idx_list] for k, v in data.items()}

    @property
    def input(self):
        """
        Return the input data for the dataset.

        :return: Dictionary containing the input points.
        :rtype: dict
        """
        return {k: v["input"] for k, v in self.conditions_dict.items()}

    def update_data(self, new_conditions_dict):
        """
        Update the dataset with new data.
        This method is used to update the dataset with new data. It replaces
        the current data with the new data provided in the new_conditions_dict
        parameter.

        :param dict new_conditions_dict: Dictionary containing the new data.
        :return: None
        """
        for condition, data in new_conditions_dict.items():
            if condition in self.conditions_dict:
                self.conditions_dict[condition].update(data)
            else:
                self.conditions_dict[condition] = data


class PinaGraphDataset(PinaDataset):
    """
    Dataset class for the PINA dataset with :class:`~torch_geometric.data.Data`
    and :class:`~pina.graph.Graph` data.
    """

    def _create_graph_batch(self, data):
        """
        Create a LabelBatch object from a list of
        :class:`~torch_geometric.data.Data` objects.

        :param data: List of items to collate in a single batch.
        :type data: list[Data] | list[Graph]
        :return: LabelBatch object all the graph collated in a single batch
            disconnected graphs.
        :rtype: LabelBatch
        """
        batch = LabelBatch.from_data_list(data)
        return batch

    def create_batch(self, data):
        """
        Create a Batch object from a list of :class:`~torch_geometric.data.Data`
        objects.

        :param data: List of items to collate in a single batch.
        :type data: list[Data] | list[Graph]
        :return: Batch object.
        :rtype: :class:`~torch_geometric.data.Batch`
            | :class:`~pina.graph.LabelBatch`
        """

        if isinstance(data[0], Data):
            return self._create_graph_batch(data)
        return self._create_tensor_batch(data)

    # Override _retrive_data method for graph handling
    def _retrive_data(self, data, idx_list):
        """
        Retrieve data from the dataset given a list of indices.

        :param dict data: Dictionary containing the data.
        :param list[int] idx_list: List of indices to retrieve.
        :return: Dictionary containing the data at the given indices.
        :rtype: dict
        """

        # Return the data from the current condition
        # If the data is a list of Data objects, create a Batch object
        # If the data is a list of torch.Tensor objects, create a torch.Tensor
        return {
            k: (
                self._create_graph_batch([v[i] for i in idx_list])
                if isinstance(v, list)
                else v[idx_list]
            )
            for k, v in data.items()
        }

    @property
    def input(self):
        """
        Return the input data for the dataset.

        :return: Dictionary containing the input points.
        :rtype: dict
        """
        return {k: v["input"] for k, v in self.conditions_dict.items()}
