"""
This module contains the PinaDataModule class, which extends the
LightningDataModule class to allow proper creation and management of
different types of Datasets defined in PINA.
"""

import warnings
from lightning.pytorch import LightningDataModule
import torch
from torch_geometric.data import Data
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from ..label_tensor import LabelTensor
from .dataset import PinaDatasetFactory, PinaTensorDataset


class DummyDataloader:

    def __init__(self, dataset):
        """
        Prepare a dataloader object that returns the entire dataset in a single
        batch. Depending on the number of GPUs, the dataset is managed
        as follows:

        - **Distributed Environment** (multiple GPUs): Divides dataset across
            processes using the rank and world size. Fetches only portion of
            data corresponding to the current process.
        - **Non-Distributed Environment** (single GPU): Fetches the entire
            dataset.

        :param PinaDataset dataset: The dataset object to be processed.

        .. note::
           This dataloader is used when the batch size is ``None``.
        """

        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            if len(dataset) < world_size:
                raise RuntimeError(
                    "Dimension of the dataset smaller than world size."
                    " Increase the size of the partition or use a single GPU"
                )
            idx, i = [], rank
            while i < len(dataset):
                idx.append(i)
                i += world_size
            self.dataset = dataset.fetch_from_idx_list(idx)
        else:
            self.dataset = dataset.get_all_data()

    def __iter__(self):
        return self

    def __len__(self):
        return 1

    def __next__(self):
        return self.dataset


class Collator:
    """
    This callable class is used to collate the data points fetched from the
    dataset. The collation is performed based on the type of dataset used and
    on the batching strategy.
    """

    def __init__(
        self, max_conditions_lengths, automatic_batching, dataset=None
    ):
        """
        Initialize the object, setting the collate function based on whether
        automatic batching is enabled or not.

        :param dict max_conditions_lengths: ``dict`` containing the maximum
            number  of data points to consider in a single batch for
            each condition.
        :param bool automatic_batching: Whether automatic PyTorch batching is
            enabled or not. For more information, see the
            :class:`~pina.data.data_module.PinaDataModule` class.
        :param PinaDataset dataset: The dataset where the data is stored.
        """

        self.max_conditions_lengths = max_conditions_lengths
        # Set the collate function based on the batching strategy
        # collate_pina_dataloader is used when automatic batching is disabled
        # collate_torch_dataloader is used when automatic batching is enabled
        self.callable_function = (
            self._collate_torch_dataloader
            if automatic_batching
            else (self._collate_pina_dataloader)
        )
        self.dataset = dataset

        # Set the function which performs the actual collation
        if isinstance(self.dataset, PinaTensorDataset):
            # If the dataset is a PinaTensorDataset, use this collate function
            self._collate = self._collate_tensor_dataset
        else:
            # If the dataset is a PinaDataset, use this collate function
            self._collate = self._collate_graph_dataset

    def _collate_pina_dataloader(self, batch):
        """
        Function used to create a batch when automatic batching is disabled.

        :param list[int] batch: List of integers representing the indices of
            the data points to be fetched.
        :return: Dictionary containing the data points fetched from the dataset.
        :rtype: dict
        """
        # Call the fetch_from_idx_list method of the dataset
        return self.dataset.fetch_from_idx_list(batch)

    def _collate_torch_dataloader(self, batch):
        """
        Function used to collate the batch

        :param list[dict] batch: List of retrieved data.
        :return: Dictionary containing the data points fetched from the dataset,
            collated.
        :rtype: dict
        """

        batch_dict = {}
        if isinstance(batch, dict):
            return batch
        conditions_names = batch[0].keys()
        # Condition names
        for condition_name in conditions_names:
            single_cond_dict = {}
            condition_args = batch[0][condition_name].keys()
            for arg in condition_args:
                data_list = [
                    batch[idx][condition_name][arg]
                    for idx in range(
                        min(
                            len(batch),
                            self.max_conditions_lengths[condition_name],
                        )
                    )
                ]
                single_cond_dict[arg] = self._collate(data_list)

            batch_dict[condition_name] = single_cond_dict
        return batch_dict

    @staticmethod
    def _collate_tensor_dataset(data_list):
        """
        Function used to collate the data when the dataset is a
        :class:`~pina.data.dataset.PinaTensorDataset`.

        :param data_list: Elements to be collated.
        :type data_list: list[torch.Tensor] | list[LabelTensor]
        :return: Batch of data.
        :rtype: dict

        :raises RuntimeError: If the data is not a :class:`torch.Tensor` or a
            :class:`~pina.label_tensor.LabelTensor`.
        """

        if isinstance(data_list[0], LabelTensor):
            return LabelTensor.stack(data_list)
        if isinstance(data_list[0], torch.Tensor):
            return torch.stack(data_list)
        raise RuntimeError("Data must be Tensors or LabelTensor ")

    def _collate_graph_dataset(self, data_list):
        """
        Function used to collate data when the dataset is a
        :class:`~pina.data.dataset.PinaGraphDataset`.

        :param data_list: Elememts to be collated.
        :type data_list: list[Data] | list[Graph]
        :return: Batch of data.
        :rtype: dict

        :raises RuntimeError: If the data is not a
            :class:`~torch_geometric.data.Data` or a :class:`~pina.graph.Graph`.
        """
        if isinstance(data_list[0], LabelTensor):
            return LabelTensor.cat(data_list)
        if isinstance(data_list[0], torch.Tensor):
            return torch.cat(data_list)
        if isinstance(data_list[0], Data):
            return self.dataset.create_batch(data_list)
        raise RuntimeError(
            "Data must be Tensors or LabelTensor or pyG "
            "torch_geometric.data.Data"
        )

    def __call__(self, batch):
        """
        Perform the collation of data fetched from the dataset. The behavoior
        of the function is set based on the batching strategy during class
        initialization.

        :param batch: List of retrieved data or sampled indices.
        :type batch: list[int] | list[dict]
        :return: Dictionary containing colleted data fetched from the dataset.
        :rtype: dict
        """

        return self.callable_function(batch)


class PinaSampler:
    """
    This class is used to create the sampler instance based on the shuffle
    parameter and the environment in which the code is running.
    """

    def __new__(cls, dataset):
        """
        Instantiate and initialize the sampler.

        :param PinaDataset dataset: The dataset from which to sample.
        :return: The sampler instance.
        :rtype: :class:`torch.utils.data.Sampler`
        """

        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            sampler = DistributedSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)
        return sampler


class PinaDataModule(LightningDataModule):
    """
    This class extends :class:`~lightning.pytorch.core.LightningDataModule`,
    allowing proper creation and management of different types of datasets
    defined in PINA.
    """

    def __init__(
        self,
        problem,
        train_size=0.7,
        test_size=0.2,
        val_size=0.1,
        batch_size=None,
        shuffle=True,
        repeat=False,
        automatic_batching=None,
        num_workers=0,
        pin_memory=False,
    ):
        """
        Initialize the object and creating datasets based on the input problem.

        :param AbstractProblem problem: The problem containing the data on which
            to create the datasets and dataloaders.
        :param float train_size: Fraction of elements in the training split. It
            must be in the range [0, 1].
        :param float test_size: Fraction of elements in the test split. It must
            be in the range [0, 1].
        :param float val_size: Fraction of elements in the validation split. It
            must be in the range [0, 1].
        :param int batch_size: The batch size used for training. If ``None``,
            the entire dataset is returned in a single batch.
            Default is ``None``.
        :param bool shuffle: Whether to shuffle the dataset before splitting.
            Default ``True``.
        :param bool repeat: If ``True``, in case of batch size larger than the
            number of elements in a specific condition, the elements are
            repeated until the batch size is reached. If ``False``, the number
            of elements in the batch is the minimum between the batch size and
            the number of elements in the condition. Default is ``False``.
        :param automatic_batching: If ``True``, automatic PyTorch batching
            is performed, which consists of extracting one element at a time
            from the dataset and collating them into a batch. This is useful
            when the dataset is too large to fit into memory. On the other hand,
            if ``False``, the items are retrieved from the dataset all at once
            avoind the overhead of collating them into a batch and reducing the
            ``__getitem__`` calls to the dataset. This is useful when the
            dataset fits into memory. Avoid using automatic batching when
            ``batch_size`` is large. Default is ``False``.
        :param int num_workers: Number of worker threads for data loading.
            Default ``0`` (serial loading).
        :param bool pin_memory: Whether to use pinned memory for faster data
            transfer to GPU. Default ``False``.

        :raises ValueError: If at least one of the splits is negative.
        :raises ValueError: If the sum of the splits is different from 1.

        .. seealso::
            For more information on multi-process data loading, see:
            https://pytorch.org/docs/stable/data.html#multi-process-data-loading

            For details on memory pinning, see:
            https://pytorch.org/docs/stable/data.html#memory-pinning
        """
        super().__init__()

        # Store fixed attributes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.repeat = repeat
        self.automatic_batching = automatic_batching

        # If batch size is None, num_workers has no effect
        if batch_size is None and num_workers != 0:
            warnings.warn(
                "Setting num_workers when batch_size is None has no effect on "
                "the DataLoading process."
            )
            self.num_workers = 0
        else:
            self.num_workers = num_workers

        # If batch size is None, pin_memory has no effect
        if batch_size is None and pin_memory:
            warnings.warn(
                "Setting pin_memory to True has no effect when "
                "batch_size is None."
            )
            self.pin_memory = False
        else:
            self.pin_memory = pin_memory

        # Collect data
        problem.collect_data()

        # Check if the splits are correct
        self._check_slit_sizes(train_size, test_size, val_size)

        # Split input data into subsets
        splits_dict = {}
        if train_size > 0:
            splits_dict["train"] = train_size
            self.train_dataset = None
        else:
            # Use the super method to create the train dataloader which
            # raises NotImplementedError
            self.train_dataloader = super().train_dataloader
        if test_size > 0:
            splits_dict["test"] = test_size
            self.test_dataset = None
        else:
            # Use the super method to create the train dataloader which
            # raises NotImplementedError
            self.test_dataloader = super().test_dataloader
        if val_size > 0:
            splits_dict["val"] = val_size
            self.val_dataset = None
        else:
            # Use the super method to create the train dataloader which
            # raises NotImplementedError
            self.val_dataloader = super().val_dataloader

        self.data_splits = self._create_splits(
            problem.collected_data, splits_dict
        )
        self.transfer_batch_to_device = self._transfer_batch_to_device

    def setup(self, stage=None):
        """
        Create the dataset objects for the given stage.
        If the stage is "fit", the training and validation datasets are created.
        If the stage is "test", the testing dataset is created.

        :param str stage: The stage for which to perform the dataset setup.

        :raises ValueError: If the stage is neither "fit" nor "test".
        """
        if stage == "fit" or stage is None:
            self.train_dataset = PinaDatasetFactory(
                self.data_splits["train"],
                max_conditions_lengths=self.find_max_conditions_lengths(
                    "train"
                ),
                automatic_batching=self.automatic_batching,
            )
            if "val" in self.data_splits.keys():
                self.val_dataset = PinaDatasetFactory(
                    self.data_splits["val"],
                    max_conditions_lengths=self.find_max_conditions_lengths(
                        "val"
                    ),
                    automatic_batching=self.automatic_batching,
                )
        elif stage == "test":
            self.test_dataset = PinaDatasetFactory(
                self.data_splits["test"],
                max_conditions_lengths=self.find_max_conditions_lengths("test"),
                automatic_batching=self.automatic_batching,
            )
        else:
            raise ValueError("stage must be either 'fit' or 'test'.")

    @staticmethod
    def _split_condition(single_condition_dict, splits_dict):
        """
        Split the condition into different stages.

        :param dict single_condition_dict: The condition to be split.
        :param dict splits_dict: The dictionary containing the number of
            elements in each stage.
        :return: A dictionary containing the split condition.
        :rtype: dict
        """

        len_condition = len(single_condition_dict["input"])

        lengths = [
            int(len_condition * length) for length in splits_dict.values()
        ]

        remainder = len_condition - sum(lengths)
        for i in range(remainder):
            lengths[i % len(lengths)] += 1

        splits_dict = {
            k: max(1, v) for k, v in zip(splits_dict.keys(), lengths)
        }
        to_return_dict = {}
        offset = 0

        for stage, stage_len in splits_dict.items():
            to_return_dict[stage] = {
                k: v[offset : offset + stage_len]
                for k, v in single_condition_dict.items()
                if k != "equation"
                # Equations are NEVER dataloaded
            }
            if offset + stage_len >= len_condition:
                offset = len_condition - 1
                continue
            offset += stage_len
        return to_return_dict

    def _create_splits(self, collector, splits_dict):
        """
        Create the dataset objects putting data in the correct splits.

        :param Collector collector: The collector object containing the data.
        :param dict splits_dict: The dictionary containing the number of
            elements in each stage.
        :return: The dictionary containing the dataset objects.
        :rtype: dict
        """

        # ----------- Auxiliary function ------------
        def _apply_shuffle(condition_dict, len_data):
            idx = torch.randperm(len_data)
            for k, v in condition_dict.items():
                if k == "equation":
                    continue
                if isinstance(v, list):
                    condition_dict[k] = [v[i] for i in idx]
                elif isinstance(v, LabelTensor):
                    condition_dict[k] = LabelTensor(v.tensor[idx], v.labels)
                elif isinstance(v, torch.Tensor):
                    condition_dict[k] = v[idx]
                else:
                    raise ValueError(f"Data type {type(v)} not supported")

        # ----------- End auxiliary function ------------

        split_names = list(splits_dict.keys())
        dataset_dict = {name: {} for name in split_names}
        for (
            condition_name,
            condition_dict,
        ) in collector.items():
            len_data = len(condition_dict["input"])
            if self.shuffle:
                _apply_shuffle(condition_dict, len_data)
            for key, data in self._split_condition(
                condition_dict, splits_dict
            ).items():
                dataset_dict[key].update({condition_name: data})
        return dataset_dict

    def _create_dataloader(self, split, dataset):
        """ "
        Create the dataloader for the given split.

        :param str split: The split on which to create the dataloader.
        :param str dataset: The dataset to be used for the dataloader.
        :return: The dataloader for the given split.
        :rtype: torch.utils.data.DataLoader
        """
        # Suppress the warning about num_workers.
        # In many cases, especially for PINNs,
        # serial data loading can outperform parallel data loading.
        warnings.filterwarnings(
            "ignore",
            message=(
                "The '(train|val|test)_dataloader' does not have many workers "
                "which may be a bottleneck."
            ),
            module="lightning.pytorch.trainer.connectors.data_connector",
        )
        # Use custom batching (good if batch size is large)
        if self.batch_size is not None:
            sampler = PinaSampler(dataset)
            if self.automatic_batching:
                collate = Collator(
                    self.find_max_conditions_lengths(split),
                    self.automatic_batching,
                    dataset=dataset,
                )
            else:
                collate = Collator(
                    None, self.automatic_batching, dataset=dataset
                )
            return DataLoader(
                dataset,
                self.batch_size,
                collate_fn=collate,
                sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
        dataloader = DummyDataloader(dataset)
        dataloader.dataset = self._transfer_batch_to_device(
            dataloader.dataset, self.trainer.strategy.root_device, 0
        )
        self.transfer_batch_to_device = self._transfer_batch_to_device_dummy
        return dataloader

    def find_max_conditions_lengths(self, split):
        """
        Define the maximum length for each conditions.

        :param dict split: The split of the dataset.
        :return: The maximum length per condition.
        :rtype: dict
        """

        max_conditions_lengths = {}
        for k, v in self.data_splits[split].items():
            if self.batch_size is None:
                max_conditions_lengths[k] = len(v["input"])
            elif self.repeat:
                max_conditions_lengths[k] = self.batch_size
            else:
                max_conditions_lengths[k] = min(
                    len(v["input"]), self.batch_size
                )
        return max_conditions_lengths

    def val_dataloader(self):
        """
        Create the validation dataloader.

        :return: The validation dataloader
        :rtype: torch.utils.data.DataLoader
        """
        return self._create_dataloader("val", self.val_dataset)

    def train_dataloader(self):
        """
        Create the training dataloader

        :return: The training dataloader
        :rtype: torch.utils.data.DataLoader
        """
        return self._create_dataloader("train", self.train_dataset)

    def test_dataloader(self):
        """
        Create the testing dataloader

        :return: The testing dataloader
        :rtype: torch.utils.data.DataLoader
        """
        return self._create_dataloader("test", self.test_dataset)

    @staticmethod
    def _transfer_batch_to_device_dummy(batch, device, dataloader_idx):
        """
        Transfer the batch to the device. This method is used when the batch
        size is None: batch has already been transferred to the device.

        :param list[tuple] batch: List of tuple where the first element of the
            tuple is the condition name and the second element is the data.
        :param torch.device device: Device to which the batch is transferred.
        :param int dataloader_idx: Index of the dataloader.
        :return: The batch transferred to the device.
        :rtype: list[tuple]
        """

        return batch

    def _transfer_batch_to_device(self, batch, device, dataloader_idx):
        """
        Transfer the batch to the device. This method is called in the
        training loop and is used to transfer the batch to the device.

        :param dict batch: The batch to be transferred to the device.
        :param torch.device device: The device to which the batch is
            transferred.
        :param int dataloader_idx: The index of the dataloader.
        :return: The batch transferred to the device.
        :rtype: list[tuple]
        """

        batch = [
            (
                k,
                super(LightningDataModule, self).transfer_batch_to_device(
                    v, device, dataloader_idx
                ),
            )
            for k, v in batch.items()
        ]

        return batch

    @staticmethod
    def _check_slit_sizes(train_size, test_size, val_size):
        """
        Check if the splits are correct. The splits sizes must be positive and
        the sum of the splits must be 1.

        :param float train_size: The size of the training split.
        :param float test_size: The size of the testing split.
        :param float val_size: The size of the validation split.

        :raises ValueError: If at least one of the splits is negative.
        :raises ValueError: If the sum of the splits is different
            from 1.
        """

        if train_size < 0 or test_size < 0 or val_size < 0:
            raise ValueError("The splits must be positive")
        if abs(train_size + test_size + val_size - 1) > 1e-6:
            raise ValueError("The sum of the splits must be 1")

    @property
    def input(self):
        """
        Return all the input points coming from all the datasets.

        :return: The input points for training.
        :rtype: dict
        """

        to_return = {}
        if hasattr(self, "train_dataset") and self.train_dataset is not None:
            to_return["train"] = self.train_dataset.input
        if hasattr(self, "val_dataset") and self.val_dataset is not None:
            to_return["val"] = self.val_dataset.input
        if hasattr(self, "test_dataset") and self.test_dataset is not None:
            to_return["test"] = self.test_dataset.input
        return to_return
