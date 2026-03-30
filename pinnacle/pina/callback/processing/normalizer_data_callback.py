"""Module for the Normalizer callback."""

import torch
from lightning.pytorch import Callback
from ...label_tensor import LabelTensor
from ...utils import check_consistency, is_function
from ...condition import InputTargetCondition
from ...data.dataset import PinaGraphDataset


class NormalizerDataCallback(Callback):
    r"""
    A Callback used to normalize the dataset inputs or targets according to
    user-provided scale and shift functions.

    The transformation is applied as:

    .. math::

        x_{\text{new}} = \frac{x - \text{shift}}{\text{scale}}

    :Example:

    >>> NormalizerDataCallback()
    >>> NormalizerDataCallback(
    ...     scale_fn: torch.std,
    ...     shift_fn: torch.mean,
    ...     stage: "all",
    ...     apply_to: "input",
    ... )
    """

    def __init__(
        self,
        scale_fn=torch.std,
        shift_fn=torch.mean,
        stage="all",
        apply_to="input",
    ):
        """
        Initialization of the :class:`NormalizerDataCallback` class.

        :param Callable scale_fn: The function to compute the scaling factor.
            Default is  ``torch.std``.
        :param Callable shift_fn: The function to compute the shifting factor.
            Default is ``torch.mean``.
        :param str stage: The stage in which normalization is applied.
            Accepted values are "train", "validate", "test", or "all".
            Default is ``"all"``.
        :param str apply_to: Whether to normalize "input" or "target" data.
            Default is ``"input"``.
        :raises ValueError: If ``scale_fn`` is not callable.
        :raises ValueError: If ``shift_fn`` is not callable.
        """
        super().__init__()

        # Validate parameters
        self.apply_to = self._validate_apply_to(apply_to)
        self.stage = self._validate_stage(stage)

        # Validate functions
        if not is_function(scale_fn):
            raise ValueError(f"scale_fn must be Callable, got {scale_fn}")
        if not is_function(shift_fn):
            raise ValueError(f"shift_fn must be Callable, got {shift_fn}")
        self.scale_fn = scale_fn
        self.shift_fn = shift_fn

        # Initialize normalizer dictionary
        self._normalizer = {}

    def _validate_apply_to(self, apply_to):
        """
        Validate the ``apply_to`` parameter.

        :param str apply_to: The candidate value for the ``apply_to`` parameter.
        :raises ValueError: If ``apply_to`` is neither "input" nor "target".
        :return: The validated ``apply_to`` value.
        :rtype: str
        """
        check_consistency(apply_to, str)
        if apply_to not in {"input", "target"}:
            raise ValueError(
                f"apply_to must be either 'input' or 'target', got {apply_to}"
            )

        return apply_to

    def _validate_stage(self, stage):
        """
        Validate the ``stage`` parameter.

        :param str stage: The candidate value for the ``stage`` parameter.
        :raises ValueError: If ``stage`` is not one of "train", "validate",
            "test", or "all".
        :return: The validated ``stage`` value.
        :rtype: str
        """
        check_consistency(stage, str)
        if stage not in {"train", "validate", "test", "all"}:
            raise ValueError(
                "stage must be one of 'train', 'validate', 'test', or 'all',"
                f" got {stage}"
            )

        return stage

    def setup(self, trainer, pl_module, stage):
        """
        Apply normalization during setup.

        :param Trainer trainer: A :class:`~pina.trainer.Trainer` instance.
        :param SolverInterface pl_module: A
            :class:`~pina.solver.solver.SolverInterface` instance.
        :param str stage: The current stage.
        :raises RuntimeError: If the training dataset is not available when
            computing normalization parameters.
        :return: The result of the parent setup.
        :rtype: Any

        :raises NotImplementedError: If the dataset is graph-based.
        """

        # Ensure datsets are not graph-based
        if isinstance(trainer.datamodule.train_dataset, PinaGraphDataset):
            raise NotImplementedError(
                "NormalizerDataCallback is not compatible with "
                "graph-based datasets."
            )

        # Extract conditions
        conditions_to_normalize = [
            name
            for name, cond in pl_module.problem.conditions.items()
            if isinstance(cond, InputTargetCondition)
        ]

        # Compute scale and shift parameters
        if not self.normalizer:
            if not trainer.datamodule.train_dataset:
                raise RuntimeError(
                    "Training dataset is not available. Cannot compute "
                    "normalization parameters."
                )
            self._compute_scale_shift(
                conditions_to_normalize, trainer.datamodule.train_dataset
            )

        # Apply normalization based on the specified stage
        if stage == "fit" and self.stage in ["train", "all"]:
            self.normalize_dataset(trainer.datamodule.train_dataset)
        if stage == "fit" and self.stage in ["validate", "all"]:
            self.normalize_dataset(trainer.datamodule.val_dataset)
        if stage == "test" and self.stage in ["test", "all"]:
            self.normalize_dataset(trainer.datamodule.test_dataset)

        return super().setup(trainer, pl_module, stage)

    def _compute_scale_shift(self, conditions, dataset):
        """
        Compute scale and shift parameters for each condition in the dataset.

        :param list conditions: The list of condition names.
        :param dataset: The `~pina.data.dataset.PinaDataset` dataset.
        """
        for cond in conditions:
            if cond in dataset.conditions_dict:
                data = dataset.conditions_dict[cond][self.apply_to]
                shift = self.shift_fn(data)
                scale = self.scale_fn(data)
                self._normalizer[cond] = {
                    "shift": shift,
                    "scale": scale,
                }

    @staticmethod
    def _norm_fn(value, scale, shift):
        """
        Normalize a value according to the scale and shift parameters.

        :param value: The input tensor to normalize.
        :type value: torch.Tensor | LabelTensor
        :param float scale: The scaling factor.
        :param float shift: The shifting factor.
        :return: The normalized tensor.
        :rtype: torch.Tensor | LabelTensor
        """
        scaled_value = (value - shift) / scale
        if isinstance(value, LabelTensor):
            scaled_value = LabelTensor(scaled_value, value.labels)

        return scaled_value

    def normalize_dataset(self, dataset):
        """
        Apply in-place normalization to the dataset.

        :param PinaDataset dataset: The dataset to be normalized.
        """
        # Initialize update dictionary
        update_dataset_dict = {}

        # Iterate over conditions and apply normalization
        for cond, norm_params in self.normalizer.items():
            points = dataset.conditions_dict[cond][self.apply_to]
            scale = norm_params["scale"]
            shift = norm_params["shift"]
            normalized_points = self._norm_fn(points, scale, shift)
            update_dataset_dict[cond] = {
                self.apply_to: (
                    LabelTensor(normalized_points, points.labels)
                    if isinstance(points, LabelTensor)
                    else normalized_points
                )
            }

        # Update the dataset in-place
        dataset.update_data(update_dataset_dict)

    @property
    def normalizer(self):
        """
        Get the dictionary of normalization parameters.

        :return: The dictionary of normalization parameters.
        :rtype: dict
        """
        return self._normalizer
