"""Module for the Processing Callbacks."""

from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.callbacks.progress.progress_bar import (
    get_standard_metrics,
)
from pina.utils import check_consistency


class PINAProgressBar(TQDMProgressBar):
    """
    PINA Implementation of a Lightning Callback for enriching the progress bar.
    """

    BAR_FORMAT = (
        "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, "
        "{rate_noinv_fmt}{postfix}]"
    )

    def __init__(self, metrics="val", **kwargs):
        """
        This class enables the display of only relevant metrics during training.

        :param metrics: Logged metrics to be shown during the training.
            Must be a subset of the conditions keys defined in
            :obj:`pina.condition.Condition`.
        :type metrics: str | list(str) | tuple(str)

        :Keyword Arguments:
            The additional keyword arguments specify the progress bar and can be
            choosen from the `pytorch-lightning TQDMProgressBar API
            <https://lightning.ai/docs/pytorch/stable/_modules/lightning/pytorch/callback/progress/tqdm_progress.html#TQDMProgressBar>`_

        Example:
            >>> pbar = PINAProgressBar(['mean'])
            >>> # ... Perform training ...
            >>> trainer = Trainer(solver, callbacks=[pbar])
        """
        super().__init__(**kwargs)
        # check consistency
        if not isinstance(metrics, (list, tuple)):
            metrics = [metrics]
        check_consistency(metrics, str)
        self._sorted_metrics = metrics

    def get_metrics(self, trainer, pl_module):
        r"""Combine progress bar metrics collected from the trainer with
        standard metrics from get_standard_metrics.
        Override this method to customize the items shown in the progress bar.
        The progress bar metrics are sorted according to ``metrics``.

        Here is an example of how to override the defaults:

        .. code-block:: python

            def get_metrics(self, trainer, model):
                # don't show the version number
                items = super().get_metrics(trainer, model)
                items.pop("v_num", None)
                return items

        :return: Dictionary with the items to be displayed in the progress bar.
        :rtype: tuple(dict)
        """
        standard_metrics = get_standard_metrics(trainer)
        pbar_metrics = trainer.progress_bar_metrics
        if pbar_metrics:
            pbar_metrics = {
                key: pbar_metrics[key]
                for key in pbar_metrics
                if key in self._sorted_metrics
            }
        return {**standard_metrics, **pbar_metrics}

    def setup(self, trainer, pl_module, stage):
        """
        Check that the initialized metrics are available and correctly logged.

        :param trainer: The trainer object managing the training process.
        :type trainer: pytorch_lightning.Trainer
        :param pl_module: Placeholder argument.
        """
        # Check if all keys in sort_keys are present in the dictionary
        for key in self._sorted_metrics:
            if (
                key not in trainer.solver.problem.conditions.keys()
                and key != "train"
                and key != "val"
            ):
                raise KeyError(f"Key '{key}' is not present in the dictionary")
        # add the loss pedix
        if trainer.batch_size is not None:
            pedix = "_loss_epoch"
        else:
            pedix = "_loss"
        self._sorted_metrics = [
            metric + pedix for metric in self._sorted_metrics
        ]
        return super().setup(trainer, pl_module, stage)
