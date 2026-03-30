"""Module for the Metric Tracker."""

import copy
import torch
from lightning.pytorch.callbacks import Callback


class MetricTracker(Callback):
    """
    Lightning Callback for Metric Tracking.
    """

    def __init__(self, metrics_to_track=None):
        """
        Tracks specified metrics during training.

        :param metrics_to_track: List of metrics to track.
            Defaults to train loss.
        :type metrics_to_track: list[str], optional
        """
        super().__init__()
        self._collection = []
        # Default to tracking 'train_loss' if not specified
        self.metrics_to_track = metrics_to_track

    def setup(self, trainer, pl_module, stage):
        """
        Called when fit, validate, test, predict, or tune begins.

        :param Trainer trainer: A :class:`~pina.trainer.Trainer` instance.
        :param SolverInterface pl_module: A
            :class:`~pina.solver.solver.SolverInterface` instance.
        :param str stage: Either 'fit', 'test' or 'predict'.
        """
        if self.metrics_to_track is None and trainer.batch_size is None:
            self.metrics_to_track = ["train_loss"]
        elif self.metrics_to_track is None:
            self.metrics_to_track = ["train_loss_epoch"]
        return super().setup(trainer, pl_module, stage)

    def on_train_epoch_end(self, trainer, pl_module):
        """
        Collect and track metrics at the end of each training epoch.

        :param trainer: The trainer object managing the training process.
        :type trainer: pytorch_lightning.Trainer
        :param pl_module: The model being trained (not used here).
        """
        # Track metrics after the first epoch onwards
        if trainer.current_epoch > 0:
            # Append only the tracked metrics to avoid unnecessary data
            tracked_metrics = {
                k: v
                for k, v in trainer.logged_metrics.items()
                if k in self.metrics_to_track
            }
            self._collection.append(copy.deepcopy(tracked_metrics))

    @property
    def metrics(self):
        """
        Aggregate collected metrics over all epochs.

        :return: A dictionary containing aggregated metric values.
        :rtype: dict
        """
        if not self._collection:
            return {}

        # Get intersection of keys across all collected dictionaries
        common_keys = set(self._collection[0]).intersection(
            *self._collection[1:]
        )

        # Stack the metric values for common keys and return
        return {
            k: torch.stack([dic[k] for dic in self._collection])
            for k in common_keys
            if k in self.metrics_to_track
        }
