"""Module for the SwitchScheduler callback."""

from lightning.pytorch.callbacks import Callback
from ...optim import TorchScheduler
from ...utils import check_consistency, check_positive_integer


class SwitchScheduler(Callback):
    """
    Callback to switch scheduler during training.
    """

    def __init__(self, new_schedulers, epoch_switch):
        """
        This callback allows switching between different schedulers during
        training, enabling the exploration of multiple optimization strategies
        without interrupting the training process.

        :param new_schedulers: The scheduler or list of schedulers to switch to.
            Use a single scheduler for single-model solvers, or a list of
            schedulers when working with multiple models.
        :type new_schedulers: pina.optim.TorchScheduler |
            list[pina.optim.TorchScheduler]
        :param int epoch_switch: The epoch at which the scheduler switch occurs.
        :raise AssertionError: If epoch_switch is less than 1.
        :raise ValueError: If each scheduler in ``new_schedulers`` is not an
            instance of :class:`pina.optim.TorchScheduler`.

        Example:
            >>> scheduler = TorchScheduler(
            >>>     torch.optim.lr_scheduler.StepLR, step_size=5
            >>> )
            >>> switch_callback = SwitchScheduler(
            >>>     new_schedulers=scheduler, epoch_switch=10
            >>> )
        """
        super().__init__()

        # Check if epoch_switch is greater than 1
        check_positive_integer(epoch_switch - 1, strict=True)

        # If new_schedulers is not a list, convert it to a list
        if not isinstance(new_schedulers, list):
            new_schedulers = [new_schedulers]

        # Check consistency
        for scheduler in new_schedulers:
            check_consistency(scheduler, TorchScheduler)

        # Store the new schedulers and epoch switch
        self._new_schedulers = new_schedulers
        self._epoch_switch = epoch_switch

    def on_train_epoch_start(self, trainer, __):
        """
        Switch the scheduler at the start of the specified training epoch.

        :param lightning.pytorch.Trainer trainer: The trainer object managing
            the training process.
        :param __: Placeholder argument (not used).
        """
        # Check if the current epoch matches the switch epoch
        if trainer.current_epoch == self._epoch_switch:
            schedulers = []

            # Hook the new schedulers to the model parameters
            for idx, scheduler in enumerate(self._new_schedulers):
                scheduler.hook(trainer.solver._pina_optimizers[idx])
                schedulers.append(scheduler)

                # Update the trainer's scheduler configs
                trainer.lr_scheduler_configs[idx].scheduler = scheduler.instance

            # Update the solver's schedulers
            trainer.solver._pina_schedulers = schedulers
