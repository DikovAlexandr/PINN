"""Module for the SwitchOptimizer callback."""

from lightning.pytorch.callbacks import Callback
from ...optim import TorchOptimizer
from ...utils import check_consistency


class SwitchOptimizer(Callback):
    """
    PINA Implementation of a Lightning Callback to switch optimizer during
    training.
    """

    def __init__(self, new_optimizers, epoch_switch):
        """
        This callback allows switching between different optimizers during
        training, enabling the exploration of multiple optimization strategies
        without interrupting the training process.

        :param new_optimizers: The model optimizers to switch to. Can be a
            single :class:`torch.optim.Optimizer` instance or a list of them
            for multiple model solver.
        :type new_optimizers: pina.optim.TorchOptimizer | list
        :param int epoch_switch: The epoch at which the optimizer switch occurs.

        Example:
            >>> optimizer = TorchOptimizer(torch.optim.Adam, lr=0.01)
            >>> switch_callback = SwitchOptimizer(
            >>>     new_optimizers=optimizer, epoch_switch=10
            >>> )
        """
        super().__init__()

        # Check if epoch_switch is greater than 1
        if epoch_switch < 1:
            raise ValueError("epoch_switch must be greater than one.")

        # If new_optimizers is not a list, convert it to a list
        if not isinstance(new_optimizers, list):
            new_optimizers = [new_optimizers]

        # Check consistency
        check_consistency(epoch_switch, int)
        for optimizer in new_optimizers:
            check_consistency(optimizer, TorchOptimizer)

        # Store the new optimizers and epoch switch
        self._new_optimizers = new_optimizers
        self._epoch_switch = epoch_switch

    def on_train_epoch_start(self, trainer, __):
        """
        Switch the optimizer at the start of the specified training epoch.

        :param lightning.pytorch.Trainer trainer: The trainer object managing
            the training process.
        :param _: Placeholder argument (not used).
        """
        # Check if the current epoch matches the switch epoch
        if trainer.current_epoch == self._epoch_switch:
            optims = []

            # Hook the new optimizers to the model parameters
            for idx, optim in enumerate(self._new_optimizers):
                optim.hook(trainer.solver._pina_models[idx].parameters())
                optims.append(optim)

            # Update the solver's optimizers
            trainer.solver._pina_optimizers = optims

            # Update the trainer's strategy optimizers
            trainer.strategy.optimizers = [o.instance for o in optims]
