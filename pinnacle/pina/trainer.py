"""Module for the Trainer."""

import sys
import warnings
import torch
import lightning
from .utils import check_consistency, custom_warning_format
from .data import PinaDataModule
from .solver import SolverInterface, PINNInterface

# set the warning for compile options
warnings.formatwarning = custom_warning_format
warnings.filterwarnings("always", category=UserWarning)


class Trainer(lightning.pytorch.Trainer):
    """
    PINA custom Trainer class to extend the standard Lightning functionality.

    This class enables specific features or behaviors required by the PINA
    framework. It modifies the standard
    :class:`lightning.pytorch.Trainer <lightning.pytorch.trainer.trainer.Trainer>`
    class to better support the training process in PINA.
    """

    def __init__(
        self,
        solver,
        batch_size=None,
        train_size=1.0,
        test_size=0.0,
        val_size=0.0,
        compile=None,
        repeat=None,
        automatic_batching=None,
        num_workers=None,
        pin_memory=None,
        shuffle=None,
        **kwargs,
    ):
        """
        Initialization of the :class:`Trainer` class.

        :param SolverInterface solver: A
            :class:`~pina.solver.solver.SolverInterface` solver used to solve a
            :class:`~pina.problem.abstract_problem.AbstractProblem`.
        :param int batch_size: The number of samples per batch to load.
            If ``None``, all samples are loaded and data is not batched.
            Default is ``None``.
        :param float train_size: The percentage of elements to include in the
            training dataset. Default is ``1.0``.
        :param float test_size: The percentage of elements to include in the
            test dataset. Default is ``0.0``.
        :param float val_size: The percentage of elements to include in the
            validation dataset. Default is ``0.0``.
        :param bool compile: If ``True``, the model is compiled before training.
            Default is ``False``. For Windows users, it is always disabled. Not
            supported for python version greater or equal than 3.14.
        :param bool repeat: Whether to repeat the dataset data in each
            condition during training. For further details, see the
            :class:`~pina.data.data_module.PinaDataModule` class. Default is
            ``False``.
        :param bool automatic_batching: If ``True``, automatic PyTorch batching
            is performed, otherwise the items are retrieved from the dataset
            all at once. For further details, see the
            :class:`~pina.data.data_module.PinaDataModule` class. Default is
            ``False``.
        :param int num_workers: The number of worker threads for data loading.
            Default is ``0`` (serial loading).
        :param bool pin_memory: Whether to use pinned memory for faster data
            transfer to GPU. Default is ``False``.
        :param bool shuffle: Whether to shuffle the data during training.
            Default is ``True``.
        :param dict kwargs: Additional keyword arguments that specify the
            training setup. These can be selected from the `pytorch-lightning
            Trainer API
            <https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-class-api>`_.
        """
        # check consistency for init types
        self._check_input_consistency(
            solver=solver,
            train_size=train_size,
            test_size=test_size,
            val_size=val_size,
            repeat=repeat,
            automatic_batching=automatic_batching,
            compile=compile,
        )
        pin_memory, num_workers, shuffle, batch_size = (
            self._check_consistency_and_set_defaults(
                pin_memory, num_workers, shuffle, batch_size
            )
        )

        # inference mode set to false when validating/testing PINNs otherwise
        # gradient is not tracked and optimization_cycle fails
        if isinstance(solver, PINNInterface):
            kwargs["inference_mode"] = False

        # Logging depends on the batch size, when batch_size is None then
        # log_every_n_steps should be zero
        if batch_size is None:
            kwargs["log_every_n_steps"] = 0
        else:
            kwargs.setdefault("log_every_n_steps", 50)  # default for lightning

        # Setting default kwargs, overriding lightning defaults
        kwargs.setdefault("enable_progress_bar", True)

        super().__init__(**kwargs)

        # checking compilation and automatic batching
        # compilation disabled for Windows and for Python 3.14+
        if (
            compile is None
            or sys.platform == "win32"
            or sys.version_info >= (3, 14)
        ):
            compile = False
            warnings.warn(
                "Compilation is disabled for Python 3.14+ and for Windows.",
                UserWarning,
            )

        repeat = repeat if repeat is not None else False

        automatic_batching = (
            automatic_batching if automatic_batching is not None else False
        )

        # set attributes
        self.compile = compile
        self.solver = solver
        self.batch_size = batch_size
        self._move_to_device()
        self.data_module = None
        self._create_datamodule(
            train_size=train_size,
            test_size=test_size,
            val_size=val_size,
            batch_size=batch_size,
            repeat=repeat,
            automatic_batching=automatic_batching,
            pin_memory=pin_memory,
            num_workers=num_workers,
            shuffle=shuffle,
        )

        # logging
        self.logging_kwargs = {
            "sync_dist": bool(
                len(self._accelerator_connector._parallel_devices) > 1
            ),
            "on_step": bool(kwargs["log_every_n_steps"] > 0),
            "prog_bar": bool(kwargs["enable_progress_bar"]),
            "on_epoch": True,
        }

    def _move_to_device(self):
        """
        Moves the ``unknown_parameters`` of an instance of
        :class:`~pina.problem.abstract_problem.AbstractProblem` to the
        :class:`Trainer` device.
        """
        device = self._accelerator_connector._parallel_devices[0]
        # move parameters to device
        pb = self.solver.problem
        if hasattr(pb, "unknown_parameters"):
            for key in pb.unknown_parameters:
                pb.unknown_parameters[key] = torch.nn.Parameter(
                    pb.unknown_parameters[key].data.to(device)
                )

    def _create_datamodule(
        self,
        train_size,
        test_size,
        val_size,
        batch_size,
        repeat,
        automatic_batching,
        pin_memory,
        num_workers,
        shuffle,
    ):
        """
        This method is designed to handle the creation of a data module when
        resampling is needed during training. Instead of manually defining and
        modifying the trainer's dataloaders, this method is called to
        automatically configure the data module.

        :param float train_size: The percentage of elements to include in the
            training dataset.
        :param float test_size: The percentage of elements to include in the
            test dataset.
        :param float val_size: The percentage of elements to include in the
            validation dataset.
        :param int batch_size: The number of samples per batch to load.
        :param bool repeat: Whether to repeat the dataset data in each
            condition during training.
        :param bool automatic_batching: Whether to perform automatic batching
            with PyTorch.
        :param bool pin_memory: Whether to use pinned memory for faster data
            transfer to GPU.
        :param int num_workers: The number of worker threads for data loading.
        :param bool shuffle: Whether to shuffle the data during training.
        :raises RuntimeError: If not all conditions are sampled.
        """
        if not self.solver.problem.are_all_domains_discretised:
            error_message = "\n".join(
                [
                    f"""{" " * 13} ---> Domain {key} {
                    "sampled" if key in self.solver.problem.discretised_domains 
                    else
                    "not sampled"}"""
                    for key in self.solver.problem.domains.keys()
                ]
            )
            raise RuntimeError(
                "Cannot create Trainer if not all conditions "
                "are sampled. The Trainer got the following:\n"
                f"{error_message}"
            )
        self.data_module = PinaDataModule(
            self.solver.problem,
            train_size=train_size,
            test_size=test_size,
            val_size=val_size,
            batch_size=batch_size,
            repeat=repeat,
            automatic_batching=automatic_batching,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
        )

    def train(self, **kwargs):
        """
        Manage the training process of the solver.

        :param dict kwargs: Additional keyword arguments. See `pytorch-lightning
            Trainer API <https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-class-api>`_
            for details.
        """
        return super().fit(self.solver, datamodule=self.data_module, **kwargs)

    def test(self, **kwargs):
        """
        Manage the test process of the solver.

        :param dict kwargs: Additional keyword arguments. See `pytorch-lightning
            Trainer API <https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-class-api>`_
            for details.
        """
        return super().test(self.solver, datamodule=self.data_module, **kwargs)

    @property
    def solver(self):
        """
        Get the solver.

        :return: The solver.
        :rtype: SolverInterface
        """
        return self._solver

    @solver.setter
    def solver(self, solver):
        """
        Set the solver.

        :param SolverInterface solver: The solver to set.
        """
        self._solver = solver

    @staticmethod
    def _check_input_consistency(
        solver,
        train_size,
        test_size,
        val_size,
        repeat,
        automatic_batching,
        compile,
    ):
        """
        Verifies the consistency of the parameters for the solver configuration.

        :param SolverInterface solver: The solver.
        :param float train_size: The percentage of elements to include in the
            training dataset.
        :param float test_size: The percentage of elements to include in the
            test dataset.
        :param float val_size: The percentage of elements to include in the
            validation dataset.
        :param bool repeat: Whether to repeat the dataset data in each
            condition during training.
        :param bool automatic_batching: Whether to perform automatic batching
            with PyTorch.
        :param bool compile: If ``True``, the model is compiled before training.
        """

        check_consistency(solver, SolverInterface)
        check_consistency(train_size, float)
        check_consistency(test_size, float)
        check_consistency(val_size, float)
        if repeat is not None:
            check_consistency(repeat, bool)
        if automatic_batching is not None:
            check_consistency(automatic_batching, bool)
        if compile is not None:
            check_consistency(compile, bool)

    @staticmethod
    def _check_consistency_and_set_defaults(
        pin_memory, num_workers, shuffle, batch_size
    ):
        """
        Checks the consistency of input parameters and sets default values
        for missing or invalid parameters.

        :param bool pin_memory: Whether to use pinned memory for faster data
            transfer to GPU.
        :param int num_workers: The number of worker threads for data loading.
        :param bool shuffle: Whether to shuffle the data during training.
        :param int batch_size: The number of samples per batch to load.
        """
        if pin_memory is not None:
            check_consistency(pin_memory, bool)
        else:
            pin_memory = False
        if num_workers is not None:
            check_consistency(num_workers, int)
        else:
            num_workers = 0
        if shuffle is not None:
            check_consistency(shuffle, bool)
        else:
            shuffle = True
        if batch_size is not None:
            check_consistency(batch_size, int)
        return pin_memory, num_workers, shuffle, batch_size

    @property
    def compile(self):
        """
        Whether compilation is required or not.

        :return: ``True`` if compilation is required, ``False`` otherwise.
        :rtype: bool
        """
        return self._compile

    @compile.setter
    def compile(self, value):
        """
        Setting the value of compile.

        :param bool value: Whether compilation is required or not.
        """
        check_consistency(value, bool)
        self._compile = value
