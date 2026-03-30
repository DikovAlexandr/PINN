"""Solver module."""

from abc import ABCMeta, abstractmethod
import lightning
import torch

from torch._dynamo import OptimizedModule
from ..problem import AbstractProblem, InverseProblem
from ..optim import Optimizer, Scheduler, TorchOptimizer, TorchScheduler
from ..loss import WeightingInterface
from ..loss.scalar_weighting import _NoWeighting
from ..utils import check_consistency, labelize_forward


class SolverInterface(lightning.pytorch.LightningModule, metaclass=ABCMeta):
    """
    Abstract base class for PINA solvers. All specific solvers must inherit
    from this interface. This class extends
    :class:`~lightning.pytorch.core.LightningModule`, providing additional
    functionalities for defining and optimizing Deep Learning models.

    By inheriting from this base class, solvers gain access to built-in training
    loops, logging utilities, and optimization techniques.
    """

    def __init__(self, problem, weighting, use_lt):
        """
        Initialization of the :class:`SolverInterface` class.

        :param AbstractProblem problem: The problem to be solved.
        :param WeightingInterface weighting: The weighting schema to be used.
            If ``None``, no weighting schema is used. Default is ``None``.
        :param bool use_lt: If ``True``, the solver uses LabelTensors as input.
        """
        super().__init__()

        # check consistency of the problem
        check_consistency(problem, AbstractProblem)
        self._check_solver_consistency(problem)
        self._pina_problem = problem

        # check consistency of the weighting and hook the condition names
        if weighting is None:
            weighting = _NoWeighting()
        check_consistency(weighting, WeightingInterface)
        self._pina_weighting = weighting
        weighting._solver = self

        # check consistency use_lt
        check_consistency(use_lt, bool)
        self._use_lt = use_lt

        # if use_lt is true add extract operation in input
        if use_lt is True:
            self.forward = labelize_forward(
                forward=self.forward,
                input_variables=problem.input_variables,
                output_variables=problem.output_variables,
            )

        # PINA private attributes (some are overridden by derived classes)
        self._pina_problem = problem
        self._pina_models = None
        self._pina_optimizers = None
        self._pina_schedulers = None

        # inverse problem handling
        if isinstance(self.problem, InverseProblem):
            self._params = self.problem.unknown_parameters
            self._clamp_params = self._clamp_inverse_problem_params
        else:
            self._params = None
            self._clamp_params = lambda: None

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Abstract method for the forward pass implementation.

        :param args: The input tensor.
        :type args: torch.Tensor | LabelTensor | Data | Graph
        :param dict kwargs: Additional keyword arguments.
        """

    @abstractmethod
    def optimization_cycle(self, batch):
        """
        The optimization cycle for the solvers.

        :param list[tuple[str, dict]] batch: A batch of data. Each element is a
            tuple containing a condition name and a dictionary of points.
        :return: The losses computed for all conditions in the batch, casted
            to a subclass of :class:`torch.Tensor`. It should return a dict
            containing the condition name and the associated scalar loss.
        :rtype: dict
        """

    def training_step(self, batch, **kwargs):
        """
        Solver training step. It computes the optimization cycle and aggregates
        the losses using the ``weighting`` attribute.

        :param list[tuple[str, dict]] batch: A batch of data. Each element is a
            tuple containing a condition name and a dictionary of points.
        :param dict kwargs: Additional keyword arguments passed to
            ``optimization_cycle``.
        :return: The loss of the training step.
        :rtype: torch.Tensor
        """
        loss = self._optimization_cycle(batch=batch, **kwargs)
        self.store_log("train_loss", loss, self.get_batch_size(batch))
        return loss

    def validation_step(self, batch, **kwargs):
        """
        Solver validation step. It computes the optimization cycle and
        averages the losses. No aggregation using the ``weighting`` attribute is
        performed.

        :param list[tuple[str, dict]] batch: A batch of data. Each element is a
            tuple containing a condition name and a dictionary of points.
        :param dict kwargs: Additional keyword arguments passed to
            ``optimization_cycle``.
        :return: The loss of the training step.
        :rtype: torch.Tensor
        """
        losses = self.optimization_cycle(batch=batch, **kwargs)
        loss = (sum(losses.values()) / len(losses)).as_subclass(torch.Tensor)
        self.store_log("val_loss", loss, self.get_batch_size(batch))
        return loss

    def test_step(self, batch, **kwargs):
        """
        Solver test step. It computes the optimization cycle and
        averages the losses. No aggregation using the ``weighting`` attribute is
        performed.

        :param list[tuple[str, dict]] batch: A batch of data. Each element is a
            tuple containing a condition name and a dictionary of points.
        :param dict kwargs: Additional keyword arguments passed to
            ``optimization_cycle``.
        :return: The loss of the training step.
        :rtype: torch.Tensor
        """
        losses = self.optimization_cycle(batch=batch, **kwargs)
        loss = (sum(losses.values()) / len(losses)).as_subclass(torch.Tensor)
        self.store_log("test_loss", loss, self.get_batch_size(batch))
        return loss

    def store_log(self, name, value, batch_size):
        """
        Store the log of the solver.

        :param str name: The name of the log.
        :param torch.Tensor value: The value of the log.
        :param int batch_size: The size of the batch.
        """

        self.log(
            name=name,
            value=value,
            batch_size=batch_size,
            **self.trainer.logging_kwargs,
        )

    def setup(self, stage):
        """
        This method is called at the start of the train and test process to
        compile the model if the :class:`~pina.trainer.Trainer`
        ``compile`` is ``True``.

        :param str stage: The current stage of the training process
            (e.g., ``fit``, ``validate``, ``test``, ``predict``).
        :return: The result of the parent class ``setup`` method.
        :rtype: Any
        """
        if self.trainer.compile and not self._is_compiled():
            self._setup_compile()
        return super().setup(stage)

    def _is_compiled(self):
        """
        Check if the model is compiled.

        :return: ``True`` if the model is compiled, ``False`` otherwise.
        :rtype: bool
        """
        for model in self._pina_models:
            if not isinstance(model, OptimizedModule):
                return False
        return True

    def _setup_compile(self):
        """
        Compile all models in the solver using ``torch.compile``.

        This method iterates through each model stored in the solver
        list and attempts to compile them for optimized execution. It supports
        models of type `torch.nn.Module` and `torch.nn.ModuleDict`. For models
        stored in a `ModuleDict`, each submodule is compiled individually.
        Models on Apple Silicon (MPS) use the 'eager' backend,
        while others use 'inductor'.

        :raises RuntimeError: If a model is neither `torch.nn.Module`
            nor `torch.nn.ModuleDict`.
        """
        for i, model in enumerate(self._pina_models):
            if isinstance(model, torch.nn.ModuleDict):
                for name, module in model.items():
                    self._pina_models[i][name] = self._compile_modules(module)
            elif isinstance(model, torch.nn.Module):
                self._pina_models[i] = self._compile_modules(model)
            else:
                raise RuntimeError(
                    "Compilation available only for "
                    "torch.nn.Module or torch.nn.ModuleDict."
                )

    def _check_solver_consistency(self, problem):
        """
        Check the consistency of the solver with the problem formulation.

        :param AbstractProblem problem: The problem to be solved.
        """
        for condition in problem.conditions.values():
            check_consistency(condition, self.accepted_conditions_types)

    def _optimization_cycle(self, batch, **kwargs):
        """
        Aggregate the loss for each condition in the batch.

        :param list[tuple[str, dict]] batch: A batch of data. Each element is a
            tuple containing a condition name and a dictionary of points.
        :param dict kwargs: Additional keyword arguments passed to
            ``optimization_cycle``.
        :return: The losses computed for all conditions in the batch, casted
            to a subclass of :class:`torch.Tensor`. It should return a dict
            containing the condition name and the associated scalar loss.
        :rtype: dict
        """
        # compute losses
        losses = self.optimization_cycle(batch)
        # clamp unknown parameters in InverseProblem (if needed)
        self._clamp_params()
        # store log
        for name, value in losses.items():
            self.store_log(
                f"{name}_loss", value.item(), self.get_batch_size(batch)
            )
        # aggregate
        loss = self.weighting.aggregate(losses).as_subclass(torch.Tensor)
        return loss

    def _clamp_inverse_problem_params(self):
        """
        Clamps the parameters of the inverse problem solver to specified ranges.
        """
        for v in self._params:
            self._params[v].data.clamp_(
                self.problem.unknown_parameter_domain.range[v][0],
                self.problem.unknown_parameter_domain.range[v][1],
            )

    @staticmethod
    def _compile_modules(model):
        """
        Perform the compilation of the model.

        This method attempts to compile the given PyTorch model
        using ``torch.compile`` to improve execution performance. The
        backend is selected based on the device on which the model resides:
        ``eager`` is used for MPS devices (Apple Silicon), and ``inductor``
        is used for all others.

        If compilation fails, the method prints the error and returns the
        original, uncompiled model.

        :param torch.nn.Module model: The model to compile.
        :raises Exception: If the compilation fails.
        :return: The compiled model.
        :rtype: torch.nn.Module
        """
        model_device = next(model.parameters()).device
        try:
            if model_device == torch.device("mps:0"):
                model = torch.compile(model, backend="eager")
            else:
                model = torch.compile(model, backend="inductor")
        except Exception as e:
            print("Compilation failed, running in normal mode.:\n", e)
        return model

    @staticmethod
    def get_batch_size(batch):
        """
        Get the batch size.

        :param list[tuple[str, dict]] batch: A batch of data. Each element is a
            tuple containing a condition name and a dictionary of points.
        :return: The size of the batch.
        :rtype: int
        """

        batch_size = 0
        for data in batch:
            batch_size += len(data[1]["input"])
        return batch_size

    @staticmethod
    def default_torch_optimizer():
        """
        Set the default optimizer to :class:`torch.optim.Adam`.

        :return: The default optimizer.
        :rtype: Optimizer
        """
        return TorchOptimizer(torch.optim.Adam, lr=0.001)

    @staticmethod
    def default_torch_scheduler():
        """
        Set the default scheduler to
        :class:`torch.optim.lr_scheduler.ConstantLR`.

        :return: The default scheduler.
        :rtype: Scheduler
        """
        return TorchScheduler(torch.optim.lr_scheduler.ConstantLR, factor=1.0)

    @property
    def problem(self):
        """
        The problem instance.

        :return: The problem instance.
        :rtype: :class:`~pina.problem.abstract_problem.AbstractProblem`
        """
        return self._pina_problem

    @property
    def use_lt(self):
        """
        Using LabelTensors as input during training.

        :return: The use_lt attribute.
        :rtype: bool
        """
        return self._use_lt

    @property
    def weighting(self):
        """
        The weighting schema.

        :return: The weighting schema.
        :rtype: :class:`~pina.loss.weighting_interface.WeightingInterface`
        """
        return self._pina_weighting


class SingleSolverInterface(SolverInterface, metaclass=ABCMeta):
    """
    Base class for PINA solvers using a single :class:`torch.nn.Module`.
    """

    def __init__(
        self,
        problem,
        model,
        optimizer=None,
        scheduler=None,
        weighting=None,
        use_lt=True,
    ):
        """
        Initialization of the :class:`SingleSolverInterface` class.

        :param AbstractProblem problem: The problem to be solved.
        :param torch.nn.Module model: The neural network model to be used.
        :param Optimizer optimizer: The optimizer to be used.
            If ``None``, the :class:`torch.optim.Adam` optimizer is
            used. Default is ``None``.
        :param Scheduler scheduler: The scheduler to be used.
            If ``None``, the :class:`torch.optim.lr_scheduler.ConstantLR`
            scheduler is used. Default is ``None``.
        :param WeightingInterface weighting: The weighting schema to be used.
            If ``None``, no weighting schema is used. Default is ``None``.
        :param bool use_lt: If ``True``, the solver uses LabelTensors as input.
        """
        if optimizer is None:
            optimizer = self.default_torch_optimizer()

        if scheduler is None:
            scheduler = self.default_torch_scheduler()

        super().__init__(problem=problem, use_lt=use_lt, weighting=weighting)

        # check consistency of models argument and encapsulate in list
        check_consistency(model, torch.nn.Module)
        # check scheduler consistency and encapsulate in list
        check_consistency(scheduler, Scheduler)
        # check optimizer consistency and encapsulate in list
        check_consistency(optimizer, Optimizer)

        # initialize the model (needed by Lightining to go to different devices)
        self._pina_models = torch.nn.ModuleList([model])
        self._pina_optimizers = [optimizer]
        self._pina_schedulers = [scheduler]

    def forward(self, x):
        """
        Forward pass implementation.

        :param x: Input tensor.
        :type x: torch.Tensor | LabelTensor | Graph | Data
        :return: Solver solution.
        :rtype: torch.Tensor | LabelTensor | Graph | Data
        """
        return self.model(x)

    def configure_optimizers(self):
        """
        Optimizer configuration for the solver.

        :return: The optimizer and the scheduler
        :rtype: tuple[list[Optimizer], list[Scheduler]]
        """
        self.optimizer.hook(self.model.parameters())
        if isinstance(self.problem, InverseProblem):
            self.optimizer.instance.add_param_group(
                {
                    "params": [
                        self._params[var]
                        for var in self.problem.unknown_variables
                    ]
                }
            )
        self.scheduler.hook(self.optimizer)
        return ([self.optimizer.instance], [self.scheduler.instance])

    @property
    def model(self):
        """
        The model used for training.

        :return: The model used for training.
        :rtype: torch.nn.Module
        """
        return self._pina_models[0]

    @property
    def scheduler(self):
        """
        The scheduler used for training.

        :return: The scheduler used for training.
        :rtype: Scheduler
        """
        return self._pina_schedulers[0]

    @property
    def optimizer(self):
        """
        The optimizer used for training.

        :return: The optimizer used for training.
        :rtype: Optimizer
        """
        return self._pina_optimizers[0]


class MultiSolverInterface(SolverInterface, metaclass=ABCMeta):
    """
    Base class for PINA solvers using multiple :class:`torch.nn.Module`.
    """

    def __init__(
        self,
        problem,
        models,
        optimizers=None,
        schedulers=None,
        weighting=None,
        use_lt=True,
    ):
        """
        Initialization of the :class:`MultiSolverInterface` class.

        :param AbstractProblem problem: The problem to be solved.
        :param models: The neural network models to be used.
        :type model: list[torch.nn.Module] | tuple[torch.nn.Module]
        :param list[Optimizer] optimizers: The optimizers to be used.
            If ``None``, the :class:`torch.optim.Adam` optimizer is used for all
            models. Default is ``None``.
        :param list[Scheduler] schedulers: The schedulers to be used.
            If ``None``, the :class:`torch.optim.lr_scheduler.ConstantLR`
            scheduler is used for all the models. Default is ``None``.
        :param WeightingInterface weighting: The weighting schema to be used.
            If ``None``, no weighting schema is used. Default is ``None``.
        :param bool use_lt: If ``True``, the solver uses LabelTensors as input.
        :raises ValueError: If the models are not a list or tuple with length
            greater than one.

        .. warning::
            :class:`MultiSolverInterface` uses manual optimization by setting
            ``automatic_optimization=False`` in
            :class:`~lightning.pytorch.core.LightningModule`. For more
            information on manual optimization please
            see `here <https://lightning.ai/docs/pytorch/stable/\
                model/manual_optimization.html>`_.
        """
        if not isinstance(models, (list, tuple)) or len(models) < 2:
            raise ValueError(
                "models should be list[torch.nn.Module] or "
                "tuple[torch.nn.Module] with len greater than "
                "one."
            )

        if optimizers is None:
            optimizers = [
                self.default_torch_optimizer() for _ in range(len(models))
            ]

        if schedulers is None:
            schedulers = [
                self.default_torch_scheduler() for _ in range(len(models))
            ]

        if any(opt is None for opt in optimizers):
            optimizers = [
                self.default_torch_optimizer() if opt is None else opt
                for opt in optimizers
            ]

        if any(sched is None for sched in schedulers):
            schedulers = [
                self.default_torch_scheduler() if sched is None else sched
                for sched in schedulers
            ]

        super().__init__(problem=problem, use_lt=use_lt, weighting=weighting)

        # check consistency of models argument and encapsulate in list
        check_consistency(models, torch.nn.Module)

        # check scheduler consistency and encapsulate in list
        check_consistency(schedulers, Scheduler)

        # check optimizer consistency and encapsulate in list
        check_consistency(optimizers, Optimizer)

        # check length consistency optimizers
        if len(models) != len(optimizers):
            raise ValueError(
                "You must define one optimizer for each model."
                f"Got {len(models)} models, and {len(optimizers)}"
                " optimizers."
            )
        if len(schedulers) != len(optimizers):
            raise ValueError(
                "You must define one scheduler for each optimizer."
                f"Got {len(schedulers)} schedulers, and {len(optimizers)}"
                " optimizers."
            )

        # initialize the model
        self._pina_models = torch.nn.ModuleList(models)
        self._pina_optimizers = optimizers
        self._pina_schedulers = schedulers

        # Set automatic optimization to False.
        # For more information on manual optimization see:
        # http://lightning.ai/docs/pytorch/stable/model/manual_optimization.html
        self.automatic_optimization = False

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """
        This method is called at the end of each training batch and overrides
        the PyTorch Lightning implementation to log checkpoints.

        :param torch.Tensor outputs: The ``model``'s output for the current
            batch.
        :param list[tuple[str, dict]] batch: A batch of data. Each element is a
            tuple containing a condition name and a dictionary of points.
        :param int batch_idx: The index of the current batch.
        """
        # increase by one the counter of optimization to save loggers
        epoch_loop = self.trainer.fit_loop.epoch_loop
        epoch_loop.manual_optimization.optim_step_progress.total.completed += 1
        return super().on_train_batch_end(outputs, batch, batch_idx)

    def configure_optimizers(self):
        """
        Optimizer configuration for the solver.

        :return: The optimizer and the scheduler
        :rtype: tuple[list[Optimizer], list[Scheduler]]
        """
        for optimizer, scheduler, model in zip(
            self.optimizers, self.schedulers, self.models
        ):
            optimizer.hook(model.parameters())
            scheduler.hook(optimizer)

        return (
            [optimizer.instance for optimizer in self.optimizers],
            [scheduler.instance for scheduler in self.schedulers],
        )

    @property
    def models(self):
        """
        The models used for training.

        :return: The models used for training.
        :rtype: torch.nn.ModuleList
        """
        return self._pina_models

    @property
    def optimizers(self):
        """
        The optimizers used for training.

        :return: The optimizers used for training.
        :rtype: list[Optimizer]
        """
        return self._pina_optimizers

    @property
    def schedulers(self):
        """
        The schedulers used for training.

        :return: The schedulers used for training.
        :rtype: list[Scheduler]
        """
        return self._pina_schedulers
