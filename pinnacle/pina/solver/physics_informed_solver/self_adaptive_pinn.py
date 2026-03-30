"""Module for the Self-Adaptive PINN solver."""

from copy import deepcopy
import torch

from ...utils import check_consistency
from ...problem import InverseProblem
from ..solver import MultiSolverInterface
from .pinn_interface import PINNInterface


class Weights(torch.nn.Module):
    """
    Implementation of the mask model for the self-adaptive weights of the
    :class:`SelfAdaptivePINN` solver.
    """

    def __init__(self, func, num_points):
        """
        Initialization of the :class:`Weights` class.

        :param torch.nn.Module func: the mask model.
        :param int num_points: the number of input points.
        """
        super().__init__()

        # Check consistency
        check_consistency(func, torch.nn.Module)

        # Initialize the weights as a learnable parameter
        self.sa_weights = torch.nn.Parameter(torch.zeros(num_points, 1))
        self.func = func

    def forward(self):
        """
        Forward pass implementation for the mask module.

        :return: evaluation of self adaptive weights through the mask.
        :rtype: torch.Tensor
        """
        return self.func(self.sa_weights)


class SelfAdaptivePINN(PINNInterface, MultiSolverInterface):
    r"""
    Self-Adaptive Physics-Informed Neural Network (SelfAdaptivePINN) solver
    class. This class implements the Self-Adaptive Physics-Informed Neural
    Network solver, using a user specified ``model`` to solve a specific
    ``problem``. It can be used to solve both forward and inverse problems.

    The Self-Adapive Physics-Informed Neural Network solver aims to find the
    solution :math:`\mathbf{u}:\Omega\rightarrow\mathbb{R}^m` of a differential
    problem:

    .. math::

        \begin{cases}
        \mathcal{A}[\mathbf{u}](\mathbf{x})=0\quad,\mathbf{x}\in\Omega\\
        \mathcal{B}[\mathbf{u}](\mathbf{x})=0\quad,
        \mathbf{x}\in\partial\Omega
        \end{cases}
    
    integrating pointwise loss evaluation using a mask :math:m and self-adaptive
    weights, which allow the model to focus on regions of the domain where the
    residual is higher.

    The loss function to solve the problem is

    .. math::

        \mathcal{L}_{\rm{problem}} = \frac{1}{N} \sum_{i=1}^{N_\Omega} m
        \left( \lambda_{\Omega}^{i} \right) \mathcal{L} \left( \mathcal{A}
        [\mathbf{u}](\mathbf{x}) \right) + \frac{1}{N} 
        \sum_{i=1}^{N_{\partial\Omega}}
        m \left( \lambda_{\partial\Omega}^{i} \right) \mathcal{L} 
        \left( \mathcal{B}[\mathbf{u}](\mathbf{x})
        \right),
    
    denoting the self adaptive weights as
    :math:`\lambda_{\Omega}^1, \dots, \lambda_{\Omega}^{N_\Omega}` and
    :math:`\lambda_{\partial \Omega}^1, \dots, 
    \lambda_{\Omega}^{N_\partial \Omega}`
    for :math:`\Omega` and :math:`\partial \Omega`, respectively.

    The Self-Adaptive Physics-Informed Neural Network solver identifies the
    solution and appropriate self adaptive weights by solving the following
    optimization problem:

    .. math::

        \min_{w} \max_{\lambda_{\Omega}^k, \lambda_{\partial \Omega}^s}
        \mathcal{L} ,
    
    where :math:`w` denotes the network parameters, and :math:`\mathcal{L}` is a
    specific loss function, , typically the MSE:

    .. math::
        \mathcal{L}(v) = \| v \|^2_2.

    .. seealso::
        **Original reference**: McClenny, Levi D., and Ulisses M. Braga-Neto.
        *Self-adaptive physics-informed neural networks.*
        Journal of Computational Physics 474 (2023): 111722.
        DOI: `10.1016/j.jcp.2022.111722
        <https://doi.org/10.1016/j.jcp.2022.111722>`_.
    """

    def __init__(
        self,
        problem,
        model,
        weight_function=torch.nn.Sigmoid(),
        optimizer_model=None,
        optimizer_weights=None,
        scheduler_model=None,
        scheduler_weights=None,
        weighting=None,
        loss=None,
    ):
        """
        Initialization of the :class:`SelfAdaptivePINN` class.

        :param AbstractProblem problem: The problem to be solved.
        :param torch.nn.Module model: The model to be used.
        :param torch.nn.Module weight_function: The Self-Adaptive mask model.
            Default is ``torch.nn.Sigmoid()``.
        :param Optimizer optimizer_model: The optimizer of the ``model``.
            If ``None``, the :class:`torch.optim.Adam` optimizer is used.
            Default is ``None``.
        :param Optimizer optimizer_weights: The optimizer of the
            ``weight_function``.
            If ``None``, the :class:`torch.optim.Adam` optimizer is used.
            Default is ``None``.
        :param Scheduler scheduler_model: Learning rate scheduler for the
            ``model``.
            If ``None``, the :class:`torch.optim.lr_scheduler.ConstantLR`
            scheduler is used. Default is ``None``.
        :param Scheduler scheduler_weights: Learning rate scheduler for the
            ``weight_function``.
            If ``None``, the :class:`torch.optim.lr_scheduler.ConstantLR`
            scheduler is used. Default is ``None``.
        :param WeightingInterface weighting: The weighting schema to be used.
            If ``None``, no weighting schema is used. Default is ``None``.
        :param torch.nn.Module loss: The loss function to be minimized.
            If ``None``, the :class:`torch.nn.MSELoss` loss is used.
            Default is `None`.
        """
        # Check consistency
        check_consistency(weight_function, torch.nn.Module)

        # Define a ModuleDict for the weights
        weights = {}
        for cond, data in problem.input_pts.items():
            weights[cond] = Weights(func=weight_function, num_points=len(data))
        weights = torch.nn.ModuleDict(weights)

        super().__init__(
            models=[model, weights],
            problem=problem,
            optimizers=[optimizer_model, optimizer_weights],
            schedulers=[scheduler_model, scheduler_weights],
            weighting=weighting,
            loss=loss,
        )

        # Extract the reduction method from the loss function
        self._reduction = self._loss_fn.reduction

        # Set the loss function to return non-aggregated losses
        self._loss_fn = type(self._loss_fn)(reduction="none")

    def training_step(self, batch, batch_idx, **kwargs):
        """
        Solver training step. It computes the optimization cycle and aggregates
        the losses using the ``weighting`` attribute.

        :param list[tuple[str, dict]] batch: A batch of data. Each element is a
            tuple containing a condition name and a dictionary of points.
        :param int batch_idx: The index of the current batch.
        :param dict kwargs: Additional keyword arguments passed to
            ``optimization_cycle``.
        :return: The loss of the training step.
        :rtype: torch.Tensor
        """
        # Weights optimization
        self.optimizer_weights.instance.zero_grad()
        loss = self._optimization_cycle(
            batch=batch, batch_idx=batch_idx, **kwargs
        )
        self.manual_backward(-loss)
        self.optimizer_weights.instance.step()
        self.scheduler_weights.instance.step()

        # Model optimization
        self.optimizer_model.instance.zero_grad()
        loss = self._optimization_cycle(
            batch=batch, batch_idx=batch_idx, **kwargs
        )
        self.manual_backward(loss)
        self.optimizer_model.instance.step()
        self.scheduler_model.instance.step()

        # Log the loss
        self.store_log("train_loss", loss, self.get_batch_size(batch))

        return loss

    @torch.set_grad_enabled(True)
    def validation_step(self, batch, **kwargs):
        """
        The validation step for the Self-Adaptive PINN solver. It returns the
        average residual computed with the ``loss`` function not aggregated.

        :param list[tuple[str, dict]] batch: A batch of data. Each element is a
            tuple containing a condition name and a dictionary of points.
        :param dict kwargs: Additional keyword arguments passed to
            ``optimization_cycle``.
        :return: The loss of the validation step.
        :rtype: torch.Tensor
        """
        losses = self.optimization_cycle(batch=batch, **kwargs)

        # Aggregate losses for each condition
        for cond, loss in losses.items():
            losses[cond] = self._apply_reduction(loss=losses[cond])

        loss = (sum(losses.values()) / len(losses)).as_subclass(torch.Tensor)
        self.store_log("val_loss", loss, self.get_batch_size(batch))
        return loss

    @torch.set_grad_enabled(True)
    def test_step(self, batch, **kwargs):
        """
        The test step for the Self-Adaptive PINN solver. It returns the average
        residual computed with the ``loss`` function not aggregated.

        :param list[tuple[str, dict]] batch: A batch of data. Each element is a
            tuple containing a condition name and a dictionary of points.
        :param dict kwargs: Additional keyword arguments passed to
            ``optimization_cycle``.
        :return: The loss of the test step.
        :rtype: torch.Tensor
        """
        losses = self.optimization_cycle(batch=batch, **kwargs)

        # Aggregate losses for each condition
        for cond, loss in losses.items():
            losses[cond] = self._apply_reduction(loss=losses[cond])

        loss = (sum(losses.values()) / len(losses)).as_subclass(torch.Tensor)
        self.store_log("test_loss", loss, self.get_batch_size(batch))
        return loss

    def loss_phys(self, samples, equation):
        """
        Computes the physics loss for the physics-informed solver based on the
        provided samples and equation.

        :param LabelTensor samples: The samples to evaluate the physics loss.
        :param EquationInterface equation: The governing equation.
        :return: The computed physics loss.
        :rtype: LabelTensor
        """
        residuals = self.compute_residual(samples, equation)
        return self._loss_fn(residuals, torch.zeros_like(residuals))

    def loss_data(self, input, target):
        """
        Compute the data loss for the Self-Adaptive PINN solver by evaluating
        the loss between the network's output and the true solution. This method
        should not be overridden, if not intentionally.

        :param input: The input to the neural network.
        :type input: LabelTensor | torch.Tensor
        :param target: The target to compare with the network's output.
        :type target: LabelTensor | torch.Tensor
        :return: The supervised loss, averaged over the number of observations.
        :rtype: LabelTensor | torch.Tensor
        """
        return self._loss_fn(self.forward(input), target)

    def forward(self, x):
        """
        Forward pass.

        :param x: Input tensor.
        :type x: torch.Tensor | LabelTensor
        :return: The output of the neural network.
        :rtype: torch.Tensor | LabelTensor
        """
        return self.model(x)

    def configure_optimizers(self):
        """
        Optimizer configuration.

        :return: The optimizers and the schedulers
        :rtype: tuple[list[Optimizer], list[Scheduler]]
        """
        # Hook the optimizers to the models
        self.optimizer_model.hook(self.model.parameters())
        self.optimizer_weights.hook(self.weights.parameters())

        # Add unknown parameters to optimization list in case of InverseProblem
        if isinstance(self.problem, InverseProblem):
            self.optimizer_model.instance.add_param_group(
                {
                    "params": [
                        self._params[var]
                        for var in self.problem.unknown_variables
                    ]
                }
            )

        # Hook the schedulers to the optimizers
        self.scheduler_model.hook(self.optimizer_model)
        self.scheduler_weights.hook(self.optimizer_weights)

        return (
            [self.optimizer_model.instance, self.optimizer_weights.instance],
            [self.scheduler_model.instance, self.scheduler_weights.instance],
        )

    def _optimization_cycle(self, batch, batch_idx, **kwargs):
        """
        Aggregate the loss for each condition in the batch.

        :param list[tuple[str, dict]] batch: A batch of data. Each element is a
            tuple containing a condition name and a dictionary of points.
        :param int batch_idx: The index of the current batch.
        :param dict kwargs: Additional keyword arguments passed to
            ``optimization_cycle``.
        :return: The losses computed for all conditions in the batch, casted
            to a subclass of :class:`torch.Tensor`. It should return a dict
            containing the condition name and the associated scalar loss.
        :rtype: dict
        """
        # Compute non-aggregated residuals
        residuals = self.optimization_cycle(batch)

        # Compute losses
        losses = {}
        for cond, res in residuals.items():

            weight_tensor = self.weights[cond]()

            # Get the correct indices for the weights. Modulus is used according
            # to the number of points in the condition, as in the PinaDataset.
            len_res = len(res)
            idx = torch.arange(
                batch_idx * len_res,
                (batch_idx + 1) * len_res,
                device=res.device,
            ) % len(self.problem.input_pts[cond])

            # Apply the weights to the residuals
            losses[cond] = self._apply_reduction(
                loss=(res * weight_tensor[idx])
            )

            # Store log
            self.store_log(
                f"{cond}_loss", losses[cond].item(), self.get_batch_size(batch)
            )

        # Clamp unknown parameters in InverseProblem (if needed)
        self._clamp_params()

        # Aggregate
        loss = self.weighting.aggregate(losses).as_subclass(torch.Tensor)

        return loss

    def _apply_reduction(self, loss):
        """
        Apply the specified reduction to the loss. The reduction is deferred
        until the end of the optimization cycle to allow self-adaptive weights
        to be applied to each point beforehand.

        :param torch.Tensor loss: The loss tensor to be reduced.
        :return: The reduced loss tensor.
        :rtype: torch.Tensor
        :raises ValueError: If the reduction method is neither "mean" nor "sum".
        """
        # Apply the specified reduction method
        if self._reduction == "mean":
            return loss.mean()
        if self._reduction == "sum":
            return loss.sum()

        # Raise an error if the reduction method is not recognized
        raise ValueError(
            f"Unknown reduction: {self._reduction}."
            " Supported reductions are 'mean' and 'sum'."
        )

    @property
    def model(self):
        """
        The model.

        :return: The model.
        :rtype: torch.nn.Module
        """
        return self.models[0]

    @property
    def weights(self):
        """
        The self-adaptive weights.

        :return: The self-adaptive weights.
        :rtype: torch.nn.Module
        """
        return self.models[1]

    @property
    def scheduler_model(self):
        """
        The scheduler associated to the model.

        :return: The scheduler for the model.
        :rtype: Scheduler
        """
        return self.schedulers[0]

    @property
    def scheduler_weights(self):
        """
        The scheduler associated to the mask model.

        :return: The scheduler for the mask model.
        :rtype: Scheduler
        """
        return self.schedulers[1]

    @property
    def optimizer_model(self):
        """
        Returns the optimizer associated to the model.

        :return: The optimizer for the model.
        :rtype: Optimizer
        """
        return self.optimizers[0]

    @property
    def optimizer_weights(self):
        """
        The optimizer associated to the mask model.

        :return: The optimizer for the mask model.
        :rtype: Optimizer
        """
        return self.optimizers[1]
