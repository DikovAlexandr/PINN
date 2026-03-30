"""Module for the Competitive PINN solver."""

import copy
import torch

from ...problem import InverseProblem
from .pinn_interface import PINNInterface
from ..solver import MultiSolverInterface


class CompetitivePINN(PINNInterface, MultiSolverInterface):
    r"""
    Competitive Physics-Informed Neural Network (CompetitivePINN) solver class.
    This class implements the Competitive Physics-Informed Neural Network
    solver, using a user specified ``model`` to solve a specific ``problem``.
    It can be used to solve both forward and inverse problems.

    The Competitive Physics-Informed Neural Network solver aims to find the
    solution :math:`\mathbf{u}:\Omega\rightarrow\mathbb{R}^m` of a differential
    problem:

    .. math::

        \begin{cases}
        \mathcal{A}[\mathbf{u}](\mathbf{x})=0\quad,\mathbf{x}\in\Omega\\
        \mathcal{B}[\mathbf{u}](\mathbf{x})=0\quad,
        \mathbf{x}\in\partial\Omega
        \end{cases}

    minimizing the loss function with respect to the model parameters, while 
    maximizing it with respect to the discriminator parameters:

    .. math::
        \mathcal{L}_{\rm{problem}} = \frac{1}{N}\sum_{i=1}^N
        \mathcal{L}(D(\mathbf{x}_i)\mathcal{A}[\mathbf{u}](\mathbf{x}_i))+
        \frac{1}{N}\sum_{i=1}^N
        \mathcal{L}(D(\mathbf{x}_i)\mathcal{B}[\mathbf{u}](\mathbf{x}_i)),

    where :math:D is the discriminator network, which identifies the points
    where the model performs worst, and :math:\mathcal{L} is a specific loss
    function, typically the MSE:

    .. math::
        \mathcal{L}(v) = \| v \|^2_2.

    .. seealso::

        **Original reference**: Zeng, Qi, et al.
        *Competitive physics informed networks.*
        International Conference on Learning Representations, ICLR 2022
        `OpenReview Preprint <https://openreview.net/forum?id=z9SIj-IM7tn>`_.
    """

    def __init__(
        self,
        problem,
        model,
        discriminator=None,
        optimizer_model=None,
        optimizer_discriminator=None,
        scheduler_model=None,
        scheduler_discriminator=None,
        weighting=None,
        loss=None,
    ):
        """
        Initialization of the :class:`CompetitivePINN` class.

        :param AbstractProblem problem: The problem to be solved.
        :param torch.nn.Module model: The neural network model to be used.
        :param torch.nn.Module discriminator: The discriminator to be used.
            If ``None``, the discriminator is a deepcopy of the ``model``.
            Default is ``None``.
        :param torch.optim.Optimizer optimizer_model: The optimizer of the
            ``model``. If ``None``, the :class:`torch.optim.Adam` optimizer is
            used. Default is ``None``.
        :param torch.optim.Optimizer optimizer_discriminator: The optimizer of
            the ``discriminator``. If ``None``, the :class:`torch.optim.Adam`
            optimizer is used. Default is ``None``.
        :param Scheduler scheduler_model: Learning rate scheduler for the
            ``model``.
            If ``None``, the :class:`torch.optim.lr_scheduler.ConstantLR`
            scheduler is used. Default is ``None``.
        :param Scheduler scheduler_discriminator: Learning rate scheduler for
            the ``discriminator``.
            If ``None``, the :class:`torch.optim.lr_scheduler.ConstantLR`
            scheduler is used. Default is ``None``.
        :param WeightingInterface weighting: The weighting schema to be used.
            If ``None``, no weighting schema is used. Default is ``None``.
        :param torch.nn.Module loss: The loss function to be minimized.
            If ``None``, the :class:`torch.nn.MSELoss` loss is used.
            Default is `None`.
        """
        if discriminator is None:
            discriminator = copy.deepcopy(model)

        super().__init__(
            models=[model, discriminator],
            problem=problem,
            optimizers=[optimizer_model, optimizer_discriminator],
            schedulers=[scheduler_model, scheduler_discriminator],
            weighting=weighting,
            loss=loss,
        )

    def forward(self, x):
        """
        Forward pass.

        :param LabelTensor x: Input tensor.
        :return: The output of the neural network.
        :rtype: LabelTensor
        """
        return self.neural_net(x)

    def training_step(self, batch):
        """
        Solver training step, overridden to perform manual optimization.

        :param list[tuple[str, dict]] batch: A batch of data. Each element is a
            tuple containing a condition name and a dictionary of points.
        :return: The aggregated loss.
        :rtype: LabelTensor
        """
        # train model
        self.optimizer_model.instance.zero_grad()
        loss = super().training_step(batch)
        self.manual_backward(loss)
        self.optimizer_model.instance.step()
        self.scheduler_model.instance.step()

        # train discriminator
        self.optimizer_discriminator.instance.zero_grad()
        loss = super().training_step(batch)
        self.manual_backward(-loss)
        self.optimizer_discriminator.instance.step()
        self.scheduler_discriminator.instance.step()

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
        # Compute discriminator bets
        discriminator_bets = self.discriminator(samples)

        # Compute residual and multiply discriminator_bets
        residual = self.compute_residual(samples=samples, equation=equation)
        residual = residual * discriminator_bets

        # Compute competitive residual.
        loss_val = self._loss_fn(
            torch.zeros_like(residual, requires_grad=True),
            residual,
        )
        return loss_val

    def loss_data(self, input, target):
        """
        Compute the data loss for the PINN solver by evaluating the loss
        between the network's output and the true solution. This method should
        not be overridden, if not intentionally.

        :param input: The input to the neural network.
        :type input: LabelTensor
        :param target: The target to compare with the network's output.
        :type target: LabelTensor
        :return: The supervised loss, averaged over the number of observations.
        :rtype: LabelTensor
        """
        return self._loss_fn(self.forward(input), target)

    def configure_optimizers(self):
        """
        Optimizer configuration.

        :return: The optimizers and the schedulers
        :rtype: tuple[list[Optimizer], list[Scheduler]]
        """
        # If the problem is an InverseProblem, add the unknown parameters
        # to the parameters to be optimized
        self.optimizer_model.hook(self.neural_net.parameters())
        self.optimizer_discriminator.hook(self.discriminator.parameters())
        if isinstance(self.problem, InverseProblem):
            self.optimizer_model.instance.add_param_group(
                {
                    "params": [
                        self._params[var]
                        for var in self.problem.unknown_variables
                    ]
                }
            )
        self.scheduler_model.hook(self.optimizer_model)
        self.scheduler_discriminator.hook(self.optimizer_discriminator)
        return (
            [
                self.optimizer_model.instance,
                self.optimizer_discriminator.instance,
            ],
            [
                self.scheduler_model.instance,
                self.scheduler_discriminator.instance,
            ],
        )

    @property
    def neural_net(self):
        """
        The model.

        :return: The model.
        :rtype: torch.nn.Module
        """
        return self.models[0]

    @property
    def discriminator(self):
        """
        The discriminator.

        :return: The discriminator.
        :rtype: torch.nn.Module
        """
        return self.models[1]

    @property
    def optimizer_model(self):
        """
        The optimizer associated to the model.

        :return: The optimizer for the model.
        :rtype: Optimizer
        """
        return self.optimizers[0]

    @property
    def optimizer_discriminator(self):
        """
        The optimizer associated to the discriminator.

        :return: The optimizer for the discriminator.
        :rtype: Optimizer
        """
        return self.optimizers[1]

    @property
    def scheduler_model(self):
        """
        The scheduler associated to the model.

        :return: The scheduler for the model.
        :rtype: Scheduler
        """
        return self.schedulers[0]

    @property
    def scheduler_discriminator(self):
        """
        The scheduler associated to the discriminator.

        :return: The scheduler for the discriminator.
        :rtype: Scheduler
        """
        return self.schedulers[1]
