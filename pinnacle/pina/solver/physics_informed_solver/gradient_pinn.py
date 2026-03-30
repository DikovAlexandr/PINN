"""Module for the Gradient PINN solver."""

import torch

from .pinn import PINN
from ...operator import grad
from ...problem import SpatialProblem


class GradientPINN(PINN):
    r"""
    Gradient Physics-Informed Neural Network (GradientPINN) solver class.
    This class implements the Gradient Physics-Informed Neural Network solver,
    using a user specified ``model`` to solve a specific ``problem``.
    It can be used to solve both forward and inverse problems.

    The Gradient Physics-Informed Neural Network solver aims to find the
    solution :math:`\mathbf{u}:\Omega\rightarrow\mathbb{R}^m` of a differential
    problem:

    .. math::

        \begin{cases}
        \mathcal{A}[\mathbf{u}](\mathbf{x})=0\quad,\mathbf{x}\in\Omega\\
        \mathcal{B}[\mathbf{u}](\mathbf{x})=0\quad,
        \mathbf{x}\in\partial\Omega
        \end{cases}

    minimizing the loss function;

    .. math::
        \mathcal{L}_{\rm{problem}} =& \frac{1}{N}\sum_{i=1}^N
        \mathcal{L}(\mathcal{A}[\mathbf{u}](\mathbf{x}_i)) +
        \frac{1}{N}\sum_{i=1}^N
        \mathcal{L}(\mathcal{B}[\mathbf{u}](\mathbf{x}_i)) + 
        &\frac{1}{N}\sum_{i=1}^N
        \nabla_{\mathbf{x}}\mathcal{L}(\mathcal{A}[\mathbf{u}](\mathbf{x}_i)) +
        \frac{1}{N}\sum_{i=1}^N
        \nabla_{\mathbf{x}}\mathcal{L}(\mathcal{B}[\mathbf{u}](\mathbf{x}_i))


    where :math:`\mathcal{L}` is a specific loss function, typically the MSE:

    .. math::
        \mathcal{L}(v) = \| v \|^2_2.

    .. seealso::

        **Original reference**: Yu, Jeremy, et al.
        *Gradient-enhanced physics-informed neural networks for forward and
        inverse PDE problems.*
        Computer Methods in Applied Mechanics and Engineering 393 (2022):114823.
        DOI: `10.1016 <https://doi.org/10.1016/j.cma.2022.114823>`_.

    .. note::
        This class is only compatible with problems that inherit from  the
        :class:`~pina.problem.spatial_problem.SpatialProblem` class. 
    """

    def __init__(
        self,
        problem,
        model,
        optimizer=None,
        scheduler=None,
        weighting=None,
        loss=None,
    ):
        """
        Initialization of the :class:`GradientPINN` class.

        :param AbstractProblem problem: The problem to be solved.
            It must inherit from at least
            :class:`~pina.problem.spatial_problem.SpatialProblem` to compute the
            gradient of the loss.
        :param torch.nn.Module model: The neural network model to be used.
        :param Optimizer optimizer: The optimizer to be used.
            If ``None``, the :class:`torch.optim.Adam` optimizer is used.
            Default is ``None``.
        :param Scheduler scheduler: Learning rate scheduler.
            If ``None``, the :class:`torch.optim.lr_scheduler.ConstantLR`
            scheduler is used. Default is ``None``.
        :param WeightingInterface weighting: The weighting schema to be used.
            If ``None``, no weighting schema is used. Default is ``None``.
        :param torch.nn.Module loss: The loss function to be minimized.
            If ``None``, the :class:`torch.nn.MSELoss` loss is used.
            Default is `None`.
        :raises ValueError: If the problem is not a SpatialProblem.
        """
        super().__init__(
            model=model,
            problem=problem,
            optimizer=optimizer,
            scheduler=scheduler,
            weighting=weighting,
            loss=loss,
        )

        if not isinstance(self.problem, SpatialProblem):
            raise ValueError(
                "Gradient PINN computes the gradient of the "
                "PINN loss with respect to the spatial "
                "coordinates, thus the PINA problem must be "
                "a SpatialProblem."
            )

    def loss_phys(self, samples, equation):
        """
        Computes the physics loss for the physics-informed solver based on the
        provided samples and equation.

        :param LabelTensor samples: The samples to evaluate the physics loss.
        :param EquationInterface equation: The governing equation.
        :return: The computed physics loss.
        :rtype: LabelTensor
        """
        # classical PINN loss
        residual = self.compute_residual(samples=samples, equation=equation)
        loss_value = self._loss_fn(
            torch.zeros_like(residual, requires_grad=True), residual
        )

        # gradient PINN loss
        loss_value = loss_value.reshape(-1, 1)
        loss_value.labels = ["__loss"]
        loss_grad = grad(loss_value, samples, d=self.problem.spatial_variables)
        g_loss_phys = self._loss_fn(
            torch.zeros_like(loss_grad, requires_grad=True), loss_grad
        )
        return loss_value + g_loss_phys
