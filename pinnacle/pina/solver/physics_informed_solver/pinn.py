"""Module for the Physics-Informed Neural Network solver."""

import torch

from .pinn_interface import PINNInterface
from ..solver import SingleSolverInterface
from ...problem import InverseProblem


class PINN(PINNInterface, SingleSolverInterface):
    r"""
    Physics-Informed Neural Network (PINN) solver class.
    This class implements Physics-Informed Neural Network solver, using a user
    specified ``model`` to solve a specific ``problem``.
    It can be used to solve both forward and inverse problems.

    The Physics Informed Neural Network solver aims to find the solution
    :math:`\mathbf{u}:\Omega\rightarrow\mathbb{R}^m` of a differential problem:

    .. math::

        \begin{cases}
        \mathcal{A}[\mathbf{u}](\mathbf{x})=0\quad,\mathbf{x}\in\Omega\\
        \mathcal{B}[\mathbf{u}](\mathbf{x})=0\quad,
        \mathbf{x}\in\partial\Omega
        \end{cases}

    minimizing the loss function:

    .. math::
        \mathcal{L}_{\rm{problem}} = \frac{1}{N}\sum_{i=1}^N
        \mathcal{L}(\mathcal{A}[\mathbf{u}](\mathbf{x}_i)) +
        \frac{1}{N}\sum_{i=1}^N
        \mathcal{L}(\mathcal{B}[\mathbf{u}](\mathbf{x}_i)),

    where :math:`\mathcal{L}` is a specific loss function, typically the MSE:

    .. math::
        \mathcal{L}(v) = \| v \|^2_2.

    .. seealso::

        **Original reference**: Karniadakis, G. E., Kevrekidis, I. G., Lu, L.,
        Perdikaris, P., Wang, S., & Yang, L. (2021).
        *Physics-informed machine learning.*
        Nature Reviews Physics, 3, 422-440.
        DOI: `10.1038 <https://doi.org/10.1038/s42254-021-00314-5>`_.
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
        Initialization of the :class:`PINN` class.

        :param AbstractProblem problem: The problem to be solved.
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
        """
        super().__init__(
            model=model,
            problem=problem,
            optimizer=optimizer,
            scheduler=scheduler,
            weighting=weighting,
            loss=loss,
        )

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

    def configure_optimizers(self):
        """
        Optimizer configuration for the PINN solver.

        :return: The optimizers and the schedulers
        :rtype: tuple[list[Optimizer], list[Scheduler]]
        """
        # If the problem is an InverseProblem, add the unknown parameters
        # to the parameters to be optimized.
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
