"""Module for the DeepEnsemble physics solver."""

import torch

from .ensemble_solver_interface import DeepEnsembleSolverInterface
from ..physics_informed_solver import PINNInterface
from ...problem import InverseProblem


class DeepEnsemblePINN(PINNInterface, DeepEnsembleSolverInterface):
    r"""
    Deep Ensemble Physics Informed Solver class. This class implements a
    Deep Ensemble for Physics Informed Neural Networks using user
    specified ``model``s to solve a specific ``problem``.

    An ensemble model is constructed by combining multiple models that solve
    the same type of problem. Mathematically, this creates an implicit
    distribution :math:`p(\mathbf{u} \mid \mathbf{s})` over the possible
    outputs :math:`\mathbf{u}`, given the original input :math:`\mathbf{s}`.
    The models :math:`\mathcal{M}_{i\in (1,\dots,r)}` in
    the ensemble work collaboratively to capture different
    aspects of the data or task, with each model contributing a distinct
    prediction :math:`\mathbf{y}_{i}=\mathcal{M}_i(\mathbf{u} \mid \mathbf{s})`.
    By aggregating these predictions, the ensemble
    model can achieve greater robustness and accuracy compared to individual
    models, leveraging the diversity of the models to reduce overfitting and
    improve generalization. Furthemore, statistical metrics can
    be computed, e.g. the ensemble mean and variance:

    .. math::
        \mathbf{\mu} = \frac{1}{N}\sum_{i=1}^r \mathbf{y}_{i}

    .. math::
        \mathbf{\sigma^2} = \frac{1}{N}\sum_{i=1}^r
        (\mathbf{y}_{i} - \mathbf{\mu})^2

    During training the PINN loss is minimized by each ensemble model:

    .. math::
        \mathcal{L}_{\rm{problem}} = \frac{1}{N}\sum_{i=1}^4
        \mathcal{L}(\mathcal{A}[\mathbf{u}](\mathbf{x}_i)) +
        \frac{1}{N}\sum_{i=1}^N
        \mathcal{L}(\mathcal{B}[\mathbf{u}](\mathbf{x}_i)),

    for the differential system:
    
    .. math::

        \begin{cases}
        \mathcal{A}[\mathbf{u}](\mathbf{x})=0\quad,\mathbf{x}\in\Omega\\
        \mathcal{B}[\mathbf{u}](\mathbf{x})=0\quad,
        \mathbf{x}\in\partial\Omega
        \end{cases}

    :math:`\mathcal{L}` indicates a specific loss function, typically the MSE:

    .. math::
        \mathcal{L}(v) = \| v \|^2_2.

    .. seealso::

        **Original reference**: Zou, Z., Wang, Z., & Karniadakis, G. E. (2025). 
        *Learning and discovering multiple solutions using physics-informed 
        neural networks with random initialization and deep ensemble*. 
        DOI: `arXiv:2503.06320 <https://arxiv.org/abs/2503.06320>`_.

    .. warning::
        This solver does not work with inverse problem. Hence in the ``problem``
        definition must not inherit from 
        :class:`~pina.problem.inverse_problem.InverseProblem`.
    """

    def __init__(
        self,
        problem,
        models,
        loss=None,
        optimizers=None,
        schedulers=None,
        weighting=None,
        ensemble_dim=0,
    ):
        """
        Initialization of the :class:`DeepEnsemblePINN` class.

        :param AbstractProblem problem: The problem to be solved.
        :param torch.nn.Module models: The neural network models to be used.
        :param torch.nn.Module loss: The loss function to be minimized.
            If ``None``, the :class:`torch.nn.MSELoss` loss is used.
            Default is ``None``.
        :param Optimizer optimizer: The optimizer to be used.
            If ``None``, the :class:`torch.optim.Adam` optimizer is used.
            Default is ``None``.
        :param Scheduler scheduler: Learning rate scheduler.
            If ``None``, the :class:`torch.optim.lr_scheduler.ConstantLR`
            scheduler is used. Default is ``None``.
        :param WeightingInterface weighting: The weighting schema to be used.
            If ``None``, no weighting schema is used. Default is ``None``.
        :param int ensemble_dim: The dimension along which the ensemble
            outputs are stacked. Default is 0.
        :raises NotImplementedError: If an inverse problem is passed.
        """
        if isinstance(problem, InverseProblem):
            raise NotImplementedError(
                "DeepEnsemblePINN can not be used to solve inverse problems."
            )
        super().__init__(
            problem=problem,
            models=models,
            loss=loss,
            optimizers=optimizers,
            schedulers=schedulers,
            weighting=weighting,
            ensemble_dim=ensemble_dim,
        )

    def loss_data(self, input, target):
        """
        Compute the data loss for the ensemble PINN solver by evaluating
        the loss between the network's output and the true solution for each
        model. This method should not be overridden, if not intentionally.

        :param input: The input to the neural network.
        :type input: LabelTensor | torch.Tensor | Graph | Data
        :param target: The target to compare with the network's output.
        :type target: LabelTensor | torch.Tensor | Graph | Data
        :return: The supervised loss, averaged over the number of observations.
        :rtype: torch.Tensor
        """
        predictions = self.forward(input)
        loss = sum(
            self._loss_fn(predictions[idx], target)
            for idx in range(self.num_ensemble)
        )
        return loss / self.num_ensemble

    def loss_phys(self, samples, equation):
        """
        Computes the physics loss for the ensemble PINN solver by evaluating
        the loss between the network's output and the true solution for each
        model. This method should not be overridden, if not intentionally.

        :param LabelTensor samples: The samples to evaluate the physics loss.
        :param EquationInterface equation: The governing equation.
        :return: The computed physics loss.
        :rtype: LabelTensor
        """
        return self._residual_loss(samples, equation)

    def _residual_loss(self, samples, equation):
        """
        Computes the physics loss for the physics-informed solver based on the
        provided samples and equation. This method should never be overridden
        by the user, if not intentionally,
        since it is used internally to compute validation loss. It overrides the
        :obj:`~pina.solver.physics_informed_solver.PINNInterface._residual_loss`
        method.

        :param LabelTensor samples: The samples to evaluate the loss.
        :param EquationInterface equation: The governing equation.
        :return: The residual loss.
        :rtype: torch.Tensor
        """
        loss = 0
        predictions = self.forward(samples)
        for idx in range(self.num_ensemble):
            residuals = equation.residual(samples, predictions[idx])
            target = torch.zeros_like(residuals, requires_grad=True)
            loss = loss + self._loss_fn(residuals, target)
        return loss / self.num_ensemble
