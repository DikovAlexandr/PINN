"""Module for the DeepEnsemble supervised solver."""

from .ensemble_solver_interface import DeepEnsembleSolverInterface
from ..supervised_solver import SupervisedSolverInterface


class DeepEnsembleSupervisedSolver(
    SupervisedSolverInterface, DeepEnsembleSolverInterface
):
    r"""
    Deep Ensemble Supervised Solver class. This class implements a
    Deep Ensemble Supervised Solver using user specified ``model``s to solve
    a specific ``problem``.

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

    During training the supervised loss is minimized by each ensemble model:

    .. math::
        \mathcal{L}_{\rm{problem}} = \frac{1}{N}\sum_{i=1}^N
        \mathcal{L}(\mathbf{u}_i - \mathcal{M}_{j}(\mathbf{s}_i)),
        \quad j \in (1,\dots,N_{ensemble})

    where :math:`\mathcal{L}` is a specific loss function, typically the MSE:

    .. math::
        \mathcal{L}(v) = \| v \|^2_2.

    In this context, :math:`\mathbf{u}_i` and :math:`\mathbf{s}_i` indicates
    the will to approximate multiple (discretised) functions given multiple
    (discretised) input functions.

    .. seealso::

        **Original reference**: Lakshminarayanan, B., Pritzel, A., & Blundell,
        C. (2017). *Simple and scalable predictive uncertainty estimation
        using deep ensembles*. Advances in neural information
        processing systems, 30.
        DOI: `arXiv:1612.01474 <https://arxiv.org/abs/1612.01474>`_.
    """

    def __init__(
        self,
        problem,
        models,
        loss=None,
        optimizers=None,
        schedulers=None,
        weighting=None,
        use_lt=False,
        ensemble_dim=0,
    ):
        """
        Initialization of the :class:`DeepEnsembleSupervisedSolver` class.

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
        :param bool use_lt: If ``True``, the solver uses LabelTensors as input.
            Default is ``True``.
        :param int ensemble_dim: The dimension along which the ensemble
            outputs are stacked. Default is 0.
        """
        super().__init__(
            problem=problem,
            models=models,
            loss=loss,
            optimizers=optimizers,
            schedulers=schedulers,
            weighting=weighting,
            use_lt=use_lt,
            ensemble_dim=ensemble_dim,
        )

    def loss_data(self, input, target):
        """
        Compute the data loss for the EnsembleSupervisedSolver by evaluating
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
