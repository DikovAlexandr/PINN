"""Module for the DeepEnsemble solver interface."""

import torch
from ..solver import MultiSolverInterface
from ...utils import check_consistency


class DeepEnsembleSolverInterface(MultiSolverInterface):
    r"""
    A class for handling ensemble models in a multi-solver training framework.
    It allows for manual optimization, as well as the ability to train,
    validate, and test multiple models as part of an ensemble.
    The ensemble dimension can be customized to control how outputs are stacked.

    By default, it is compatible with problems defined by
    :class:`~pina.problem.abstract_problem.AbstractProblem`,
    and users can choose the problem type the solver is meant to address.

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
        optimizers=None,
        schedulers=None,
        weighting=None,
        use_lt=True,
        ensemble_dim=0,
    ):
        """
        Initialization of the :class:`DeepEnsembleSolverInterface` class.

        :param AbstractProblem problem: The problem to be solved.
        :param torch.nn.Module models: The neural network models to be used.
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
            problem, models, optimizers, schedulers, weighting, use_lt
        )
        # check consistency
        check_consistency(ensemble_dim, int)
        self._ensemble_dim = ensemble_dim

    def forward(self, x, ensemble_idx=None):
        """
        Forward pass through the ensemble models. If an `ensemble_idx` is
        provided, it returns the output of the specific model
        corresponding to that index. If no index is given, it stacks the outputs
        of all models along the ensemble dimension.

        :param LabelTensor x: The input tensor to the models.
        :param int ensemble_idx: Optional index to select a specific
            model from the ensemble. If ``None`` results for all models are
            stacked in ``ensemble_dim`` dimension. Default is ``None``.
        :return: The output of the selected model or the stacked
            outputs from all models.
        :rtype: LabelTensor
        """
        # if an index is passed, return the specific model output for that index
        if ensemble_idx is not None:
            return self.models[ensemble_idx].forward(x)
        # otherwise return the stacked output
        return torch.stack(
            [self.forward(x, idx) for idx in range(self.num_ensemble)],
            dim=self.ensemble_dim,
        )

    def training_step(self, batch):
        """
        Training step for the solver, overridden for manual optimization.
        This method performs a forward pass, calculates the loss, and applies
        manual backward propagation and optimization steps for each model in
        the ensemble.

        :param list[tuple[str, dict]] batch: A batch of training data.
            Each element is a tuple containing a condition name and a
            dictionary of points.
        :return: The aggregated loss after the training step.
        :rtype: torch.Tensor
        """
        # zero grad for optimizer
        for opt in self.optimizers:
            opt.instance.zero_grad()
        # perform forward passes and aggregate losses
        loss = super().training_step(batch)
        # perform backpropagation
        self.manual_backward(loss)
        # optimize
        for opt, sched in zip(self.optimizers, self.schedulers):
            opt.instance.step()
            sched.instance.step()
        return loss

    @property
    def ensemble_dim(self):
        """
        The dimension along which the ensemble outputs are stacked.

        :return: The ensemble dimension.
        :rtype: int
        """
        return self._ensemble_dim

    @property
    def num_ensemble(self):
        """
        The number of models in the ensemble.

        :return: The number of models in the ensemble.
        :rtype: int
        """
        return len(self.models)
