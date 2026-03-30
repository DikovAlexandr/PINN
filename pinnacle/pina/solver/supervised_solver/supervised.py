"""Module for the Supervised solver."""

from .supervised_solver_interface import SupervisedSolverInterface
from ..solver import SingleSolverInterface


class SupervisedSolver(SupervisedSolverInterface, SingleSolverInterface):
    r"""
    Supervised Solver solver class. This class implements a Supervised Solver,
    using a user specified ``model`` to solve a specific ``problem``.

    The  Supervised Solver class aims to find a map between the input
    :math:`\mathbf{s}:\Omega\rightarrow\mathbb{R}^m` and the output
    :math:`\mathbf{u}:\Omega\rightarrow\mathbb{R}^m`.

    Given a model :math:`\mathcal{M}`, the following loss function is
    minimized during training:

    .. math::
        \mathcal{L}_{\rm{problem}} = \frac{1}{N}\sum_{i=1}^N
        \mathcal{L}(\mathbf{u}_i - \mathcal{M}(\mathbf{s}_i)),

    where :math:`\mathcal{L}` is a specific loss function, typically the MSE:

    .. math::
        \mathcal{L}(v) = \| v \|^2_2.

    In this context, :math:`\mathbf{u}_i` and :math:`\mathbf{s}_i` indicates
    the will to approximate multiple (discretised) functions given multiple
    (discretised) input functions.
    """

    def __init__(
        self,
        problem,
        model,
        loss=None,
        optimizer=None,
        scheduler=None,
        weighting=None,
        use_lt=True,
    ):
        """
        Initialization of the :class:`SupervisedSolver` class.

        :param AbstractProblem problem: The problem to be solved.
        :param torch.nn.Module model: The neural network model to be used.
        :param torch.nn.Module loss: The loss function to be minimized.
            If ``None``, the :class:`torch.nn.MSELoss` loss is used.
            Default is `None`.
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
        """
        super().__init__(
            model=model,
            problem=problem,
            loss=loss,
            optimizer=optimizer,
            scheduler=scheduler,
            weighting=weighting,
            use_lt=use_lt,
        )

    def loss_data(self, input, target):
        """
        Compute the data loss for the Supervised solver by evaluating the loss
        between the network's output and the true solution. This method should
        not be overridden, if not intentionally.

        :param input: The input to the neural network.
        :type input: LabelTensor | torch.Tensor | Graph | Data
        :param target: The target to compare with the network's output.
        :type target: LabelTensor | torch.Tensor | Graph | Data
        :return: The supervised loss, averaged over the number of observations.
        :rtype: LabelTensor | torch.Tensor | Graph | Data
        """
        return self._loss_fn(self.forward(input), target)
