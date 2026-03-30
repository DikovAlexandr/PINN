"""Module for the Supervised solver interface."""

from abc import abstractmethod

import torch

from torch.nn.modules.loss import _Loss
from ..solver import SolverInterface
from ...utils import check_consistency
from ...loss.loss_interface import LossInterface
from ...condition import InputTargetCondition


class SupervisedSolverInterface(SolverInterface):
    r"""
    Base class for Supervised solvers. This class implements a Supervised Solver
    , using a user specified ``model`` to solve a specific ``problem``.

    The ``SupervisedSolverInterface`` class can be used to define
    Supervised solvers that work with one or multiple optimizers and/or models.
    By default, it is compatible with problems defined by
    :class:`~pina.problem.abstract_problem.AbstractProblem`,
    and users can choose the problem type the solver is meant to address.
    """

    accepted_conditions_types = InputTargetCondition

    def __init__(self, loss=None, **kwargs):
        """
        Initialization of the :class:`SupervisedSolver` class.

        :param AbstractProblem problem: The problem to be solved.
        :param torch.nn.Module loss: The loss function to be minimized.
            If ``None``, the :class:`torch.nn.MSELoss` loss is used.
            Default is `None`.
        :param kwargs: Additional keyword arguments to be passed to the
            :class:`~pina.solver.solver.SolverInterface` class.
        """
        if loss is None:
            loss = torch.nn.MSELoss()

        super().__init__(**kwargs)

        # check consistency
        check_consistency(loss, (LossInterface, _Loss), subclass=False)

        # assign variables
        self._loss_fn = loss

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
        condition_loss = {}
        for condition_name, points in batch:
            condition_loss[condition_name] = self.loss_data(
                input=points["input"], target=points["target"]
            )
        return condition_loss

    @abstractmethod
    def loss_data(self, input, target):
        """
        Compute the data loss for the Supervised. This method is abstract and
        should be override by derived classes.

        :param input: The input to the neural network.
        :type input: LabelTensor | torch.Tensor | Graph | Data
        :param target: The target to compare with the network's output.
        :type target: LabelTensor | torch.Tensor | Graph | Data
        :return: The supervised loss, averaged over the number of observations.
        :rtype: LabelTensor | torch.Tensor | Graph | Data
        """

    @property
    def loss(self):
        """
        The loss function to be minimized.

        :return: The loss function to be minimized.
        :rtype: torch.nn.Module
        """
        return self._loss_fn
