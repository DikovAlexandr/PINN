"""Module for the Physics-Informed Neural Network Interface."""

from abc import ABCMeta, abstractmethod
import warnings
import torch

from ...utils import custom_warning_format
from ..supervised_solver import SupervisedSolverInterface
from ...condition import (
    InputTargetCondition,
    InputEquationCondition,
    DomainEquationCondition,
)

# set the warning for torch >= 2.8 compile
warnings.formatwarning = custom_warning_format
warnings.filterwarnings("always", category=UserWarning)


class PINNInterface(SupervisedSolverInterface, metaclass=ABCMeta):
    """
    Base class for Physics-Informed Neural Network (PINN) solvers, implementing
    the :class:`~pina.solver.solver.SolverInterface` class.

    The `PINNInterface` class can be used to define PINNs that work with one or
    multiple optimizers and/or models. By default, it is compatible with
    problems defined by :class:`~pina.problem.abstract_problem.AbstractProblem`,
    and users can choose the problem type the solver is meant to address.
    """

    accepted_conditions_types = (
        InputTargetCondition,
        InputEquationCondition,
        DomainEquationCondition,
    )

    def __init__(self, **kwargs):
        """
        Initialization of the :class:`PINNInterface` class.

        :param AbstractProblem problem: The problem to be solved.
        :param torch.nn.Module loss: The loss function to be minimized.
            If ``None``, the :class:`torch.nn.MSELoss` loss is used.
            Default is `None`.
        :param kwargs: Additional keyword arguments to be passed to the
            :class:`~pina.solver.supervised_solver.SupervisedSolverInterface`
            class.
        """
        kwargs["use_lt"] = True
        super().__init__(**kwargs)

        # current condition name
        self.__metric = None

    def setup(self, stage):
        """
        Setup method executed at the beginning of training and testing.

        This method compiles the model only if the installed torch version
        is earlier than 2.8, due to known issues with later versions
        (see https://github.com/mathLab/PINA/issues/621).

        .. warning::
            For torch >= 2.8, compilation is disabled. Forcing compilation
            on these versions may cause runtime errors or unstable behavior.

        :param str stage: The current stage of the training process
            (e.g., ``fit``, ``validate``, ``test``, ``predict``).
        :return: The result of the parent class ``setup`` method.
        :rtype: Any
        """
        # Override the compilation, compiling only for torch < 2.8, see
        # related issue at https://github.com/mathLab/PINA/issues/621
        if torch.__version__ >= "2.8":
            self.trainer.compile = False
            warnings.warn(
                "Compilation is disabled for torch >= 2.8. "
                "Forcing compilation may cause runtime errors or instability.",
                UserWarning,
            )
        return super().setup(stage)

    def optimization_cycle(self, batch, loss_residuals=None):
        """
        The optimization cycle for the PINN solver.

        This method allows to call `_run_optimization_cycle` with the physics
        loss as argument, thus distinguishing the training step from the
        validation and test steps.

        :param list[tuple[str, dict]] batch: A batch of data. Each element is a
            tuple containing a condition name and a dictionary of points.
        :return: The losses computed for all conditions in the batch, casted
            to a subclass of :class:`torch.Tensor`. It should return a dict
            containing the condition name and the associated scalar loss.
        :rtype: dict
        """
        # which losses to use
        if loss_residuals is None:
            loss_residuals = self.loss_phys
        # compute optimization cycle
        condition_loss = {}
        for condition_name, points in batch:
            self.__metric = condition_name
            # if equations are passed
            if "target" not in points:
                input_pts = points["input"]
                condition = self.problem.conditions[condition_name]
                loss = loss_residuals(
                    input_pts.requires_grad_(), condition.equation
                )
            # if data are passed
            else:
                input_pts = points["input"]
                output_pts = points["target"]
                loss = self.loss_data(
                    input=input_pts.requires_grad_(), target=output_pts
                )
            # append loss
            condition_loss[condition_name] = loss
        return condition_loss

    @torch.set_grad_enabled(True)
    def validation_step(self, batch):
        """
        The validation step for the PINN solver. It returns the average residual
        computed with the ``loss`` function not aggregated.

        :param list[tuple[str, dict]] batch: A batch of data. Each element is a
            tuple containing a condition name and a dictionary of points.
        :return: The loss of the validation step.
        :rtype: torch.Tensor
        """
        return super().validation_step(
            batch, loss_residuals=self._residual_loss
        )

    @torch.set_grad_enabled(True)
    def test_step(self, batch):
        """
        The test step for the PINN solver. It returns the average residual
        computed with the ``loss`` function not aggregated.

        :param list[tuple[str, dict]] batch: A batch of data. Each element is a
            tuple containing a condition name and a dictionary of points.
        :return: The loss of the test step.
        :rtype: torch.Tensor
        """
        return super().test_step(batch, loss_residuals=self._residual_loss)

    def loss_data(self, input, target):
        """
        Compute the data loss for the PINN solver by evaluating the loss
        between the network's output and the true solution. This method should
        be overridden by the derived class.

        :param LabelTensor input: The input to the neural network.
        :param LabelTensor target: The target to compare with the
            network's output.
        :return: The supervised loss, averaged over the number of observations.
        :rtype: LabelTensor
        :raises NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError(
            "PINN is being used in a supervised learning context, but the "
            "'loss_data' method has not been implemented. "
        )

    @abstractmethod
    def loss_phys(self, samples, equation):
        """
        Computes the physics loss for the physics-informed solver based on the
        provided samples and equation. This method must be overridden in
        subclasses. It distinguishes different types of PINN solvers.

        :param LabelTensor samples: The samples to evaluate the physics loss.
        :param EquationInterface equation: The governing equation.
        :return: The computed physics loss.
        :rtype: LabelTensor
        """

    def compute_residual(self, samples, equation):
        """
        Compute the residuals of the equation.

        :param LabelTensor samples: The samples to evaluate the loss.
        :param EquationInterface equation: The governing equation.
        :return: The residual of the solution of the model.
        :rtype: LabelTensor
        """
        residual = equation.residual(
            samples, self.forward(samples), self._params
        )
        return residual

    def _residual_loss(self, samples, equation):
        """
        Computes the physics loss for the physics-informed solver based on the
        provided samples and equation. This method should never be overridden
        by the user, if not intentionally,
        since it is used internally to compute validation loss.


        :param LabelTensor samples: The samples to evaluate the loss.
        :param EquationInterface equation: The governing equation.
        :return: The residual loss.
        :rtype: torch.Tensor
        """
        residuals = self.compute_residual(samples, equation)
        return self._loss_fn(residuals, torch.zeros_like(residuals))

    @property
    def current_condition_name(self):
        """
        The current condition name.

        :return: The current condition name.
        :rtype: str
        """
        return self.__metric
