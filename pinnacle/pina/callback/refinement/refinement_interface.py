"""
RefinementInterface class for handling the refinement of points in a neural
network training process.
"""

from abc import ABCMeta, abstractmethod
from lightning.pytorch import Callback
from ...utils import check_consistency
from ...solver.physics_informed_solver import PINNInterface


class RefinementInterface(Callback, metaclass=ABCMeta):
    """
    Interface class of Refinement approaches.
    """

    def __init__(self, sample_every, condition_to_update=None):
        """
        Initializes the RefinementInterface.

        :param int sample_every: The number of epochs between each refinement.
        :param condition_to_update: The conditions to update during the
            refinement process. If None, all conditions with a domain will be
            updated. Default is None.
        :type condition_to_update: list(str) | tuple(str) | str

        """
        # check consistency of the input
        check_consistency(sample_every, int)
        if condition_to_update is not None:
            if isinstance(condition_to_update, str):
                condition_to_update = [condition_to_update]
            if not isinstance(condition_to_update, (list, tuple)):
                raise ValueError(
                    "'condition_to_update' must be iter of strings."
                )
            check_consistency(condition_to_update, str)
        # store
        self.sample_every = sample_every
        self._condition_to_update = condition_to_update
        self._dataset = None
        self._initial_population_size = None

    def on_train_start(self, trainer, solver):
        """
        Called when the training begins. It initializes the conditions and
        dataset.

        :param ~lightning.pytorch.trainer.trainer.Trainer trainer: The trainer
            object.
        :param ~pina.solver.solver.SolverInterface solver: The solver
            object associated with the trainer.
        :raises RuntimeError: If the solver is not a PINNInterface.
        :raises RuntimeError: If the conditions do not have a domain to sample
            from.
        """
        # check we have valid conditions names
        if self._condition_to_update is None:
            self._condition_to_update = [
                name
                for name, cond in solver.problem.conditions.items()
                if hasattr(cond, "domain")
            ]

        for cond in self._condition_to_update:
            if cond not in solver.problem.conditions:
                raise RuntimeError(
                    f"Condition '{cond}' not found in "
                    f"{list(solver.problem.conditions.keys())}."
                )
            if not hasattr(solver.problem.conditions[cond], "domain"):
                raise RuntimeError(
                    f"Condition '{cond}' does not contain a domain to "
                    "sample from."
                )
        # check solver
        if not isinstance(solver, PINNInterface):
            raise RuntimeError(
                "Refinment strategies are currently implemented only "
                "for physics informed based solvers. Please use a Solver "
                "inheriting from 'PINNInterface'."
            )
        # store dataset
        self._dataset = trainer.datamodule.train_dataset
        # compute initial population size
        self._initial_population_size = self._compute_population_size(
            self._condition_to_update
        )
        return super().on_train_epoch_start(trainer, solver)

    def on_train_epoch_end(self, trainer, solver):
        """
        Performs the refinement at the end of each training epoch (if needed).

        :param ~lightning.pytorch.trainer.trainer.Trainer: The trainer object.
        :param PINNInterface solver: The solver object.
        """
        if (trainer.current_epoch % self.sample_every == 0) and (
            trainer.current_epoch != 0
        ):
            self._update_points(solver)
        return super().on_train_epoch_end(trainer, solver)

    @abstractmethod
    def sample(self, current_points, condition_name, solver):
        """
        Samples new points based on the condition.

        :param current_points: Current points in the domain.
        :param condition_name: Name of the condition to update.
        :param PINNInterface solver: The solver object.
        :return: New points sampled based on the R3 strategy.
        :rtype: LabelTensor
        """

    @property
    def dataset(self):
        """
        Returns the dataset for training.
        """
        return self._dataset

    @property
    def initial_population_size(self):
        """
        Returns the dataset for training size.
        """
        return self._initial_population_size

    def _update_points(self, solver):
        """
        Performs the refinement of the points.

        :param PINNInterface solver: The solver object.
        """
        new_points = {}
        for name in self._condition_to_update:
            current_points = self.dataset.conditions_dict[name]["input"]
            new_points[name] = {
                "input": self.sample(current_points, name, solver)
            }
        self.dataset.update_data(new_points)

    def _compute_population_size(self, conditions):
        """
        Computes the number of points in the dataset for each condition.

        :param conditions: List of conditions to compute the number of points.
        :return: Dictionary with the population size for each condition.
        :rtype: dict
        """
        return {
            cond: len(self.dataset.conditions_dict[cond]["input"])
            for cond in conditions
        }
