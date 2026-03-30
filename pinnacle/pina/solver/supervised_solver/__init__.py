"""Module for the Supervised solvers."""

__all__ = [
    "SupervisedSolverInterface",
    "SupervisedSolver",
    "ReducedOrderModelSolver",
]

from .supervised_solver_interface import SupervisedSolverInterface
from .supervised import SupervisedSolver
from .reduced_order_model import ReducedOrderModelSolver
