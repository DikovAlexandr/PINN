"""Module for the Ensemble solver classes."""

__all__ = [
    "DeepEnsembleSolverInterface",
    "DeepEnsembleSupervisedSolver",
    "DeepEnsemblePINN",
]

from .ensemble_solver_interface import DeepEnsembleSolverInterface
from .ensemble_supervised import DeepEnsembleSupervisedSolver
from .ensemble_pinn import DeepEnsemblePINN
