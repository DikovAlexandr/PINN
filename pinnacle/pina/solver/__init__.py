"""Module for the solver classes."""

__all__ = [
    "SolverInterface",
    "SingleSolverInterface",
    "MultiSolverInterface",
    "PINNInterface",
    "PINN",
    "GradientPINN",
    "CausalPINN",
    "CompetitivePINN",
    "SelfAdaptivePINN",
    "RBAPINN",
    "SupervisedSolverInterface",
    "SupervisedSolver",
    "ReducedOrderModelSolver",
    "DeepEnsembleSolverInterface",
    "DeepEnsembleSupervisedSolver",
    "DeepEnsemblePINN",
    "GAROM",
]

from .solver import SolverInterface, SingleSolverInterface, MultiSolverInterface
from .physics_informed_solver import (
    PINNInterface,
    PINN,
    GradientPINN,
    CausalPINN,
    CompetitivePINN,
    SelfAdaptivePINN,
    RBAPINN,
)
from .supervised_solver import (
    SupervisedSolverInterface,
    SupervisedSolver,
    ReducedOrderModelSolver,
)
from .ensemble_solver import (
    DeepEnsembleSolverInterface,
    DeepEnsembleSupervisedSolver,
    DeepEnsemblePINN,
)
from .garom import GAROM
