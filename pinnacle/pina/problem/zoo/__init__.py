"""Module for implemented problems."""

__all__ = [
    "SupervisedProblem",
    "HelmholtzProblem",
    "AllenCahnProblem",
    "AdvectionProblem",
    "Poisson2DSquareProblem",
    "DiffusionReactionProblem",
    "InversePoisson2DSquareProblem",
    "AcousticWaveProblem",
]

from .supervised_problem import SupervisedProblem
from .helmholtz import HelmholtzProblem
from .allen_cahn import AllenCahnProblem
from .advection import AdvectionProblem
from .poisson_2d_square import Poisson2DSquareProblem
from .diffusion_reaction import DiffusionReactionProblem
from .inverse_poisson_2d_square import InversePoisson2DSquareProblem
from .acoustic_wave import AcousticWaveProblem
