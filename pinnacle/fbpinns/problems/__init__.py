"""
Problems package for FBPINN / PINN experiments.

This package contains problem definitions and boundary condition utilities.
"""

from problems.problems import (
    _Problem,
    Cos1D_1,
    Cos_multi1D_1,
    Sin1D_2,
    Cos_Cos2D_1,
    Sin2D_1,
    Sin2x2D,
    CavityFlow,
)

__all__ = [
    "_Problem",
    "Cos1D_1",
    "Cos_multi1D_1",
    "Sin1D_2",
    "Cos_Cos2D_1",
    "Sin2D_1",
    "Sin2x2D",
    "CavityFlow",
]
