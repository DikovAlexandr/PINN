"""Module for loss functions and weighting functions."""

__all__ = [
    "LossInterface",
    "LpLoss",
    "PowerLoss",
    "WeightingInterface",
    "ScalarWeighting",
    "NeuralTangentKernelWeighting",
    "SelfAdaptiveWeighting",
    "LinearWeighting",
]

from .loss_interface import LossInterface
from .power_loss import PowerLoss
from .lp_loss import LpLoss
from .weighting_interface import WeightingInterface
from .scalar_weighting import ScalarWeighting
from .ntk_weighting import NeuralTangentKernelWeighting
from .self_adaptive_weighting import SelfAdaptiveWeighting
from .linear_weighting import LinearWeighting
