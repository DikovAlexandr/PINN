"""Module for the Neural model classes."""

__all__ = [
    "FeedForward",
    "ResidualFeedForward",
    "MultiFeedForward",
    "DeepONet",
    "MIONet",
    "FNO",
    "FourierIntegralKernel",
    "KernelNeuralOperator",
    "AveragingNeuralOperator",
    "LowRankNeuralOperator",
    "Spline",
    "GraphNeuralOperator",
    "PirateNet",
    "EquivariantGraphNeuralOperator",
    "SINDy",
]

from .feed_forward import FeedForward, ResidualFeedForward
from .multi_feed_forward import MultiFeedForward
from .deeponet import DeepONet, MIONet
from .fourier_neural_operator import FNO, FourierIntegralKernel
from .kernel_neural_operator import KernelNeuralOperator
from .average_neural_operator import AveragingNeuralOperator
from .low_rank_neural_operator import LowRankNeuralOperator
from .spline import Spline
from .spline_surface import SplineSurface
from .graph_neural_operator import GraphNeuralOperator
from .pirate_network import PirateNet
from .equivariant_graph_neural_operator import EquivariantGraphNeuralOperator
from .sindy import SINDy
