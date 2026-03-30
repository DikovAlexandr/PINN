"""Module for the building blocks of the neural models."""

__all__ = [
    "ContinuousConvBlock",
    "ResidualBlock",
    "EnhancedLinear",
    "SpectralConvBlock1D",
    "SpectralConvBlock2D",
    "SpectralConvBlock3D",
    "FourierBlock1D",
    "FourierBlock2D",
    "FourierBlock3D",
    "PODBlock",
    "OrthogonalBlock",
    "PeriodicBoundaryEmbedding",
    "FourierFeatureEmbedding",
    "AVNOBlock",
    "LowRankBlock",
    "RBFBlock",
    "GNOBlock",
    "PirateNetBlock",
]

from .convolution_2d import ContinuousConvBlock
from .residual import ResidualBlock, EnhancedLinear
from .spectral import (
    SpectralConvBlock1D,
    SpectralConvBlock2D,
    SpectralConvBlock3D,
)
from .fourier_block import FourierBlock1D, FourierBlock2D, FourierBlock3D
from .pod_block import PODBlock
from .orthogonal import OrthogonalBlock
from .embedding import PeriodicBoundaryEmbedding, FourierFeatureEmbedding
from .average_neural_operator_block import AVNOBlock
from .low_rank_block import LowRankBlock
from .rbf_block import RBFBlock
from .gno_block import GNOBlock
from .pirate_network_block import PirateNetBlock
