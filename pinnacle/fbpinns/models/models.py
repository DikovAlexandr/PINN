"""Standard PyTorch models used in FBPINN / PINN experiments.

This module defines fully connected neural network architectures that are used
by :mod:`config.constants` when setting up FBPINN / PINN problems.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def total_params(model: nn.Module) -> int:
    """Return the total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters())


class FCN(nn.Module):
    """Fully connected feed-forward network."""

    def __init__(self, N_INPUT: int, N_OUTPUT: int, N_HIDDEN: int, N_LAYERS: int) -> None:
        """Initialise the FCN architecture.

        :param int N_INPUT: Input dimensionality.
        :param int N_OUTPUT: Output dimensionality.
        :param int N_HIDDEN: Width of hidden layers.
        :param int N_LAYERS: Total number of hidden layers.
        """
        super().__init__()

        activation = nn.Tanh

        # Input -> first hidden
        self.fcs = nn.Sequential(
            nn.Linear(N_INPUT, N_HIDDEN),
            activation(),
        )

        # Hidden stack
        self.fch = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(N_HIDDEN, N_HIDDEN),
                    activation(),
                )
                for _ in range(N_LAYERS - 1)
            ]
        )

        # Last hidden -> output
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

        # Helper attributes / methods for analysing computational complexity
        d1, d2, h, l = N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS
        self.size = d1 * h + h + (l - 1) * (h * h + h) + h * d2 + d2
        # assumes Tanh uses 5 FLOPS
        self._single_flop = (
            2 * d1 * h
            + h
            + 5 * h
            + (l - 1) * (2 * h * h + h + 5 * h)
            + 2 * h * d2
            + d2
        )
        self.flops = lambda batch_size: batch_size * self._single_flop
        assert self.size == total_params(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the FCN."""
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x


class BiFCN(nn.Module):
    """Parallel fully connected network.

    Two networks are defined in parallel:
    - Solution network
    - Parameter network

    The parameter network takes the first two dimensions of the input as input.
    The combined output has two components.
    """

    def __init__(self, N_INPUT: int, N_OUTPUT: int, N_HIDDEN: int, N_LAYERS: int) -> None:
        """Initialise the BiFCN architecture.

        :param int N_INPUT: Input dimensionality.
        :param int N_OUTPUT: Output dimensionality (must be 2).
        :param int N_HIDDEN: Width of hidden layers.
        :param int N_LAYERS: Total number of hidden layers per branch.
        """
        super().__init__()
        assert N_OUTPUT == 2

        activation = nn.Tanh

        # Branch 1 (solution)
        self.fcs = nn.Sequential(
            nn.Linear(N_INPUT, N_HIDDEN),
            activation(),
        )
        self.fch = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(N_HIDDEN, N_HIDDEN),
                    activation(),
                )
                for _ in range(N_LAYERS - 1)
            ]
        )
        self.fce = nn.Linear(N_HIDDEN, 1)

        # Branch 2 (parameters; takes first two dims of input)
        self.fcs2 = nn.Sequential(
            nn.Linear(2, N_HIDDEN),
            activation(),
        )
        self.fch2 = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(N_HIDDEN, N_HIDDEN),
                    activation(),
                )
                for _ in range(N_LAYERS - 1)
            ]
        )
        self.fce2 = nn.Linear(N_HIDDEN, 1)

        # Helper attributes / methods for analysing computational complexity
        d1, d2, h, l = N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS
        self.size = d1 * h + h + (l - 1) * (h * h + h) + h * d2 + d2
        self._single_flop = 1  # dummy placeholder
        self.flops = lambda batch_size: batch_size * self._single_flop
        # assert self.size == total_params(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the BiFCN."""
        m1, m2 = self.fcs(x), self.fcs2(x[:, :2])
        mm1, mm2 = self.fch(m1), self.fch2(m2)
        o1, o2 = self.fce(mm1), self.fce2(mm2)
        return torch.concat([o1, o2], dim=-1)
