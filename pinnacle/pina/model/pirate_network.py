"""Module for the PirateNet model class."""

import torch
from .block import FourierFeatureEmbedding, PirateNetBlock
from ..utils import check_consistency, check_positive_integer


class PirateNet(torch.nn.Module):
    """
    Implementation of Physics-Informed residual adaptive network (PirateNet).

    The model consists of a Fourier feature embedding layer, multiple PirateNet
    blocks, and a final output layer. Each PirateNet block consist of three
    dense layers with dual gating mechanism and an adaptive residual connection,
    whose contribution is controlled by a trainable parameter ``alpha``.

    The PirateNet, augmented with random weight factorization, is designed to
    mitigate spectral bias in deep networks.

    .. seealso::

        **Original reference**:
        Wang, S., Sankaran, S., Stinis., P., Perdikaris, P. (2025).
        *Simulating Three-dimensional Turbulence with Physics-informed Neural
        Networks*.
        DOI: `arXiv preprint arXiv:2507.08972.
        <https://arxiv.org/abs/2507.08972>`_
    """

    def __init__(
        self,
        input_dimension,
        inner_size,
        output_dimension,
        embedding=None,
        n_layers=3,
        activation=torch.nn.Tanh,
    ):
        """
        Initialization of the :class:`PirateNet` class.

        :param int input_dimension: The number of input features.
        :param int inner_size: The number of hidden units in the dense layers.
        :param int output_dimension: The number of output features.
        :param torch.nn.Module embedding: The embedding module used to transform
            the input into a higher-dimensional feature space. If ``None``, a
            default :class:`~pina.model.block.FourierFeatureEmbedding` with
            scaling factor of 2 is used. Default is ``None``.
        :param int n_layers: The number of PirateNet blocks in the model.
            Default is 3.
        :param torch.nn.Module activation: The activation function to be used in
            the blocks. Default is :class:`torch.nn.Tanh`.
        """
        super().__init__()

        # Check consistency
        check_consistency(activation, torch.nn.Module, subclass=True)
        check_positive_integer(input_dimension, strict=True)
        check_positive_integer(inner_size, strict=True)
        check_positive_integer(output_dimension, strict=True)
        check_positive_integer(n_layers, strict=True)

        # Initialize the activation function
        self.activation = activation()

        # Initialize the Fourier embedding
        self.embedding = embedding or FourierFeatureEmbedding(
            input_dimension=input_dimension,
            output_dimension=inner_size,
            sigma=2.0,
        )

        # Initialize the shared dense layers
        self.linear1 = torch.nn.Linear(inner_size, inner_size)
        self.linear2 = torch.nn.Linear(inner_size, inner_size)

        # Initialize the PirateNet blocks
        self.blocks = torch.nn.ModuleList(
            [PirateNetBlock(inner_size, activation) for _ in range(n_layers)]
        )

        # Initialize the output layer
        self.output_layer = torch.nn.Linear(inner_size, output_dimension)

    def forward(self, input_):
        """
        Forward pass of the PirateNet model. It applies the Fourier feature
        embedding, computes the shared gating tensors U and V, and passes the
        input through each block in the network. Finally, it applies the output
        layer to produce the final output.

        :param input_: The input tensor for the model.
        :type input_: torch.Tensor | LabelTensor
        :return: The output tensor of the model.
        :rtype: torch.Tensor | LabelTensor
        """
        # Apply the Fourier feature embedding
        x = self.embedding(input_)

        # Compute U and V from the shared dense layers
        U = self.activation(self.linear1(x))
        V = self.activation(self.linear2(x))

        # Pass through each block in the network
        for block in self.blocks:
            x = block(x, U, V)

        return self.output_layer(x)

    @property
    def alpha(self):
        """
        Return the alpha values of all PirateNetBlock layers.

        :return: A list of alpha values from each block.
        :rtype: list
        """
        return [block.alpha.item() for block in self.blocks]
