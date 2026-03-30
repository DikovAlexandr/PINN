"""Module for residual blocks and enhanced linear layers."""

import torch
from torch import nn
from ...utils import check_consistency


class ResidualBlock(nn.Module):
    """
    Residual block class.

    .. seealso::

        **Original reference**: He, Kaiming, et al.
        *Deep residual learning for image recognition.*
        Proceedings of the IEEE conference on computer vision and pattern
        recognition. 2016.
        DOI: `<https://arxiv.org/pdf/1512.03385.pdf>`_.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        spectral_norm=False,
        activation=torch.nn.ReLU(),
    ):
        """
        Initialization of the :class:`ResidualBlock` class.

        :param int input_dim: The input dimension.
        :param int output_dim: The output dimension.
        :param int hidden_dim: The hidden dimension.
        :param bool spectral_norm: If ``True``, the spectral normalization is
            applied to the feedforward layers. Default is ``False``.
        :param torch.nn.Module activation: The activation function.
            Default is :class:`torch.nn.ReLU`.

        """
        super().__init__()
        # check consistency
        check_consistency(spectral_norm, bool)
        check_consistency(input_dim, int)
        check_consistency(output_dim, int)
        check_consistency(hidden_dim, int)
        check_consistency(activation, torch.nn.Module)

        # assign variables
        self._spectral_norm = spectral_norm
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._hidden_dim = hidden_dim
        self._activation = activation

        # create layers
        self._l1 = self._spect_norm(nn.Linear(input_dim, hidden_dim))
        self._l2 = self._spect_norm(nn.Linear(hidden_dim, output_dim))
        self._l3 = self._spect_norm(nn.Linear(input_dim, output_dim))

    def forward(self, x):
        """
        Forward pass.

        :param torch.Tensor x: The input tensor.
        :return: The output tensor.
        :rtype: torch.Tensor
        """
        y = self._activation(self._l1(x))
        y = self._l2(y)
        x = self._l3(x)
        return y + x

    def _spect_norm(self, x):
        """
        Perform spectral normalization on the network layers.

        :param torch.nn.Module x: A :class:`torch.nn.Linear` layer.
        :return: The spectral norm of the layer
        :rtype: torch.nn.Module
        """
        return nn.utils.spectral_norm(x) if self._spectral_norm else x


class EnhancedLinear(torch.nn.Module):
    """
    Enhanced Linear layer class.

    This class is a wrapper for enhancing a linear layer with activation and/or
    dropout.
    """

    def __init__(self, layer, activation=None, dropout=None):
        """
        Initialization of the :class:`EnhancedLinear` class.

        :param torch.nn.Module layer: The linear layer to be enhanced.
        :param torch.nn.Module activation: The activation function. Default is
            ``None``.
        :param float dropout: The dropout probability. Default is ``None``.

        :Example:

            >>> linear_layer = torch.nn.Linear(10, 20)
            >>> activation = torch.nn.ReLU()
            >>> dropout_prob = 0.5
            >>> enhanced_linear = EnhancedLinear(
            ...     linear_layer,
            ...     activation,
            ...     dropout_prob
            ... )
        """
        super().__init__()

        # check consistency
        check_consistency(layer, nn.Module)
        if activation is not None:
            check_consistency(activation, nn.Module)
        if dropout is not None:
            check_consistency(dropout, float)

        # assign forward
        if (dropout is None) and (activation is None):
            self._model = torch.nn.Sequential(layer)

        elif (dropout is None) and (activation is not None):
            self._model = torch.nn.Sequential(layer, activation)

        elif (dropout is not None) and (activation is None):
            self._model = torch.nn.Sequential(layer, self._drop(dropout))

        elif (dropout is not None) and (activation is not None):
            self._model = torch.nn.Sequential(
                layer, activation, self._drop(dropout)
            )

    def forward(self, x):
        """
        Forward pass.

        :param torch.Tensor x: The input tensor.
        :return: The output tensor.
        :rtype: torch.Tensor
        """
        return self._model(x)

    def _drop(self, p):
        """
        Apply dropout with probability p.

        :param float p: Dropout probability.
        :return: Dropout layer with the specified probability.
        :rtype: torch.nn.Dropout
        """
        return torch.nn.Dropout(p)
