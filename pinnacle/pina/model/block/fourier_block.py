"""Module for the Fourier Neural Operator Block class."""

import torch
from torch import nn
from ...utils import check_consistency

from .spectral import (
    SpectralConvBlock1D,
    SpectralConvBlock2D,
    SpectralConvBlock3D,
)


class FourierBlock1D(nn.Module):
    """
    The inner block of the Fourier Neural Operator for 1-dimensional input
    tensors.

    The module computes the spectral convolution of the input with a linear
    kernel in the fourier space, and then it maps the input back to the physical
    space. The output is then added to a Linear tranformation of the input in
    the physical space. Finally an activation function is applied to the output.

    .. seealso::

        **Original reference**: Li, Z., Kovachki, N., Azizzadenesheli, K.,
        Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2020).
        *Fourier neural operator for parametric partial differential equations*.
        DOI: `arXiv preprint arXiv:2010.08895.
        <https://arxiv.org/abs/2010.08895>`_

    """

    def __init__(
        self,
        input_numb_fields,
        output_numb_fields,
        n_modes,
        activation=torch.nn.Tanh,
    ):
        r"""
        Initialization of the :class:`FourierBlock1D` class.

        :param int input_numb_fields: The number of channels for the input.
        :param int output_numb_fields: The number of channels for the output.
        :param n_modes: The number of modes to select for each dimension.
            It must be at most equal to :math:`\floor(Nx/2)+1`.
        :type n_modes: list[int] | tuple[int]
        :param torch.nn.Module activation: The activation function.
            Default is :class:`torch.nn.Tanh`.
        """

        super().__init__()

        # check type consistency
        check_consistency(activation(), nn.Module)

        # assign variables
        self._spectral_conv = SpectralConvBlock1D(
            input_numb_fields=input_numb_fields,
            output_numb_fields=output_numb_fields,
            n_modes=n_modes,
        )
        self._activation = activation()
        self._linear = nn.Conv1d(input_numb_fields, output_numb_fields, 1)

    def forward(self, x):
        """
        Forward pass of the block. It performs a spectral convolution and a
        linear transformation of the input. Then, it sums the results.

        :param torch.Tensor x: The input tensor for performing the computation.
        :return: The output tensor.
        :rtype: torch.Tensor
        """
        return self._activation(self._spectral_conv(x) + self._linear(x))


class FourierBlock2D(nn.Module):
    """
    The inner block of the Fourier Neural Operator for 2-dimensional input
    tensors.

    The module computes the spectral convolution of the input with a linear
    kernel in the fourier space, and then it maps the input back to the physical
    space. The output is then added to a Linear tranformation of the input in
    the physical space. Finally an activation function is applied to the output.

    .. seealso::

        **Original reference**: Li, Z., Kovachki, N., Azizzadenesheli, K.,
        Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2020).
        *Fourier neural operator for parametric partial differential equations*.
        DOI: `arXiv preprint arXiv:2010.08895.
        <https://arxiv.org/abs/2010.08895>`_
    """

    def __init__(
        self,
        input_numb_fields,
        output_numb_fields,
        n_modes,
        activation=torch.nn.Tanh,
    ):
        r"""
        Initialization of the :class:`FourierBlock2D` class.

        :param int input_numb_fields: The number of channels for the input.
        :param int output_numb_fields: The number of channels for the output.
        :param n_modes: The number of modes to select for each dimension.
            It must be at most equal to :math:`\floor(Nx/2)+1`,
            :math:`\floor(Ny/2)+1`.
        :type n_modes: list[int] | tuple[int]
        :param torch.nn.Module activation: The activation function.
            Default is :class:`torch.nn.Tanh`.
        """
        super().__init__()

        # check type consistency
        check_consistency(activation(), nn.Module)

        # assign variables
        self._spectral_conv = SpectralConvBlock2D(
            input_numb_fields=input_numb_fields,
            output_numb_fields=output_numb_fields,
            n_modes=n_modes,
        )
        self._activation = activation()
        self._linear = nn.Conv2d(input_numb_fields, output_numb_fields, 1)

    def forward(self, x):
        """
        Forward pass of the block. It performs a spectral convolution and a
        linear transformation of the input. Then, it sums the results.

        :param torch.Tensor x: The input tensor for performing the computation.
        :return: The output tensor.
        :rtype: torch.Tensor
        """
        return self._activation(self._spectral_conv(x) + self._linear(x))


class FourierBlock3D(nn.Module):
    """
    The inner block of the Fourier Neural Operator for 3-dimensional input
    tensors.

    The module computes the spectral convolution of the input with a linear
    kernel in the fourier space, and then it maps the input back to the physical
    space. The output is then added to a Linear tranformation of the input in
    the physical space. Finally an activation function is applied to the output.

    .. seealso::

        **Original reference**: Li, Z., Kovachki, N., Azizzadenesheli, K.,
        Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2020).
        *Fourier neural operator for parametric partial differential equations*.
        DOI: `arXiv preprint arXiv:2010.08895.
        <https://arxiv.org/abs/2010.08895>`_
    """

    def __init__(
        self,
        input_numb_fields,
        output_numb_fields,
        n_modes,
        activation=torch.nn.Tanh,
    ):
        r"""
        Initialization of the :class:`FourierBlock3D` class.

        :param int input_numb_fields: The number of channels for the input.
        :param int output_numb_fields: The number of channels for the output.
        :param n_modes: The number of modes to select for each dimension.
            It must be at most equal to :math:`\floor(Nx/2)+1`,
            :math:`\floor(Ny/2)+1`, :math:`\floor(Nz/2)+1`.
        :type n_modes: list[int] | tuple[int]
        :param torch.nn.Module activation: The activation function.
            Default is :class:`torch.nn.Tanh`.
        """
        super().__init__()

        # check type consistency
        check_consistency(activation(), nn.Module)

        # assign variables
        self._spectral_conv = SpectralConvBlock3D(
            input_numb_fields=input_numb_fields,
            output_numb_fields=output_numb_fields,
            n_modes=n_modes,
        )
        self._activation = activation()
        self._linear = nn.Conv3d(input_numb_fields, output_numb_fields, 1)

    def forward(self, x):
        """
        Forward pass of the block. It performs a spectral convolution and a
        linear transformation of the input. Then, it sums the results.

        :param torch.Tensor x: The input tensor for performing the computation.
        :return: The output tensor.
        :rtype: torch.Tensor
        """
        return self._activation(self._spectral_conv(x) + self._linear(x))
