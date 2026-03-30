"""Module for spectral convolution blocks."""

import torch
from torch import nn
from ...utils import check_consistency


######## 1D Spectral Convolution ###########
class SpectralConvBlock1D(nn.Module):
    """
    Spectral Convolution Block for one-dimensional tensors.

    This class computes the spectral convolution of the input with a linear
    kernel in the fourier space, and then it maps the input back to the physical
    space.
    The block expects an input of size [``batch``, ``input_numb_fields``, ``N``]
    and returns an output of size [``batch``, ``output_numb_fields``, ``N``].
    """

    def __init__(self, input_numb_fields, output_numb_fields, n_modes):
        r"""
        Initialization of the :class:`SpectralConvBlock1D` class.

        :param int input_numb_fields: The number of channels for the input.
        :param int output_numb_fields: The number of channels for the output.
        :param int n_modes: The number of modes to select for each dimension.
            It must be at most equal to :math:`\floor(Nx/2)+1`.
        """
        super().__init__()

        # check type consistency
        check_consistency(input_numb_fields, int)
        check_consistency(output_numb_fields, int)

        # assign variables
        self._modes = n_modes
        self._input_channels = input_numb_fields
        self._output_channels = output_numb_fields

        # scaling factor
        scale = 1.0 / (self._input_channels * self._output_channels)
        self._weights = nn.Parameter(
            scale
            * torch.rand(
                self._input_channels,
                self._output_channels,
                self._modes,
                dtype=torch.cfloat,
            )
        )

    def _compute_mult1d(self, input, weights):
        """
        Compute the matrix multiplication of the input and the linear kernel
        weights.

        :param torch.Tensor input: The input tensor. Expected of size
            [``batch``, ``input_numb_fields``, ``N``].
        :param torch.Tensor weights: The kernel weights. Expected of size
            [``input_numb_fields``, ``output_numb_fields``, ``N``].
        :return: The result of the matrix multiplication.
        :rtype: torch.Tensor
        """
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        """
        Forward pass.

        :param torch.Tensor x: The input tensor. Expected of size
            [``batch``, ``input_numb_fields``, ``N``].
        :return: The input tensor. Expected of size
            [``batch``, ``output_numb_fields``, ``N``].
        :rtype: torch.Tensor
        """
        batch_size = x.shape[0]

        # Compute Fourier transform of the input
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batch_size,
            self._output_channels,
            x.size(-1) // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )
        out_ft[:, :, : self._modes] = self._compute_mult1d(
            x_ft[:, :, : self._modes], self._weights
        )

        # Return to physical space
        return torch.fft.irfft(out_ft, n=x.size(-1))


######## 2D Spectral Convolution ###########
class SpectralConvBlock2D(nn.Module):
    """
    Spectral Convolution Block for two-dimensional tensors.

    This class computes the spectral convolution of the input with a linear
    kernel in the fourier space, and then it maps the input back to the physical
    space.
    The block expects an input of size
    [``batch``, ``input_numb_fields``, ``Nx``, ``Ny``]
    and returns an output of size
    [``batch``, ``output_numb_fields``, ``Nx``, ``Ny``].
    """

    def __init__(self, input_numb_fields, output_numb_fields, n_modes):
        r"""
        Initialization of the :class:`SpectralConvBlock2D` class.

        :param int input_numb_fields: The number of channels for the input.
        :param int output_numb_fields: The number of channels for the output.
        :param n_modes: The number of modes to select for each dimension.
            It must be at most equal to :math:`\floor(Nx/2)+1`,
            :math:`\floor(Ny/2)+1`.
        :type n_modes: list[int] | tuple[int]
        :raises ValueError: If the number of modes is not consistent.
        :raises ValueError: If the number of modes is not a list or tuple.
        """
        super().__init__()

        # check type consistency
        check_consistency(input_numb_fields, int)
        check_consistency(output_numb_fields, int)
        check_consistency(n_modes, int)
        if isinstance(n_modes, (tuple, list)):
            if len(n_modes) != 2:
                raise ValueError(
                    "Expected n_modes to be a list or tuple of len two, "
                    "with each entry corresponding to the number of modes "
                    "for each dimension "
                )
        elif isinstance(n_modes, int):
            n_modes = [n_modes] * 2
        else:
            raise ValueError(
                "Expected n_modes to be a list or tuple of len two, "
                "with each entry corresponding to the number of modes "
                "for each dimension; or an int value representing the "
                "number of modes for all dimensions"
            )

        # assign variables
        self._modes = n_modes
        self._input_channels = input_numb_fields
        self._output_channels = output_numb_fields

        # scaling factor
        scale = 1.0 / (self._input_channels * self._output_channels)
        self._weights1 = nn.Parameter(
            scale
            * torch.rand(
                self._input_channels,
                self._output_channels,
                self._modes[0],
                self._modes[1],
                dtype=torch.cfloat,
            )
        )
        self._weights2 = nn.Parameter(
            scale
            * torch.rand(
                self._input_channels,
                self._output_channels,
                self._modes[0],
                self._modes[1],
                dtype=torch.cfloat,
            )
        )

    def _compute_mult2d(self, input, weights):
        """
        Compute the matrix multiplication of the input and the linear kernel
        weights.

        :param torch.Tensor input: The input tensor. Expected of size
            [``batch``, ``input_numb_fields``, ``Nx``, ``Ny``].
        :param torch.Tensor weights: The kernel weights. Expected of size
            [``input_numb_fields``, ``output_numb_fields``, ``Nx``, ``Ny``].
        :return: The result of the matrix multiplication.
        :rtype: torch.Tensor
        """
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        """
        Forward pass.

        :param torch.Tensor x: The input tensor. Expected of size
            [``batch``, ``input_numb_fields``, ``Nx``, ``Ny``].
        :return: The input tensor. Expected of size
            [``batch``, ``output_numb_fields``, ``Nx``, ``Ny``].
        :rtype: torch.Tensor
        """

        batch_size = x.shape[0]

        # Compute Fourier transform of the input
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batch_size,
            self._output_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )
        out_ft[:, :, : self._modes[0], : self._modes[1]] = self._compute_mult2d(
            x_ft[:, :, : self._modes[0], : self._modes[1]], self._weights1
        )
        out_ft[:, :, -self._modes[0] :, : self._modes[1] :] = (
            self._compute_mult2d(
                x_ft[:, :, -self._modes[0] :, : self._modes[1]], self._weights2
            )
        )

        # Return to physical space
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


######## 3D Spectral Convolution ###########
class SpectralConvBlock3D(nn.Module):
    """
    Spectral Convolution Block for three-dimensional tensors.

    This class computes the spectral convolution of the input with a linear
    kernel in the fourier space, and then it maps the input back to the physical
    space.
    The block expects an input of size
    [``batch``, ``input_numb_fields``, ``Nx``, ``Ny``, ``Nz``]
    and returns an output of size
    [``batch``, ``output_numb_fields``, ``Nx``, ``Ny``, ``Nz``].
    """

    def __init__(self, input_numb_fields, output_numb_fields, n_modes):
        r"""
        Initialization of the :class:`SpectralConvBlock3D` class.

        :param int input_numb_fields: The number of channels for the input.
        :param int output_numb_fields: The number of channels for the output.
        :param n_modes: The number of modes to select for each dimension.
            It must be at most equal to :math:`\floor(Nx/2)+1`,
            :math:`\floor(Ny/2)+1`, :math:`\floor(Nz/2)+1`.
        :type n_modes: list[int] | tuple[int]
        :raises ValueError: If the number of modes is not consistent.
        :raises ValueError: If the number of modes is not a list or tuple.
        """
        super().__init__()

        # check type consistency
        check_consistency(input_numb_fields, int)
        check_consistency(output_numb_fields, int)
        check_consistency(n_modes, int)
        if isinstance(n_modes, (tuple, list)):
            if len(n_modes) != 3:
                raise ValueError(
                    "Expected n_modes to be a list or tuple of len three, "
                    "with each entry corresponding to the number of modes "
                    "for each dimension "
                )
        elif isinstance(n_modes, int):
            n_modes = [n_modes] * 3
        else:
            raise ValueError(
                "Expected n_modes to be a list or tuple of len three, "
                "with each entry corresponding to the number of modes "
                "for each dimension; or an int value representing the "
                "number of modes for all dimensions"
            )

        # assign variables
        self._modes = n_modes
        self._input_channels = input_numb_fields
        self._output_channels = output_numb_fields

        # scaling factor
        scale = 1.0 / (self._input_channels * self._output_channels)
        self._weights1 = nn.Parameter(
            scale
            * torch.rand(
                self._input_channels,
                self._output_channels,
                self._modes[0],
                self._modes[1],
                self._modes[2],
                dtype=torch.cfloat,
            )
        )
        self._weights2 = nn.Parameter(
            scale
            * torch.rand(
                self._input_channels,
                self._output_channels,
                self._modes[0],
                self._modes[1],
                self._modes[2],
                dtype=torch.cfloat,
            )
        )
        self._weights3 = nn.Parameter(
            scale
            * torch.rand(
                self._input_channels,
                self._output_channels,
                self._modes[0],
                self._modes[1],
                self._modes[2],
                dtype=torch.cfloat,
            )
        )
        self._weights4 = nn.Parameter(
            scale
            * torch.rand(
                self._input_channels,
                self._output_channels,
                self._modes[0],
                self._modes[1],
                self._modes[2],
                dtype=torch.cfloat,
            )
        )

    def _compute_mult3d(self, input, weights):
        """
        Compute the matrix multiplication of the input and the linear kernel
        weights.

        :param torch.Tensor input: The input tensor. Expected of size
            [``batch``, ``input_numb_fields``, ``Nx``, ``Ny``, ``Nz``].
        :param torch.Tensor weights: The kernel weights. Expected of size
            [``input_numb_fields``, ``output_numb_fields``, ``Nx``, ``Ny``,
            ``Nz``].
        :return: The result of the matrix multiplication.
        :rtype: torch.Tensor
        """
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        """
        Forward pass.

        :param torch.Tensor x: The input tensor. Expected of size
            [``batch``, ``input_numb_fields``, ``Nx``, ``Ny``, ``Nz``].
        :return: The input tensor. Expected of size
            [``batch``, ``output_numb_fields``, ``Nx``, ``Ny``, ``Nz``].
        :rtype: torch.Tensor
        """

        batch_size = x.shape[0]

        # Compute Fourier transform of the input
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batch_size,
            self._output_channels,
            x.size(-3),
            x.size(-2),
            x.size(-1) // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )

        slice0 = (
            slice(None),
            slice(None),
            slice(self._modes[0]),
            slice(self._modes[1]),
            slice(self._modes[2]),
        )
        out_ft[slice0] = self._compute_mult3d(x_ft[slice0], self._weights1)

        slice1 = (
            slice(None),
            slice(None),
            slice(self._modes[0]),
            slice(-self._modes[1], None),
            slice(self._modes[2]),
        )
        out_ft[slice1] = self._compute_mult3d(x_ft[slice1], self._weights2)

        slice2 = (
            slice(None),
            slice(None),
            slice(-self._modes[0], None),
            slice(self._modes[1]),
            slice(self._modes[2]),
        )
        out_ft[slice2] = self._compute_mult3d(x_ft[slice2], self._weights3)

        slice3 = (
            slice(None),
            slice(None),
            slice(-self._modes[0], None),
            slice(-self._modes[1], None),
            slice(self._modes[2]),
        )
        out_ft[slice3] = self._compute_mult3d(x_ft[slice3], self._weights4)

        # Return to physical space
        return torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
