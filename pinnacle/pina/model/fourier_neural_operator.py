"""Module for the Fourier Neural Operator model class."""

import warnings
import torch
from torch import nn
from ..label_tensor import LabelTensor
from ..utils import check_consistency
from .block.fourier_block import FourierBlock1D, FourierBlock2D, FourierBlock3D
from .kernel_neural_operator import KernelNeuralOperator


class FourierIntegralKernel(torch.nn.Module):
    """
    Fourier Integral Kernel model class.

    This class implements the Fourier Integral Kernel network, which
    performs global convolution in the Fourier space.

    .. seealso::

        **Original reference**: Li, Z., Kovachki, N., Azizzadenesheli, K., Liu,
        B., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2020).
        *Fourier neural operator for parametric partial differential equations*.
        DOI: `arXiv preprint arXiv:2010.08895.
        <https://arxiv.org/abs/2010.08895>`_
    """

    def __init__(
        self,
        input_numb_fields,
        output_numb_fields,
        n_modes,
        dimensions=3,
        padding=8,
        padding_type="constant",
        inner_size=20,
        n_layers=2,
        func=nn.Tanh,
        layers=None,
    ):
        """
        Initialization of the :class:`FourierIntegralKernel` class.

        :param int input_numb_fields: The number of input fields.
        :param int output_numb_fields: The number of output fields.
        :param n_modes: The number of modes.
        :type n_modes: int | list[int]
        :param int dimensions: The number of dimensions. It can be set to ``1``,
            ``2``, or ``3``. Default is ``3``.
        :param int padding: The padding size. Default is ``8``.
        :param str padding_type: The padding strategy. Default is ``constant``.
        :param int inner_size: The inner size. Default is ``20``.
        :param int n_layers: The number of layers. Default is ``2``.
        :param func: The activation function. If a list is passed, it must have
            the same length as ``n_layers``. If a single function is passed, it
            is used for all layers, except for the last one.
            Default is :class:`torch.nn.Tanh`.
        :type func: torch.nn.Module | list[torch.nn.Module]
        :param list[int] layers: The list of the dimension of inner layers.
            If ``None``, ``n_layers`` of dimension ``inner_size`` are used.
            Otherwise, it overrides the values passed to ``n_layers`` and
            ``inner_size``. Default is ``None``.
        :raises RuntimeError: If the number of layers and functions are
            inconsistent.
        :raises RunTimeError: If the number of layers and modes are
            inconsistent.
        """
        super().__init__()

        # check type consistency
        self._check_consistency(
            dimensions,
            padding,
            padding_type,
            inner_size,
            n_layers,
            func,
            layers,
            n_modes,
        )

        # assign padding
        self._padding = padding

        # initialize fourier layer for each dimension
        fourier_layer = self._get_fourier_block(dimensions)

        # Here we build the FNO kernels by stacking Fourier Blocks

        # 1. Assign output dimensions for each FNO layer
        if layers is None:
            layers = [inner_size] * n_layers

        # 2. Assign activation functions for each FNO layer
        if isinstance(func, list):
            if len(layers) != len(func):
                raise RuntimeError(
                    "Inconsistent number of layers and functions."
                )
            _functions = func
        else:
            _functions = [func for _ in range(len(layers) - 1)]
        _functions.append(torch.nn.Identity)

        # 3. Assign modes functions for each FNO layer
        if isinstance(n_modes, list):
            if all(isinstance(i, list) for i in n_modes) and len(layers) != len(
                n_modes
            ):
                raise RuntimeError("Inconsistent number of layers and modes.")
            if all(isinstance(i, int) for i in n_modes):
                n_modes = [n_modes] * len(layers)
        else:
            n_modes = [n_modes] * len(layers)

        # 4. Build the FNO network
        tmp_layers = [input_numb_fields] + layers + [output_numb_fields]
        self._layers = nn.Sequential(
            *[
                fourier_layer(
                    input_numb_fields=tmp_layers[i],
                    output_numb_fields=tmp_layers[i + 1],
                    n_modes=n_modes[i],
                    activation=_functions[i],
                )
                for i in range(len(layers))
            ]
        )

        # 5. Padding values for spectral conv
        if isinstance(padding, int):
            padding = [padding] * dimensions
        self._ipad = [-pad if pad > 0 else None for pad in padding[:dimensions]]
        self._padding_type = padding_type
        self._pad = [
            val for pair in zip([0] * dimensions, padding) for val in pair
        ]

    def forward(self, x):
        """
        Forward pass for the :class:`FourierIntegralKernel` model.

        :param x: The input tensor for performing the computation. Depending
            on the ``dimensions`` in the initialization, it expects a tensor
            with the following shapes:
            * 1D tensors: ``[batch, X, channels]``
            * 2D tensors: ``[batch, X, Y, channels]``
            * 3D tensors: ``[batch, X, Y, Z, channels]``
        :type x: torch.Tensor | LabelTensor
        :raises Warning: If a LabelTensor is passed as input.
        :return: The output tensor.
        :rtype: torch.Tensor
        """
        if isinstance(x, LabelTensor):
            warnings.warn(
                "LabelTensor passed as input is not allowed,"
                " casting LabelTensor to Torch.Tensor"
            )
            x = x.as_subclass(torch.Tensor)
        # permuting the input [batch, channels, x, y, ...]
        permutation_idx = [0, x.ndim - 1, *list(range(1, x.ndim - 1))]
        x = x.permute(permutation_idx)

        # padding the input
        x = torch.nn.functional.pad(x, pad=self._pad, mode=self._padding_type)

        # apply fourier layers
        x = self._layers(x)

        # remove padding
        idxs = [slice(None), slice(None)] + [slice(pad) for pad in self._ipad]
        x = x[idxs]

        # permuting back [batch, x, y, ..., channels]
        permutation_idx = [0, *list(range(2, x.ndim)), 1]
        x = x.permute(permutation_idx)

        return x

    @staticmethod
    def _check_consistency(
        dimensions,
        padding,
        padding_type,
        inner_size,
        n_layers,
        func,
        layers,
        n_modes,
    ):
        """
        Check the consistency of the input parameters.


        :param int dimensions: The number of dimensions.
        :param int padding: The padding size.
        :param str padding_type: The padding strategy.
        :param int inner_size: The inner size.
        :param int n_layers: The number of layers.
        :param func: The activation function.
        :type func: torch.nn.Module | list[torch.nn.Module]
        :param list[int] layers: The list of the dimension of inner layers.
        :param n_modes: The number of modes.
        :type n_modes: int | list[int]
        :raises ValueError: If the input is not consistent.
        """
        check_consistency(dimensions, int)
        check_consistency(padding, int)
        check_consistency(padding_type, str)
        check_consistency(inner_size, int)
        check_consistency(n_layers, int)
        check_consistency(func, nn.Module, subclass=True)

        if layers is not None:
            if isinstance(layers, (tuple, list)):
                check_consistency(layers, int)
            else:
                raise ValueError("layers must be tuple or list of int.")
        if not isinstance(n_modes, (list, tuple, int)):
            raise ValueError(
                "n_modes must be a int or list or tuple of valid modes."
                " More information on the official documentation."
            )

    @staticmethod
    def _get_fourier_block(dimensions):
        """
        Retrieve the Fourier Block class based on the number of dimensions.

        :param int dimensions: The number of dimensions.
        :raises NotImplementedError: If the number of dimensions is not 1, 2,
            or 3.
        :return: The Fourier Block class.
        :rtype: FourierBlock1D | FourierBlock2D | FourierBlock3D
        """
        if dimensions == 1:
            return FourierBlock1D
        if dimensions == 2:
            return FourierBlock2D
        if dimensions == 3:
            return FourierBlock3D
        raise NotImplementedError("FNO implemented only for 1D/2D/3D data.")


class FNO(KernelNeuralOperator):
    """
    Fourier Neural Operator model class.

    The Fourier Neural Operator (FNO) is a general architecture for learning
    operators, which  map functions to functions. It can be trained both with
    Supervised and Physics_Informed learning strategies. The Fourier Neural
    Operator performs global convolution in the Fourier space.

    .. seealso::

        **Original reference**: Li, Z., Kovachki, N., Azizzadenesheli, K.,
        Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2020).
        *Fourier neural operator for parametric partial differential equations*.
        DOI: `arXiv preprint arXiv:2010.08895.
        <https://arxiv.org/abs/2010.08895>`_
    """

    def __init__(
        self,
        lifting_net,
        projecting_net,
        n_modes,
        dimensions=3,
        padding=8,
        padding_type="constant",
        inner_size=20,
        n_layers=2,
        func=nn.Tanh,
        layers=None,
    ):
        """
        :param torch.nn.Module lifting_net: The lifting neural network mapping
            the input to its hidden dimension.
        :param torch.nn.Module projecting_net: The projection neural network
            mapping the hidden representation to the output function.
        :param n_modes: The number of modes.
        :type n_modes: int | list[int]
        :param int dimensions: The number of dimensions. It can be set to ``1``,
            ``2``, or ``3``. Default is ``3``.
        :param int padding: The padding size. Default is ``8``.
        :param str padding_type: The padding strategy. Default is ``constant``.
        :param int inner_size: The inner size. Default is ``20``.
        :param int n_layers: The number of layers. Default is ``2``.
        :param func: The activation function. If a list is passed, it must have
            the same length as ``n_layers``. If a single function is passed, it
            is used for all layers, except for the last one.
            Default is :class:`torch.nn.Tanh`.
        :type func: torch.nn.Module | list[torch.nn.Module]
        :param list[int] layers: The list of the dimension of inner layers.
            If ``None``, ``n_layers`` of dimension ``inner_size`` are used.
            Otherwise, it overrides the values passed to ``n_layers`` and
            ``inner_size``. Default is ``None``.
        """
        lifting_operator_out = lifting_net(
            torch.rand(size=next(lifting_net.parameters()).size())
        ).shape[-1]
        super().__init__(
            lifting_operator=lifting_net,
            projection_operator=projecting_net,
            integral_kernels=FourierIntegralKernel(
                input_numb_fields=lifting_operator_out,
                output_numb_fields=next(projecting_net.parameters()).size(),
                n_modes=n_modes,
                dimensions=dimensions,
                padding=padding,
                padding_type=padding_type,
                inner_size=inner_size,
                n_layers=n_layers,
                func=func,
                layers=layers,
            ),
        )

    def forward(self, x):
        """
        Forward pass for the :class:`FourierNeuralOperator` model.

        The ``lifting_net`` maps the input to the hidden dimension.
        Then, several layers of Fourier blocks are applied. Finally, the
        ``projection_net`` maps the hidden representation to the output
        function.

        :param x: The input tensor for performing the computation. Depending
            on the ``dimensions`` in the initialization, it expects a tensor
            with the following shapes:

            * 1D tensors: ``[batch, X, channels]``
            * 2D tensors: ``[batch, X, Y, channels]``
            * 3D tensors: ``[batch, X, Y, Z, channels]``

        :type x: torch.Tensor | LabelTensor
        :return: The output tensor.
        :rtype: torch.Tensor
        """

        if isinstance(x, LabelTensor):
            x = x.as_subclass(torch.Tensor)
        return super().forward(x)
