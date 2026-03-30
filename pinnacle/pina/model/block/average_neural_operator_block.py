"""Module for the Averaging Neural Operator Block class."""

import torch
from torch import nn
from ...utils import check_consistency


class AVNOBlock(nn.Module):
    r"""
    The inner block of the Averaging Neural Operator.

    The operator layer performs an affine transformation where the convolution
    is approximated with a local average. Given the input function
    :math:`v(x)\in\mathbb{R}^{\rm{emb}}` the layer computes the operator update
    :math:`K(v)` as:

    .. math::
        K(v) = \sigma\left(Wv(x) + b + \frac{1}{|\mathcal{A}|}\int v(y)dy\right)

    where:

    *   :math:`\mathbb{R}^{\rm{emb}}` is the embedding (hidden) size
        corresponding to the ``hidden_size`` object
    *   :math:`\sigma` is a non-linear activation, corresponding to the
        ``func`` object
    *   :math:`W\in\mathbb{R}^{\rm{emb}\times\rm{emb}}` is a tunable matrix.
    *   :math:`b\in\mathbb{R}^{\rm{emb}}` is a tunable bias.

    .. seealso::

        **Original reference**: Lanthaler S., Li, Z., Stuart, A. (2020).
        *The Nonlocal Neural Operator: Universal Approximation*.
        DOI: `arXiv preprint arXiv:2304.13221.
        <https://arxiv.org/abs/2304.13221>`_
    """

    def __init__(self, hidden_size=100, func=nn.GELU):
        """
        Initialization of the :class:`AVNOBlock` class.

        :param int hidden_size: The size of the hidden layer.
            Defaults is ``100``.
        :param func: The activation function.
            Default is :class:`torch.nn.GELU`.
        """
        super().__init__()

        # Check type consistency
        check_consistency(hidden_size, int)
        check_consistency(func, nn.Module, subclass=True)
        # Assignment
        self._nn = nn.Linear(hidden_size, hidden_size)
        self._func = func()

    def forward(self, x):
        r"""
        Forward pass of the block. It performs a sum of local average and an
        affine transformation of the field.

        :param torch.Tensor x: The input tensor for performing the computation.
        :return: The output tensor.
        :rtype: torch.Tensor
        """
        return self._func(self._nn(x) + torch.mean(x, dim=1, keepdim=True))
