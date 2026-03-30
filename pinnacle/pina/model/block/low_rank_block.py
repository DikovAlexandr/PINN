"""Module for the Low Rank Neural Operator Block class."""

import torch

from ...utils import check_consistency


class LowRankBlock(torch.nn.Module):
    """
    The inner block of the Low Rank Neural Operator.

    .. seealso::

        **Original reference**: Kovachki, N., Li, Z., Liu, B.,
        Azizzadenesheli, K., Bhattacharya, K., Stuart, A., & Anandkumar, A.
        (2023). *Neural operator: Learning maps between function
        spaces with applications to PDEs*. Journal of Machine Learning
        Research, 24(89), 1-97.
    """

    def __init__(
        self,
        input_dimensions,
        embedding_dimenion,
        rank,
        inner_size=20,
        n_layers=2,
        func=torch.nn.Tanh,
        bias=True,
    ):
        r"""
        Initialization of the :class:`LowRankBlock` class.

        :param int input_dimensions: The input dimension of the field.
        :param int embedding_dimenion: The embedding dimension of the field.
        :param int rank: The rank of the low rank approximation. The expected
            value is :math:`2d`, where :math:`d` is the rank of each basis
            function.
        :param int inner_size: The number of neurons for each hidden layer in
            the basis function neural network. Default is ``20``.
        :param int n_layers: The number of hidden layers in the basis function
            neural network. Default is ``2``.
        :param func: The activation function. If a list is passed, it must have
            the same length as ``n_layers``. If a single function is passed, it
            is used for all layers, except for the last one.
            Default is :class:`torch.nn.Tanh`.
        :type func: torch.nn.Module | list[torch.nn.Module]
        :param bool bias: If ``True`` bias is considered for the basis function
            neural network. Default is ``True``.
        """
        super().__init__()
        from ..feed_forward import FeedForward

        # Assignment (check consistency inside FeedForward)
        self._basis = FeedForward(
            input_dimensions=input_dimensions,
            output_dimensions=2 * rank * embedding_dimenion,
            inner_size=inner_size,
            n_layers=n_layers,
            func=func,
            bias=bias,
        )
        self._nn = torch.nn.Linear(embedding_dimenion, embedding_dimenion)

        check_consistency(rank, int)
        self._rank = rank
        self._func = func()

    def forward(self, x, coords):
        r"""
        Forward pass of the block. It performs an affine transformation of the
        field, followed by a low rank approximation. The latter is performed by
        means of a dot product of the basis :math:`\psi^{(i)}` with the vector
        field :math:`v` to compute coefficients used to expand
        :math:`\phi^{(i)}`, evaluated in the spatial input :math:`x`.

        :param torch.Tensor x: The input tensor for performing the computation.
        :param torch.Tensor coords: The coordinates for which the field is
            evaluated to perform the computation.
        :return: The output tensor.
        :rtype: torch.Tensor
        """
        # extract basis
        coords = coords.as_subclass(torch.Tensor)
        basis = self._basis(coords)
        # reshape [B, N, D, 2*rank]
        shape = list(basis.shape[:-1]) + [-1, 2 * self.rank]
        basis = basis.reshape(shape)
        # divide
        psi = basis[..., : self.rank]
        phi = basis[..., self.rank :]
        # compute dot product
        coeff = torch.einsum("...dr,...d->...r", psi, x)
        # expand the basis
        expansion = torch.einsum("...r,...dr->...d", coeff, phi)
        # apply linear layer and return
        return self._func(self._nn(x) + expansion)

    @property
    def rank(self):
        """
        The basis rank.

        :return: The basis rank.
        :rtype: int
        """
        return self._rank
