"""Modules for the the Embedding blocks."""

import torch
from pina.utils import check_consistency


class PeriodicBoundaryEmbedding(torch.nn.Module):
    r"""
    Enforcing hard-constrained periodic boundary conditions by embedding the
    input.

    A function :math:`u:\mathbb{R}^{\rm{in}} \rightarrow\mathbb{R}^{\rm{out}}`
    is periodic with respect to the spatial coordinates :math:`\mathbf{x}`
    with period :math:`\mathbf{L}` if:

    .. math::
        u(\mathbf{x})  = u(\mathbf{x} + n \mathbf{L})\;\;
        \forall n\in\mathbb{N}.

    The :class:`PeriodicBoundaryEmbedding` augments the input as follows:

    .. math::
        \mathbf{x} \rightarrow \tilde{\mathbf{x}} = \left[1,
        \cos\left(\frac{2\pi}{L_1} x_1 \right),
        \sin\left(\frac{2\pi}{L_1}x_1\right), \cdots,
        \cos\left(\frac{2\pi}{L_{\rm{in}}}x_{\rm{in}}\right),
        \sin\left(\frac{2\pi}{L_{\rm{in}}}x_{\rm{in}}\right)\right],

    where :math:`\text{dim}(\tilde{\mathbf{x}}) = 3\text{dim}(\mathbf{x})`.

    .. seealso::
        **Original reference**:
            1.  Dong, Suchuan, and Naxian Ni (2021).
                *A method for representing periodic functions and enforcing
                exactly periodic boundary conditions with deep neural networks*.
                Journal of Computational Physics 435, 110242.
                DOI: `10.1016/j.jcp.2021.110242.
                <https://doi.org/10.1016/j.jcp.2021.110242>`_
            2.  Wang, S., Sankaran, S., Wang, H., & Perdikaris, P. (2023).
                *An expert's guide to training physics-informed neural
                networks*.
                DOI: `arXiv preprint arXiv:2308.0846.
                <https://arxiv.org/abs/2308.08468>`_

    .. warning::
        The embedding is a truncated fourier expansion, and enforces periodic
        boundary conditions only for the function, and not for its derivatives.
        Enforcement of the approximate periodicity in the derivatives can be
        performed. Extensive tests have shown (see referenced papers) that this
        implementation can correctly enforce the periodic boundary conditions on
        the derivatives up to the order :math:`\sim 2,3`. This is not guaranteed
        for orders :math:`>3`. The PINA module is tested only for periodic
        boundary conditions on the function itself.
    """

    def __init__(self, input_dimension, periods, output_dimension=None):
        """
        Initialization of the :class:`PeriodicBoundaryEmbedding` block.

        :param int input_dimension: The dimension of the input tensor.
        :param periods: The periodicity with respect to each dimension for the
            input data. If ``float`` or ``int`` is passed, the period is assumed
            to be constant over all the dimensions of the data. If a ``dict`` is
            passed the `dict.values` represent periods, while the ``dict.keys``
            represent the dimension where the periodicity is enforced.
            The `dict.keys` can either be `int` if working with
            :class:`torch.Tensor`, or ``str`` if working with
            :class:`pina.label_tensor.LabelTensor`.
        :type periods: float | int | dict
        :param int output_dimension: The dimension of the output after the
            fourier embedding. If not ``None``, a :class:`torch.nn.Linear` layer
            is applied to the fourier embedding output to match the desired
            dimensionality. Default is ``None``.
        :raises TypeError: If the periods dict is not consistent.
        """
        super().__init__()

        # check input consistency
        check_consistency(periods, (float, int, dict))
        check_consistency(input_dimension, int)
        if output_dimension is not None:
            check_consistency(output_dimension, int)
            self._layer = torch.nn.Linear(input_dimension * 3, output_dimension)
        else:
            self._layer = torch.nn.Identity()

        # checks on the periods
        if isinstance(periods, dict):
            if not all(
                isinstance(dim, (str, int)) and isinstance(period, (float, int))
                for dim, period in periods.items()
            ):
                raise TypeError(
                    "In dictionary periods, keys must be integers"
                    " or strings, and values must be float or int."
                )
            self._period = periods
        else:
            self._period = {k: periods for k in range(input_dimension)}

    def forward(self, x):
        """
        Forward pass.

        :param x: The input tensor.
        :type x: torch.Tensor | LabelTensor
        :return: Periodic embedding of the input.
        :rtype: torch.Tensor
        """
        omega = torch.stack(
            [
                torch.pi * 2.0 / torch.tensor([val], device=x.device)
                for val in self._period.values()
            ],
            dim=-1,
        )
        x = self._get_vars(x, list(self._period.keys()))
        return self._layer(
            torch.cat(
                [
                    torch.ones_like(x),
                    torch.cos(omega * x),
                    torch.sin(omega * x),
                ],
                dim=-1,
            )
        )

    def _get_vars(self, x, indeces):
        """
        Get the variables from input tensor ordered by specific indeces.

        :param x: The input tensor from which to extract.
        :type x: torch.Tensor | LabelTensor
        :param indeces: The indeces to extract.
        :type indeces: list[int] | list[str]
        :raises RuntimeError: If the indeces are not consistent.
        :raises RuntimeError: If the extraction is not possible.
        :return: The extracted tensor.
        :rtype: torch.Tensor | LabelTensor
        """
        if isinstance(indeces[0], str):
            try:
                return x.extract(indeces)
            except AttributeError as e:
                raise RuntimeError(
                    "Not possible to extract input variables from tensor."
                    " Ensure that the passed tensor is a LabelTensor or"
                    " pass list of integers to extract variables. For"
                    " more information refer to warning in the documentation."
                ) from e
        elif isinstance(indeces[0], int):
            return x[..., indeces]
        else:
            raise RuntimeError(
                "Not able to extract correct indeces for tensor."
                " For more information refer to warning in the documentation."
            )

    @property
    def period(self):
        """
        The period of the function.

        :return: The period of the function.
        :rtype: dict | float | int
        """
        return self._period


class FourierFeatureEmbedding(torch.nn.Module):
    r"""
    Fourier Feature Embedding class to encode the input features using random
    Fourier features.

    This class applies a Fourier transformation to the input features, which can
    help in learning high-frequency variations in data. The class supports
    multiscale feature embedding, creating embeddings for each scale specified
    by the ``sigma`` parameter.

    The Fourier Feature Embedding augments the input features as follows
    (3.10 of original paper):

    .. math::
        \mathbf{x} \rightarrow \tilde{\mathbf{x}} = \left[
        \cos\left( \mathbf{B} \mathbf{x} \right),
        \sin\left( \mathbf{B} \mathbf{x} \right)\right],

    where :math:`\mathbf{B}_{ij} \sim \mathcal{N}(0, \sigma^2)`.

    If multiple ``sigma`` are passed, the resulting embeddings are concateneted:

    .. math::
        \mathbf{x} \rightarrow \tilde{\mathbf{x}} = \left[
        \cos\left( \mathbf{B}^1 \mathbf{x} \right),
        \sin\left( \mathbf{B}^1 \mathbf{x} \right),
        \cos\left( \mathbf{B}^2 \mathbf{x} \right),
        \sin\left( \mathbf{B}^3 \mathbf{x} \right),
        \dots,
        \cos\left( \mathbf{B}^M \mathbf{x} \right),
        \sin\left( \mathbf{B}^M \mathbf{x} \right)\right],

    where :math:`\mathbf{B}^k_{ij} \sim \mathcal{N}(0, \sigma_k^2) \quad k \in
    (1, \dots, M)`.

    .. seealso::
        **Original reference**:
        Wang, S., Wang, H., and Perdikaris, P. (2021).
        *On the eigenvector bias of Fourier feature networks: From regression to
        solving multi-scale PDEs with physics-informed neural networks.*
        Computer Methods in Applied Mechanics and Engineering 384 (2021):
        113938.
        DOI: `10.1016/j.cma.2021.113938.
        <https://doi.org/10.1016/j.cma.2021.113938>`_
    """

    def __init__(self, input_dimension, output_dimension, sigma):
        """
        Initialization of the :class:`FourierFeatureEmbedding` block.

        :param int input_dimension: The dimension of the input tensor.
        :param int output_dimension: The dimension of the output tensor. The
            output is obtained as a concatenation of cosine and sine embeddings.
        :param sigma: The standard deviation used for the Fourier Embedding.
            This value must reflect the granularity of the scale in the
            differential equation solution.
        :type sigma: float | int
        :raises RuntimeError: If the output dimension is not an even number.
        """
        super().__init__()

        # check consistency
        check_consistency(sigma, (int, float))
        check_consistency(output_dimension, int)
        check_consistency(input_dimension, int)
        if output_dimension % 2:
            raise RuntimeError(
                "Expected output_dimension to be a even number, "
                f"got {output_dimension}."
            )

        # assign sigma
        self._sigma = sigma

        # create non-trainable matrices
        self._matrix = (
            torch.rand(
                size=(input_dimension, output_dimension // 2),
                requires_grad=False,
            )
            * self.sigma
        )

    def forward(self, x):
        """
        Forward pass.

        :param x: The input tensor.
        :type x: torch.Tensor | LabelTensor
        :return: Fourier embedding of the input.
        :rtype: torch.Tensor
        """
        # compute random matrix multiplication
        out = torch.mm(x, self._matrix.to(device=x.device, dtype=x.dtype))
        # return embedding
        return torch.cat(
            [torch.cos(2 * torch.pi * out), torch.sin(2 * torch.pi * out)],
            dim=-1,
        )

    @property
    def sigma(self):
        """
        The standard deviation used for the Fourier Embedding.

        :return: The standard deviation used for the Fourier Embedding.
        :rtype: float | int
        """
        return self._sigma
