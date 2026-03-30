"""Module for Base Continuous Convolution class."""

import warnings
import torch


class PODBlock(torch.nn.Module):
    """
    Proper Orthogonal Decomposition block.

    This block projects the input field on the proper orthogonal decomposition
    basis. Before being used, it must be fitted to the data with the ``fit``
    method, which invokes the singular value decomposition. This block is not
    trainable.

    .. note::
        All the POD modes are stored in memory, avoiding to recompute them when
        the rank changes, leading to increased memory usage.
    """

    def __init__(self, rank, scale_coefficients=True):
        """
        Initialization of the :class:`PODBlock` class.

        :param int rank: The rank of the POD layer.
        :param bool scale_coefficients: If ``True``, the coefficients are scaled
            after the projection to have zero mean and unit variance.
            Default is ``True``.
        """
        super().__init__()
        self.__scale_coefficients = scale_coefficients
        self.register_buffer("_basis", None)
        self._singular_values = None
        self.register_buffer("_std", None)
        self.register_buffer("_mean", None)
        self._rank = rank

    @property
    def rank(self):
        """
        The rank of the POD layer.

        :return: The rank of the POD layer.
        :rtype: int
        """
        return self._rank

    @rank.setter
    def rank(self, value):
        """
        Set the rank of the POD layer.

        :param int value: The new rank of the POD layer.
        :raises ValueError: If the rank is not a positive integer.
        """
        if value < 1 or not isinstance(value, int):
            raise ValueError("The rank must be positive integer")

        self._rank = value

    @property
    def basis(self):
        """
        The POD basis. It is a matrix whose columns are the first ``rank`` POD
        modes.

        :return: The POD basis.
        :rtype: torch.Tensor
        """
        if self._basis is None:
            return None

        return self._basis[: self.rank]

    @property
    def singular_values(self):
        """
        The singular values of the POD basis.

        :return: The singular values.
        :rtype: torch.Tensor
        """
        if self._singular_values is None:
            return None

        return self._singular_values[: self.rank]

    @property
    def scaler(self):
        """
        Return the scaler dictionary, having keys ``mean`` and ``std``
        corresponding to the mean and the standard deviation of the
        coefficients, respectively.

        :return: The scaler dictionary.
        :rtype: dict
        """
        if self._std is None:
            return None

        return {
            "mean": self._mean[: self.rank],
            "std": self._std[: self.rank],
        }

    @property
    def scale_coefficients(self):
        """
        The flag indicating if the coefficients are scaled after the projection.

        :return: The flag indicating if the coefficients are scaled.
        :rtype: bool
        """
        return self.__scale_coefficients

    def fit(self, X, randomized=True):
        """
        Set the POD basis by performing the singular value decomposition of the
        given tensor. If ``self.scale_coefficients`` is True, the coefficients
        are scaled after the projection to have zero mean and unit variance.

        :param torch.Tensor X: The input tensor to be reduced.
        :param bool randomized: If ``True``, a randomized algorithm is used to
            compute the POD basis. In general, this leads to faster
            computations, but the results may be less accurate. Default is
            ``True``.
        """
        self._fit_pod(X, randomized)

        if self.__scale_coefficients:
            self._fit_scaler(torch.matmul(self._basis, X.T))

    def _fit_scaler(self, coeffs):
        """
        Compute the mean and the standard deviation of the given coefficients,
        which are then stored in ``self._scaler``.

        :param torch.Tensor coeffs: The coefficients to be scaled.
        """
        self._std = torch.std(coeffs, dim=1)  # pylint: disable=W0201
        self._mean = torch.mean(coeffs, dim=1)  # pylint: disable=W0201

    def _fit_pod(self, X, randomized):
        """
        Compute the POD basis of the given tensor, which is then stored in
        ``self._basis``.

        :param torch.Tensor X: The tensor to be reduced.
        """
        if X.device.type == "mps":  #  svd_lowrank not arailable for mps
            warnings.warn(
                "svd_lowrank not available for mps, using svd instead."
                "This may slow down computations.",
                ResourceWarning,
            )
            u, s, _ = torch.svd(X.T)
        else:
            if randomized:
                warnings.warn(
                    "Considering a randomized algorithm to compute the POD "
                    "basis"
                )
                u, s, _ = torch.svd_lowrank(X.T, q=X.shape[0])

            else:
                u, s, _ = torch.svd(X.T)
        self._basis = u.T  # pylint: disable=W0201
        self._singular_values = s

    def forward(self, X):
        """
        The forward pass of the POD layer.

        :param torch.Tensor X: The input tensor to be reduced.
        :return: The reduced tensor.
        :rtype: torch.Tensor
        """
        return self.reduce(X)

    def reduce(self, X):
        """
        Reduce the input tensor to its POD representation. The POD layer must
        be fitted before being used.

        :param torch.Tensor X: The input tensor to be reduced.
        :raises RuntimeError: If the POD layer is not fitted.
        :return: The reduced tensor.
        :rtype: torch.Tensor
        """
        if self._basis is None:
            raise RuntimeError(
                "The POD layer needs to be fitted before being used."
            )

        coeff = torch.matmul(self.basis, X.T)
        if coeff.ndim == 1:
            coeff = coeff.unsqueeze(1)

        coeff = coeff.T
        if self.__scale_coefficients:
            coeff = (coeff - self.scaler["mean"]) / self.scaler["std"]

        return coeff

    def expand(self, coeff):
        """
        Expand the given coefficients to the original space. The POD layer needs
        to be fitted before being used.

        :param torch.Tensor coeff: The coefficients to be expanded.
        :raises RuntimeError: If the POD layer is not fitted.
        :return: The expanded tensor.
        :rtype: torch.Tensor
        """
        if self._basis is None:
            raise RuntimeError(
                "The POD layer needs to be trained before being used."
            )

        if self.__scale_coefficients:
            coeff = coeff * self.scaler["std"] + self.scaler["mean"]
        predicted = torch.matmul(self.basis.T, coeff.T).T

        if predicted.ndim == 1:
            predicted = predicted.unsqueeze(0)

        return predicted
