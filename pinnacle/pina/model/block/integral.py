"""Module to perform integration for continuous convolution."""

import torch


class Integral:
    """
    Class allowing integration for continous convolution.
    """

    def __init__(self, param):
        """
        Initializzation of the :class:`Integral` class.

        :param param: The type of continuous convolution.
        :type param: string
        :raises TypeError: If the parameter is neither ``discrete``
            nor ``continuous``.
        """
        if param == "discrete":
            self.make_integral = self.integral_param_disc
        elif param == "continuous":
            self.make_integral = self.integral_param_cont
        else:
            raise TypeError

    def __call__(self, *args, **kwds):
        """
        Call the integral function

        :param list args: Arguments for the integral function.
        :param dict kwds: Keyword arguments for the integral function.
        :return: The integral of the input.
        :rtype: torch.tensor
        """
        return self.make_integral(*args, **kwds)

    def _prepend_zero(self, x):
        """
        Create bins to perform integration.

        :param torch.Tensor x: The input tensor.
        :return: The bins for the integral.
        :rtype: torch.Tensor
        """
        return torch.cat((torch.zeros(1, dtype=x.dtype, device=x.device), x))

    def integral_param_disc(self, x, y, idx):
        """
        Perform discrete integration with discrete parameters.

        :param torch.Tensor x: The first input tensor.
        :param torch.Tensor y: The second input tensor.
        :param list[int] idx: The indices for different strides.
        :return: The discrete integral.
        :rtype: torch.Tensor
        """
        cs_idxes = self._prepend_zero(torch.cumsum(torch.tensor(idx), 0))
        cs = self._prepend_zero(torch.cumsum(x.flatten() * y.flatten(), 0))
        return cs[cs_idxes[1:]] - cs[cs_idxes[:-1]]

    def integral_param_cont(self, x, y, idx):
        """
        Perform continuous integration with continuous parameters.

        :param torch.Tensor x: The first input tensor.
        :param torch.Tensor y: The second input tensor.
        :param list[int] idx: The indices for different strides.
        :raises NotImplementedError: The method is not implemented.
        """
        raise NotImplementedError
