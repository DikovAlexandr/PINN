"""Module for the PowerLoss class."""

import torch

from ..utils import check_consistency
from .loss_interface import LossInterface


class PowerLoss(LossInterface):
    r"""
    Implementation of the Power Loss. It defines a criterion to measures the 
    pointwise error between values in the input :math:`x` and values in the
    target :math:`y`.

    If ``reduction`` is set to ``none``, the loss can be written as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \frac{1}{D}\left[\sum_{i=1}^{D} 
        \left| x_n^i - y_n^i \right|^p\right],
    
    If ``relative`` is set to ``True``, the relative error is computed:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \frac{ \sum_{i=1}^{D} | x_n^i - y_n^i|^p }
        {\sum_{i=1}^{D}|y_n^i|^p},

    where :math:`N` is the batch size.
    
    If ``reduction`` is not ``none``, then:

    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{`mean';}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{`sum'.}
        \end{cases}
    """

    def __init__(self, p=2, reduction="mean", relative=False):
        """
        Initialization of the :class:`PowerLoss` class.

        :param int p: Degree of the Lp norm. It specifies the norm to be
            computed. Default is ``2`` (euclidean norm).
        :param str reduction: The reduction method for the loss.
            Available options: ``none``, ``mean``, ``sum``.
            If ``none``, no reduction is applied. If ``mean``, the sum of the
            loss values is divided by the number of values. If ``sum``, the loss
            values are summed. Default is ``mean``.
        :param bool relative: If ``True``, the relative error is computed.
            Default is ``False``.
        """
        super().__init__(reduction=reduction)

        # check consistency
        check_consistency(p, (str, int, float))
        check_consistency(relative, bool)

        self.p = p
        self.relative = relative

    def forward(self, input, target):
        """
        Forward method of the loss function.

        :param torch.Tensor input: Input tensor from real data.
        :param torch.Tensor target: Model tensor output.
        :return: Loss evaluation.
        :rtype: torch.Tensor
        """
        loss = torch.abs((input - target)).pow(self.p).mean(-1)
        if self.relative:
            loss = loss / torch.abs(input).pow(self.p).mean(-1)
        return self._reduction(loss)
