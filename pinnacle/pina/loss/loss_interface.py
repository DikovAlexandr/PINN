"""Module for the Loss Interface."""

from abc import ABCMeta, abstractmethod
from torch.nn.modules.loss import _Loss
import torch


class LossInterface(_Loss, metaclass=ABCMeta):
    """
    Abstract base class for all losses. All classes defining a loss function
    should inherit from this interface.
    """

    def __init__(self, reduction="mean"):
        """
        Initialization of the :class:`LossInterface` class.

        :param str reduction: The reduction method for the loss.
            Available options: ``none``, ``mean``, ``sum``.
            If ``none``, no reduction is applied. If ``mean``, the sum of the
            loss values is divided by the number of values. If ``sum``, the loss
            values are summed. Default is ``mean``.
        """
        super().__init__(reduction=reduction, size_average=None, reduce=None)

    @abstractmethod
    def forward(self, input, target):
        """
        Forward method of the loss function.

        :param torch.Tensor input: Input tensor from real data.
        :param torch.Tensor target: Model tensor output.
        """

    def _reduction(self, loss):
        """
        Apply the reduction to the loss.

        :param torch.Tensor loss: The tensor containing the pointwise losses.
        :raises ValueError: If the reduction method is not valid.
        :return: Reduced loss.
        :rtype: torch.Tensor
        """
        if self.reduction == "none":
            ret = loss
        elif self.reduction == "mean":
            ret = torch.mean(loss, keepdim=True, dim=-1)
        elif self.reduction == "sum":
            ret = torch.sum(loss, keepdim=True, dim=-1)
        else:
            raise ValueError(self.reduction + " is not valid")
        return ret
