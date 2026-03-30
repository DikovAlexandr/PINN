"""Module for the Multi Feed Forward model class."""

from abc import ABC, abstractmethod
import torch
from .feed_forward import FeedForward


class MultiFeedForward(torch.nn.Module, ABC):
    """
    Multi Feed Forward neural network model class.

    This model allows to create a network with multiple Feed Forward neural
    networks combined together. The user is required to define the ``forward``
    method to choose how to combine the networks.
    """

    def __init__(self, ffn_dict):
        """
        Initialization of the :class:`MultiFeedForward` class.

        :param dict ffn_dict: A dictionary containing the Feed Forward neural
            networks to be combined.
        :raises TypeError: If the input is not a dictionary.
        """
        super().__init__()

        if not isinstance(ffn_dict, dict):
            raise TypeError

        for name, constructor_args in ffn_dict.items():
            setattr(self, name, FeedForward(**constructor_args))

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Forward pass for the :class:`MultiFeedForward` model.

        The user is required to define this method to choose how to combine the
        networks.
        """
