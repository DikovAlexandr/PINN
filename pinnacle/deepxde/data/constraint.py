from .data import Data
from .. import config
from ..backend import torch


class Constraint(Data):
    """General constraints."""

    def __init__(self, constraint, train_x, test_x):
        self.constraint = constraint
        self.train_x = train_x
        self.test_x = test_x

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        # PyTorch uses model.training attribute to check training mode
        if model.net.training:
            f = self.constraint(inputs, outputs, self.train_x)
        else:
            f = self.constraint(inputs, outputs, self.test_x)
        
        # Create zeros tensor with same shape and dtype as f
        zeros = torch.zeros_like(f, dtype=config.real(torch))
        return loss_fn(zeros, f)

    def train_next_batch(self, batch_size=None):
        return self.train_x, None

    def test(self):
        return self.test_x, None
