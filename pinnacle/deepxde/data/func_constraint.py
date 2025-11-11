import numpy as np

from .data import Data
from .. import config
from ..backend import torch
from ..utils import run_if_any_none


class FuncConstraint(Data):
    """Function approximation with constraints."""

    def __init__(
        self, geom, constraint, func, num_train, anchors, num_test, dist_train="uniform"
    ):
        self.geom = geom
        self.constraint = constraint
        self.func = func
        self.num_train = num_train
        self.anchors = anchors
        self.num_test = num_test
        self.dist_train = dist_train

        self.train_x, self.train_y = None, None
        self.test_x, self.test_y = None, None

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        self.train_next_batch()
        self.test()

        n = 0
        if self.anchors is not None:
            n += len(self.anchors)

        # PyTorch uses model.training attribute to check training mode
        if model.net.training:
            f = self.constraint(inputs, outputs, self.train_x)
        else:
            f = self.constraint(inputs, outputs, self.test_x)
        
        # Create zeros tensor with same shape and dtype as f
        zeros = torch.zeros_like(f, dtype=config.real(torch))
        return [
            loss_fn(targets[:n], outputs[:n]),
            loss_fn(zeros, f),
        ]

    @run_if_any_none("train_x", "train_y")
    def train_next_batch(self, batch_size=None):
        if self.dist_train == "uniform":
            self.train_x = self.geom.uniform_points(self.num_train, False)
        elif self.dist_train == "log uniform":
            self.train_x = self.geom.log_uniform_points(self.num_train, False)
        else:
            self.train_x = self.geom.random_points(
                self.num_train, random=self.dist_train
            )
        if self.anchors is not None:
            self.train_x = np.vstack((self.anchors, self.train_x))
        self.train_y = self.func(self.train_x)
        return self.train_x, self.train_y

    @run_if_any_none("test_x", "test_y")
    def test(self):
        self.test_x = self.geom.uniform_points(self.num_test, True)
        self.test_y = self.func(self.test_x)
        return self.test_x, self.test_y
