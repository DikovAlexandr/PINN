"""Training utilities package for FBPINN / PINN experiments.

This package exposes common training-time components such as loss functions,
trainer base classes, and active schedulers.
"""

from training.losses import (
    l1_loss,
    l1_rel_err,
    l2_loss,
    l2_rel_err,
    err_csv,
    max_err,
)
from training.trainersBase import _Trainer
from training.active_schedulers import (
    _ActiveScheduler,
    AllActiveSchedulerND,
    PointActiveSchedulerND,
    LineActiveSchedulerND,
    PlaneActiveSchedulerND,
    ManualActiveSchedulerND,
)

__all__ = [
    "l1_loss",
    "l1_rel_err",
    "l2_loss",
    "l2_rel_err",
    "err_csv",
    "max_err",
    "_Trainer",
    "_ActiveScheduler",
    "AllActiveSchedulerND",
    "PointActiveSchedulerND",
    "LineActiveSchedulerND",
    "PlaneActiveSchedulerND",
    "ManualActiveSchedulerND",
]
