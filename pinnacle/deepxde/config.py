import os
import random

import numpy as np

from . import backend as bkd
from .backend import backend_name, torch
from .real import Real

# Default float type
real = Real(32)
# Random seed
random_seed = None
# XLA not supported in PyTorch backend
xla_jit = False

# TODO: Double check is Pytorch actually not support XLA and implement if is no

def default_float():
    """Returns the default float type, as a string."""
    if real.precision == 64:
        return "float64"
    return "float32"


def set_default_float(value):
    """Sets the default float type (PyTorch backend only).

    The default floating point type is 'float32'.

    Args:
        value (String): 'float16', 'float32', or 'float64'.
    """
    assert backend_name == "pytorch", f"Only PyTorch backend is supported, got: {backend_name}"
    
    if value == "float16":
        print("Set the default float type to float16")
        real.set_float16()
    elif value == "float32":
        print("Set the default float type to float32")
        real.set_float32()
    elif value == "float64":
        print("Set the default float type to float64")
        real.set_float64()
    else:
        raise ValueError(f"{value} not supported in deepXDE")
    
    torch.set_default_dtype(real(torch))


def set_random_seed(seed):
    """Sets all random seeds for the program (Python random, NumPy, and PyTorch backend).

    You can use this to make the program fully deterministic. This means that if the
    program is run multiple times with the same inputs on the same hardware, it will
    have the exact same outputs each time. This is useful for debugging models, and for
    obtaining fully reproducible results.

    Warning:
        Note that determinism in general comes at the expense of lower performance and
        so your model may run slower when determinism is enabled.

    Args:
        seed (int): The desired seed.
    """
    assert backend_name == "pytorch", f"Only PyTorch backend is supported, got: {backend_name}"
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    global random_seed
    random_seed = seed


def enable_xla_jit(mode=True):
    """XLA is not supported in PyTorch backend.

    Args:
        mode (bool): Whether to enable just-in-time compilation with XLA (``True``) or
            disable just-in-time compilation with XLA (``False``).
    
    Raises:
        ValueError: PyTorch backend does not support XLA.
    """
    assert backend_name == "pytorch", f"Only PyTorch backend is supported, got: {backend_name}"
    
    if mode:
        raise ValueError("Backend PyTorch does not support XLA.")
    
    global xla_jit
    xla_jit = False
    print("XLA is not supported in PyTorch backend.\n")


def disable_xla_jit():
    """Disables just-in-time compilation with XLA.
    
    PyTorch backend does not support XLA, so this is a no-op.

    This is equivalent with ``enable_xla_jit(False)``.
    """
    enable_xla_jit(False)
