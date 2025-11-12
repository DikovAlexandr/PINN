import importlib
import sys

from .config import LBFGS_options, set_LBFGS_options, Muon_options, set_Muon_options
from ..backend import backend_name


def _load_backend(mod_name):
    """Load PyTorch backend (only supported backend)."""
    assert mod_name == "pytorch", f"Only PyTorch backend is supported, got: {mod_name}"
    mod = importlib.import_module(".pytorch", __name__)
    thismod = sys.modules[__name__]
    for api, obj in mod.__dict__.items():
        setattr(thismod, api, obj)


_load_backend(backend_name)
