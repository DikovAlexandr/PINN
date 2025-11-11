# References: https://github.com/dmlc/dgl/tree/master/python/dgl/backend
# Modified to support only PyTorch backend

import importlib
import os
import sys

from . import backend

_enabled_apis = set()


def _gen_missing_api(api, mod_name):
    def _missing_api(*args, **kwargs):
        raise ImportError(
            'API "%s" is not supported by backend "%s".' % (api, mod_name)
        )

    return _missing_api


def load_backend(mod_name):
    if mod_name != "pytorch":
        raise NotImplementedError(
            "Only PyTorch backend is supported. Got: %s" % mod_name
        )

    print("Using backend: %s\n" % mod_name, file=sys.stderr, flush=True)
    mod = importlib.import_module(".pytorch", __name__)
    thismod = sys.modules[__name__]
    # log backend name
    setattr(thismod, "backend_name", "pytorch")
    for api in backend.__dict__.keys():
        if api.startswith("__"):
            # ignore python builtin attributes
            continue
        if api == "data_type_dict":
            # load data type
            if api not in mod.__dict__:
                raise ImportError(
                    'API "data_type_dict" is required but missing for backend "pytorch".'
                )
            data_type_dict = mod.__dict__[api]()
            for name, dtype in data_type_dict.items():
                setattr(thismod, name, dtype)

            # override data type dict function
            setattr(thismod, "data_type_dict", data_type_dict)
            setattr(
                thismod,
                "reverse_data_type_dict",
                {v: k for k, v in data_type_dict.items()},
            )
        else:
            # load functions
            if api in mod.__dict__:
                _enabled_apis.add(api)
                setattr(thismod, api, mod.__dict__[api])
            else:
                setattr(thismod, api, _gen_missing_api(api, "pytorch"))


def get_preferred_backend():
    """Always return PyTorch as the backend."""
    # Check environment variables for backward compatibility
    backend_name = None
    if "DDE_BACKEND" in os.environ:
        backend_name = os.getenv("DDE_BACKEND")
    elif "DDEBACKEND" in os.environ:
        backend_name = os.getenv("DDEBACKEND")
    
    # Only accept pytorch
    if backend_name and backend_name.lower() != "pytorch":
        print(
            f"Warning: Backend '{backend_name}' is not supported. Using PyTorch.",
            file=sys.stderr,
        )
    
    return "pytorch"


load_backend(get_preferred_backend())


def is_enabled(api):
    """Return true if the api is enabled by the current backend.

    Args:
        api (string): The api name.

    Returns:
        bool: ``True`` if the API is enabled by the current backend.
    """
    return api in _enabled_apis
