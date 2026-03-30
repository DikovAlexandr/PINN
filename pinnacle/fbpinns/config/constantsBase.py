"""FBPINNs constants base (vendored).

This module provides a lightweight base class for configuration objects
(`ConstantsBase`) and a helper for pretty-printing multiple config dicts.
It also contains small filesystem helpers for setting up output directories
and saving run configuration.
"""

from __future__ import annotations

import pickle
import sys
from typing import Dict, Iterable, List, Mapping

sys.path.insert(0, "./shared_modules/")
import io_utils


class ConstantsBase:
    """Base class for FBPINN and PINN constants.

    Instances of this class behave like simple attribute containers with
    limited item access (``__getitem__``/``__setitem__``) so that constants
    can be updated via dictionary-style syntax while still validating keys.
    """

    def __getitem__(self, key: str):
        """Return the value associated with *key*.

        :param str key: Name of the attribute to retrieve.
        :raises KeyError: If *key* is not present in the instance ``__dict__``.
        """
        if key not in self.__dict__:
            raise KeyError(f'key "{key}" not in self.__dict__')
        return self.__dict__[key]

    def __setitem__(self, key: str, item):
        """Set the value associated with *key*.

        Only existing keys are allowed to be updated. This helps prevent
        accidental creation of misspelled configuration fields.

        :param str key: Name of the attribute to update.
        :param item: New value for the attribute.
        :raises KeyError: If *key* is not present in the instance ``__dict__``.
        """
        if key not in self.__dict__:
            raise KeyError(f'key "{key}" not in self.__dict__')
        self.__dict__[key] = item

    def __str__(self) -> str:
        """Return a human-readable representation of all constants."""
        lines = [f"{k}: {self[k]}" for k in vars(self)]
        return "\n".join(lines)

    # The methods below assume that RUN, SUMMARY_OUT_DIR and MODEL_OUT_DIR
    # attributes exist on the subclass.

    def get_outdirs(self) -> None:
        """Create/clean the summary and model output directories.

        ``SUMMARY_OUT_DIR`` is typically of the form
        ``\"results/summaries/<RUN>/\"`` and ``MODEL_OUT_DIR`` of the form
        ``\"results/models/<RUN>/\"``. The helper ensures the directories
        exist and are cleared before a new run starts.
        """
        io_utils.get_dir(self.SUMMARY_OUT_DIR)
        io_utils.clear_dir(self.SUMMARY_OUT_DIR)
        io_utils.get_dir(self.MODEL_OUT_DIR)
        io_utils.clear_dir(self.MODEL_OUT_DIR)

    def save_constants_file(self) -> None:
        """Persist the current constants to text and pickle files.

        Text and pickle files are written under ``SUMMARY_OUT_DIR`` using
        the pattern ``constants_<RUN>.txt`` / ``constants_<RUN>.pickle``.
        Note that pickling saves functions/classes/modules by name, so the
        unpickling environment must have access to the corresponding code.
        """
        txt_path = f"{self.SUMMARY_OUT_DIR}constants_{self.RUN}.txt"
        pkl_path = f"{self.SUMMARY_OUT_DIR}constants_{self.RUN}.pickle"

        with open(txt_path, "w") as f_txt:
            for k, v in self.__dict__.items():
                f_txt.write(f"{k}: {v}\n")

        with open(pkl_path, "wb") as f_pkl:
            pickle.dump(self.__dict__, f_pkl)


def print_c_dicts(c_dicts: Iterable[Mapping[str, object]]) -> None:
    """Pretty-print a list of configuration dictionaries.

    The function aligns values for each key across all provided dictionaries
    so that differences between configurations are easy to compare.

    :param c_dicts: Iterable of dictionaries containing configuration values.
    """
    # Collect the union of all keys, preserving first-seen order.
    keys: List[str] = []
    for c_dict in reversed(list(c_dicts)):
        for k in c_dict.keys():
            if k not in keys:
                keys.append(k)

    for k in keys:
        row: List[str] = []
        for c_dict in c_dicts:
            item = c_dict.get(k, "None")
            row.append(str(item))
        print(f"{k}: " + " | ".join(row))
