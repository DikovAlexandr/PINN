"""FBPINNs helper I/O utilities (vendored).

This module provides small filesystem helpers used by `constantsBase.py`.
It intentionally keeps behavior conservative to avoid accidentally deleting
the current working directory.
"""

from __future__ import annotations

import glob
import os
import shutil
from typing import Iterable


_DANGEROUS_DIR_NAMES = {"..", ".", "", "/", "./", "../", "*"}


def get_dir(directory: str) -> str:
    """Create `directory` if it does not exist, and return it."""
    os.makedirs(directory, exist_ok=True)
    return directory


def clear_dir(directory: str) -> None:
    """Remove all files/subdirectories inside `directory` (but keep the directory)."""
    # Important: if None passed to os.listdir, current directory is wiped.
    if not isinstance(directory, str):
        raise TypeError(f"directory must be str, got {type(directory)!r}")
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"{directory} is not a directory")
    if directory in _DANGEROUS_DIR_NAMES:
        raise ValueError(
            "Refusing to clear a dangerous directory name: " f"{directory!r}"
        )

    for entry in os.listdir(directory):
        path = os.path.join(directory, entry)
        try:
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        except Exception as exc:
            # Best-effort cleanup; keep legacy behavior of printing.
            print(exc)


def clear_files(glob_expression: str) -> None:
    """Remove all files matching `glob_expression`."""
    for path in glob.glob(glob_expression):
        if os.path.isfile(path):
            os.remove(path)


def remove_dir(directory: str) -> None:
    """Recursively remove `directory`."""
    if not isinstance(directory, str):
        raise TypeError(f"directory must be str, got {type(directory)!r}")
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"{directory} is not a directory")
    if directory in _DANGEROUS_DIR_NAMES:
        raise ValueError(
            "Refusing to remove a dangerous directory name: " f"{directory!r}"
        )

    shutil.rmtree(directory)
