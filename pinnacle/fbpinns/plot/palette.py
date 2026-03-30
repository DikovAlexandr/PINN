"""Centralised colormap choices for the vendored FBPINNs plotting code.

Goal: keep plots publication-friendly and consistent across 2D/3D "field"
figures. In particular, we default to a turquoise-forward sequential palette
for scalar fields, and a diverging turquoise↔orange palette for differences.

You can override the palettes without editing code via environment variables:
- ``FBPINNS_FIELD_CMAP``: any matplotlib colormap name (e.g. "GnBu", "viridis").
- ``FBPINNS_DIFF_CMAP``: any matplotlib colormap name (e.g. "RdBu_r", "seismic").
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from matplotlib.colors import LinearSegmentedColormap


def _maybe_override(env_key: str) -> str | None:
    val = os.environ.get(env_key, "").strip()
    return val or None


@lru_cache(maxsize=1)
def field_cmap() -> Any:
    """Colormap used for scalar solution/reference fields.

    Returns either a matplotlib colormap object or a cmap name (string).
    """
    override = _maybe_override("FBPINNS_FIELD_CMAP")
    if override is not None:
        return override

    # A custom sequential palette with a strong turquoise tone.
    return LinearSegmentedColormap.from_list(
        "fbpinns_turquoise",
        [
            "#0b132b",  # deep navy
            "#1c2541",  # indigo
            "#3a506b",  # blue-grey
            "#2a9d8f",  # teal
            "#5bc0be",  # turquoise
            "#e0fbfc",  # very light cyan
        ],
        N=256,
    )


@lru_cache(maxsize=1)
def diff_cmap() -> Any:
    """Colormap used for (prediction - reference) differences."""
    override = _maybe_override("FBPINNS_DIFF_CMAP")
    if override is not None:
        return override

    # Diverging palette, turquoise for negative values and warm orange for positive.
    return LinearSegmentedColormap.from_list(
        "fbpinns_turquoise_orange",
        [
            "#006d77",  # deep teal (negative)
            "#ffffff",  # zero
            "#e76f51",  # warm orange (positive)
        ],
        N=256,
    )
