"""
Domain definitions for FBPINNs.

This package contains modules for defining and managing N-dimensional domains
with overlapping rectangular subdomains, including base classes, window functions,
and active domain management.
"""

from domain.domainsBase import _RectangularDomainND
from domain.domains import ActiveRectangularDomainND
from domain.windows import construct_window_function_ND

__all__ = [
    "_RectangularDomainND",
    "ActiveRectangularDomainND",
    "construct_window_function_ND",
]
