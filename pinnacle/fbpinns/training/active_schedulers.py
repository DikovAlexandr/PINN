"""Active schedulers for FBPINN training.

This module defines iterable schedulers that control which FBPINN subdomains
are active / fixed / inactive at each training step. The schedulers are used
by :mod:`config.constants` when configuring FBPINN problems.
"""

from __future__ import annotations

import itertools

import numpy as np


class _ActiveScheduler:
    """Base class for scheduling updates to the active array."""

    name: str | None = None

    def __init__(self, N_STEPS: int, D) -> None:
        """Store domain geometry for an active scheduler.

        :param N_STEPS: Total number of training steps.
        :param D: ActiveRectangularDomainND-like object with attributes
            ``nd`` (number of dimensions), ``nm`` (number of models per
            dimension) and ``xx`` (grid coordinates).
        """
        self.N_STEPS = N_STEPS
        # ActiveRectangularDomainND.nd, NUMBER OF DIMENSIONS
        self.nd = D.nd
        # Number of middle segments / subdomains / models in each dimension
        self.nm = D.nm
        # Shape: (nd, nm+1)
        self.xx = D.xx.copy()

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.N_STEPS

    def __iter__(self):
        """Yield active maps over time (to be implemented by subclasses)."""
        # returns None if active map not to be changed, otherwise active map
        raise NotImplementedError


# ALL ACTIVE SCHEDULER


class AllActiveSchedulerND(_ActiveScheduler):
    """Scheduler where all models are active at all times."""

    name = "All"

    def __iter__(self):
        """Yield full-active map once, then ``None`` (no changes)."""
        for i in range(self.N_STEPS):
            if i == 0:
                # (nm)
                yield np.ones(self.nm, dtype=int)
            else:
                # Active domains do not change
                yield None


# POINT-BASED ACTIVE SCHEDULERS


class _SubspacePointActiveSchedulerND(_ActiveScheduler):
    """Slowly expand radially outwards from a point in a subspace of the domain.

    Radii are measured in x-units. This is the base class for point/line/plane
    schedulers.
    """

    def __init__(self, N_STEPS: int, D, point, iaxes) -> None:
        super().__init__(N_STEPS, D)

        # point in constrained axes
        point = np.array(point)
        # unconstrained axes
        iaxes = list(iaxes)

        # validation
        if point.ndim != 1:
            raise ValueError("point ndim != 1")
        if len(point) > self.nd:
            raise ValueError("len(point) > nd")
        if len(iaxes) + len(point) != self.nd:
            raise ValueError("len(iaxes) + len(point) != nd")

        self.point = point
        self.iaxes = iaxes

    def _get_radii(self, point: np.ndarray, xx: np.ndarray):
        """Get the radii from a point in a subspace of ``xx``.

        Returns the min and max distances from the point to the bounding
        corners of each subdomain cell.
        """
        # get subspace dimensions
        nd, nm = xx.shape[0], tuple(s - 1 for s in xx.shape[1:])
        assert len(nm) == nd
        # make sure they match with point
        assert len(point) == nd

        # get xmin, xmax of each model
        # self.xx (nd, nm+1)
        xmins = xx[(slice(None),) + (slice(None, -1),) * nd]  # (nd, nm)
        xmaxs = xx[(slice(None),) + (slice(1, None),) * nd]  # (nd, nm)

        # whether point is inside model
        point = point.copy().reshape((nd,) + (1,) * nd)  # (nd, (1,)*nd)
        # point is broadcast
        c_inside = (point >= xmins) & (point < xmaxs)
        c_inside = np.prod(c_inside, axis=0).astype(bool)  # (nm) must be true across all dims

        # get bounding corners of each model
        x = np.stack([xmins, xmaxs], axis=0)  # (2, nd, nm)
        bb = np.zeros((2**nd, nd) + nm)  # (2**nd, nd, nm)
        for ic, offsets in enumerate(itertools.product(*([[0, 1]] * nd))):
            # for each corner
            for i, o in enumerate(offsets):  # for each dimension
                bb[(ic, i) + (slice(None),) * nd] = x[(o, i) + (slice(None),) * nd]

        # distance from each corner to point
        point = point.copy().reshape((1, nd) + (1,) * nd)  # (1, nd, (1,)*nd)
        # (2**nd, nm) point is broadcast
        r = np.sqrt(np.sum((bb - point) ** 2, axis=1))
        rmin, rmax = np.min(r, axis=0), np.max(r, axis=0)  # (nm)

        # set rmin=0 where point is inside model
        rmin[c_inside] = 0.0

        return rmin, rmax

    def __iter__(self):
        """Yield active maps for a point-based growth strategy."""
        # slice constrained axes
        ic = [i for i in range(self.nd) if i not in self.iaxes]  # constrained axes
        sl = tuple([ic, *[slice(None) if i in ic else 0 for i in range(self.nd)]])
        xx = self.xx[sl]  # (nd-uc, nm-uc)

        # get subspace radii
        rmin, rmax = self._get_radii(self.point, xx)

        # insert unconstrained axes back in (for broadcasting below)
        # (nm with 1s)
        rmin, rmax = np.expand_dims(rmin, axis=self.iaxes), np.expand_dims(rmax, axis=self.iaxes)

        # initialise active array, start scheduling
        active = np.zeros(self.nm, dtype=int)  # (nm)
        r_min, r_max = np.min(rmin), np.max(rmax)
        # rmin, rmax has dim nm, while r_min, r_max are scalars
        # For a total of N_STEPS, not each radius N_STEPS
        for i in range(self.N_STEPS):
            # advance radius
            rt = r_min + (r_max - r_min) * (i / (self.N_STEPS))

            # get filters
            c_inactive = active == 0
            c_active = active == 1  # (nm) active filter
            # (nm) circle (surface) inside box (approximately!)
            # (only uses corners)
            c_radius = (rt >= rmin) & (rt < rmax)
            # c_radius is broadcast
            c_to_active = c_inactive & c_radius
            c_to_fixed = c_active & (~c_radius)

            # set values
            if c_to_active.any() or c_to_fixed.any():
                active[c_to_active] = 1
                active[c_to_fixed] = 2
                yield active
            else:
                yield None


class PointActiveSchedulerND(_SubspacePointActiveSchedulerND):
    """Slowly expands outwards from a point in the domain (in x units)."""

    name = "Point"

    def __init__(self, N_STEPS: int, D, point) -> None:
        if len(point) != D.nd:
            raise ValueError(f"point incorrect shape {point.shape!r}")
        super().__init__(N_STEPS, D, point, iaxes=[])


class LineActiveSchedulerND(_SubspacePointActiveSchedulerND):
    """Slowly expands outwards from a line in the domain (in x units)."""

    name = "Line"

    def __init__(self, N_STEPS: int, D, point, iaxis: int) -> None:
        if D.nd < 2:
            raise ValueError("requires nd >= 2")
        if len(point) != D.nd - 1:
            raise ValueError(f"point incorrect shape {point.shape!r}")

        super().__init__(N_STEPS, D, point, iaxes=[iaxis])


class PlaneActiveSchedulerND(_SubspacePointActiveSchedulerND):
    """Slowly expands outwards from a plane in the domain (in x units)."""

    name = "Plane"

    def __init__(self, N_STEPS: int, D, point, iaxes) -> None:
        if D.nd < 3:
            raise ValueError("requires nd >= 3")
        if len(point) != D.nd - 2:
            raise ValueError(f"point incorrect shape {point.shape!r}")

        super().__init__(N_STEPS, D, point, iaxes=iaxes)


class ManualActiveSchedulerND(_ActiveScheduler):
    """Scheduler where the active array sequence is hard-coded."""

    name = "Manual"

    def __init__(self, N_STEPS: int, D, actives) -> None:
        super().__init__(N_STEPS, D)
        self.actives = actives
        self.n_steps = N_STEPS / len(actives)

    def __iter__(self):
        """Yield active maps according to the pre-defined sequence."""
        current = 0
        for i in range(self.N_STEPS):
            if i * len(self.actives) >= current * self.N_STEPS:
                current += 1
                yield self.actives[current - 1]
            else:
                # Active domains do not change
                yield None
