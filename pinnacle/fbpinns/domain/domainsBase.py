"""
Module for base rectangular domain definitions.

This module defines the base class :class:`_RectangularDomainND` used by
:mod:`domains` to represent N-dimensional domains with overlapping rectangular
subdomains.
"""

from __future__ import annotations

from bisect import bisect_left
import itertools

import numpy as np


class _RectangularDomainND:
    """
    Base class for N-dimensional domains with hyperrectangular subdomains.

    FBPINN uses :class:`ActiveRectangularDomainND` inherited from this class.
    PINN does not use this class directly.

    :param subdomain_xs: List of rectangle edges along each dimension.
    :type subdomain_xs: list[np.ndarray]
    :param subdomain_ws: List of overlap widths along each dimension.
    :type subdomain_ws: list[np.ndarray]
    """

    def __init__(self, subdomain_xs, subdomain_ws):
        # Validation
        if len(subdomain_xs) != len(subdomain_ws):
            raise Exception(
                f"ERROR: lengths of subdomain_xs ({len(subdomain_xs)}) "
                f"and subdomain_ws ({len(subdomain_ws)}) do not match!"
            )

        for i in range(len(subdomain_xs)):
            if len(subdomain_xs[i]) != len(subdomain_ws[i]):
                raise Exception(
                    f"ERROR: length of subdomain_x does not equal "
                    f"length of subdomain_w at index {i}"
                )

        subdomain_xs = [np.array(x).copy() for x in subdomain_xs]
        subdomain_ws = [np.array(w).copy() for w in subdomain_ws]

        # Preparation
        # Get dimensions
        nd = len(subdomain_xs)  # NUMBER OF DIMENSIONS
        nm = tuple([len(x) - 1 for x in subdomain_xs])
        # Number of Middle Segments / Subdomains / Models IN EACH DIMENSION
        # For example, for a division like 田, nd=2, nm=(2,2),
        # for a division like 目, nd=2, nm=(1,3)

        # Make widths zero on boundaries
        for w in subdomain_ws:
            w[-1] = w[0] = 0
        # In other words, the "overlap width" at the left and right endpoint
        # is set to 0.

        # Expand out xs and ws
        xs = np.meshgrid(*subdomain_xs, indexing="ij")  # nd x [(nm,)+1]
        ws = np.meshgrid(*subdomain_ws, indexing="ij")  # nd x [(nm,)+1]

        # Expand out iis and make initial maps
        iis = np.meshgrid(*[range(n) for n in nm], indexing="ij")  # nd x [(nm,)]
        sm0 = ms0 = np.expand_dims(np.stack(iis, 0), 0)  # (1,nd,nm)

        # Convenience slicer class
        class NDSlicer:
            def __getitem__(self, s):
                return (s,) * nd
                # Expands to nd dimensions, returns (s,s,...,s) where s appears nd times

            def cut_edges(self, constrained_axes):
                return tuple(
                    [slice(None)] * 2
                    + [
                        slice(None, -1) if i in constrained_axes else slice(None)
                        for i in range(nd)
                    ]
                )
                # Cuts edges of constrained axes

        sl = NDSlicer()

        # Compute segments / maps
        # NOTE: models_segments will contain segment indices outside of grid
        # (in order to maintain tensor structure). These need to be discarded
        # in outside code.

        segments = []
        segments_models = []
        models_segments = []
        ca2idx = dict()

        # For increasing numbers of constrained axes
        for order in range(0, nd + 1):
            # For each set of constrained axis combinations
            for constrained_axes in itertools.combinations(range(nd), order):
                # Example: nd=3 => constrained_axes is
                # (), (0,), (1,), (2,), (0,1,), (0,2,), (1,2,), (0,1,2,)
                ca2idx[constrained_axes] = len(segments)

                # Segments
                s = np.zeros((2, nd) + nm)  # (2,nd,nm) (left or right, dim, grid)

                # For each dimension
                for i in range(nd):
                    if i in constrained_axes:  # if this dimension is constrained
                        # xn - wn/2, xn + wn/2 (illustration: ====, overlap)
                        s[0, i], s[1, i] = (
                            xs[i][sl[1:]] - ws[i][sl[1:]] / 2,
                            xs[i][sl[1:]] + ws[i][sl[1:]] / 2,
                        )
                    else:
                        # x{n-1} + w{n-1}/2, xn - wn/2 (illustration: ----, non-overlap)
                        s[0, i], s[1, i] = (
                            xs[i][sl[:-1]] + ws[i][sl[:-1]] / 2,
                            xs[i][sl[1:]] - ws[i][sl[1:]] / 2,
                        )
                s = s[sl.cut_edges(constrained_axes)]  # cut constrained edges
                segments.append(s)  # now s shape is (2,nd,nm') where nm' is nm after cut_edges

                # Maps
                n_overlap = 2 ** len(constrained_axes)  # number of overlapping models
                sm = np.concatenate([sm0.copy() for _ in range(n_overlap)])  # (ne,nd,nm)
                ms = np.concatenate([ms0.copy() for _ in range(n_overlap)])  # (ne,nd,nm)

                # For all overlapping elements
                for iel, offsets in enumerate(
                    itertools.product(*([[0, 1]] * len(constrained_axes)))
                ):
                    # For each constrained dimension
                    for ic, offset in enumerate(offsets):  # add their offsets to map
                        sm[iel, constrained_axes[ic]] += offset
                        ms[iel, constrained_axes[ic]] -= offset
                sm = sm[sl.cut_edges(constrained_axes)]  # cut constrained edges
                segments_models.append(sm)
                models_segments.append(ms)

        # Save self attributes
        self.nd = nd
        self.nm = nm
        self.segments = segments
        self.segments_models = segments_models
        self.models_segments = models_segments

        # Final validation
        for ioa, s in enumerate(self.segments):
            if np.any(s[0] > s[1]):
                raise Exception(f"ERROR: segments are negative! ({ioa})")

        # Helpers
        self.xx = np.stack(xs, 0)  # (nd,nm+1)
        self.ww = np.stack(ws, 0)  # (nd,nm+1)

        # Helper for determining which segment a point belongs to (onm)
        self.xmarks = list()
        for ithd in range(self.nd):
            x = subdomain_xs[ithd]
            w = subdomain_ws[ithd]
            # x0+w0/2, x1-w1/2, x1+w1/2, ..., xn-wn/2
            xmark = []
            for xx, ww in zip(x, w):
                xmark.append(xx - ww / 2)
                xmark.append(xx + ww / 2)
            self.xmarks.append(xmark[1:-1])
        self.ca2idx = ca2idx

    def get_onm(self, x):
        """
        Get the segment index for a given point.

        :param x: Point coordinates.
        :type x: np.ndarray
        :return: Segment index tuple (order, dim_indices...).
        :rtype: tuple
        """
        constrained_axes = tuple()
        dim_index = [None for _ in range(self.nd)]
        for ithd in range(self.nd):
            pos = bisect_left(self.xmarks[ithd], x[ithd]) - 1
            pos = min(max(pos, 0), len(self.xmarks[ithd]) - 2)
            # x[ithd] is in the pos-th interval of xmark[ithd]
            if pos % 2 == 0:  # dim is not constrained
                dim_index[ithd] = pos // 2
            else:  # dim is constrained
                constrained_axes = constrained_axes + (ithd,)
                dim_index[ithd] = pos // 2
        return (self.ca2idx[constrained_axes],) + tuple(dim_index)

    def __str__(self):
        st = "nd:" + str(self.nd) + "\n"
        st += "nm:" + str(self.nm) + "\n"

        st += "segments:" + "\n"
        for s in self.segments:
            st += str(s.shape) + " " + str(s) + "\n"

        st += "segments_models:" + "\n"
        for sm in self.segments_models:
            st += str(sm.shape) + " " + str(sm) + "\n"

        st += "models_segments:" + "\n"
        for ms in self.models_segments:
            st += str(ms.shape) + " " + str(ms) + "\n"

        return st[:-1]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    colors = (
        [
            "tab:blue",
            "tab:orange",
            "tab:green",
            "tab:red",
            "tab:purple",
            "tab:brown",
            "tab:pink",
            "tab:gray",
            "tab:olive",
            "tab:cyan",
        ]
        * 100
    )
    itergrid = lambda shape: enumerate(
        itertools.product(*[np.arange(d) for d in shape])
    )

    # 1D test
    for subdomain_xs, subdomain_ws in [
        [[np.array([-5, 0, 3, 6, 10])], [0.2 * np.array([1, 2, 3, 4, 5])]],
        [[np.array([-5, 0])], [0.2 * np.array([1, 2])]],
    ]:
        D = _RectangularDomainND(subdomain_xs, subdomain_ws)
        print(D)

        ss, xx = D.segments, D.xx
        plt.figure(figsize=(8 * len(ss), 6))
        for iplot, (s, sm, ms) in enumerate(
            zip(D.segments, D.segments_models, D.models_segments)
        ):
            # Create a sub-plot for each combination of constrained_axes
            ax = plt.subplot(1, len(ss), iplot + 1)

            # Plot subdomains (same for every subplot)
            for im, (i,) in itergrid(D.nm):  # for each subdomain
                plt.plot(
                    [subdomain_xs[0][i], subdomain_xs[0][i + 1]], [0, 0], color=colors[im]
                )

            # Plot segments (for this combination of axes)
            for iseg, (i,) in itergrid(s.shape[2:]):  # for each segment
                plt.plot(
                    [s[0, 0, i], s[1, 0, i]], [0.1, 0.1], color=colors[iseg]
                )  # shape of s: (left or right, which dim, grid)

            # Plot maps (for this combination of axes)
            for iel in range(sm.shape[0]):  # iel: ith element
                for iseg, (i,) in itergrid(sm.shape[2:]):
                    plt.scatter(
                        xx[0, i] + sm[iel, 0, i],
                        [0.2],
                        c=colors[iseg],
                        s=80,
                    )

            for iel in range(ms.shape[0]):
                for iseg, (i,) in itergrid(ms.shape[2:]):
                    plt.scatter(
                        xx[0, i] + ms[iel, 0, i],
                        [0.3],
                        c=colors[iseg],
                        s=40,
                        linewidths=1,
                        edgecolor="k",
                    )

            plt.autoscale()
        plt.show()

    # 2D test
    for subdomain_xs, subdomain_ws in [
        [
            [np.array([-5, 0, 3, 6, 10]), np.array([5, 15, 35])],
            [0.2 * np.array([1, 2, 3, 4, 5]), 0.2 * np.array([5, 6, 7])],
        ],
        [
            [np.array([-5, 0]), np.array([5, 15, 35])],
            [0.2 * np.array([1, 2]), 0.2 * np.array([5, 6, 7])],
        ],
    ]:
        D = _RectangularDomainND(subdomain_xs, subdomain_ws)
        print(D)

        ss, xx = D.segments, D.xx
        plt.figure(figsize=(4 * len(ss), 10))
        for iplot, (s, sm, ms) in enumerate(
            zip(D.segments, D.segments_models, D.models_segments)
        ):
            ax = plt.subplot(1, len(ss), iplot + 1)

            # Plot subdomains
            for im, (i, j) in itergrid(D.nm):  # for each subdomain
                rect = patches.Rectangle(
                    (subdomain_xs[0][i], subdomain_xs[1][j]),  # xy
                    subdomain_xs[0][i + 1] - subdomain_xs[0][i],  # width
                    subdomain_xs[1][j + 1] - subdomain_xs[1][j],  # height
                    linewidth=1,
                    edgecolor=colors[im],
                    facecolor="none",
                )
                ax.add_patch(rect)

            # Plot segments
            for iseg, (i, j) in itergrid(s.shape[2:]):  # for each segment
                rect = patches.Rectangle(
                    (s[0, 0, i, j], s[0, 1, i, j]),  # xy (x,y)
                    s[1, 0, i, j] - s[0, 0, i, j],  # width (along x)
                    s[1, 1, i, j] - s[0, 1, i, j],  # height (along y)
                    linewidth=2,
                    edgecolor=colors[iseg],
                    facecolor="none",
                )
                ax.add_patch(rect)

            # Plot maps
            for iel in range(sm.shape[0]):
                for iseg, (i, j) in itergrid(sm.shape[2:]):
                    plt.scatter(
                        xx[0, i, j] + sm[iel, 0, i, j],
                        xx[1, i, j] + sm[iel, 1, i, j],
                        c=colors[iseg],
                        s=80,
                    )

            for iel in range(ms.shape[0]):
                for iseg, (i, j) in itergrid(ms.shape[2:]):
                    plt.scatter(
                        xx[0, i, j] + ms[iel, 0, i, j],
                        xx[1, i, j] + ms[iel, 1, i, j],
                        c=colors[iseg],
                        s=40,
                        linewidths=1,
                        edgecolor="k",
                    )

            ax.set_aspect("equal")
            plt.autoscale()
        plt.show()

    # 3D test
    for subdomain_xs, subdomain_ws in [
        [
            [
                np.array([-5, 0, 3, 6, 10]),
                np.array([5, 15, 35, 45]),
                np.array([-2, 5, 12]),
            ],
            [
                0.2 * np.array([1, 2, 3, 4, 5]),
                0.2 * np.array([2, 3, 4, 5]),
                0.2 * np.array([4, 5, 6]),
            ],
        ],
        [
            [np.array([-5, 0]), np.array([5, 15, 35, 45]), np.array([-2, 5])],
            [0.2 * np.array([1, 2]), 0.2 * np.array([2, 3, 4, 5]), 0.2 * np.array([4, 5])],
        ],
    ]:
        D = _RectangularDomainND(subdomain_xs, subdomain_ws)
        print(D)

        ss, xx = D.segments, D.xx
        plt.figure(figsize=(4 * len(ss), 10))
        for iplot, (s, sm, ms) in enumerate(
            zip(D.segments, D.segments_models, D.models_segments)
        ):
            ax = plt.subplot(1, len(ss), iplot + 1)

            # Plot subdomains
            for im, (i, j, k) in itergrid(D.nm):  # for each subdomain
                rect = patches.Rectangle(
                    (subdomain_xs[0][i], subdomain_xs[1][j]),  # xy
                    subdomain_xs[0][i + 1] - subdomain_xs[0][i],  # width
                    subdomain_xs[1][j + 1] - subdomain_xs[1][j],  # height
                    linewidth=1,
                    edgecolor=colors[im],
                    facecolor="none",
                )
                ax.add_patch(rect)

            # Plot segments
            for iseg, (i, j, k) in itergrid(s.shape[2:]):  # for each segment
                rect = patches.Rectangle(
                    (s[0, 0, i, j, k], s[0, 1, i, j, k]),  # xy (x,y)
                    s[1, 0, i, j, k] - s[0, 0, i, j, k],  # width (along x)
                    s[1, 1, i, j, k] - s[0, 1, i, j, k],  # height (along y)
                    linewidth=2,
                    edgecolor=colors[iseg],
                    facecolor="none",
                )
                ax.add_patch(rect)

            # Plot maps
            for iel in range(sm.shape[0]):
                for iseg, (i, j, k) in itergrid(sm.shape[2:]):
                    plt.scatter(
                        xx[0, i, j, k] + sm[iel, 0, i, j, k],
                        xx[1, i, j, k] + sm[iel, 1, i, j, k],
                        c=colors[iseg],
                        s=80,
                    )

            for iel in range(ms.shape[0]):
                for iseg, (i, j, k) in itergrid(ms.shape[2:]):
                    plt.scatter(
                        xx[0, i, j, k] + ms[iel, 0, i, j, k],
                        xx[1, i, j, k] + ms[iel, 1, i, j, k],
                        c=colors[iseg],
                        s=40,
                        linewidths=1,
                        edgecolor="k",
                    )

            ax.set_aspect("equal")
            plt.autoscale()
        plt.show()
