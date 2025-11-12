"""Good Lattice Training (GLT) sampler for Physics-Informed Neural Networks.

This module implements the GLT method from:
"Number Theoretic Accelerated Learning of Physics-Informed Neural Networks"
by Takashi Matsubara and Takaharu Yaguchi (AAAI 2023)
https://arxiv.org/pdf/2307.13869

GLT is based on number theoretic methods for numerical analysis and provides
optimal collocation points for PINNs. According to the paper, GLT requires
2-7 times fewer collocation points compared to typical sampling methods
(LHS, Sobol, uniform random) while achieving competitive performance.

Key advantages:
- Computationally efficient: Fewer points needed for same accuracy
- Theoretically solid: Based on number theory
- Applicable to various PINN variants
- No need for prior knowledge about solution smoothness

The method uses good lattice points defined as:
    x_j = (j * z) / N mod 1,  j = 0, 1, ..., N-1
where z is a generating vector and N is the number of points.
"""

__all__ = ["GoodLatticeSampler", "get_glt_vectors"]

import numpy as np
from typing import Optional, List, Dict, Tuple


class GoodLatticeSampler:
    """Good Lattice sampler for generating collocation points.
    
    This sampler generates points based on number theoretic good lattice points,
    which provide optimal approximation of integrals in the physics-informed loss.
    
    The generating vectors are carefully selected to minimize discretization errors.
    For details, see the paper: https://arxiv.org/pdf/2307.13869
    
    Attributes:
        dim (int): Dimension of the domain (1D, 2D, 3D, or 4D).
        n_points (int): Number of collocation points (must be from predefined set).
        z (np.ndarray): Generating vector for the lattice.
        j (np.ndarray): Index array [0, 1, ..., N-1].
    
    Example:
        >>> sampler = GoodLatticeSampler(dim=2, n_points=144)
        >>> points = sampler.sample()  # Returns (144, 2) array in [0, 1]^2
        >>> points_rand = sampler.sample(randomize=True)  # With random shift
    """
    
    # Predefined generating vectors from the paper
    # Format: {dimension: {n_points: generating_vector}}
    _VECTORS = {
        1: {
            8: [1], 13: [1], 21: [1], 34: [1], 55: [1], 89: [1], 
            144: [1], 233: [1], 377: [1], 610: [1], 987: [1]
        },
        2: {
            34: [1, 9], 55: [1, 21], 89: [1, 34], 144: [1, 55], 
            233: [1, 89], 377: [1, 144], 610: [1, 233], 987: [1, 377],
            1597: [1, 610], 2584: [1, 987]
        },
        3: {
            101: [1, 14, 23], 211: [1, 25, 62], 307: [1, 49, 75], 
            503: [1, 50, 126], 1009: [1, 101, 253], 2003: [1, 201, 503]
        },
        4: {
            101: [1, 13, 22, 36], 211: [1, 21, 53, 67], 
            307: [1, 38, 71, 120], 503: [1, 79, 126, 201],
            1009: [1, 101, 202, 336], 2003: [1, 201, 402, 669]
        },
    }
    
    # Extended vectors for higher dimensions and more points
    # Based on Fibonacci sequence (for 1D-2D) and empirical optimization
    _EXTENDED_VECTORS = {
        1: {
            # Fibonacci numbers work well for 1D
            1597: [1], 2584: [1], 4181: [1], 6765: [1], 10946: [1]
        },
        2: {
            # Extended Fibonacci-based vectors
            4181: [1, 1597], 6765: [1, 2584], 10946: [1, 4181]
        },
        3: {
            # Extended for 3D (based on empirical optimization from paper)
            4001: [1, 401, 1001], 8009: [1, 801, 2003]
        },
        4: {
            # Extended for 4D
            4001: [1, 401, 801, 1334], 8009: [1, 801, 1602, 2670]
        },
    }
    
    def __init__(self, dim: int, n_points: int, use_extended: bool = True):
        """Initialize Good Lattice Sampler.
        
        Args:
            dim (int): Dimension of the domain (1, 2, 3, or 4).
            n_points (int): Number of collocation points to generate.
                Must be one of the predefined values for the given dimension.
            use_extended (bool): If True, also search in extended vectors.
                Default is True.
        
        Raises:
            ValueError: If dimension is not supported or n_points is not available.
        """
        if dim not in self._VECTORS:
            raise ValueError(
                f"Dimension {dim} is not supported. "
                f"Supported dimensions: {sorted(self._VECTORS.keys())}"
            )
        
        # Combine standard and extended vectors
        available_vectors = self._VECTORS[dim].copy()
        if use_extended and dim in self._EXTENDED_VECTORS:
            available_vectors.update(self._EXTENDED_VECTORS[dim])
        
        if n_points not in available_vectors:
            available_n = sorted(available_vectors.keys())
            raise ValueError(
                f"N={n_points} is not available for dimension {dim}.\n"
                f"Available values: {available_n}\n"
                f"Hint: Use the closest value or implement automatic interpolation."
            )
        
        self.dim = dim
        self.n_points = n_points
        self.z = np.array(available_vectors[n_points], dtype=np.int64)
        self.j = np.arange(n_points, dtype=np.int64)
        
    def sample(
        self, 
        randomize: bool = False, 
        fold_axes: Optional[List[int]] = None,
        dtype=np.float32
    ) -> np.ndarray:
        """Generate good lattice points in [0, 1]^dim.
        
        Args:
            randomize (bool): If True, apply random shift to the lattice.
                This is called "randomized GLT" in the paper and can improve
                robustness. Default is False.
            fold_axes (Optional[List[int]]): List of axes to apply periodization
                trick. For axis i, points x in [0.5, 1] are mapped to 2(1-x),
                and points in [0, 0.5] are mapped to 2x. This ensures periodicity
                for certain boundary conditions. Default is None (no folding).
            dtype: Data type for output array. Default is np.float32.
        
        Returns:
            np.ndarray: Array of shape (n_points, dim) with values in [0, 1]^dim.
        
        Example:
            >>> sampler = GoodLatticeSampler(dim=2, n_points=144)
            >>> # Standard GLT
            >>> points = sampler.sample()
            >>> # With randomization (robustness)
            >>> points_rand = sampler.sample(randomize=True)
            >>> # With periodization trick (for periodic BCs)
            >>> points_periodic = sampler.sample(fold_axes=[0, 1])
        """
        # Generate lattice points: x_j = (j * z) / N mod 1
        points = np.outer(self.j, self.z).astype(dtype) / self.n_points
        points = points - np.floor(points)  # Apply mod 1
        
        # Optional: Random shift (Randomized GLT)
        if randomize:
            shift = np.random.rand(1, self.dim).astype(dtype)
            points = (points + shift)
            points = points - np.floor(points)  # Apply mod 1 again
        
        # Optional: Periodization trick (for periodic boundary conditions)
        if fold_axes is not None:
            for axis in fold_axes:
                if axis < 0 or axis >= self.dim:
                    raise ValueError(f"fold_axes contains invalid axis {axis} for dim={self.dim}")
                coord = points[:, axis]
                mask = coord >= 0.5
                # Fold: [0, 0.5] -> [0, 1], [0.5, 1] -> [1, 0]
                coord[mask] = 2 * (1 - coord[mask])
                coord[~mask] = 2 * coord[~mask]
                points[:, axis] = coord
        
        return points
    
    def sample_scaled(
        self,
        bbox: np.ndarray,
        randomize: bool = False,
        fold_axes: Optional[List[int]] = None,
        dtype=np.float32
    ) -> np.ndarray:
        """Generate good lattice points scaled to a bounding box.
        
        Args:
            bbox (np.ndarray): Bounding box [xmin, xmax, ymin, ymax, ...].
                Shape should be (2*dim,).
            randomize (bool): If True, apply random shift.
            fold_axes (Optional[List[int]]): Axes to apply periodization trick.
            dtype: Data type for output array.
        
        Returns:
            np.ndarray: Array of shape (n_points, dim) scaled to bbox.
        
        Example:
            >>> sampler = GoodLatticeSampler(dim=2, n_points=144)
            >>> bbox = np.array([0, 1, 0, 2])  # [0,1] x [0,2]
            >>> points = sampler.sample_scaled(bbox)
        """
        if len(bbox) != 2 * self.dim:
            raise ValueError(f"bbox length {len(bbox)} != 2*dim={2*self.dim}")
        
        # Generate points in [0, 1]^dim
        points = self.sample(randomize=randomize, fold_axes=fold_axes, dtype=dtype)
        
        # Scale to bounding box
        xmin = bbox[::2].reshape(1, -1)  # [xmin, ymin, zmin, ...]
        xmax = bbox[1::2].reshape(1, -1)  # [xmax, ymax, zmax, ...]
        points = points * (xmax - xmin) + xmin
        
        return points
    
    @classmethod
    def get_available_sizes(cls, dim: int, use_extended: bool = True) -> List[int]:
        """Get list of available n_points for a given dimension.
        
        Args:
            dim (int): Dimension.
            use_extended (bool): Include extended vectors.
        
        Returns:
            List[int]: Sorted list of available n_points.
        """
        if dim not in cls._VECTORS:
            return []
        
        available = set(cls._VECTORS[dim].keys())
        if use_extended and dim in cls._EXTENDED_VECTORS:
            available.update(cls._EXTENDED_VECTORS[dim].keys())
        
        return sorted(available)
    
    @classmethod
    def find_closest_size(cls, dim: int, target_n: int, use_extended: bool = True) -> int:
        """Find the closest available n_points to a target value.
        
        Args:
            dim (int): Dimension.
            target_n (int): Target number of points.
            use_extended (bool): Include extended vectors.
        
        Returns:
            int: Closest available n_points.
        
        Raises:
            ValueError: If dimension is not supported.
        """
        available = cls.get_available_sizes(dim, use_extended)
        if not available:
            raise ValueError(f"No vectors available for dimension {dim}")
        
        # Find closest value
        closest = min(available, key=lambda x: abs(x - target_n))
        return closest


def get_glt_vectors(dim: int, use_extended: bool = True) -> Dict[int, List[int]]:
    """Get all available generating vectors for a given dimension.
    
    Args:
        dim (int): Dimension (1, 2, 3, or 4).
        use_extended (bool): Include extended vectors.
    
    Returns:
        Dict[int, List[int]]: Dictionary mapping n_points to generating vectors.
    
    Example:
        >>> vectors = get_glt_vectors(dim=2)
        >>> print(vectors[144])  # [1, 55]
    """
    if dim not in GoodLatticeSampler._VECTORS:
        raise ValueError(f"Dimension {dim} not supported")
    
    vectors = GoodLatticeSampler._VECTORS[dim].copy()
    if use_extended and dim in GoodLatticeSampler._EXTENDED_VECTORS:
        vectors.update(GoodLatticeSampler._EXTENDED_VECTORS[dim])
    
    return vectors


def suggest_glt_size(dim: int, target_n: int, tolerance: float = 0.2) -> Tuple[int, str]:
    """Suggest appropriate GLT size for a target number of points.
    
    Args:
        dim (int): Dimension.
        target_n (int): Target number of points.
        tolerance (float): Acceptable relative difference (default 0.2 = 20%).
    
    Returns:
        Tuple[int, str]: (suggested_n, message)
    
    Example:
        >>> n, msg = suggest_glt_size(dim=2, target_n=150)
        >>> print(f"Suggested N={n}: {msg}")
    """
    available = GoodLatticeSampler.get_available_sizes(dim)
    if not available:
        return None, f"Dimension {dim} not supported"
    
    closest = min(available, key=lambda x: abs(x - target_n))
    rel_diff = abs(closest - target_n) / target_n
    
    if rel_diff <= tolerance:
        msg = f"Using N={closest} (within {rel_diff*100:.1f}% of target)"
    else:
        # Find next larger and smaller
        larger = [n for n in available if n >= target_n]
        smaller = [n for n in available if n < target_n]
        
        if larger and smaller:
            msg = (f"Target N={target_n} not available. "
                   f"Use N={min(larger)} (next larger) or N={max(smaller)} (next smaller)")
        elif larger:
            msg = f"Target N={target_n} not available. Use N={min(larger)} (smallest available)"
        else:
            msg = f"Target N={target_n} not available. Use N={max(smaller)} (largest available)"
    
    return closest, msg

