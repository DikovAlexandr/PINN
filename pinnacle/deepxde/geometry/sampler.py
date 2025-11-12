__all__ = ["sample", "sample_glt"]

import numpy as np
import skopt

from .. import config
from .glt import GoodLatticeSampler, suggest_glt_size


def sample(n_samples, dimension, sampler="pseudo"):
    """Generate pseudorandom or quasirandom samples in [0, 1]^dimension.

    Args:
        n_samples (int): The number of samples.
        dimension (int): Space dimension.
        sampler (string): One of the following: "pseudo" (pseudorandom), "LHS" (Latin
            hypercube sampling), "Halton" (Halton sequence), "Hammersley" (Hammersley
            sequence), "Sobol" (Sobol sequence), "GLT" (Good Lattice Training), or
            "GLT-rand" (randomized GLT).
    
    Note:
        For GLT, if n_samples is not exactly available, the closest available
        value will be used and a warning will be printed.
    """
    if sampler == "pseudo":
        return pseudorandom(n_samples, dimension)
    if sampler in ["LHS", "Halton", "Hammersley", "Sobol"]:
        return quasirandom(n_samples, dimension, sampler)
    if sampler in ["GLT", "GLT-rand", "glt", "glt-rand"]:
        randomize = sampler.lower().endswith("-rand")
        return sample_glt(n_samples, dimension, randomize=randomize)
    raise ValueError(f"{sampler} sampling is not available.")


def pseudorandom(n_samples, dimension):
    """Pseudo random."""
    # If random seed is set, then the rng based code always returns the same random
    # number, which may not be what we expect.
    # rng = np.random.default_rng(config.random_seed)
    # return rng.random(size=(n_samples, dimension), dtype=config.real(np))
    return np.random.random(size=(n_samples, dimension)).astype(config.real(np))


def quasirandom(n_samples, dimension, sampler):
    # Certain points should be removed:
    # - Boundary points such as [..., 0, ...]
    # - Special points [0, 0, 0, ...] and [0.5, 0.5, 0.5, ...], which cause error in
    #   Hypersphere.random_points() and Hypersphere.random_boundary_points()
    skip = 0
    if sampler == "LHS":
        sampler = skopt.sampler.Lhs()
    elif sampler == "Halton":
        # 1st point: [0, 0, ...]
        sampler = skopt.sampler.Halton(min_skip=1, max_skip=1)
    elif sampler == "Hammersley":
        # 1st point: [0, 0, ...]
        if dimension == 1:
            sampler = skopt.sampler.Hammersly(min_skip=1, max_skip=1)
        else:
            sampler = skopt.sampler.Hammersly()
            skip = 1
    elif sampler == "Sobol":
        # 1st point: [0, 0, ...], 2nd point: [0.5, 0.5, ...]
        sampler = skopt.sampler.Sobol(randomize=False)
        if dimension < 3:
            skip = 1
        else:
            skip = 2
    space = [(0.0, 1.0)] * dimension
    return np.asarray(
        sampler.generate(space, n_samples + skip)[skip:], dtype=config.real(np)
    )


def sample_glt(n_samples, dimension, randomize=False, fold_axes=None):
    """Generate Good Lattice Training (GLT) samples in [0, 1]^dimension.
    
    GLT is based on number theoretic methods for numerical analysis and provides
    optimal collocation points for PINNs. According to the paper (Matsubara & Yaguchi, 
    AAAI 2023), GLT requires 2-7 times fewer collocation points compared to typical 
    sampling methods while achieving competitive performance.
    
    Reference: https://arxiv.org/pdf/2307.13869
    
    Args:
        n_samples (int): The number of samples (target value).
        dimension (int): Space dimension (1, 2, 3, or 4).
        randomize (bool): If True, apply random shift to lattice points.
            This is called "randomized GLT" and can improve robustness.
            Default is False.
        fold_axes (list): List of axes to apply periodization trick.
            For periodic boundary conditions, this folds the domain.
            Default is None (no folding).
    
    Returns:
        np.ndarray: Array of shape (actual_n_samples, dimension) in [0, 1]^dimension.
            Note: actual_n_samples may differ from n_samples if exact value not available.
    
    Raises:
        ValueError: If dimension is not supported (only 1-4D supported).
    
    Example:
        >>> # Standard GLT
        >>> points = sample_glt(144, 2)
        >>> # Randomized GLT (more robust)
        >>> points_rand = sample_glt(144, 2, randomize=True)
        >>> # GLT with periodization (for periodic BCs)
        >>> points_periodic = sample_glt(144, 2, fold_axes=[0, 1])
    
    Note:
        GLT requires specific numbers of points (based on generating vectors).
        If n_samples is not available, the closest value will be used automatically.
        Use GoodLatticeSampler.get_available_sizes(dim) to see all available values.
    """
    if dimension not in [1, 2, 3, 4]:
        raise ValueError(
            f"GLT only supports dimensions 1-4. Got dimension={dimension}. "
            f"For higher dimensions (5D+), use Monte Carlo or other methods."
        )
    
    # Find closest available size
    try:
        actual_n, msg = suggest_glt_size(dimension, n_samples, tolerance=0.2)
        
        # Print warning if size differs significantly
        if abs(actual_n - n_samples) > 0:
            import sys
            print(f"GLT: {msg}", file=sys.stderr, flush=True)
        
        # Create sampler and generate points
        sampler = GoodLatticeSampler(dim=dimension, n_points=actual_n)
        points = sampler.sample(
            randomize=randomize, 
            fold_axes=fold_axes, 
            dtype=config.real(np)
        )
        
        return points
        
    except Exception as e:
        raise ValueError(
            f"Failed to generate GLT samples for dim={dimension}, n={n_samples}: {e}"
        )
