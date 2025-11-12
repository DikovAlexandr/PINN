"""GPU utilities for PINN training.

Simple and efficient utilities for GPU device selection, memory monitoring,
and cache management. No overhead, no complex optimization logic.
"""

import torch
import numpy as np
from typing import Dict, Tuple
from contextlib import contextmanager
import warnings


__all__ = [
    "get_optimal_device",
    "get_gpu_memory_info",
    "clear_gpu_cache",
    "gpu_memory_context",
    "estimate_tensor_memory",
]


def get_optimal_device(verbose: bool = True) -> torch.device:
    """Get the optimal device for computation.
    
    Args:
        verbose: If True, print device information.
    
    Returns:
        Best available device (CUDA if available, otherwise CPU).
        
    Example:
        >>> device = get_optimal_device()
        Using GPU: NVIDIA GeForce RTX 3090
        GPU Memory: 24.0 GB
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        if verbose:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"Using GPU: {gpu_name}")
            print(f"GPU Memory: {gpu_memory:.1f} GB")
        return device
    else:
        if verbose:
            print("CUDA not available, using CPU")
        return torch.device('cpu')


def get_gpu_memory_info() -> Dict[str, float]:
    """Get GPU memory information in GB.
    
    Returns:
        Dictionary with memory information:
        - allocated: Currently allocated memory
        - reserved: Reserved/cached memory
        - total: Total GPU memory
        - free: Free memory
        - utilization: Memory utilization percentage
        
    Example:
        >>> info = get_gpu_memory_info()
        >>> print(f"GPU Memory: {info['allocated']:.2f}/{info['total']:.2f} GB")
        GPU Memory: 2.50/24.00 GB
    """
    if not torch.cuda.is_available():
        return {
            'error': 'CUDA not available',
            'allocated': 0.0,
            'reserved': 0.0,
            'total': 0.0,
            'free': 0.0,
            'utilization': 0.0
        }
    
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    free = total - reserved
    utilization = (allocated / total) * 100 if total > 0 else 0
    
    return {
        'allocated': allocated,
        'reserved': reserved,
        'total': total,
        'free': free,
        'utilization': utilization
    }


def clear_gpu_cache(verbose: bool = False):
    """Clear GPU cache to free memory.
    
    Args:
        verbose: If True, print memory freed.
        
    Example:
        >>> clear_gpu_cache(verbose=True)
        GPU cache cleared. Freed: 2.5 GB
    """
    if torch.cuda.is_available():
        if verbose:
            before = torch.cuda.memory_reserved() / 1e9
        torch.cuda.empty_cache()
        if verbose:
            after = torch.cuda.memory_reserved() / 1e9
            freed = before - after
            if freed > 0:
                print(f"GPU cache cleared. Freed: {freed:.2f} GB")
            else:
                print("GPU cache cleared.")


@contextmanager
def gpu_memory_context(clear_cache: bool = True, verbose: bool = False):
    """Context manager for GPU memory management.
    
    Automatically clears cache before and after operations.
    
    Args:
        clear_cache: If True, clear cache before and after.
        verbose: If True, print memory usage.
        
    Example:
        >>> with gpu_memory_context(verbose=True):
        ...     # Your GPU operations here
        ...     model.train()
        Initial GPU memory: 1.20 GB
        Final GPU memory: 3.45 GB
        Memory used: 2.25 GB
    """
    if not torch.cuda.is_available():
        yield
        return
    
    if clear_cache:
        torch.cuda.empty_cache()
    
    initial_memory = torch.cuda.memory_allocated() / 1e9
    
    if verbose:
        print(f"Initial GPU memory: {initial_memory:.2f} GB")
    
    try:
        yield
    finally:
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated() / 1e9
            if verbose:
                print(f"Final GPU memory: {final_memory:.2f} GB")
                print(f"Memory used: {final_memory - initial_memory:.2f} GB")
            
            if clear_cache:
                torch.cuda.empty_cache()


def estimate_tensor_memory(tensor_shape: Tuple[int, ...], dtype: torch.dtype = torch.float32) -> float:
    """Estimate memory required for a tensor in GB.
    
    Args:
        tensor_shape: Shape of the tensor.
        dtype: Data type of the tensor.
        
    Returns:
        Memory required in GB.
        
    Example:
        >>> memory = estimate_tensor_memory((10000, 100), torch.float32)
        >>> print(f"Estimated memory: {memory:.4f} GB")
        Estimated memory: 0.0040 GB
    """
    num_elements = np.prod(tensor_shape)
    
    dtype_sizes = {
        torch.float32: 4,
        torch.float64: 8,
        torch.float16: 2,
        torch.int32: 4,
        torch.int64: 8,
        torch.int16: 2,
        torch.int8: 1,
        torch.uint8: 1,
        torch.bool: 1,
    }
    
    bytes_per_element = dtype_sizes.get(dtype, 4)
    total_bytes = num_elements * bytes_per_element
    
    return total_bytes / 1e9
