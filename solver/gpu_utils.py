"""GPU utilities and optimization helpers for PINN solver."""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from contextlib import contextmanager


def get_optimal_device() -> torch.device:
    """Get the optimal device for computation.
    
    Returns:
        Best available device (CUDA if available, otherwise CPU).
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return device
    else:
        print("CUDA not available, using CPU")
        return torch.device('cpu')


def get_gpu_memory_info() -> Dict[str, float]:
    """Get GPU memory information.
    
    Returns:
        Dictionary with memory information in GB.
    """
    if not torch.cuda.is_available():
        return {'error': 'CUDA not available'}
    
    return {
        'allocated': torch.cuda.memory_allocated() / 1e9,
        'cached': torch.cuda.memory_reserved() / 1e9,
        'total': torch.cuda.get_device_properties(0).total_memory / 1e9,
        'free': (torch.cuda.get_device_properties(0).total_memory - 
                torch.cuda.memory_reserved()) / 1e9
    }


def clear_gpu_cache():
    """Clear GPU cache to free memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextmanager
def gpu_memory_context():
    """Context manager for GPU memory management."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def optimize_tensor_operations(tensor: torch.Tensor) -> torch.Tensor:
    """Optimize tensor for GPU operations.
    
    Args:
        tensor: Input tensor.
        
    Returns:
        Optimized tensor.
    """
    # Ensure tensor is contiguous and on the right device
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    
    # Use half precision if possible for memory efficiency
    if tensor.dtype == torch.float32 and torch.cuda.is_available():
        # Only use half precision for large tensors
        if tensor.numel() > 10000:
            tensor = tensor.half()
    
    return tensor


def batch_tensor_operations(tensor: torch.Tensor, batch_size: int) -> List[torch.Tensor]:
    """Split tensor into batches for memory-efficient processing.
    
    Args:
        tensor: Input tensor.
        batch_size: Size of each batch.
        
    Returns:
        List of tensor batches.
    """
    batches = []
    num_points = len(tensor)
    
    for i in range(0, num_points, batch_size):
        end_idx = min(i + batch_size, num_points)
        batches.append(tensor[i:end_idx])
    
    return batches


def compute_derivatives_vectorized(u: torch.Tensor, inputs: List[torch.Tensor], 
                                 order: int = 1) -> Dict[str, torch.Tensor]:
    """Compute derivatives using vectorized operations for efficiency.
    
    Args:
        u: Output tensor.
        inputs: List of input tensors for which to compute derivatives.
        order: Order of derivatives to compute.
        
    Returns:
        Dictionary of derivatives.
    """
    derivatives = {}
    
    for i, input_tensor in enumerate(inputs):
        if order >= 1:
            # First derivative
            grad_outputs = torch.ones_like(u)
            first_deriv = torch.autograd.grad(
                u, input_tensor, grad_outputs=grad_outputs,
                create_graph=True, allow_unused=True, retain_graph=True
            )[0]
            derivatives[f'du_dx{i}'] = first_deriv
            
            if order >= 2:
                # Second derivative
                second_deriv = torch.autograd.grad(
                    first_deriv, input_tensor, grad_outputs=torch.ones_like(first_deriv),
                    create_graph=True, allow_unused=True, retain_graph=True
                )[0]
                derivatives[f'd2u_dx{i}2'] = second_deriv
    
    return derivatives


def adaptive_batch_size(available_memory: float, tensor_size: int, 
                       dtype_size: int = 4) -> int:
    """Calculate adaptive batch size based on available memory.
    
    Args:
        available_memory: Available memory in GB.
        tensor_size: Size of each tensor.
        dtype_size: Size of data type in bytes.
        
    Returns:
        Optimal batch size.
    """
    # Estimate memory needed per sample (rough approximation)
    memory_per_sample = tensor_size * dtype_size * 4  # Factor of 4 for gradients, etc.
    
    # Use 80% of available memory
    usable_memory = available_memory * 0.8 * 1e9  # Convert to bytes
    
    batch_size = int(usable_memory / memory_per_sample)
    
    # Ensure minimum batch size
    return max(batch_size, 1)


def profile_memory_usage(func):
    """Decorator to profile memory usage of functions."""
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated()
            
        result = func(*args, **kwargs)
        
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated()
            memory_used = (peak_memory - initial_memory) / 1e9
            print(f"Memory used by {func.__name__}: {memory_used:.2f} GB")
            
        return result
    
    return wrapper


class MemoryEfficientPINN:
    """Memory-efficient PINN wrapper for large-scale problems."""
    
    def __init__(self, pinn, max_memory_gb: float = 8.0):
        """Initialize memory-efficient PINN.
        
        Args:
            pinn: Base PINN instance.
            max_memory_gb: Maximum memory usage in GB.
        """
        self.pinn = pinn
        self.max_memory_gb = max_memory_gb
        self.device = pinn.device
        
    def _get_optimal_batch_size(self, tensor_size: int) -> int:
        """Get optimal batch size based on available memory."""
        if self.device.type == 'cuda':
            available_memory = self.max_memory_gb
        else:
            import psutil
            available_memory = psutil.virtual_memory().available / 1e9
            
        return adaptive_batch_size(available_memory, tensor_size)
    
    def train_memory_efficient(self) -> None:
        """Train PINN with memory-efficient batching."""
        # Calculate optimal batch sizes
        ic_batch_size = self._get_optimal_batch_size(len(self.pinn.x_initial))
        bc_batch_size = self._get_optimal_batch_size(len(self.pinn.x_boundary))
        eq_batch_size = self._get_optimal_batch_size(len(self.pinn.x_equation))
        
        print(f"Memory-efficient batch sizes: IC={ic_batch_size}, BC={bc_batch_size}, EQ={eq_batch_size}")
        
        # Update batch sizes
        self.pinn.batch_size = min(ic_batch_size, bc_batch_size, eq_batch_size)
        self.pinn._setup_batching()
        
        # Train normally
        self.pinn.train()
    
    def predict_memory_efficient(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict with memory-efficient batching."""
        optimal_batch_size = self._get_optimal_batch_size(len(x))
        return self.pinn.predict_batch(x, t, optimal_batch_size)


def optimize_for_inference(model: torch.nn.Module) -> torch.nn.Module:
    """Optimize model for inference.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Optimized model.
    """
    model.eval()
    
    # Enable optimizations
    if hasattr(torch, 'jit'):
        try:
            model = torch.jit.script(model)
        except:
            pass  # JIT compilation failed, continue with regular model
    
    # Optimize for inference
    torch.backends.cudnn.benchmark = True
    
    return model


def benchmark_derivative_computation(model: torch.nn.Module, 
                                   input_tensors: List[torch.Tensor],
                                   num_runs: int = 100) -> Dict[str, float]:
    """Benchmark derivative computation performance.
    
    Args:
        model: Neural network model.
        input_tensors: List of input tensors.
        num_runs: Number of benchmark runs.
        
    Returns:
        Performance metrics.
    """
    model.eval()
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(torch.cat(input_tensors, dim=1))
    
    # Benchmark
    times = []
    
    for _ in range(num_runs):
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if torch.cuda.is_available():
            start_time.record()
        else:
            import time
            start = time.time()
        
        with torch.enable_grad():
            output = model(torch.cat(input_tensors, dim=1))
            # Compute some derivatives
            if len(input_tensors) > 0:
                grad = torch.autograd.grad(output, input_tensors[0], 
                                         grad_outputs=torch.ones_like(output),
                                         create_graph=True)[0]
        
        if torch.cuda.is_available():
            end_time.record()
            torch.cuda.synchronize()
            times.append(start_time.elapsed_time(end_time) / 1000.0)  # Convert to seconds
        else:
            times.append(time.time() - start)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times)
    }
