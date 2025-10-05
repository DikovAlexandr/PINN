"""Batch processing utilities for large-scale PINN problems."""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import math


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    batch_size: int
    num_workers: int = 0
    pin_memory: bool = True
    drop_last: bool = False
    shuffle: bool = True


class DataLoader:
    """Custom data loader for PINN training data."""
    
    def __init__(self, x: torch.Tensor, t: torch.Tensor, u: Optional[torch.Tensor] = None,
                 config: BatchConfig = None):
        """Initialize data loader.
        
        Args:
            x: Spatial coordinates.
            t: Time coordinates.
            u: Solution values (optional).
            config: Batch configuration.
        """
        self.x = x
        self.t = t
        self.u = u
        self.config = config or BatchConfig(batch_size=1000)
        self.num_samples = len(x)
        self.num_batches = math.ceil(self.num_samples / self.config.batch_size)
        
    def __len__(self) -> int:
        """Get number of batches."""
        return self.num_batches
    
    def __iter__(self):
        """Iterate over batches."""
        if self.config.shuffle:
            indices = torch.randperm(self.num_samples)
        else:
            indices = torch.arange(self.num_samples)
        
        for i in range(self.num_batches):
            start_idx = i * self.config.batch_size
            end_idx = min(start_idx + self.config.batch_size, self.num_samples)
            batch_indices = indices[start_idx:end_idx]
            
            x_batch = self.x[batch_indices]
            t_batch = self.t[batch_indices]
            
            if self.u is not None:
                u_batch = self.u[batch_indices]
                yield x_batch, t_batch, u_batch
            else:
                yield x_batch, t_batch


class AdaptiveBatchScheduler:
    """Adaptive batch size scheduler for training optimization."""
    
    def __init__(self, initial_batch_size: int = 1000, 
                 max_batch_size: int = 10000,
                 memory_threshold: float = 0.8):
        """Initialize adaptive batch scheduler.
        
        Args:
            initial_batch_size: Starting batch size.
            max_batch_size: Maximum allowed batch size.
            memory_threshold: Memory usage threshold for adjustment.
        """
        self.initial_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.memory_threshold = memory_threshold
        self.current_batch_size = initial_batch_size
        self.adjustment_factor = 1.2
        
    def get_batch_size(self) -> int:
        """Get current batch size."""
        return self.current_batch_size
    
    def update_batch_size(self, memory_usage: float, loss_trend: float) -> int:
        """Update batch size based on memory usage and loss trend.
        
        Args:
            memory_usage: Current memory usage (0-1).
            loss_trend: Recent loss trend (positive = increasing, negative = decreasing).
            
        Returns:
            Updated batch size.
        """
        if memory_usage < self.memory_threshold and loss_trend > 0:
            # Memory available and loss increasing - increase batch size
            self.current_batch_size = min(
                int(self.current_batch_size * self.adjustment_factor),
                self.max_batch_size
            )
        elif memory_usage > self.memory_threshold:
            # Memory pressure - decrease batch size
            self.current_batch_size = max(
                int(self.current_batch_size / self.adjustment_factor),
                self.initial_batch_size
            )
        
        return self.current_batch_size


class BatchProcessor:
    """Batch processor for PINN training data."""
    
    def __init__(self, device: torch.device):
        """Initialize batch processor.
        
        Args:
            device: Device for computations.
        """
        self.device = device
        
    def create_batches(self, x: torch.Tensor, t: torch.Tensor, 
                      u: Optional[torch.Tensor] = None,
                      batch_size: int = 1000) -> List[Tuple[torch.Tensor, ...]]:
        """Create batches from input data.
        
        Args:
            x: Spatial coordinates.
            t: Time coordinates.
            u: Solution values (optional).
            batch_size: Size of each batch.
            
        Returns:
            List of batches.
        """
        batches = []
        num_samples = len(x)
        
        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            
            x_batch = x[i:end_idx].to(self.device)
            t_batch = t[i:end_idx].to(self.device)
            
            if u is not None:
                u_batch = u[i:end_idx].to(self.device)
                batches.append((x_batch, t_batch, u_batch))
            else:
                batches.append((x_batch, t_batch))
        
        return batches
    
    def process_batches(self, batches: List[Tuple[torch.Tensor, ...]], 
                       process_func: Callable) -> List[torch.Tensor]:
        """Process batches with given function.
        
        Args:
            batches: List of batches.
            process_func: Function to process each batch.
            
        Returns:
            List of processed results.
        """
        results = []
        
        for batch in batches:
            result = process_func(*batch)
            results.append(result)
        
        return results


class MemoryManager:
    """Memory manager for large-scale PINN training."""
    
    def __init__(self, device: torch.device, max_memory_gb: float = 8.0):
        """Initialize memory manager.
        
        Args:
            device: Device for computations.
            max_memory_gb: Maximum memory usage in GB.
        """
        self.device = device
        self.max_memory_gb = max_memory_gb
        self.memory_usage_history = []
        
    def get_memory_usage(self) -> float:
        """Get current memory usage as fraction of total."""
        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated()
            total = torch.cuda.get_device_properties(0).total_memory
            return allocated / total
        else:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024**3) / self.max_memory_gb
    
    def should_clear_cache(self) -> bool:
        """Check if cache should be cleared."""
        current_usage = self.get_memory_usage()
        self.memory_usage_history.append(current_usage)
        
        # Keep only last 10 measurements
        if len(self.memory_usage_history) > 10:
            self.memory_usage_history.pop(0)
        
        # Clear cache if usage is high
        return current_usage > 0.8
    
    def clear_cache_if_needed(self):
        """Clear cache if memory usage is high."""
        if self.should_clear_cache():
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            print("Cache cleared due to high memory usage")
    
    def get_optimal_batch_size(self, tensor_size: int, 
                             dtype_size: int = 4) -> int:
        """Calculate optimal batch size based on available memory.
        
        Args:
            tensor_size: Size of each tensor.
            dtype_size: Size of data type in bytes.
            
        Returns:
            Optimal batch size.
        """
        available_memory = self.max_memory_gb * 0.8  # Use 80% of available memory
        memory_per_sample = tensor_size * dtype_size * 4  # Factor of 4 for gradients
        
        batch_size = int(available_memory * 1e9 / memory_per_sample)
        return max(batch_size, 1)


class DistributedBatchProcessor:
    """Distributed batch processor for multi-GPU training."""
    
    def __init__(self, world_size: int, rank: int):
        """Initialize distributed processor.
        
        Args:
            world_size: Total number of processes.
            rank: Rank of current process.
        """
        self.world_size = world_size
        self.rank = rank
        
    def split_data(self, x: torch.Tensor, t: torch.Tensor, 
                  u: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]:
        """Split data across processes.
        
        Args:
            x: Spatial coordinates.
            t: Time coordinates.
            u: Solution values (optional).
            
        Returns:
            Split data for current process.
        """
        chunk_size = len(x) // self.world_size
        start_idx = self.rank * chunk_size
        end_idx = start_idx + chunk_size if self.rank < self.world_size - 1 else len(x)
        
        x_local = x[start_idx:end_idx]
        t_local = t[start_idx:end_idx]
        
        if u is not None:
            u_local = u[start_idx:end_idx]
            return x_local, t_local, u_local
        else:
            return x_local, t_local
    
    def gather_results(self, local_results: List[torch.Tensor]) -> List[torch.Tensor]:
        """Gather results from all processes.
        
        Args:
            local_results: Results from current process.
            
        Returns:
            Gathered results from all processes.
        """
        # This would be implemented with torch.distributed in practice
        # For now, return local results
        return local_results


class BatchOptimizer:
    """Optimizer for batch processing parameters."""
    
    def __init__(self, initial_config: BatchConfig):
        """Initialize batch optimizer.
        
        Args:
            initial_config: Initial batch configuration.
        """
        self.config = initial_config
        self.performance_history = []
        
    def optimize_batch_size(self, performance_metric: float) -> BatchConfig:
        """Optimize batch size based on performance.
        
        Args:
            performance_metric: Current performance metric (e.g., loss, time).
            
        Returns:
            Optimized batch configuration.
        """
        self.performance_history.append(performance_metric)
        
        if len(self.performance_history) < 3:
            return self.config
        
        # Simple optimization: increase batch size if performance is improving
        recent_trend = np.mean(self.performance_history[-3:]) - np.mean(self.performance_history[-6:-3])
        
        if recent_trend < 0:  # Performance improving
            new_batch_size = min(self.config.batch_size * 2, 10000)
        else:  # Performance degrading
            new_batch_size = max(self.config.batch_size // 2, 100)
        
        self.config.batch_size = new_batch_size
        return self.config


def create_optimized_dataloader(x: torch.Tensor, t: torch.Tensor, 
                               u: Optional[torch.Tensor] = None,
                               batch_size: int = 1000,
                               device: torch.device = None) -> DataLoader:
    """Create optimized data loader.
    
    Args:
        x: Spatial coordinates.
        t: Time coordinates.
        u: Solution values (optional).
        batch_size: Batch size.
        device: Device for computations.
        
    Returns:
        Optimized data loader.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move data to device
    x = x.to(device)
    t = t.to(device)
    if u is not None:
        u = u.to(device)
    
    # Create optimized configuration
    config = BatchConfig(
        batch_size=batch_size,
        pin_memory=device.type == 'cuda',
        shuffle=True
    )
    
    return DataLoader(x, t, u, config)


def benchmark_batch_processing(x: torch.Tensor, t: torch.Tensor,
                             u: Optional[torch.Tensor] = None,
                             batch_sizes: List[int] = None) -> Dict[int, float]:
    """Benchmark batch processing with different batch sizes.
    
    Args:
        x: Spatial coordinates.
        t: Time coordinates.
        u: Solution values (optional).
        batch_sizes: List of batch sizes to test.
        
    Returns:
        Performance metrics for each batch size.
    """
    if batch_sizes is None:
        batch_sizes = [100, 500, 1000, 2000, 5000]
    
    results = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for batch_size in batch_sizes:
        dataloader = create_optimized_dataloader(x, t, u, batch_size, device)
        
        # Benchmark
        times = []
        for _ in range(10):  # 10 iterations
            start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            
            if torch.cuda.is_available():
                start_time.record()
            else:
                import time
                start = time.time()
            
            for batch in dataloader:
                # Simulate processing
                if u is not None:
                    x_batch, t_batch, u_batch = batch
                    _ = x_batch + t_batch + u_batch
                else:
                    x_batch, t_batch = batch
                    _ = x_batch + t_batch
            
            if torch.cuda.is_available():
                end_time.record()
                torch.cuda.synchronize()
                times.append(start_time.elapsed_time(end_time) / 1000.0)
            else:
                times.append(time.time() - start)
        
        results[batch_size] = np.mean(times)
    
    return results
