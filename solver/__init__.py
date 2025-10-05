"""
PINN Solver Module for DeepXDE

This module provides Physics-Informed Neural Network (PINN) implementations
for solving partial differential equations, specifically designed to work
as an extension to the DeepXDE library.

Main Components:
- PINN: Main neural network class for solving PDEs
- Geometry: Classes for defining spatial domains (1D and 2D)
- Conditions: Classes for initial and boundary conditions
- PDE: Classes for defining partial differential equations
- Enhancements: Advanced techniques like RAR and hybrid optimization
- Visualizations: Plotting and visualization utilities
- Metrics: Evaluation metrics for solution quality

Example:
    >>> from solver import PINN, Interval, TimeDomain, PDE
    >>> from solver.conditions import InitialConditions, BoundaryConditions
    >>> 
    >>> # Define geometry and time domain
    >>> geom = Interval(0, 1)
    >>> time_domain = TimeDomain(0, 1)
    >>> 
    >>> # Define PDE
    >>> pde = PDE('heat', alpha=0.1)
    >>> 
    >>> # Create PINN solver
    >>> pinn = PINN(problem, net_params)
"""

from .pinn import PINN
from .pinn_optimized import OptimizedPINN
from .geometry import Interval, Rectangle, Circle, Ellipse
from .timedomain import TimeDomain
from .pde import PDE
from .conditions import (
    InitialConditions, 
    BoundaryConditions, 
    Equation, 
    Problem, 
    Solution
)
from .enhancements import (
    LossWeightAdjuster, 
    HybridOptimizer, 
    EarlyStopping,
    rar_points
)
from .metrics import (
    accuracy,
    l2_norm,
    l2_relative_error,
    max_value_error,
    calculate_error
)
from .visualizations import (
    solution_gif,
    conditions_plot,
    solution_surface_plot,
    comparison_plot,
    loss_history_plot,
    error_plot,
    evolution_gif
)
from .callbacks import ModelCheckpoint, Timer
from .siren import SirenParams, SineLayer, Sin
from .utils import NetParams, create_folder, to_numpy
from .gpu_utils import get_optimal_device, get_gpu_memory_info, MemoryEfficientPINN
from .batch_utils import create_optimized_dataloader, BatchConfig, AdaptiveBatchScheduler

__version__ = "1.0.0"
__author__ = "Dikov Alexandr"

__all__ = [
    # Main classes
    "PINN",
    "OptimizedPINN",
    "PDE",
    "Problem",
    "Solution",
    
    # Geometry classes
    "Interval",
    "Rectangle", 
    "Circle",
    "Ellipse",
    
    # Time domain
    "TimeDomain",
    
    # Condition classes
    "InitialConditions",
    "BoundaryConditions", 
    "Equation",
    
    # Enhancement classes
    "LossWeightAdjuster",
    "HybridOptimizer",
    "EarlyStopping",
    
    # Functions
    "rar_points",
    "accuracy",
    "l2_norm", 
    "l2_relative_error",
    "max_value_error",
    "calculate_error",
    "solution_gif",
    "conditions_plot",
    "solution_surface_plot", 
    "comparison_plot",
    "loss_history_plot",
    "error_plot",
    "evolution_gif",
    
    # Utility classes
    "ModelCheckpoint",
    "Timer",
    "SirenParams",
    "SineLayer",
    "Sin",
    "NetParams",
    
    # Utility functions
    "create_folder",
    "to_numpy",
    
    # GPU utilities
    "get_optimal_device",
    "get_gpu_memory_info",
    "MemoryEfficientPINN",
    
    # Batch utilities
    "create_optimized_dataloader",
    "BatchConfig",
    "AdaptiveBatchScheduler",
]
