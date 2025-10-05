"""Example usage of optimized PINN with GPU support and batching."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import time

from solver.pinn_optimized import OptimizedPINN
from solver.gpu_utils import get_optimal_device, get_gpu_memory_info, benchmark_derivative_computation
from solver.batch_utils import create_optimized_dataloader, benchmark_batch_processing
from solver import Interval, TimeDomain, PDE
from solver.conditions import InitialConditions, BoundaryConditions, Equation, Problem
from solver.utils import NetParams


def initial_condition_1d(x):
    """Initial condition for 1D heat equation: u(x,0) = sin(πx)."""
    return torch.sin(np.pi * x)


def boundary_condition_1d(x, t):
    """Boundary conditions for 1D heat equation: u(0,t) = u(1,t) = 0."""
    return torch.zeros_like(t)


def analytical_solution_1d(x, t, alpha=0.1):
    """Analytical solution for 1D heat equation."""
    return torch.exp(-alpha**2 * np.pi**2 * t) * torch.sin(np.pi * x)


def main():
    """Main function demonstrating optimized PINN usage."""
    print("Optimized PINN Solver - GPU and Batching Example")
    print("=" * 55)
    
    # Get optimal device
    device = get_optimal_device()
    print(f"Using device: {device}")
    
    # Problem parameters
    alpha = 0.1  # Thermal diffusivity
    x_left, x_right = 0.0, 1.0
    t_start, t_end = 0.0, 1.0
    
    # Create geometry and time domain
    geom = Interval(x_left, x_right)
    time_domain = TimeDomain(t_start, t_end)
    
    # Create PDE
    pde = PDE('heat', alpha=alpha)
    
    # Setup initial conditions
    initial_conditions = InitialConditions()
    initial_conditions.set_initial_conditions(
        geom, time_domain, initial_condition_1d, 
        num_points=200, random=False, device=device.type
    )
    
    # Setup boundary conditions
    boundary_conditions = BoundaryConditions()
    boundary_conditions.set_boundary_conditions(
        geom, time_domain, boundary_condition_1d,
        num_points=200, random=False, device=device.type
    )
    
    # Setup equation points
    equation = Equation()
    equation.set_equation(
        pde, geom, time_domain, 
        num_points=2000, random=False, device=device.type
    )
    
    # Create problem
    problem = Problem(
        initial_conditions, boundary_conditions, equation,
        None, geom, time_domain, alpha
    )
    
    # Setup network parameters
    net_params = NetParams()
    net_params.set_params(
        input_dim=2, output_dim=1, 
        hidden_layers=[64, 64, 64, 64],
        epochs=3000, batch_size=1000, lr=0.001,
        activation='tanh', training_mode='train',
        regularization='None', lambda_reg=0.0,
        optimizer='Adam', scheduler='ReduceLROnPlateau',
        early_stopping=True, use_rar=False,
        use_weights_adjuster=False, display_interval=100,
        model_save_path='models', output_path='output',
        save_loss=True, initial_weights_path=None,
        siren_params=None
    )
    
    # Create optimized PINN
    print("Creating optimized PINN...")
    pinn = OptimizedPINN(problem, net_params, device=device.type, batch_size=500)
    
    # Show memory usage before training
    if device.type == 'cuda':
        print("GPU Memory before training:")
        memory_info = get_gpu_memory_info()
        for key, value in memory_info.items():
            print(f"  {key}: {value:.2f} GB")
    
    # Benchmark derivative computation
    print("\nBenchmarking derivative computation...")
    x_test = torch.linspace(0, 1, 100, device=device).reshape(-1, 1)
    t_test = torch.linspace(0, 1, 100, device=device).reshape(-1, 1)
    
    benchmark_results = benchmark_derivative_computation(
        pinn.net, [x_test, t_test], num_runs=50
    )
    print("Derivative computation benchmark:")
    for key, value in benchmark_results.items():
        print(f"  {key}: {value:.4f} seconds")
    
    # Train PINN
    print("\nTraining optimized PINN...")
    start_time = time.time()
    pinn.train()
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Show memory usage after training
    if device.type == 'cuda':
        print("\nGPU Memory after training:")
        memory_info = get_gpu_memory_info()
        for key, value in memory_info.items():
            print(f"  {key}: {value:.2f} GB")
    
    # Generate test points
    x_test = torch.linspace(x_left, x_right, 100, device=device).reshape(-1, 1)
    t_test = torch.linspace(t_start, t_end, 100, device=device).reshape(-1, 1)
    
    # Create meshgrid for visualization
    X, T = torch.meshgrid(x_test.squeeze(), t_test.squeeze(), indexing='ij')
    x_flat = X.flatten().reshape(-1, 1)
    t_flat = T.flatten().reshape(-1, 1)
    
    # Make predictions with batching
    print("\nMaking predictions with batching...")
    start_time = time.time()
    u_pred = pinn.predict_batch(x_flat, t_flat, batch_size=500)
    prediction_time = time.time() - start_time
    print(f"Prediction completed in {prediction_time:.2f} seconds")
    
    # Compute analytical solution
    u_analytical = analytical_solution_1d(x_flat, t_flat, alpha)
    
    # Reshape for plotting
    u_pred_2d = u_pred.reshape(X.shape)
    u_analytical_2d = u_analytical.reshape(X.shape)
    
    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # PINN solution
    im1 = ax1.contourf(X.cpu().numpy(), T.cpu().numpy(), u_pred_2d.cpu().detach().numpy(), levels=20)
    ax1.set_title('Optimized PINN Solution')
    ax1.set_xlabel('x')
    ax1.set_ylabel('t')
    plt.colorbar(im1, ax=ax1)
    
    # Analytical solution
    im2 = ax2.contourf(X.cpu().numpy(), T.cpu().numpy(), u_analytical_2d.cpu().detach().numpy(), levels=20)
    ax2.set_title('Analytical Solution')
    ax2.set_xlabel('x')
    ax2.set_ylabel('t')
    plt.colorbar(im2, ax=ax2)
    
    # Error
    error = torch.abs(u_pred_2d - u_analytical_2d)
    im3 = ax3.contourf(X.cpu().numpy(), T.cpu().numpy(), error.cpu().detach().numpy(), levels=20)
    ax3.set_title('Absolute Error')
    ax3.set_xlabel('x')
    ax3.set_ylabel('t')
    plt.colorbar(im3, ax=ax3)
    
    plt.tight_layout()
    plt.savefig('output/optimized_heat_equation_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Compute metrics
    from solver.metrics import l2_relative_error, max_value_error
    
    l2_error = l2_relative_error(u_analytical, u_pred)
    max_error = max_value_error(u_analytical, u_pred)
    
    print(f"\nResults:")
    print(f"L2 Relative Error: {l2_error:.6f}")
    print(f"Max Value Error: {max_error:.6f}")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Prediction Time: {prediction_time:.2f} seconds")
    
    # Benchmark batch processing
    print("\nBenchmarking batch processing...")
    batch_results = benchmark_batch_processing(
        x_flat.cpu(), t_flat.cpu(), u_flat.cpu() if 'u_flat' in locals() else None,
        batch_sizes=[100, 500, 1000, 2000]
    )
    
    print("Batch processing benchmark:")
    for batch_size, time_taken in batch_results.items():
        print(f"  Batch size {batch_size}: {time_taken:.4f} seconds")
    
    # Save model
    pinn.save_weights()
    print("\nModel saved successfully!")
    
    # Show final memory usage
    if device.type == 'cuda':
        print("\nFinal GPU Memory usage:")
        memory_info = pinn.get_memory_usage()
        for key, value in memory_info.items():
            print(f"  {key}: {value:.2f} GB")


def large_scale_example():
    """Example with large-scale problem to demonstrate batching."""
    print("\n" + "="*60)
    print("Large-Scale Problem Example")
    print("="*60)
    
    device = get_optimal_device()
    
    # Create larger problem
    geom = Interval(0, 1)
    time_domain = TimeDomain(0, 1)
    pde = PDE('heat', alpha=0.1)
    
    # More points for large-scale testing
    initial_conditions = InitialConditions()
    initial_conditions.set_initial_conditions(
        geom, time_domain, initial_condition_1d, 
        num_points=1000, random=False, device=device.type
    )
    
    boundary_conditions = BoundaryConditions()
    boundary_conditions.set_boundary_conditions(
        geom, time_domain, boundary_condition_1d,
        num_points=1000, random=False, device=device.type
    )
    
    equation = Equation()
    equation.set_equation(
        pde, geom, time_domain, 
        num_points=10000, random=False, device=device.type
    )
    
    problem = Problem(
        initial_conditions, boundary_conditions, equation,
        None, geom, time_domain, 0.1
    )
    
    # Setup for large-scale training
    net_params = NetParams()
    net_params.set_params(
        input_dim=2, output_dim=1, 
        hidden_layers=[128, 128, 128, 128],
        epochs=1000, batch_size=2000, lr=0.001,
        activation='tanh', training_mode='train',
        regularization='None', lambda_reg=0.0,
        optimizer='Adam', scheduler='ReduceLROnPlateau',
        early_stopping=True, use_rar=False,
        use_weights_adjuster=False, display_interval=50,
        model_save_path='models', output_path='output',
        save_loss=True, initial_weights_path=None,
        siren_params=None
    )
    
    # Create optimized PINN with smaller batch size for memory efficiency
    pinn = OptimizedPINN(problem, net_params, device=device.type, batch_size=1000)
    
    print(f"Training large-scale PINN with {len(problem.equation.x)} equation points...")
    start_time = time.time()
    pinn.train()
    training_time = time.time() - start_time
    print(f"Large-scale training completed in {training_time:.2f} seconds")
    
    # Test prediction on large dataset
    x_test = torch.linspace(0, 1, 500, device=device).reshape(-1, 1)
    t_test = torch.linspace(0, 1, 500, device=device).reshape(-1, 1)
    
    print("Testing prediction on large dataset...")
    start_time = time.time()
    u_pred = pinn.predict_batch(x_test, t_test, batch_size=500)
    prediction_time = time.time() - start_time
    print(f"Large-scale prediction completed in {prediction_time:.2f} seconds")
    
    print(f"Predicted {len(u_pred)} points in {prediction_time:.2f} seconds")
    print(f"Prediction rate: {len(u_pred)/prediction_time:.0f} points/second")


if __name__ == "__main__":
    main()
    large_scale_example()
