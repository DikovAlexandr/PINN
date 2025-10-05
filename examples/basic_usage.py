"""Basic usage example for PINN solver module."""

import torch
import numpy as np
import matplotlib.pyplot as plt

from solver import PINN, Interval, TimeDomain, PDE
from solver.conditions import InitialConditions, BoundaryConditions, Equation, Problem
from solver.utils import NetParams
from solver.visualizations import solution_surface_plot, comparison_plot


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
    """Main function demonstrating basic PINN usage."""
    print("PINN Solver - Basic Usage Example")
    print("=" * 40)
    
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
        num_points=100, random=False, device="cpu"
    )
    
    # Setup boundary conditions
    boundary_conditions = BoundaryConditions()
    boundary_conditions.set_boundary_conditions(
        geom, time_domain, boundary_condition_1d,
        num_points=100, random=False, device="cpu"
    )
    
    # Setup equation points
    equation = Equation()
    equation.set_equation(
        pde, geom, time_domain, 
        num_points=1000, random=False, device="cpu"
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
        hidden_layers=[40, 40, 40, 40],
        epochs=5000, batch_size=1000, lr=0.001,
        activation='tanh', training_mode='train',
        regularization='None', lambda_reg=0.0,
        optimizer='Adam', scheduler=None,
        early_stopping=True, use_rar=False,
        use_weights_adjuster=False, display_interval=100,
        model_save_path='models', output_path='output',
        save_loss=True, initial_weights_path=None,
        siren_params=None
    )
    
    # Create and train PINN
    print("Creating PINN...")
    pinn = PINN(problem, net_params, device="cpu")
    
    print("Training PINN...")
    pinn.train()
    
    # Generate test points
    x_test = torch.linspace(x_left, x_right, 50).reshape(-1, 1)
    t_test = torch.linspace(t_start, t_end, 50).reshape(-1, 1)
    
    # Create meshgrid for visualization
    X, T = torch.meshgrid(x_test.squeeze(), t_test.squeeze(), indexing='ij')
    x_flat = X.flatten().reshape(-1, 1)
    t_flat = T.flatten().reshape(-1, 1)
    
    # Make predictions
    print("Making predictions...")
    u_pred = pinn.predict(x_flat, t_flat)
    
    # Compute analytical solution
    u_analytical = analytical_solution_1d(x_flat, t_flat, alpha)
    
    # Reshape for plotting
    u_pred_2d = u_pred.reshape(X.shape)
    u_analytical_2d = u_analytical.reshape(X.shape)
    
    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # PINN solution
    im1 = ax1.contourf(X.numpy(), T.numpy(), u_pred_2d.detach().numpy(), levels=20)
    ax1.set_title('PINN Solution')
    ax1.set_xlabel('x')
    ax1.set_ylabel('t')
    plt.colorbar(im1, ax=ax1)
    
    # Analytical solution
    im2 = ax2.contourf(X.numpy(), T.numpy(), u_analytical_2d.detach().numpy(), levels=20)
    ax2.set_title('Analytical Solution')
    ax2.set_xlabel('x')
    ax2.set_ylabel('t')
    plt.colorbar(im2, ax=ax2)
    
    # Error
    error = torch.abs(u_pred_2d - u_analytical_2d)
    im3 = ax3.contourf(X.numpy(), T.numpy(), error.detach().numpy(), levels=20)
    ax3.set_title('Absolute Error')
    ax3.set_xlabel('x')
    ax3.set_ylabel('t')
    plt.colorbar(im3, ax=ax3)
    
    plt.tight_layout()
    plt.savefig('output/heat_equation_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Compute metrics
    from solver.metrics import l2_relative_error, max_value_error
    
    l2_error = l2_relative_error(u_analytical, u_pred)
    max_error = max_value_error(u_analytical, u_pred)
    
    print(f"\nResults:")
    print(f"L2 Relative Error: {l2_error:.6f}")
    print(f"Max Value Error: {max_error:.6f}")
    
    # Save model
    pinn.save_weights()
    print("Model saved successfully!")


if __name__ == "__main__":
    main()
