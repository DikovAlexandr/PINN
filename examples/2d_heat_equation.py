"""2D heat equation example using PINN solver module."""

import torch
import numpy as np
import matplotlib.pyplot as plt

from solver import PINN, Rectangle, TimeDomain, PDE
from solver.conditions import InitialConditions, BoundaryConditions, Equation, Problem
from solver.utils import NetParams
from solver.visualizations import solution_surface_plot


def initial_condition_2d(x):
    """Initial condition for 2D heat equation."""
    x_coords = x[:, 0].reshape(-1, 1)
    y_coords = x[:, 1].reshape(-1, 1)
    return torch.sin(np.pi * x_coords) * torch.sin(np.pi * y_coords)


def boundary_condition_2d(x, t):
    """Boundary conditions for 2D heat equation."""
    return torch.zeros_like(t)


def analytical_solution_2d(x, t, alpha=0.1):
    """Analytical solution for 2D heat equation."""
    x_coords = x[:, 0].reshape(-1, 1)
    y_coords = x[:, 1].reshape(-1, 1)
    return (torch.exp(-2 * alpha**2 * np.pi**2 * t) * 
            torch.sin(np.pi * x_coords) * torch.sin(np.pi * y_coords))


def main():
    """Main function for 2D heat equation example."""
    print("PINN Solver - 2D Heat Equation Example")
    print("=" * 45)
    
    # Problem parameters
    alpha = 0.1  # Thermal diffusivity
    x_min, x_max = 0.0, 1.0
    y_min, y_max = 0.0, 1.0
    t_start, t_end = 0.0, 0.5
    
    # Create geometry and time domain
    geom = Rectangle(x_min, x_max, y_min, y_max)
    time_domain = TimeDomain(t_start, t_end)
    
    # Create PDE
    pde = PDE('heat2D', alpha=alpha)
    
    # Setup initial conditions
    initial_conditions = InitialConditions()
    initial_conditions.set_initial_conditions(
        geom, time_domain, initial_condition_2d,
        num_points=400, random=False, device="cpu"
    )
    
    # Setup boundary conditions
    boundary_conditions = BoundaryConditions()
    boundary_conditions.set_boundary_conditions(
        geom, time_domain, boundary_condition_2d,
        num_points=400, random=False, device="cpu"
    )
    
    # Setup equation points
    equation = Equation()
    equation.set_equation(
        pde, geom, time_domain,
        num_points=2000, random=False, device="cpu"
    )
    
    # Create problem
    problem = Problem(
        initial_conditions, boundary_conditions, equation,
        None, geom, time_domain, alpha
    )
    
    # Setup network parameters
    net_params = NetParams()
    net_params.set_params(
        input_dim=3, output_dim=1,  # x, y, t
        hidden_layers=[50, 50, 50, 50],
        epochs=3000, batch_size=2000, lr=0.001,
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
    
    # Generate test points for final time
    x_test = torch.linspace(x_min, x_max, 30)
    y_test = torch.linspace(y_min, y_max, 30)
    X, Y = torch.meshgrid(x_test, y_test, indexing='ij')
    
    x_flat = X.flatten().reshape(-1, 1)
    y_flat = Y.flatten().reshape(-1, 1)
    xy_flat = torch.cat([x_flat, y_flat], dim=1)
    t_final = torch.ones(xy_flat.shape[0], 1) * t_end
    
    # Make predictions
    print("Making predictions...")
    u_pred = pinn.predict(xy_flat, t_final)
    u_pred_2d = u_pred.reshape(X.shape)
    
    # Compute analytical solution
    u_analytical = analytical_solution_2d(xy_flat, t_final, alpha)
    u_analytical_2d = u_analytical.reshape(X.shape)
    
    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # PINN solution
    im1 = ax1.contourf(X.numpy(), Y.numpy(), u_pred_2d.detach().numpy(), levels=20)
    ax1.set_title('PINN Solution (t=0.5)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_aspect('equal')
    plt.colorbar(im1, ax=ax1)
    
    # Analytical solution
    im2 = ax2.contourf(X.numpy(), Y.numpy(), u_analytical_2d.detach().numpy(), levels=20)
    ax2.set_title('Analytical Solution (t=0.5)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_aspect('equal')
    plt.colorbar(im2, ax=ax2)
    
    # Error
    error = torch.abs(u_pred_2d - u_analytical_2d)
    im3 = ax3.contourf(X.numpy(), Y.numpy(), error.detach().numpy(), levels=20)
    ax3.set_title('Absolute Error')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_aspect('equal')
    plt.colorbar(im3, ax=ax3)
    
    plt.tight_layout()
    plt.savefig('output/2d_heat_equation_comparison.png', dpi=300, bbox_inches='tight')
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
