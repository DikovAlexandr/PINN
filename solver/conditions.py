import numpy as np
import torch

def set_boundary_conditions(size, num_points, T, device="cuda:0"):
    """
    Set the boundary conditions for a simulation.

    Parameters:
    - size (float): The size of the domain.
    - num_points (int): The number of points in the domain.
    - T (float): The final time of the simulation.

    Returns:
    - boundary_conditions (BoundaryConditions): 
      An instance of class, containing the x, t, and u values.
    """
    x_boundary_right = torch.ones(num_points, device=device) * size
    x_boundary_left = torch.zeros(num_points, device=device)
    x_boundary = torch.cat([x_boundary_right, x_boundary_left])

    t_boundary = torch.linspace(0, T, len(x_boundary), device=device)

    u_boundary = torch.zeros_like(x_boundary, device=device)

    class BoundaryConditions:
        def __init__(self, x, t, u):
            self.x = x
            self.t = t
            self.u = u

    boundary_conditions = BoundaryConditions(x_boundary, t_boundary, u_boundary)

    return boundary_conditions


def set_initial_conditions(initial_distribution_func, size, num_points, 
                           dx, random=False, device="cuda:0"):
    """
    Generate initial conditions for a simulation.

    Parameters:
        initial_distribution_func (function): The initial distribution function.
        size (float): The size of the domain.
        num_points (int): The number of points to generate.
        dx (float): The spacing between the points.
        random (bool, optional): If True, generate random points in the domain.

    Returns:
        initial_conditions (InitialConditions): 
        An instance of class containing the initial x, t, and u values.
    """
    if random:
        x_initial = torch.rand(num_points, device=device) * (size - dx) + dx
    else:
        x_initial = torch.linspace(dx, size - dx, num_points, device=device)
    
    t_initial = torch.zeros_like(x_initial, device=device)
    
    u_initial = initial_distribution_func(x_initial).to(device)

    class InitialConditions:
        def __init__(self, x, t, u):
            self.x = x
            self.t = t
            self.u = u

    initial_conditions = InitialConditions(x_initial, t_initial, u_initial)
    
    return initial_conditions


def set_equation(size, num_points, T, dx, random=False, device="cuda:0"):
    """
    Generate a set of equations for a given size, number of points, time, and spacing.

    Parameters:
        size (int): The size of the equation.
        num_points (int): The number of points in the equation.
        T (float): The final time of the simulation.
        dx (float): The spacing between points in the equation.
                    It is necessary, just as the equation may not hold at the boundary.
        random (bool, optional): If True, generate random points in the domain.

    Returns:
        equation (Equation): 
        An instance of class containing the x, t values of the equation.
    """
    if random:
        x_equation = torch.rand(num_points, device=device) * (size - dx) + dx
    else:
        x_equation = torch.linspace(dx, size - dx, num_points, device=device)

    t_equation = torch.linspace(0, T, num_points, device=device)

    x, t = torch.meshgrid(x_equation, t_equation)

    x_equation = x.flatten()
    t_equation = t.flatten()

    class Equation:
        def __init__(self, x, t):
            self.x = x
            self.t = t

    equation = Equation(x_equation, t_equation)

    return equation


def set_test(size, num_points, time, random=False, device="cuda:0"):
    """
    Generates a test dataset for a given size, number of points, and time.

    Parameters:
        size (float): The maximum value for the test dataset.
        num_points (int): The number of points in the test dataset.
        time (float): The time value for all points in the test dataset.
        random (bool, optional): If True, generates a random test dataset.

    Returns:
        test (Test):
        An instance of class containing the x, t values of the generated test dataset.
    """
    if random:
        x_test = torch.rand(num_points, device=device) * size
    else:
        x_test = torch.linspace(0, size, num_points, device=device)

    t_test = torch.ones(len(x_test), device=device) * time

    class Test:
        def __init__(self, x, t):
            self.x = x
            self.t = t

    test = Test(x_test, t_test)

    return test