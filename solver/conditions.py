import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Callable

from solver.geometry import Geometry
from solver.timedomain import TimeDomain
from solver.pde import PDE

class MathConditions:
    def __init__(self):
        self.x = None
        self.t = None
        self.u = None


class Equation(MathConditions):
    def __init__(self):
        pass

    def set_equation(self, pde: PDE, 
                     geom: Geometry, 
                     time: TimeDomain, 
                     num_points: int, 
                     random: bool, 
                     device="cuda:0"):
        """
        Set up the equation domain.

        Parameters:
            pde (PDE): Class that stores the pde.
            geom (Geometry): Class that stores the geometry domain data.
            time (TimeDomain): Class that stores the time domain data.
            num_points (int): Number of points for equation.
            random (bool): If True, random points in domain will be used.
            device (str): Device to use. Default is 'cuda:0'.
        """
        self.pde = pde

        if geom.get_dimension() == 1:
            x_equation = geom.inners(int(np.sqrt(num_points)), 
                                     device=device, random=random)
            t_equation = time.inners(int(np.sqrt(num_points)), 
                                     device=device, random=random)
            x, t = torch.meshgrid(x_equation, t_equation)
            self.x = x.reshape(-1, 1)
            self.t = t.reshape(-1, 1)

        if geom.get_dimension() == 2:
            x_equation = geom.inners(int(np.cbrt(num_points)), 
                                     device=device, random=random)
            t_equation = time.inners(int(np.cbrt(num_points)), 
                                     device=device, random=random)
            x_coords = x_equation[:, 0]
            y_coords = x_equation[:, 1]
            x, y, t = torch.meshgrid(x_coords, y_coords, t_equation)
            self.x = torch.stack([x.flatten(), y.flatten()], dim=-1) 
            self.t = t.reshape(-1, 1)

    def get_equation_points(self):
        """
        Return the equation points.
        """
        return self.x.requires_grad_(True), self.t.requires_grad_(True)


class InitialConditions(MathConditions):
    def __init__(self):
        pass

    def set_initial_conditions(self, geom: Geometry, 
                               time: TimeDomain, 
                               initial_func: Callable, 
                               num_points: int, 
                               random: bool, 
                               device="cuda:0"):
        """
        Set up the initial conditions.

        Parameters:
            geom (Geometry): Class that stores the geometry domain data.
            time (TimeDomain): Class that stores the time domain data.
            initial_func (Callable): Function that stores the initial conditions.
            num_points (int): Number of points for initial conditions.
            random (bool): If True, random points in domain will be used.
            device (str): Device to use. Default is 'cuda:0'.
        """
        self.x = geom.inners(num_points, device=device, random=random)
        if geom.get_dimension() == 1:
            self.x = self.x.reshape(-1, 1)
        elif geom.get_dimension() == 2:
            self.x = self.x.reshape(-1, 2)
        self.t = time.initial(num_points, device=device, random=random)
        self.t = self.t[:self.x.shape[0]].reshape(-1, 1)
        self.u = initial_func(self.x).to(device).reshape(-1, 1)
        self.initial_func = initial_func
    
    def get_initial_conditions(self):
        """
        Return the initial conditions.
        """
        return self.x, self.t, self.u


class BoundaryConditions(MathConditions):
    def __init__(self):
        pass

    def set_boundary_conditions(self, geom: Geometry, 
                                time: TimeDomain, 
                                boundary_func: Callable, 
                                num_points: int, 
                                random: bool, 
                                device="cuda:0"):
        """
        Set up the boundary conditions.

        Parameters:
            geom (Geometry): Class that stores the geometry domain data.
            time (TimeDomain): Class that stores the time domain data.
            boundary_func (Callable): Function that stores the boundary conditions.
            num_points (int): Number of points for boundary conditions.
            random (bool): If True, random points in domain will be used.
            device (str): Device to use. Default is 'cuda:0'.
        """
        x_boundary = geom.boundary(int(np.sqrt(num_points)), 
                                   device=device, random=random)
        t_boundary = time.inners(int(np.sqrt(num_points)), 
                                 device=device, random=random)
        if x_boundary.dim() == 1:
            x, t = torch.meshgrid(x_boundary, t_boundary)
            self.x = x.reshape(-1, 1)
            self.t = t.reshape(-1, 1)

        if x_boundary.dim() == 2:
            # x_coords = x_boundary[:, 0]
            # y_coords = x_boundary[:, 1]
            # x, y, t = torch.meshgrid(x_coords, y_coords, t_boundary)
            # self.x = torch.stack([x.flatten(), y.flatten()], dim=-1)
            # self.t = t.flatten()

            t_boundary = t_boundary.view(-1, 1).expand(-1, x_boundary.size(0))
            x_boundary = x_boundary.repeat(t_boundary.size(0), 1, 1)
            self.x = x_boundary.view(-1, 2)
            self.t = t_boundary.reshape(-1, 1)

        self.u = boundary_func(self.x, self.t).to(device).reshape(-1, 1)
        self.boundary_func = boundary_func

    def get_boundary_conditions(self):
        """
        Return the boundary conditions.
        """
        return self.x, self.t, self.u


class Test(MathConditions):
    def __init__(self):
        pass
    
    def set_test(self, geom: Geometry, 
                 time: TimeDomain, 
                 num_points: int, 
                 random: bool, 
                 device="cuda:0"):
        """
        Set up the test conditions. Used final time.

        Parameters:
            geom (Geometry): Class that stores the geometry domain data.
            time (TimeDomain): Class that stores the time domain data.
            num_points (int): Number of points for test conditions.
            random (bool): If True, random points in domain will be used.
            device (str): Device to use. Default is 'cuda:0'.
        """
        self.x = geom.inners(num_points, 
                             device=device, 
                             random=random)
        if geom.get_dimension() == 1:
            self.x = self.x.reshape(-1, 1)
        elif geom.get_dimension() == 2:
            self.x = self.x.reshape(-1, 2)
        self.t = time.final(num_points, 
                            device=device, 
                            random=random)[:self.x.shape[0]].reshape(-1, 1)

    def get_test(self):
        """
        Return the test conditions.
        """
        return self.x, self.t


class Problem:
    def __init__(self, 
                 initial_conditions: InitialConditions, 
                 boundary_conditions: BoundaryConditions,
                 equation: Equation, 
                 test: Test, 
                 geom: Geometry, 
                 period: TimeDomain, 
                 alpha: float):
        """
        Initialize the problem.

        Parameters:
            initial_conditions (InitialConditions): Class that stores the initial conditions.
            boundary_conditions (BoundaryConditions): Class that stores the boundary conditions.
            equation (Equation): Class that stores the equation.
            test (Test): Class that stores the test conditions.
            geom (Geometry): Class that stores the geometry domain data.
            time (TimeDomain): Class that stores the time domain data.
            alpha (float): Coefficient of the thermal conductivity.
        """
        self.initial_conditions = initial_conditions
        self.boundary_conditions = boundary_conditions
        self.equation = equation
        self.test = test
        self.geom = geom
        self.period = period
        self.alpha = alpha

    def get_problem(self):
        pass

class Solution:
    def __init__(self):
        pass

    def set_solution(self, x: torch.Tensor, 
                     t: torch.Tensor, 
                     u: torch.Tensor):
        """
        Set up the solution.

        Parameters:
            x (torch.Tensor): Solution points.
            t (torch.Tensor): Solution times.
            u (torch.Tensor): Solution values.
        """
        self.x = x
        self.t = t
        self.u = u

    def get_solution(self):
        """
        Return the solution.
        """
        return self.x, self.t, self.u