import numpy as np
import torch
import matplotlib.pyplot as plt

class MathConditions:
    def __init__(self):
        self.x = None
        self.t = None
        self.u = None


class Equation(MathConditions):
    def __init__(self):
        pass

    def set_equation(self, geom, time, 
                     num_points, random, device="cuda:0"):
        x_equation = geom.inners(int(np.sqrt(num_points)), 
                                 device=device, random=random)
        t_equation = time.inners(int(np.sqrt(num_points)), 
                                 device=device, random=random)
        x, t = torch.meshgrid(x_equation, t_equation)
        self.x = x.flatten()
        self.t = t.flatten()

    def get_equation(self):
        return self.x.requires_grad_(True), self.t.requires_grad_(True)


class InitialConditions(MathConditions):
    def __init__(self):
        pass

    def set_initial_conditions(self, geom, time, initial_func, 
                               num_points, random, device="cuda:0"):
        self.x = geom.inners(num_points, device=device, random=random)
        self.t = time.initial(num_points, device=device, random=random)
        self.u = initial_func(self.x)
        self.initial_func = initial_func
    
    def get_initial_conditions(self):
        return self.x, self.t, self.u.unsqueeze(1)


class BoundaryConditions(MathConditions):
    def __init__(self):
        pass

    def set_boundary_conditions(self, geom, time, boundary_func, 
                                num_points, random, device="cuda:0"):
        x_boundary = geom.boundary(int(np.sqrt(num_points)), 
                                   device=device, random=random)
        t_boundary = time.inners(int(np.sqrt(num_points)), 
                                 device=device, random=random)
        x, t = torch.meshgrid(x_boundary, t_boundary)
        self.x = x.flatten()
        self.t = t.flatten()
        self.u = boundary_func(self.x, self.t)
        self.boundary_func = boundary_func

    def get_boundary_conditions(self):
        return self.x, self.t, self.u.unsqueeze(1)


class Test(MathConditions):
    def __init__(self):
        pass
    
    def set_test(self, geom, time, num_points, random, device="cuda:0"):
        self.x = geom.inners(num_points, device=device, random=random)
        self.t = time.final(num_points, device=device, random=random)

    def get_test(self):
        return self.x, self.t


class Problem:
    def __init__(self, initial_conditions, boundary_conditions,
                 equation, test, geom, time, alpha):
        self.initial_conditions = initial_conditions
        self.boundary_conditions = boundary_conditions
        self.equation = equation
        self.test = test
        self.geom = geom
        self.time = time
        self.alpha = alpha

    def get_problem(self):
        pass

class Solution:
    def __init__(self):
        pass

    def set_solution(self, x, t, u):
        self.x = x
        self.t = t
        self.u = u

    def get_solution(self):
        return self.x, self.t, self.u