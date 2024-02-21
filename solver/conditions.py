import numpy as np
import torch

class MathConditions:
    def __init__(self, x, t, u):
        self.x = x
        self.t = t
        self.u = u


class Equation(MathConditions):
    def __init__(self, x, t):
        super().__init__(x, t, None)

    def set_equation(self, geom, time, num_points, random, device="cuda:0"):
        x_equation = geom.inners(int(np.sqrt(num_points)), device=device, random=random)
        t_equation = time.inners(int(np.sqrt(num_points)), device=device, random=random)
        x, t = torch.meshgrid(x_equation, t_equation)
        self.x = x.flatten()
        self.t = t.flatten()


class InitialConditions(MathConditions):
    def __init__(self, x, t, u):
        super().__init__(x, t, u)

    def set_initial_conditions(self, geom, time, initial_func, num_points, random, device="cuda:0"):
        self.x = geom.inners(num_points, device=device, random=random)
        self.t = time.initial(num_points, device=device, random=random)
        self.u = initial_func(self.x, self.t)


class BoundaryConditions(MathConditions):
    def __init__(self, x, t, u):
        super().__init__(x, t, u)

    def set_boundary_conditions(self, geom, time, boundary_func, num_points, random, device="cuda:0"):
        self.x = geom.boundaries(num_points, device=device, random=random)
        self.t = time.inners(num_points, device=device, random=random)
        self.u = boundary_func(self.x, self.t)


class Test(MathConditions):
    def __init__(self, x, t):
        super().__init__(x, t, None)
    
    def set_test(self, geom, time, num_points, random, device="cuda:0"):
        self.x = geom.inners(num_points, device=device, random=random)
        self.t = time.final(num_points, device=device, random=random)


class Problem:
    def __init__(self, initial_conditions, boundary_conditions, equation):
        self.initial_conditions = initial_conditions
        self.boundary_conditions = boundary_conditions
        self.equation = equation