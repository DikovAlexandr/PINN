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
        return self.x, self.t


class InitialConditions(MathConditions):
    def __init__(self):
        pass

    def set_initial_conditions(self, geom, time, initial_func, 
                               num_points, random, device="cuda:0"):
        self.x = geom.inners(num_points, device=device, random=random)
        self.t = time.initial(num_points, device=device, random=random)
        self.u = initial_func(self.x)
    
    def get_initial_conditions(self):
        return self.x, self.t, self.u


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

    def get_boundary_conditions(self):
        return self.x, self.t, self.u


class Test(MathConditions):
    def __init__(self):
        pass
    
    def set_test(self, geom, time, num_points, random, device="cuda:0"):
        self.x = geom.inners(num_points, device=device, random=random)
        self.t = time.final(num_points, device=device, random=random)

    def get_test(self):
        return self.x, self.t


class Problem:
    def __init__(self, initial_conditions, boundary_conditions, test,
                 equation, geom, time, alpha):
        self.initial_conditions = initial_conditions
        self.boundary_conditions = boundary_conditions
        self.equation = equation
        self.test = test
        self.geom = geom
        self.time = time
        self.alpha = alpha
        
    def to_numpy(self, tensor):
        return tensor.cpu().detach().numpy()

    def visualize(self, t, output_folder=None):
        x_ic, _, u_ic = self.initial_conditions.get_initial_conditions()
        x_bc, t_bc, u_bc = self.boundary_conditions.get_boundary_conditions()
        time_index = self.find_index(t_bc, t)

        if len(x_ic.shape) == 1:
            # 1D problem
            if t == 0:
                plt.scatter(self.to_numpy(x_ic),
                            self.to_numpy(u_ic),
                            marker='o', label="IC")
                plt.scatter(self.to_numpy(x_bc[time_index]),
                            self.to_numpy(u_bc[time_index]),
                            marker='o', label="BC")
            else:
                plt.scatter(self.to_numpy(x_bc[time_index]),
                            self.to_numpy(u_bc[time_index]),
                            marker='o', label="BC")
            plt.xlim(self.geom.limits()[0], self.geom.limits()[1])
            plt.ylim(0, max(self.to_numpy(u_ic).max(),
                            self.to_numpy(u_bc).max()))
            plt.xlabel("x")
            plt.ylabel("u")
        else:
            # 2D problem
            if t == 0:
                plt.scatter(self.to_numpy(x_ic[:, 0]),
                            self.to_numpy(x_ic[:, 1]),
                            c=u_ic, cmap='viridis',
                            marker='o', label="IC")
                plt.scatter(self.to_numpy(x_bc[:, 0][time_index]),
                            self.to_numpy(x_bc[:, 1][time_index]),
                            c=u_bc[time_index], cmap='viridis',
                            marker='o', label="BC")
            else:
                plt.scatter(self.to_numpy(x_bc[:, 0][time_index]),
                            self.to_numpy(x_bc[:, 1][time_index]),
                            c=u_bc[time_index], cmap='viridis',
                            marker='o', label="BC")
            plt.xlim(self.geom.limits()[0][0], 
                     self.geom.limits()[0][1])
            plt.ylim(self.geom.limits()[1][0], 
                     self.geom.limits()[1][1])
            plt.colorbar(label='u')
            plt.xlabel("x")
            plt.ylabel("y")
        plt.legend()

        # Save the figure or display it
        if output_folder:
            plt.savefig(output_folder)
            plt.show()
        else:
            plt.show()

    def find_index(self, array, value):
        return torch.where(torch.abs(array - value) <= 0.5 * self.time.grid_spacing_inners())[0]