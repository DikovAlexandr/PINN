import numpy as np
import torch

from .utils import split_number

class Geometry1D:
    def length(self) -> float:
        pass
    
    def inside(self, x) -> bool:
        pass

    def boundary(self, num_points, 
                 device="cuda:0", random=False) -> torch.Tensor:
        pass

    def inners(self, num_points, 
               device="cuda:0", random=False) -> torch.Tensor:
        pass

    def grid_spacing(self) -> float:
        pass

class Geometry2D:
    def inside(self, x) -> bool:
        pass

    def boundary(self, num_points, 
                 device="cuda:0", random=False) -> torch.Tensor:
        pass

    def inners(self, num_points, 
               device="cuda:0", random=False) -> torch.Tensor:
        pass

    def grid_spacing(self) -> float:
        pass

class Interval(Geometry1D):
    def __init__(self, x_left, x_right):
        self.x_left = x_left
        self.x_right = x_right
        self.spacing = None

    def __str__(self):
        return f"Interval({self.x_left}, {self.x_right})"
    
    def length(self):
        return self.x_right - self.x_left
    
    def inside(self, x):
        return (x >= self.x_left) & (x <= self.x_right)

    def boundary(self, num_points, device="cuda:0", random=False):
        x_boundary_right = torch.ones(num_points, device=device) * self.x_right
        x_boundary_left = torch.ones(num_points, device=device) * self.x_left
        return torch.cat([x_boundary_right, x_boundary_left])
    
    def inners(self, num_points, device="cuda:0", random=False):
        if random:
            x_inners = (torch.rand(num_points, device=device) * 
                        (self.x_right - self.x_left) + self.x_left)
        else:
            x_inners = torch.linspace(self.x_left, self.x_right, num_points, device=device)
            self.spacing = (x_inners[1] - x_inners[0]).item()
        return x_inners
    
    def grid_spacing_inners(self):
        return self.spacing


class Rectangle(Geometry2D):
    def __init__(self, x_min, x_max, y_min, y_max):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.aspect_ratio = (x_max - x_min) / (y_max - y_min)
        self.spacing_boundaries = None
        self.spacing_inners = None

    def __str__(self):
        return (f"Rectangle({self.x_min}, {self.x_max}, "
                f"{self.y_min}, {self.y_max})")
    
    def size (self):
        return (self.x_max - self.x_min, self.y_max - self.y_min)

    def inside(self, x):
        x = np.asarray(x)
        return ((x[0] >= self.x_min) & (x[0] <= self.x_max) & 
                (x[1] >= self.y_min) & (x[1] <= self.y_max))

    def boundary(self, num_points, device="cuda:0", random=False):
        if random:
            top_boundary, bottom_boundary, left_boundary, right_boundary = split_number(num_points)
            # Top
            x_boundary = (torch.rand(top_boundary, device=device) * 
                            (self.x_max - self.x_min) + self.x_min)
            y_boundary_top = torch.ones(top_boundary, device=device) * self.y_max
            top = torch.stack([x_boundary, y_boundary_top], dim=1)

            # Bottom
            x_boundary = (torch.rand(bottom_boundary, device=device) * 
                            (self.x_max - self.x_min) + self.x_min)
            y_boundary_bottom = torch.ones(bottom_boundary, device=device) * self.y_min
            bottom = torch.stack([x_boundary, y_boundary_bottom], dim=1)

            # Left
            y_boundary = (torch.rand(left_boundary, device=device) * 
                            (self.y_max - self.y_min) + self.y_min)
            x_boundary_left = torch.ones(left_boundary, device=device) * self.x_min
            left = torch.stack([x_boundary_left, y_boundary], dim=1)

            # Right
            y_boundary = (torch.rand(right_boundary, device=device) * 
                            (self.y_max - self.y_min) + self.y_min)
            x_boundary_right = torch.ones(right_boundary, device=device) * self.x_max
            right = torch.stack([x_boundary_right, y_boundary], dim=1)

            return torch.cat((top, right, bottom, left))
        else:
            # Top and bottom
            x_boundary = torch.linspace(self.x_min, self.x_max, 
                                        int(num_points/4 * self.aspect_ratio), device=device)
            y_boundary_top = torch.ones(int(num_points/4 * self.aspect_ratio), device=device) * self.y_max
            y_boundary_bottom = torch.ones(int(num_points/4 * self.aspect_ratio), device=device) * self.y_min
            top = torch.stack([x_boundary, y_boundary_top], dim=1)
            bottom = torch.stack([x_boundary, y_boundary_bottom], dim=1)

            # Left and right
            y_boundary = torch.linspace(self.y_min, self.y_max, 
                                        int(num_points/4 / self.aspect_ratio), device=device)
            x_boundary_left = torch.ones(int(num_points/4 / self.aspect_ratio), device=device) * self.x_min
            x_boundary_right = torch.ones(int(num_points/4 / self.aspect_ratio), device=device) * self.x_max
            left = torch.stack([x_boundary_left, y_boundary], dim=1)
            right = torch.stack([x_boundary_right, y_boundary], dim=1)

            # Calculate spacing
            self.spacing_boundaries = ((x_boundary[1] - x_boundary[0]).item(),
                                       (y_boundary[1] - y_boundary[0]).item())
            return torch.cat((top, right, bottom, left))
    
    def inners(self, num_points, device="cuda:0", random=False):
        if random:
            x_inners = (torch.rand(num_points, device=device) * 
                        (self.x_max - self.x_min) + self.x_min)
            y_inners = (torch.rand(num_points, device=device) * 
                        (self.y_max - self.y_min) + self.y_min)
            return torch.stack([x_inners, y_inners], dim=1)
        else:
            x_inners = torch.linspace(self.x_min, self.x_max, 
                                      int(np.sqrt(num_points) * self.aspect_ratio), device=device)
            y_inners = torch.linspace(self.y_min, self.y_max, 
                                      int(np.sqrt(num_points) / self.aspect_ratio), device=device)
            X, Y = torch.meshgrid(x_inners, y_inners)
            X = X.reshape(-1, 1)
            Y = Y.reshape(-1, 1)

            # Calculate spacing
            self.spacing_inners = ((x_inners[1] - x_inners[0]).item(),
                                   (y_inners[1] - y_inners[0]).item())
            return torch.cat([X, Y], dim=1)
        
    def grid_spacing_boundary(self):
        return self.spacing_boundaries
    
    def grid_spacing_inners(self):
        return self.spacing_inners

class Ellipse(Geometry2D):
    def __init__(self, x_center, y_center, x_radius, y_radius):
        self.x_center = x_center
        self.y_center = y_center
        self.x_radius = x_radius
        self.y_radius = y_radius
        self.spacing_boundaries = None
        self.spacing_inners = None

    def __str__(self):
        return (f"Ellipse(x_center={self.x_center}, y_center={self.y_center}, "
                f"x_radius={self.x_radius}, y_radius={self.y_radius})")

    def inside(self, x):
        x = np.asarray(x)
        return (((x[0] - self.x_center) / self.x_radius) ** 2 + 
                ((x[1] - self.y_center) / self.y_radius) ** 2 <= 1)

    def boundary(self, num_points, device="cuda:0", random=False):
        if random:
            theta = torch.rand(num_points, device=device) * 2 * np.pi
            x_boundary = self.x_center + self.x_radius * torch.cos(theta)
            y_boundary = self.y_center + self.y_radius * torch.sin(theta)
            return torch.stack([x_boundary, y_boundary], dim=1)
        else:
            theta = torch.linspace(0, 2 * np.pi, num_points, device=device)
            x_boundary = self.x_center + self.x_radius * torch.cos(theta)
            y_boundary = self.y_center + self.y_radius * torch.sin(theta)

            # Calculate spacing
            # TODO: check if this is correct, spacing may depend on theta
            self.spacing_boundaries = ((x_boundary[1] - x_boundary[0]).item(),
                                       (y_boundary[1] - y_boundary[0]).item())
            return torch.stack([x_boundary, y_boundary], dim=1)
        
    def inners(self, num_points, device="cuda:0", random=False):
        if random:
            theta = torch.rand(num_points, device=device) * 2 * np.pi
            r = torch.sqrt(torch.rand(num_points, device=device)) * self.x_radius
            x_interior = self.x_center + r * torch.cos(theta)
            y_interior = self.y_center + r * torch.sin(theta)
            return torch.stack([x_interior, y_interior], dim=1)
        else:
            x_inners = torch.linspace(self.x_center - self.x_radius, self.x_center + self.x_radius, int(np.sqrt(num_points)), device=device)
            y_inners = torch.linspace(self.y_center - self.y_radius, self.y_center + self.y_radius, int(np.sqrt(num_points)), device=device)
            X, Y = torch.meshgrid(x_inners, y_inners)
            x = X.reshape(-1, 1)
            y = Y.reshape(-1, 1)
            points = torch.stack([x, y], dim=1)
            mask = self.inside(points)

            # Calculate spacing
            self.spacing_inners = ((x_inners[1] - x_inners[0]).item(),
                                   (y_inners[1] - y_inners[0]).item())
            return points[mask.squeeze()]
        
    def grid_spacing_boundary(self):
        return self.spacing_boundaries
    
    def grid_spacing_inners(self):
        return self.spacing_inners

class Circle(Ellipse):
    def __init__(self, x_center, y_center, radius):
        super().__init__(x_center, y_center, radius, radius)