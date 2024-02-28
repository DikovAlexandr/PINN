import numpy as np
import torch
from typing import Tuple

from .utils import split_number

class Geometry1D:
    def length(self) -> float:
        pass
    
    def inside(self, x) -> bool:
        pass

    def boundary(self, num_points, 
                 device="cuda:0", random=False) -> torch.Tensor:
        pass

    def get_boundary(self) -> torch.Tensor:
        pass

    def add_boundary(self, new_boundary_points) -> None:
        pass

    def inners(self, num_points, 
               device="cuda:0", random=False) -> torch.Tensor:
        pass

    def get_inners(self) -> torch.Tensor:
        pass

    def add_inners(self, new_inners_points) -> None:
        pass

    def grid_spacing(self) -> float:
        pass

class Geometry2D:
    def inside(self, x) -> bool:
        pass

    def boundary(self, num_points, 
                 device="cuda:0", random=False) -> torch.Tensor:
        pass

    def get_boundary(self) -> torch.Tensor:
        pass

    def add_boundary(self, new_boundary_points) -> None:
        pass

    def inners(self, num_points, 
               device="cuda:0", random=False) -> torch.Tensor:
        pass

    def get_inners(self) -> torch.Tensor:
        pass

    def add_inners(self, new_inners_points) -> None:
        pass

    def grid_spacing(self) -> float:
        pass

class Interval(Geometry1D):
    def __init__(self, x_left, x_right):
        self.x_left = x_left
        self.x_right = x_right
        self.spacing = None

        self.boundary_points = None
        self.inners_points = None

    def __str__(self) -> str:
        return f"Interval({self.x_left}, {self.x_right})"
    
    def length(self) -> float:
        return self.x_right - self.x_left
    
    def inside(self, x) -> bool:
        return (x >= self.x_left) & (x <= self.x_right)

    def boundary(self, num_points, device="cuda:0", random=False) -> torch.Tensor:
        x_boundary_right = torch.ones(num_points, device=device) * self.x_right
        x_boundary_left = torch.ones(num_points, device=device) * self.x_left
        self.boundary_points = torch.cat([x_boundary_right, x_boundary_left])
        return self.boundary_points.requires_grad_(True)
    
    def get_boundary(self) -> torch.Tensor:
        return self.boundary_points.requires_grad_(True)
    
    def add_boundary(self, new_boundary_points) -> None:
        self.boundary_points = torch.cat([self.boundary_points, new_boundary_points])
    
    def inners(self, num_points, device="cuda:0", random=False) -> torch.Tensor:
        if random:
            self.inners_points = (torch.rand(num_points, device=device) * 
                        (self.x_right - self.x_left) + self.x_left)
        else:
            self.inners_points = torch.linspace(self.x_left, self.x_right, 
                                                num_points, device=device)
            self.spacing = (self.inners_points[1] - self.inners_points[0]).item()
        return self.inners_points.requires_grad_(True)
    
    def get_inners(self) -> torch.Tensor:
        return self.inners_points.requires_grad_(True)
    
    def add_inners(self, new_inners_points) -> None:
        self.inners_points = torch.cat([self.inners_points, new_inners_points])
    
    def grid_spacing_inners(self) -> float:
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

        self.boundary_points = None
        self.inners_points = None

    def __str__(self) -> str:
        return (f"Rectangle({self.x_min}, {self.x_max}, "
                f"{self.y_min}, {self.y_max})")
    
    def size (self) -> Tuple[float, float]:
        return (self.x_max - self.x_min, self.y_max - self.y_min)

    def inside(self, x) -> bool:
        x = np.asarray(x)
        return ((x[0] >= self.x_min) & (x[0] <= self.x_max) & 
                (x[1] >= self.y_min) & (x[1] <= self.y_max))

    def boundary(self, num_points, device="cuda:0", random=False) -> torch.Tensor:
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

            self.boundary_points = torch.cat((top, right, bottom, left))
        else:
            # Top and bottom
            x_boundary = torch.linspace(self.x_min, self.x_max, 
                                        int(num_points/4 * self.aspect_ratio), 
                                        device=device)
            y_boundary_top = torch.ones(int(num_points/4 * self.aspect_ratio), 
                                        device=device) * self.y_max
            y_boundary_bottom = torch.ones(int(num_points/4 * self.aspect_ratio), 
                                           device=device) * self.y_min
            top = torch.stack([x_boundary, y_boundary_top], dim=1)
            bottom = torch.stack([x_boundary, y_boundary_bottom], dim=1)

            # Left and right
            y_boundary = torch.linspace(self.y_min, self.y_max, 
                                        int(num_points/4 / self.aspect_ratio), 
                                        device=device)
            x_boundary_left = torch.ones(int(num_points/4 / self.aspect_ratio), 
                                         device=device) * self.x_min
            x_boundary_right = torch.ones(int(num_points/4 / self.aspect_ratio), 
                                          device=device) * self.x_max
            left = torch.stack([x_boundary_left, y_boundary], dim=1)
            right = torch.stack([x_boundary_right, y_boundary], dim=1)

            # Calculate spacing
            self.spacing_boundaries = ((x_boundary[1] - x_boundary[0]).item(),
                                       (y_boundary[1] - y_boundary[0]).item())
            self.boundary_points = torch.cat((top, right, bottom, left))
        return self.boundary_points.requires_grad_(True)
    
    def get_boundary(self) -> torch.Tensor:
        return self.boundary_points.requires_grad_(True)
    
    def add_boundary(self, new_boundary_points) -> None:
        self.boundary_points = torch.cat((self.boundary_points, new_boundary_points))
    
    def inners(self, num_points, device="cuda:0", random=False) -> torch.Tensor:
        if random:
            x_inners = (torch.rand(num_points, device=device) * 
                        (self.x_max - self.x_min) + self.x_min)
            y_inners = (torch.rand(num_points, device=device) * 
                        (self.y_max - self.y_min) + self.y_min)
            self.inners_points = torch.stack([x_inners, y_inners], dim=1)
        else:
            x_inners = torch.linspace(self.x_min, self.x_max, 
                                      int(np.sqrt(num_points) * self.aspect_ratio), 
                                      device=device)
            y_inners = torch.linspace(self.y_min, self.y_max, 
                                      int(np.sqrt(num_points) / self.aspect_ratio), 
                                      device=device)
            X, Y = torch.meshgrid(x_inners, y_inners)
            X = X.reshape(-1, 1)
            Y = Y.reshape(-1, 1)

            # Calculate spacing
            self.spacing_inners = ((x_inners[1] - x_inners[0]).item(),
                                   (y_inners[1] - y_inners[0]).item())
            self.inners_points = torch.cat([X, Y], dim=1)
        return self.inners_points.requires_grad_(True)
    
    def get_inners(self) -> torch.Tensor:
        return self.inners_points.requires_grad_(True)
    
    def add_inners(self, new_inners_points) -> None:
        self.inners_points = torch.cat((self.inners_points, new_inners_points))
        
    def grid_spacing_boundary(self) -> Tuple[float, float]:
        return self.spacing_boundaries
    
    def grid_spacing_inners(self) -> Tuple[float, float]:
        return self.spacing_inners

class Ellipse(Geometry2D):
    def __init__(self, x_center, y_center, x_radius, y_radius):
        self.x_center = x_center
        self.y_center = y_center
        self.x_radius = x_radius
        self.y_radius = y_radius
        self.spacing_boundaries = None
        self.spacing_inners = None

        self.boundary_points = None
        self.inners_points = None

    def __str__(self) -> str:
        return (f"Ellipse(x_center={self.x_center}, y_center={self.y_center}, "
                f"x_radius={self.x_radius}, y_radius={self.y_radius})")

    def inside(self, x) -> bool:
        x = np.asarray(x)
        return (((x[0] - self.x_center) / self.x_radius) ** 2 + 
                ((x[1] - self.y_center) / self.y_radius) ** 2 <= 1)

    def boundary(self, num_points, device="cuda:0", random=False) -> torch.Tensor:
        if random:
            theta = torch.rand(num_points, device=device) * 2 * np.pi
            x_boundary = self.x_center + self.x_radius * torch.cos(theta)
            y_boundary = self.y_center + self.y_radius * torch.sin(theta)
            self.boundary_points = torch.stack([x_boundary, y_boundary], dim=1)
        else:
            theta = torch.linspace(0, 2 * np.pi, num_points, device=device)
            x_boundary = self.x_center + self.x_radius * torch.cos(theta)
            y_boundary = self.y_center + self.y_radius * torch.sin(theta)

            # Calculate spacing
            # TODO: check if this is correct, spacing may depend on theta
            self.spacing_boundaries = ((x_boundary[1] - x_boundary[0]).item(),
                                       (y_boundary[1] - y_boundary[0]).item())
            self.boundary_points = torch.stack([x_boundary, y_boundary], dim=1)
        return self.boundary_points.requires_grad_(True)
    
    def get_boundary(self) -> torch.Tensor:
        return self.boundary_points.requires_grad_(True)
    
    def add_boundary(self, new_boundary_points) -> None:
        self.boundary_points = torch.cat((self.boundary_points, new_boundary_points))
        
    def inners(self, num_points, device="cuda:0", random=False) -> torch.Tensor:
        if random:
            theta = torch.rand(num_points, device=device) * 2 * np.pi
            r = torch.sqrt(torch.rand(num_points, device=device)) * self.x_radius
            x_interior = self.x_center + r * torch.cos(theta)
            y_interior = self.y_center + r * torch.sin(theta)
            self.inners_points = torch.stack([x_interior, y_interior], dim=1)
        else:
            x_inners = torch.linspace(self.x_center - self.x_radius, 
                                      self.x_center + self.x_radius, 
                                      int(np.sqrt(num_points)), device=device)
            y_inners = torch.linspace(self.y_center - self.y_radius, 
                                      self.y_center + self.y_radius, 
                                      int(np.sqrt(num_points)), device=device)
            X, Y = torch.meshgrid(x_inners, y_inners)
            x = X.reshape(-1, 1)
            y = Y.reshape(-1, 1)
            points = torch.stack([x, y], dim=1)
            mask = self.inside(points)

            # Calculate spacing
            self.spacing_inners = ((x_inners[1] - x_inners[0]).item(),
                                   (y_inners[1] - y_inners[0]).item())
            self.inners_points = points[mask.squeeze()]
        return self.inners_points.requires_grad_(True)
    
    def get_inners(self) -> torch.Tensor:
        return self.inners_points.requires_grad_(True)
    
    def add_inners(self, new_inners_points) -> None:
        self.inners_points = torch.cat([self.inners_points, new_inners_points])
        
    def grid_spacing_boundary(self):
        return self.spacing_boundaries
    
    def grid_spacing_inners(self):
        return self.spacing_inners

class Circle(Ellipse):
    def __init__(self, x_center, y_center, radius):
        super().__init__(x_center, y_center, radius, radius)