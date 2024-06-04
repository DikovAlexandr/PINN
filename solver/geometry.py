from abc import ABC, abstractmethod
import numpy as np
import torch
from typing import Tuple, Union, List

from .utils import split_number


class Geometry(ABC):
    @abstractmethod
    def inside(self) -> bool:
        raise NotImplementedError
    
    @abstractmethod
    def boundary(self) -> torch.Tensor:
        raise NotImplementedError
    
    @abstractmethod
    def get_boundary(self) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod       
    def add_boundary(self) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def inners(self) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod   
    def get_inners(self) -> torch.Tensor:
        raise NotImplementedError
    
    @abstractmethod
    def add_inners(self, new_inners_points) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def get_dimension(self) -> int:
        raise NotImplementedError
    
    @abstractmethod
    def grid_spacing_inners(self) -> float:
        raise NotImplementedError
    

class Geometry1D(Geometry):
    @abstractmethod
    def length(self) -> float:
        pass
    
    @abstractmethod
    def limits(self) -> Tuple[float, float]:
        pass

    def get_dimension(self) -> int:
        return 1


class Geometry2D(Geometry):
    @abstractmethod
    def inside(self) -> bool:
        pass
    
    @abstractmethod
    def limits(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        pass
    
    @abstractmethod
    def grid_spacing_boundary(self) -> float:
        pass

    def get_dimension(self) -> int:
        return 2


class Interval(Geometry1D):
    def __init__(self, x_left, x_right):
        """
        Parameters:
            x_left (float): Left boundary.
            x_right (float): Right boundary.
        """
        self.x_left = x_left
        self.x_right = x_right
        self.spacing = None

        self.boundary_points = None
        self.inners_points = None

    def __str__(self) -> str:
        return f"Interval({self.x_left}, {self.x_right})"
    
    def length(self) -> float:
        """
        Returns the length of the interval.
        """
        return self.x_right - self.x_left
    
    def limits(self) -> Tuple[float, float]:
        """
        Returns the limits of the interval as a tuple.
        """
        return (self.x_left, self.x_right)
    
    def inside(self, x: Union[float, list, torch.Tensor]) -> Union[bool, torch.Tensor]:
        """
        Returns True if the point is inside the interval.
        """
        if isinstance(x, float):
            return (x >= self.x_left) & (x <= self.x_right)
        elif isinstance(x, list) or isinstance(x, torch.Tensor):
            return (x[0] >= self.x_left) & (x[0] <= self.x_right)
        else:
            raise TypeError("Unsupported type for x")

    def boundary(self, num_points: int, 
                 device="cuda:0", random=False) -> torch.Tensor:
        """
        Set the boundary points.

        Parameters:
            num_points (int): Number of points to set.
            device (str, optional): Device on which to set the points. Defaults to "cuda:0".
            random (bool, optional): If True, set the points randomly. Defaults to False.
        """
        x_boundary_right = torch.ones(int(num_points/2), device=device) * self.x_right
        x_boundary_left = torch.ones(int(num_points/2), device=device) * self.x_left
        self.boundary_points = torch.cat([x_boundary_right, x_boundary_left])
        return self.boundary_points
    
    def get_boundary(self) -> torch.Tensor:
        """
        Returns the boundary points.
        """
        return self.boundary_points
    
    def add_boundary(self, new_boundary_points: torch.Tensor) -> None:
        """
        Adds new boundary points.
        """
        self.boundary_points = torch.cat([self.boundary_points, new_boundary_points])
    
    def inners(self, num_points: int, 
               device="cuda:0", random=False) -> torch.Tensor:
        """
        Set the inner points.

        Parameters: 
            num_points (int): Number of points to set.
            device (str, optional): Device on which to set the points. Defaults to "cuda:0".
            random (bool, optional): If True, set the points randomly. Defaults to False.
        """
        if random:
            self.inners_points = (torch.rand(num_points, device=device) * 
                                  (self.x_right - self.x_left) + self.x_left)
        else:
            self.inners_points = torch.linspace(self.x_left, self.x_right, 
                                                num_points, device=device)
            self.spacing = (self.inners_points[1] - self.inners_points[0]).item()
        return self.inners_points
    
    def get_inners(self) -> torch.Tensor:
        """
        Returns the inner points.
        """                
        return self.inners_points
    
    def add_inners(self, new_inners_points: torch.Tensor) -> None:
        """
        Adds new inner points.
        """
        self.inners_points = torch.cat([self.inners_points, new_inners_points])
    
    def grid_spacing_inners(self) -> float:
        """
        Returns the grid spacing.
        """                
        return self.spacing


class Rectangle(Geometry2D):
    def __init__(self, 
                 x_min, x_max,
                 y_min, y_max):
        """
        Parameters:
            x_min (float): Left boundary x coordinate.
            x_max (float): Right boundary x coordinate.
            y_min (float): Bottom boundary y coordinate.
            y_max (float): Top boundary y coordinate.
        """
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
    
    def size(self) -> Tuple[float, float]:
        """
        Returns the size of the rectangle as a tuple.
        """
        return (self.x_max - self.x_min, self.y_max - self.y_min)
    
    def limits(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Returns the limits of the rectangle as a tuple in the form (left, right) and (bottom, top).
        """ 
        return ((self.x_min, self.x_max), (self.y_min, self.y_max))

    def inside(self, x: Union[np.ndarray, List[float], Tuple[float, ...], torch.Tensor]) -> Union[bool, torch.Tensor]:
        """
        Returns True if the point is inside the rectangle.
        """
        if isinstance(x, torch.Tensor):
            return ((x[0] >= self.x_min) & (x[0] <= self.x_max) & 
                    (x[1] >= self.y_min) & (x[1] <= self.y_max))
        else:
            x = np.asarray(x)
            return ((x[0] >= self.x_min) & (x[0] <= self.x_max) & 
                    (x[1] >= self.y_min) & (x[1] <= self.y_max))

    def boundary(self, num_points: int, 
                 device="cuda:0", random=False) -> torch.Tensor:
        """
        Set the boundary points.

        Parameters:
            num_points (int): Number of points to set.
            device (str, optional): Device on which to set the points. Defaults to "cuda:0".
            random (bool, optional): If True, set the points randomly. Defaults to False.
        """
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
            
            # Concatenate
            self.boundary_points = torch.cat((top, right, bottom, left))
        return self.boundary_points
    
    def get_boundary(self) -> torch.Tensor:
        """
        Returns the boundary points.
        """
        return self.boundary_points
    
    def add_boundary(self, new_boundary_points: torch.Tensor) -> None:
        """
        Adds new boundary points to the boundary.
        """
        self.boundary_points = torch.cat((self.boundary_points, new_boundary_points))
    
    def inners(self, num_points: int, 
               device="cuda:0", random=False) -> torch.Tensor:
        """
        Set the inner points.

        Parameters:
            num_points (int): Number of points to set.
            device (str, optional): Device on which to set the points. Defaults to "cuda:0".
            random (bool, optional): If True, set the points randomly. Defaults to False.
        """
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
        return self.inners_points
    
    def get_inners(self) -> torch.Tensor:
        """
        Returns the inner points.
        """
        return self.inners_points
    
    def add_inners(self, new_inners_points: torch.Tensor) -> None:
        """
        Adds new inner points to the inner points.
        """
        self.inners_points = torch.cat((self.inners_points, new_inners_points))
        
    def grid_spacing_boundary(self) -> Tuple[float, float]:
        """
        Returns the spacing of the boundary points.
        """
        return self.spacing_boundaries
    
    def grid_spacing_inners(self) -> Tuple[float, float]:
        """
        Returns the spacing of the inner points.
        """
        return self.spacing_inners


class Ellipse(Geometry2D):
    def __init__(self, x_center, y_center, x_radius, y_radius):
        """
        Parameters:
            x_center (float): x coordinate of the ellipses center point.
            y_center (float): y coordinate of the ellipses center point.
            x_radius (float): Semi-major axis of the ellipse (along the x-axis).
            y_radius (float): Semi-minor axis of the ellipse (along the y-axis).
        """
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
    
    def limits(self) -> Tuple[Tuple[float]]:
        """
        Returns the limits of the ellipse in the form (x_min, x_max, y_min, y_max).
        """
        return ((self.x_center - self.x_radius, self.x_center + self.x_radius),
                (self.y_center - self.y_radius, self.y_center + self.y_radius))

    def inside(self, x: Union[np.ndarray, List[float], Tuple[float, ...], torch.Tensor]) -> Union[bool, torch.Tensor]:
        """
        Returns True if x is inside the ellipse.
        """
        if isinstance(x, torch.Tensor):
            return (((x[0] - self.x_center) / self.x_radius) ** 2 + 
                    ((x[1] - self.y_center) / self.y_radius) ** 2 <= 1)
        else:
            x = np.asarray(x)
            return (((x[0] - self.x_center) / self.x_radius) ** 2 + 
                    ((x[1] - self.y_center) / self.y_radius) ** 2 <= 1)

    def boundary(self, num_points: int, 
                 device="cuda:0", random=False) -> torch.Tensor:
        """
        Set the boundary points.

        Parameters:
            num_points (int): Number of points to set.
            device (str, optional): Device on which to set the points. Defaults to "cuda:0".
            random (bool, optional): If True, set the points randomly. Defaults to False.
        """
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
        return self.boundary_points
    
    def get_boundary(self) -> torch.Tensor:
        """
        Returns the boundary points.
        """
        return self.boundary_points
    
    def add_boundary(self, new_boundary_points) -> None:
        """
        Adds new boundary points to the boundary points.
        """     
        self.boundary_points = torch.cat((self.boundary_points, new_boundary_points))
        
    def inners(self, num_points: int, 
               device="cuda:0", random=False) -> torch.Tensor:
        """
        Set the inner points.

        Parameters:
            num_points (int): Number of points to set.
            device (str, optional): Device on which to set the points. Defaults to "cuda:0".
            random (bool, optional): If True, set the points randomly. Defaults to False.
        """
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
        return self.inners_points
    
    def get_inners(self) -> torch.Tensor:
        """
        Returns the inner points.
        """
        return self.inners_points
    
    def add_inners(self, new_inners_points) -> None:
        """
        Adds new inner points to the inner points.
        """
        self.inners_points = torch.cat([self.inners_points, new_inners_points])
        
    def grid_spacing_boundary(self):
        """
        Returns the spacing of the boundary points.
        """
        return self.spacing_boundaries
    
    def grid_spacing_inners(self):
        """
        Returns the spacing of the inner points.
        """
        return self.spacing_inners


class Circle(Ellipse):
    def __init__(self, x_center, y_center, radius):
        """
        Parameters:
            x_center (float): x-coordinate of the center.
            y_center (float): y-coordinate of the center.
            radius (float): radius of the circle.
        """
        super().__init__(x_center, y_center, radius, radius)