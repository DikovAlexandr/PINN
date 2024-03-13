import torch
import numpy as np


class TimeDomain:
    def __init__(self, t0, t1):
        self.t0 = t0
        self.t1 = t1
        self.spacing = None

    def __str__(self):
        return f"TimeDomain({self.t0}, {self.t1})"
    
    def inside(self, t):
        return (t >= self.t0) & (t <= self.t1)
    
    def initial(self, num_points, device="cuda:0", random=False): 
        return torch.ones(num_points, device=device) * self.t0
    
    def final(self, num_points, device="cuda:0", random=False):
        return torch.ones(num_points, device=device) * self.t1
    
    def inners(self, num_points, device="cuda:0", random=False):
        if random:
            t_inners = (torch.rand(num_points, device=device) * 
                        (self.t1 - self.t0) + self.t0)
        else:
            t_inners = torch.linspace(self.t0, self.t1, 
                                      num_points, device=device)

            # Calculate spacing
            self.spacing = (t_inners[1] - t_inners[0]).item()
        return t_inners
    
    def grid_spacing_inners(self):
        return self.spacing