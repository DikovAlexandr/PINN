import os
import torch
import numpy as np

class SirenParams:
    def __init__(self, first_omega_0, hidden_omega_0, outermost_linear):
        self.first_omega_0 = first_omega_0
        self.hidden_omega_0 = hidden_omega_0
        self.outermost_linear = outermost_linear

class SineLayer(torch.nn.Module):
    def __init__(self, input, output, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = input
        self.linear = torch.nn.Linear(input, output, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0) # Read paper for details
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
class Sin():
    def __init__(self):
        pass

    def __repr__(self):
        return 'Sin()'