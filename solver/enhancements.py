import numpy as np
import torch

from solver.geometry import Geometry
from solver.timedomain import TimeDomain

# Class for loss weight adjustment based on the loss values
class LossWeightAdjuster:
    def __init__(self, max_weight, min_weight, threshold, scaling_factor):
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.threshold = threshold
        self.scaling_factor = scaling_factor

    def adjust_weights(self, weights, losses):
        adjusted_weights = []
        for weight, loss in zip(weights, losses):
            if loss < self.threshold:
                weight = max(weight / self.scaling_factor, self.min_weight)
            else:
                weight = min(weight * self.scaling_factor, self.max_weight)
            adjusted_weights.append(weight)
        return adjusted_weights

# Collocation points resampling algorithm based on the RAR from the paper "DeepXDE"
def rar_points(geom: Geometry, 
               period: TimeDomain, 
               X: torch.Tensor, T: torch.Tensor, 
               errors: torch.Tensor, 
               num_points: int, epsilon: float, random=True) -> tuple:
    """
    Residual-based Adaptive Refinement (RAR) for collocation point placement.

    This function identifies the point with the maximum absolute error and adds new points 
    around it to improve the model's accuracy in regions of high error.

    Parameters:
        geom (Geometry): An object representing the spatial domain.
        period (TimeDomain): An object representing the time domain.
        X (torch.Tensor): Spatial coordinates of existing collocation points.
        T (torch.Tensor): Time coordinates of existing collocation points.
        errors (torch.Tensor): Array of errors at each existing collocation point.
        num_points (int): Number of new points to add.
        epsilon (float): Radius around the maximum error point to sample new points.
        random (bool, optional): Whether to sample points randomly or on a grid. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - x (torch.Tensor): Spatial coordinates of the new collocation points.
            - t (torch.Tensor): Time coordinates of the new collocation points.
            - center_coords (tuple): Coordinates (x, t) of the point with the maximum error.
    """

    device = X.device
    max_index = torch.argmax(torch.abs(errors))
    center_coords = [X[max_index], T[max_index]]
    dimension = len(center_coords[0])

    if random:
        x_extra = center_coords[0] + torch.randn(num_points, dimension) * epsilon
        t_extra = center_coords[1] + torch.randn(num_points) * epsilon
    else:
        n = int((num_points ** (1/(1+dimension))) / 2)

        if dimension == 1:
            x_extra = center_coords[0] + torch.linspace(-geom.grid_spacing_inners() * n,
                                                        geom.grid_spacing_inners() * n, 
                                                        2 * n + 1, dtype=X.dtype).to(device)
            x_extra = x_extra.reshape(-1, 1)
        elif dimension == 2:
            x_extra = []
            for i in range(dimension):
                x_i = center_coords[0][i] + torch.linspace(-geom.grid_spacing_inners()[i] * n,
                                                           geom.grid_spacing_inners()[i] * n, 
                                                           2 * n + 1, dtype=X.dtype).to(device)
                x_extra.append(x_i)
            x_extra = torch.stack(x_extra, dim=1)
        
        t_extra = center_coords[1] + torch.linspace(-period.grid_spacing_inners() * n,
                                                    period.grid_spacing_inners() * n, 
                                                    2*n + 1, dtype=T.dtype).to(device)
        # Make a grid
        if dimension == 1:
            x_extra, t_extra = torch.meshgrid(x_extra.squeeze(), t_extra)
            x_extra = x_extra.flatten().reshape(-1, 1)
            t_extra = t_extra.flatten().reshape(-1, 1)
        elif dimension == 2:
            x_extra, y_extra, t_extra = torch.meshgrid(x_extra[:, 0],
                                                       x_extra[:, 1], 
                                                       t_extra)
            x_extra = torch.stack((x_extra.flatten(), y_extra.flatten()), dim=1)
            t_extra = t_extra.flatten().reshape(-1, 1)
    
    # Clip offsets to the boundaries
    new_points = []
    for x, t in zip(x_extra, t_extra):
        if geom.inside(x.tolist()) and period.inside(t.item()):
            new_points.append([x, t])
    x = torch.stack([item[0] for item in new_points], dim=0).to(X.dtype)
    t = torch.stack([item[1] for item in new_points], dim=0).to(T.dtype)
    return x, t, (center_coords[0], center_coords[1])

# Class for hybrid optimization based on the paper
# "The Old and the New: Can Physics-Informed Deep-Learning Replace Traditional Linear Solvers?"
class HybridOptimizer:
    def __init__(self, model, switch_epoch=2000, switch_threshold=1e-2):
        self.model = model
        self.switch_iter = switch_epoch
        self.switch_threshold = switch_threshold
        self.current_optim = None
        self.epoch_of_switch = None

    def use_optimizer_adam(self):
        if self.optim_adam is not None:
            self.current_optim = self.optim_adam
        else:
            print("Adam optimizer is not available. Current optimizer is L-BFGS")

    def set_optimizer_adam(self, optim_adam):
        self.optim_adam = optim_adam

    def use_optimizer_lbfgs(self):
        if self.optim_lbfgs is not None:
            self.current_optim = self.optim_lbfgs
        else:
            print("L-BFGS optimizer is not available. Current optimizer is ADAM")

    def set_optimizer_lbfgs(self, optim_lbfgs):
        self.optim_lbfgs = optim_lbfgs
    
    def get_current_optimizer(self):
        return self.current_optim

    def get_optimizer_parameters(self):
        return self.current_optim.param_groups[0]['params']

    def get_switch_info(self):
        return {'epoch_of_switch': self.epoch_of_switch, '\n'
                'switch_loss_threshold': self.switch_threshold}
    
    def zero_grad(self):
        self.current_optim.zero_grad()

    def step(self, iter, closure):        
        if iter >= self.switch_iter and isinstance(self.current_optim, torch.optim.Adam):
            # Switch to L-BFGS if conditions are met
            loss = closure()
            if loss.item() < self.switch_threshold:
                print(f'Switching to L-BFGS at intation {iter + 1} with loss {loss.item()}')
                self.epoch_of_switch = iter    
                self.use_optimizer_lbfgs()

        # Perform optimization step with the current optimizer
        if isinstance(self.current_optim, torch.optim.LBFGS):
            self.current_optim.step(closure)
        else:
            closure()
            self.current_optim.step()

class EarlyStopping:
    def __init__(self, patience=50):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def step(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                                                