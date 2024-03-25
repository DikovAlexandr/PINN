import numpy as np
import torch

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
def rar_points(geom, period, X, T, errors, num_points, epsilon, random=True):
    max_index = np.argmax(np.absolute(errors))
    center_coords = [X[max_index], T[max_index]]
    dimension = len(center_coords[0])

    if random:
        x_extra = center_coords[0] + np.random.normal(0, epsilon, size=(num_points, dimension))
        t_extra = center_coords[1] + np.random.normal(0, epsilon, num_points)
    else:
        n = int((num_points ** (1/(1+dimension))) / 2)
        x_extra = []
        for i in range(dimension):
            x_i = center_coords[0][i] + np.linspace(-geom.grid_spacing_inners()[i] * n,
                                                    geom.grid_spacing_inners()[i] * n, 2 * n + 1)
            x_extra.append(x_i)
        x_extra = np.column_stack(x_extra)
        t_extra = center_coords[1] + np.linspace(-period.grid_spacing_inners() * n,
                                         period.grid_spacing_inners() * n, 2*n + 1)
        # Make a grid
        if dimension == 1:
            x_extra, t_extra = np.meshgrid(x_extra,  t_extra)
            x_extra = x_extra.flatten()
            t_extra = t_extra.flatten()
        elif dimension == 2:
            x_extra, y_extra, t_extra = np.meshgrid(x_extra[:, 0],  x_extra[:, 1], t_extra)
            x_extra = np.column_stack((x_extra.flatten(), y_extra.flatten()))
            t_extra = t_extra.flatten()
    
    # Clip offsets to the boundaries
    new_points = []
    for x, t in zip(x_extra, t_extra):
        new_points.append([x.tolist(), t])
    new_points = [point for point in new_points if geom.inside(point[0]) and period.inside(point[1])]
    x = np.array([item[0] for item in new_points])
    t = np.array([item[1] for item in new_points])
    return x, t, (center_coords[0], center_coords[1])

# Class for hybrid optimization based on the paper
# "The Old and the New: Can Physics-Informed Deep-Learning Replace Traditional Linear Solvers?"
class HybridOptimizer:
    def __init__(self, model, switch_epoch=2000, switch_threshold=1e-3):
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