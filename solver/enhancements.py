import numpy as np


# Class for loss weight adjustment based on the loss values
class LossWeightAdjuster:
    def __init__(self, max_weight, min_weight, threshold, scaling_factor):
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.threshold = threshold
        self.scaling_factor = scaling_factor

    def adjust_weights(self, weights, losses):
        for i in range(len(weights)):
            if losses[i] < self.threshold:
                weights[i] = weights[i] / self.scaling_factor
            else:
                weights[i] = losses[i] * self.scaling_factor
            weights[i] = max(self.min_weight, min(weights[i], self.max_weight))

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
    def __init__(self, model, criterion, 
                 optim_adam=None, optim_lbfgs=None, 
                 switch_epoch=None, switch_threshold=None,
                 early_stopping=None):
        self.model = model
        self.criterion = criterion
        self.optim_adam = optim_adam
        self.optim_lbfgs = optim_lbfgs
        self.switch_epoch = switch_epoch
        self.switch_threshold = switch_threshold
        self.current_optim = optim_adam if optim_lbfgs is None else optim_lbfgs
        self.early_stopping = early_stopping
        
        self.epoch_of_switch = None

    def set_optimizer_adam(self):
        if self.optim_adam is not None:
            self.current_optim = self.optim_adam
        else:
            print("ADAM optimizer is not available. Current optimizer is L-BFGS")

    def set_optimizer_lbfgs(self):
        if self.optim_lbfgs is not None:
            self.current_optim = self.optim_lbfgs
        else:
            print("L-BFGS optimizer is not available. Current optimizer is ADAM")

    def get_current_optimizer(self):
        return self.current_optim

    def get_optimizer_parameters(self):
        return self.current_optim.param_groups[0]['params']

    def get_switch_info(self):
        return {'epoch_of_switch': self.epoch_of_switch, '\n'
                'switch_loss_threshold': self.switch_threshold}

    def closure(self):
        self.current_optim.zero_grad()
        predictions = self.model(self.X)
        loss = self.criterion(predictions, self.y)
        loss.backward()
        return loss

    def step(self, epoch, X, y):
        # Update trining data
        self.X = X
        self.y = y

        # Reset gradients
        self.current_optim.zero_grad()
        
        if epoch >= self.switch_epoch and self.current_optim == self.optim_adam:
            # Switch to L-BFGS if conditions are met
            loss = self.criterion(self.model(self.X), self.y)
            if loss.item() < self.switch_threshold:
                print(f'Switching to L-BFGS at epoch {epoch + 1} with loss {loss.item()}')
                self.epoch_of_switch = epoch
                self.current_optim = self.optim_lbfgs

        # Perform optimization step with the current optimizer
        self.current_optim.step(self.closure)

        # Check early stopping condition
        if self.early_stopping is not None:
            current_loss = self.criterion(self.model(self.X), self.y).item()
            if self.early_stopping(current_loss):
                print(f'Early stopping at epoch {epoch + 1} with loss {current_loss}')

class EarlyStopping:
    def __init__(self, patience=5):
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

# class EarlyStopping:
#     def __init__(self, patience=100, min_delta=0.0001):
#         self.patience = patience
#         self.min_delta = min_delta
#         self.counter = 0
#         self.best_loss = None
#         self.early_stop = False

#     def __call__(self, current_loss):
#         if self.best_loss is None:
#             self.best_loss = current_loss
#         elif current_loss > self.best_loss - self.min_delta:
#             self.counter += 1
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_loss = current_loss
#             self.counter = 0