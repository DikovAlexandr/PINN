import os
import torch
import numpy as np

import solver.callbacks as callbacks
import solver.conditions as conditions
import solver.metrics as metrics
import solver.enhancements as enhancements
import solver.siren as siren
import solver.utils as utils


def pde(dudt, d2udx2, alpha):
    """
    Calculate residual of differential heat equation.

    Parameters:
        dudt (torch.Tensor): First derivative with respect to time.
        d2udx2 (torch.Tensor): Second derivative with respect to space.
        alpha (torch.Tensor): A constant parameter (thermal diffusivity coefficient).

    Returns:
        torch.Tensor: Result of the PDE calculation
    """
    return dudt - alpha**2 * d2udx2

def pde2d(dudt, d2udx2, d2udy2, alpha):
    """
    Calculate residual of differential heat equation.

    Parameters:
        dudt (torch.Tensor): First derivative with respect to time.
        d2udx2 (torch.Tensor): Second derivative with respect to space (x).
        d2udy2 (torch.Tensor): Second derivative with respect to space (y).
        alpha (torch.Tensor): A constant parameter (thermal diffusivity coefficient).

    Returns:
        torch.Tensor: Result of the PDE calculation
    """
    return dudt - alpha**2 * (d2udx2 + d2udy2)


class PINN():
    def __init__(self, problem, net_params, device='cuda:0'):
        # Initial points
        self.x_initial, self.t_initial, self.u_initial = problem.initial_conditions.get_initial_conditions()
        
        # Boundary points
        self.x_boundary, self.t_boundary, self.u_boundary = problem.boundary_conditions.get_boundary_conditions()
        
        # Equation points
        self.x_equation, self.t_equation = problem.equation.get_equation_points()

        # PDE
        self.pde = problem.equation.pde

        # Coefficient of thermal diffusivity
        self.alpha = problem.alpha

        # Problem dimensions
        self.dims = problem.geom.get_dimension()
        print(self.dims)

        # Device
        self.device = device

        # Construct network:
        self.input = net_params.input
        self.output = net_params.output
        self.hidden_layers = net_params.hidden_layers

        # Training parameters
        self.epochs = net_params.epochs
        self.batch_size = net_params.batch_size
        self.training_mode = net_params.training_mode # (train on sample or full data)

        # Activation function
        activations = {
            'tanh': torch.nn.Tanh(),
            'sigmoid': torch.nn.Sigmoid(),
            'sin': siren.Sin()
        }
        if net_params.activation in activations:
            self.activation = activations[net_params.activation]
        else:
            raise ValueError(f"Unsupported activation function: {net_params.activation}")
        
        # Siren parameters
        if net_params.siren_params != None:
            self.first_omega_0 = net_params.siren_params.first_omega_0
            self.hidden_omega_0 = net_params.siren_params.hidden_omega_0
            self.outermost_linear = net_params.siren_params.outermost_linear

        # Initialize network
        self.network()
        self.print_network()

        # Regularization
        self.regularization = net_params.regularization
        self.lambda_reg = net_params.lambda_reg

        # Optimizer
        if net_params.optimizer == 'LBFGS':
            self.optimizer = torch.optim.LBFGS(self.net.parameters(), 
                                               lr=net_params.lr,
                                               max_iter=10000,
                                               max_eval=100000,
                                               tolerance_grad = 1e-05,
                                               tolerance_change = 1e-09,
                                               history_size = 100,
                                               line_search_fn = 'strong_wolfe')
            # From Navier-Stokes:
            # lr=1, max_iter=10000, max_eval=100000, history_size=100, tolerance_grad=1e-15, 
            # tolerance_change=0.5 * np.finfo(float).eps, line_search_fn="strong_wolfe")

            # From: https://github.com/omniscientoctopus/Physics-Informed-Neural-Networks/blob/main/PyTorch/Burgers'%20Equation/Burgers.ipynb
            # lr=0.1,  max_iter = 250, max_eval = None, tolerance_grad = 1e-05, 
            # tolerance_change = 1e-09, history_size = 100, line_search_fn = 'strong_wolfe')
        elif net_params.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.net.parameters(), 
                                              lr=net_params.lr)
        elif net_params.optimizer == 'Hybrid':
            self.optimizer = enhancements.HybridOptimizer(self.net.parameters(),
                                                          switch_epoch=1000,
                                                          switch_threshold=10)
            self.optimizer.set_optimizer_adam(torch.optim.Adam(self.net.parameters()))
            self.optimizer.set_optimizer_lbfgs(torch.optim.LBFGS(self.net.parameters()))
            self.optimizer.use_optimizer_adam()
        else:
            raise ValueError(f"Unsupported optimizer: {net_params.optimizer}")    

        # Scheduler
        if net_params.optimizer == 'Hybrid':
            self.scheduler = None
        else:
            schedulers = {
                'StepLR': torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.5),
                'ExponentialLR': torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9),
                'ReduceLROnPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=10),
                None: None
            }
            if net_params.scheduler in schedulers:
                self.scheduler = schedulers[net_params.scheduler]
            else:
                raise ValueError(f"Unsupported scheduler: {net_params.scheduler}")
        
        # Early stopping
        if net_params.early_stopping:
            self.early_stopping = enhancements.EarlyStopping(100)
        else:
            self.early_stopping = None

        # Use RAR
        if net_params.use_rar:
            self.use_rar = True
        else:
            self.use_rar = False

        # Weights adjustment
        if net_params.use_weights_adjuster:
            self.adjuster = enhancements.LossWeightAdjuster(1e6, 1, 1e-3, 10)
        else:
            self.adjuster = None

        # Display interval
        self.display_interval = net_params.display_interval

        # Save model
        self.model_save_path = net_params.model_save_path

        # Save plots and loss history
        if net_params.output_path:
            self.output_path = net_params.output_path
            # utils.create_or_clear_folder(self.output_path)
            utils.create_folder(self.output_path)
        else:
            self.output_path = './output'
            utils.create_or_clear_folder(self.output_path)

        # Save loss history
        if net_params.save_loss:
            self.save_loss = net_params.save_loss
        else:
            self.save_loss = False

        # Initial weights for fine tuning
        if net_params.initial_weights_path:
            self.load_weights(net_params.initial_weights_path)

        # Iteration number
        self.iter = 0

        # Loss
        self.loss = 0
        self.mse = torch.nn.MSELoss()
        self.weight_eq = 1000
        self.weight_bc = 10000
        self.weight_ic = 1000
        self.null = torch.zeros((len(self.x_equation), 1), device=self.device)

        # Model name
        self.model_name = self.generate_name()

    def generate_name(self):
        base_name = f"HL_{len(self.hidden_layers)}_A_{self.activation}_N_{self.hidden_layers[-1]}"
        
        existing_models = os.listdir(self.output_path)
        existing_numbers = [int(name.split('_')[-1].split('.')[0]) for name in existing_models if name.startswith(base_name) and name.split('_')[-1].split('.')[0].isdigit()]
        
        next_number = 0 if not existing_numbers else max(existing_numbers) + 1
        print("Next number:", next_number)
        model_name = f"{base_name}_{next_number}"
        return model_name

    def load_weights(self, file_path):
        model_weights = torch.load(file_path)
        self.net.load_state_dict(model_weights)
        print("Model weights loaded from", file_path)

    def randomize_data(self, x, t, u=None):
        # Get the total number of examples
        total_examples = x.shape[0]
        random_indices = torch.randperm(total_examples)[:self.batch_size].to(self.device)
        sampled_x = x[random_indices, :]
        sampled_t = t[random_indices, :]

        if u is None:
            return sampled_x, sampled_t
        else:
            sampled_u = u[random_indices, :]
            return sampled_x, sampled_t, sampled_u
        
    def sort_data(self, x, t, u=None):
        # Sort the data by t and x in ascending order
        if u is None:
            combined_tensors = list(zip(x, t))
            combined_tensors.sort(key=lambda item: (item[1], item[0]))
            sorted_x, sorted_t = zip(*combined_tensors)
            sorted_x = torch.tensor(sorted_x).to(self.device)
            sorted_t = torch.tensor(sorted_t).to(self.device)
            return sorted_x, sorted_t
        else:
            combined_tensors = list(zip(x, t, u))
            combined_tensors.sort(key=lambda item: (item[1], item[0]))
            sorted_x, sorted_t, sorted_u = zip(*combined_tensors)
            sorted_x = torch.tensor(sorted_x).to(self.device)
            sorted_t = torch.tensor(sorted_t).to(self.device)
            sorted_u = torch.tensor(sorted_u).to(self.device)
            return sorted_x, sorted_t, sorted_u
        
    def network(self):
        if isinstance(self.activation, siren.Sin): # SIREN
            # First linear layer
            layers = [siren.SineLayer(self.input, self.hidden_layers[0], 
                                      is_first=True, omega_0=self.first_omega_0)]

            # Hidden layers
            for i in range(1, len(self.hidden_layers)):
                layers.append(siren.SineLayer(self.hidden_layers[i-1], self.hidden_layers[i], 
                                              is_first=False, omega_0=self.hidden_omega_0))

            # Last linear layer
            if self.outermost_linear:
                final_linear = torch.nn.Linear(self.hidden_layers[-1], self.output)
                with torch.no_grad():
                    final_linear.weight.uniform_(-np.sqrt(6 / self.hidden_layers[-1]) / self.hidden_omega_0, 
                                                 np.sqrt(6 / self.hidden_layers[-1]) / self.hidden_omega_0)
                layers.append(final_linear)
            else:
                layers.append(siren.SineLayer(self.hidden_layers[-1], self.output, 
                                              is_first=False, omega_0=self.hidden_omega_0))
        else: # FCN
            # First linear layer
            layers = [torch.nn.Linear(self.input, self.hidden_layers[0]), self.activation]

            # Hidden layers
            for i in range(1, len(self.hidden_layers)):
                layers.append(torch.nn.Linear(self.hidden_layers[i-1], self.hidden_layers[i]))
                layers.append(self.activation)
            
            # Last linear layer
            layers.append(torch.nn.Linear(self.hidden_layers[-1], self.output))

        self.net = torch.nn.Sequential(*layers)
        self.net.to(self.device)

    def print_network(self):
        print("Activation Function:", self.activation.__class__.__name__)
        print("Hidden Dimensions:", len(self.hidden_layers))
        print("Number of neurons:", self.hidden_layers[-1])

        print("----------")
        layer_num = 0
        for i, layer in enumerate(self.net):
            if isinstance(layer, torch.nn.Linear) or isinstance(layer, siren.SineLayer):
                next_layer = self.net[i + 1] if i + 1 < len(self.net) else None
            
                if next_layer and isinstance(next_layer, torch.nn.Module):
                    print(f"Layer {layer_num}: {layer} -> {next_layer.__class__.__name__}")
                else:
                    print(f"Layer {layer_num}: {layer}")
                layer_num += 1
        print("----------")

    def function(self, x, t, derivatives=[], is_equation=False):
        results = {}

        if is_equation:            
            if self.dims == 1:
                u_pred = self.net(torch.cat((x, t), dim=1))
                if 'dudx' in derivatives:
                    dudx = torch.autograd.grad(u_pred, x,
                                               grad_outputs=torch.ones_like(u_pred), 
                                               create_graph=True, allow_unused=True)[0]
                    results['dudx'] = dudx

                if 'd2udx2' in derivatives:
                    dudx = torch.autograd.grad(u_pred, x,
                                               grad_outputs=torch.ones_like(u_pred), 
                                               create_graph=True, allow_unused=True)[0]
                    d2udx2 = torch.autograd.grad(dudx, x,
                                                 grad_outputs=torch.ones_like(dudx), 
                                                 create_graph=True, allow_unused=True)[0]
                    results['d2udx2'] = d2udx2

            elif self.dims == 2:
                x_component, y_component = x[:, 0].reshape(-1, 1), x[:, 1].reshape(-1, 1)
                u_pred = self.net(torch.cat((x_component, y_component, t), dim=1))
                if 'dudx' in derivatives:
                    dudx = torch.autograd.grad(u_pred, x_component,
                                               grad_outputs=torch.ones_like(u_pred), 
                                               create_graph=True, allow_unused=True)[0]
                    results['dudx'] = dudx

                if 'dudy' in derivatives:
                    dudy = torch.autograd.grad(u_pred, y_component,
                                               grad_outputs=torch.ones_like(u_pred), 
                                               create_graph=True, allow_unused=True)[0]
                    results['dudy'] = dudy

                if 'd2udx2' in derivatives:
                    dudx = torch.autograd.grad(u_pred, x_component,
                                               grad_outputs=torch.ones_like(u_pred), 
                                               create_graph=True, allow_unused=True)[0]
                    d2udx2 = torch.autograd.grad(dudx, x_component,
                                                 grad_outputs=torch.ones_like(dudx), 
                                                 create_graph=True, allow_unused=True)[0]
                    results['d2udx2'] = d2udx2

                if 'd2udy2' in derivatives:
                    dudy = torch.autograd.grad(u_pred, y_component,
                                               grad_outputs=torch.ones_like(u_pred), 
                                               create_graph=True, allow_unused=True)[0]
                    d2udy2 = torch.autograd.grad(dudy, y_component,
                                                 grad_outputs=torch.ones_like(dudy), 
                                                 create_graph=True, allow_unused=True)[0]
                    results['d2udy2'] = d2udy2

                if 'd2udxy' in derivatives:
                    dudx = torch.autograd.grad(u_pred, x_component,
                                               grad_outputs=torch.ones_like(u_pred), 
                                               create_graph=True, allow_unused=True)[0]
                    d2udxy = torch.autograd.grad(dudx, y_component,
                                                 grad_outputs=torch.ones_like(dudx), 
                                                 create_graph=True, allow_unused=True)[0]
                    results['d2udxy'] = d2udxy

            else:
                raise ValueError("x must be a 1D or 2D tensor")

            if 'dudt' in derivatives:
                dudt = torch.autograd.grad(u_pred, t,
                                           grad_outputs=torch.ones_like(u_pred), 
                                           create_graph=True, allow_unused=True)[0]
                results['dudt'] = dudt

            if 'd2udt2' in derivatives:
                dudt = torch.autograd.grad(u_pred, t,
                                           grad_outputs=torch.ones_like(u_pred), 
                                           create_graph=True, allow_unused=True)[0]
                d2udt2 = torch.autograd.grad(dudt, t,
                                             grad_outputs=torch.ones_like(dudt), 
                                             create_graph=True, allow_unused=True)[0]
                results['d2udt2'] = d2udt2
            return u_pred, results
        else:
            if self.dims == 1:
                u_pred = self.net(torch.cat((x, t), dim=1))
            if self.dims == 2:
                x_component, y_component = x[:, 0].reshape(-1, 1), x[:, 1].reshape(-1, 1)
                u_pred = self.net(torch.cat((x_component, y_component, t), dim=1))
            return u_pred

    def closure(self):
        if hasattr(self.optimizer, 'step_closure'):
            return self.optimizer.step_closure(self)
        else:
            # Reset gradients
            self.optimizer.zero_grad()

            # Initial loss
            u_prediction = self.function(self.x_initial, self.t_initial)
            self.initial_loss = self.mse(u_prediction, self.u_initial)

            # Boundary loss
            u_prediction = self.function(self.x_boundary, self.t_boundary)
            self.boundary_loss = self.mse(u_prediction, self.u_boundary)

            # Equation loss
            # _, dudt, d2udx2 = self.function(self.x_equation, 
            #                                 self.t_equation, 
            #                                 is_equation=True)
            # heat_eq_prediction = pde(dudt, d2udx2, self.alpha)
            # self.equation_loss = self.mse(heat_eq_prediction, self.null)

            # New equation loss
            derivatives_names = ['dudt', 'd2udx2', 'd2udy2']
            _, derivatives = self.function(self.x_equation, self.t_equation, 
                                               derivatives=derivatives_names,
                                               is_equation=True)
            # dudt = derivatives.get('dudt')
            # d2udx2 = derivatives.get('d2udx2')
            # d2udy2 = derivatives.get('d2udy2')
            
            heat_eq_prediction = self.pde.substitute_into_equation(derivatives)
            self.equation_loss = self.mse(heat_eq_prediction, self.null)

            # Total loss
            self.loss = (self.weight_ic * self.initial_loss + 
                         self.weight_bc * self.boundary_loss + 
                         self.weight_eq * self.equation_loss)
            
            # Lasso, Ridge or Elastic regularization
            if self.regularization == "Lasso":
                self.loss += self.lasso(self.lambda_reg)
            elif self.regularization == "Ridge":
                self.loss += self.ridge(self.lambda_reg)
            elif self.regularization == "Elastic":
                self.loss += self.elastic(self.lambda_reg)

            # Derivative with respect to weights
            self.loss.backward(retain_graph=True)
            self.iter += 1

            # Print current loss
            if self.iter % self.display_interval == 0:
                exact_loss = self.initial_loss + self.boundary_loss + self.equation_loss
                print(f'Iteration {self.iter}: Loss {exact_loss}')

            if self.iter % self.display_interval == 0 and self.save_loss:
                # Write to file
                exact_loss = self.initial_loss + self.boundary_loss + self.equation_loss
                with open(f'{self.output_path}/{self.model_name}.csv', 'a') as f:
                    f.write(f"{self.iter}, {exact_loss}\n")

            return self.loss
        
    def lasso(self, lambda_reg):
        l1_reg = torch.tensor(0., requires_grad=True)
        for param in self.net.parameters():
            l1_reg = l1_reg + torch.norm(param, p=1)
        return lambda_reg * l1_reg

    def ridge(self, lambda_reg):
        l2_reg = torch.tensor(0., requires_grad=True)
        for param in self.net.parameters():
            l2_reg = l2_reg + torch.norm(param)
        return lambda_reg * l2_reg
    
    def elastic(self, lambda_reg):
        l1_reg = torch.tensor(0., requires_grad=True)
        l2_reg = torch.tensor(0., requires_grad=True)
        l1_ratio = 0.5
        for param in self.net.parameters():
            l1_reg = l1_reg + torch.norm(param, p=1)
            l2_reg = l2_reg + torch.norm(param)
        return lambda_reg * l1_ratio * l1_reg + lambda_reg * (1 - l1_ratio) * l2_reg

    def train(self):
        self.net.train()
        print(f"Oprtimizer: {self.optimizer.__class__.__name__}")

        if isinstance(self.optimizer, torch.optim.LBFGS):
            self.optimizer.zero_grad()
            # self.loss = self.closure()
            self.loss = self.optimizer.step(self.closure)
            self.post_optimization_actions(self.iter)
        elif isinstance(self.optimizer, torch.optim.Adam):
            for epoch in range(self.epochs):
                self.optimizer.zero_grad()
                self.loss = self.closure()
                self.optimizer.step()
                self.post_optimization_actions(epoch)

                # Early stopping
                if self.early_stopping is not None:
                    self.early_stopping.step(self.loss.item())
                    if self.early_stopping.early_stop:
                        print("Early stopping triggered at epoch", epoch)
                        break
        else:
            # Hybrid optimizer
            local_epoch = 0
            while isinstance(self.optimizer.get_current_optimizer(), torch.optim.Adam):
                self.optimizer.zero_grad()
                self.loss = self.closure()
                self.optimizer.step(self.iter, self.closure)
                self.post_optimization_actions(local_epoch)
                local_epoch += 1

                # Early stopping
                if self.early_stopping is not None:
                    self.early_stopping.step(self.loss.item())
                    if self.early_stopping.early_stop:
                        print("Early stopping triggered at epoch", local_epoch)
                        break

            if isinstance(self.optimizer.get_current_optimizer(), torch.optim.LBFGS):
                self.iter = local_epoch
                self.optimizer.zero_grad()
                self.loss = self.closure()
                self.optimizer.step(self.iter, self.closure)

    def post_optimization_actions(self, epoch):
        # Update scheduler
        if self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(self.loss.item())
            else:
                self.scheduler.step()

        # Weighs calibration
        if epoch % 10 == 0 and self.adjuster is not None:
            weights = [self.weight_ic, self.weight_bc, self.weight_eq]
            # print(f"Current weights: {self.weight_ic}, {self.weight_bc}, {self.weight_eq}")
            losses = [self.initial_loss.item(), 
                    self.boundary_loss.item(), 
                    self.equation_loss.item()]
            self.weight_ic, self.weight_bc, self.weight_eq = self.adjuster.adjust_weights(weights, losses)
            print(f"Adjusted weights: {self.weight_ic}, {self.weight_bc}, {self.weight_eq}")

    def predict(self, x, t):
        self.net.eval()

        with torch.no_grad():
            u_pred = self.function(x, t)

        return u_pred
    
    def get_loss_history(self):
        if self.save_loss:
            return f'{self.output_path}/{self.model_name}.csv'
        else:
            return None
        
    def save_weights(self):
        utils.create_folder(self.model_save_path)
        # torch.save(self.net, f'{self.model_save_path}/model.pth')
        model_weights = {}
        for name, param in self.net.named_parameters():
            model_weights[name] = param.data
        torch.save(model_weights, f'{self.model_save_path}/{self.model_name}_weights.pth')
        print("Model weights saved to", self.model_save_path)