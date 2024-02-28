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
    dudt (torch.Tensor): the first derivative with respect to time
    d2udx2 (torch.Tensor): the second derivative with respect to space
    alpha (torch.Tensor): a constant parameter (thermal diffusivity coefficient)

    Returns:
    torch.Tensor: the result of the PDE calculation
    """
    return dudt - alpha**2 * d2udx2


class PINN():
    def __init__(self, problem, net_params, device='cuda:0'):
        # Initial points
        self.x_initial, self.t_initial, self.u_initial = problem.initial_conditions.get_initial_conditions()
        
        # Boundary points
        self.x_boundary, self.t_boundary, self.u_boundary = problem.boundary_conditions.get_boundary_conditions()
        
        # Equation points
        self.x_equation, self.t_equation = problem.equation.get_equation()

        # Coefficient of thermal diffusivity
        self.alpha = problem.alpha 

        # Device
        self.device = device

        # Siren parameters
        if net_params.siren_params != None:
            print("Siren parameters:", net_params.siren_params)
            self.first_omega_0 = net_params.siren_params.first_omega_0
            self.hidden_omega_0 = net_params.siren_params.hidden_omega_0
            self.outermost_linear = net_params.siren_params.outermost_linear

        # Initialize network:
        self.input = net_params.input
        self.output = net_params.output
        self.hidden_layers = net_params.hidden_layers
        self.activation = net_params.activation
        self.network()
        self.print_network()

        # Optimizer
        if net_params.optimizer == 'LBFGS':
            self.optimizer = torch.optim.LBFGS(self.net.parameters(), 
                                                lr=1,
                                                max_iter=10000, 
                                                max_eval=100000,
                                                history_size=100, 
                                                tolerance_grad=1e-15, 
                                                tolerance_change=0.5 * np.finfo(float).eps,
                                                line_search_fn="strong_wolfe")
        elif net_params.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        elif net_params.optimizer == 'Adam+LBFGS':
            # TODO: Implement Adam+LBFGS optimizer
            pass

        # From: https://github.com/omniscientoctopus/Physics-Informed-Neural-Networks/blob/main/PyTorch/Burgers'%20Equation/Burgers.ipynb
        # optimizer = torch.optim.LBFGS(PINN.parameters(), lr=0.1, 
        #                               max_iter = 250, max_eval = None, 
        #                               tolerance_grad = 1e-05, tolerance_change = 1e-09, 
        #                               history_size = 100, line_search_fn = 'strong_wolfe')

        # Number of epochs
        self.epochs = net_params.epochs 
        
        # Iteration number
        self.iter = 0

        # Training mode (train on sample or full data)
        self.training_mode = net_params.training_mode

        # Number of samples for training
        self.batch_size = net_params.batch_size
        
        # Null vector is needed in equation loss
        self.null = torch.zeros((len(self.x_equation), 1), device=self.device)

        # Loss function
        self.mse = torch.nn.MSELoss()

        # Loss
        self.loss = 0

        # Loss weights
        self.weight_eq = 1
        self.weight_bc = 1
        self.weight_ic = 1

        # Weights adjustment
        self.adjust_weights = enhancements.LossWeightAdjuster(1e6, 1e-6, 1e-8, 10)

        # Save plots
        self.output_path = net_params.output_path
        utils.create_or_clear_folder(self.output_path)

    def sample_training_data(self, x, t, u=None):
        # Get the total number of examples
        total_examples = x.shape[0]

        # Generate random indices for sampling
        random_indices = torch.randperm(total_examples)[:self.batch_size].to(self.device)

        # Sample the data
        sampled_x = x[random_indices, :]
        sampled_t = t[random_indices, :]

        if u is None:
            return sampled_x, sampled_t
        else:
            sampled_u = u[random_indices, :]
            return sampled_x, sampled_t, sampled_u
        
    def network(self):
        if self.activation == "sin":
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
            
            self.net = torch.nn.Sequential(*layers)
            self.net.to(self.device)
        else:
            layers = [torch.nn.Linear(self.input, self.hidden_layers[0]), self.activation]

            for i in range(1, len(self.hidden_layers)):
                layers.append(torch.nn.Linear(self.hidden_layers[i-1], self.hidden_layers[i]))
                layers.append(self.activation)
            
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

    def function(self, x, t, is_equation=False):
        u_pred = self.net(torch.hstack((x, t)))

        if is_equation:
            dudx = torch.autograd.grad(u_pred, x, 
                                       grad_outputs=torch.ones_like(u_pred), 
                                       create_graph=True)[0]
            d2udx2 = torch.autograd.grad(dudx, x, 
                                         grad_outputs=torch.ones_like(dudx), 
                                         create_graph=True)[0]
            dudt = torch.autograd.grad(u_pred, t, 
                                       grad_outputs=torch.ones_like(u_pred), 
                                       create_graph=True)[0]
            return u_pred, dudt, d2udx2

        return u_pred

    def closure(self):
        # Reset gradients
        self.optimizer.zero_grad()

        # Initial loss
        if self.training_mode == 'sample':
            sampled_x, sampled_t, sampled_u = self.sample_training_data(self.x_initial, 
                                                                        self.t_initial, 
                                                                        self.u_initial, 
                                                                        self.batch_size)
            u_prediction = self.function(sampled_x, sampled_t)
            initial_loss = self.mse(u_prediction, sampled_u)
        else:
            u_prediction = self.function(self.x_initial, self.t_initial)
            initial_loss = self.mse(u_prediction, self.u_initial)

        # Boundary loss
        if self.training_mode == 'sample':
            sampled_x, sampled_t, sampled_u = self.sample_training_data(self.x_boundary, 
                                                                        self.t_boundary, 
                                                                        self.u_boundary, 
                                                                        self.batch_size)
            u_prediction = self.function(sampled_x, sampled_t)
            boundary_loss = self.mse(u_prediction, sampled_u)
        else:
            u_prediction = self.function(self.x_boundary, self.t_boundary)
            boundary_loss = self.mse(u_prediction, self.u_boundary)

        # Equation loss
        if self.training_mode == 'sample':    
            sampled_x, sampled_t = self.sample_training_data(self.x_equation,
                                                             self.t_equation,
                                                             None,
                                                             self.batch_size)
            _, dudt, d2udx2 = self.function(sampled_x, sampled_t, is_equation=True)
        else:
            _, dudt, d2udx2 = self.function(self.x_equation, self.t_equation, is_equation=True)
        heat_eq_prediction = pde(dudt, d2udx2, self.alpha)
        equation_loss = self.mse(heat_eq_prediction, self.null)

        # Total loss
        self.loss = (self.weight_ic * initial_loss + 
                     self.weight_bc * boundary_loss + 
                     self.weight_eq * equation_loss)

        # Derivative with respect to weights
        self.loss.backward()
        self.optimizer.step()

        self.iter += 1

        # Weighs calibration
        if self.iter % 100 == 0:
            self.adjust_weights([self.weight_ic, self.weight_bc, self.weight_eq], 
                                [initial_loss, boundary_loss, equation_loss])

        if self.iter % 100 == 0:
            # Write to file
            with open(f'{self.output_path}/HL_{len(self.hidden_layers)}_A_{self.activation}_N_{self.hidden_layers[-1]}.csv', 'a') as f:
                f.write(f"{self.iter}, {self.loss}\n")

            # print('Iteration: {:}, Loss: {:0.6f}'.format(self.iter, self.loss))

            test = set_test(self.size, self.equation_points, 0.5)
            u_test = self.predict(test.x, test.t)

            # Plot numerical vs analytical solution
            self.plot_test(test.x, u_test, self.iter)

        return self.loss
    
    def get_loss_history(self):
        return f'{self.output_path}/HL_{len(self.hidden_layers)}_A_{self.activation}_N_{self.hidden_layers[-1]}.csv'

    def train(self):
        # Training loop
        self.net.train()
        # self.optimizer.step(self.closure)

        # For Adam
        # early_stopping = EarlyStopping(patience=10, min_delta=0.0001)

        for epoch in range(self.epochs):
            self.optimizer.step(self.closure)
            # current_loss = self.closure()
            # early_stopping(current_loss)
            
            # if early_stopping.early_stop:
            #     print("Early stopping")
            #     break

    def predict(self, x, t):
        x = torch.tensor(x[:, np.newaxis], dtype=torch.float32, requires_grad=True)
        t = torch.tensor(t[:, np.newaxis], dtype=torch.float32, requires_grad=True)
        
        self.net.eval()

        with torch.no_grad():
            u_pred = self.function(x, t)

        self.net.train()

        return u_pred