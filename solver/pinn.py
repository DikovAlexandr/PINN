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

        # Construct network:
        self.input = net_params.input
        self.output = net_params.output
        self.hidden_layers = net_params.hidden_layers

        # Activation function
        activations = {
            'relu': torch.nn.ReLU(),
            'tanh': torch.nn.Tanh(),
            'sigmoid': torch.nn.Sigmoid(),
            'sin': siren.Sin()
        }
        if net_params.activation in activations:
            self.activation = activations[net_params.activation]
        else:
            raise ValueError(f"Unsupported activation function: {net_params.activation}")
        
        # Initialize network
        self.network()
        self.print_network()

        # Optimizer
        if net_params.optimizer == 'LBFGS':
            self.optimizer = torch.optim.LBFGS(self.net.parameters(), 
                                               lr=net_params.learning_rate)
            # From Navier-Stokes:
            # lr=1, max_iter=10000, max_eval=100000, history_size=100, tolerance_grad=1e-15, 
            # tolerance_change=0.5 * np.finfo(float).eps, line_search_fn="strong_wolfe")

            # From: https://github.com/omniscientoctopus/Physics-Informed-Neural-Networks/blob/main/PyTorch/Burgers'%20Equation/Burgers.ipynb
            # lr=0.1,  max_iter = 250, max_eval = None, tolerance_grad = 1e-05, 
            # tolerance_change = 1e-09, history_size = 100, line_search_fn = 'strong_wolfe')
        elif net_params.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.net.parameters(), 
                                              lr=net_params.learning_rate)
        elif net_params.optimizer == 'Hybrid':
            # TODO: Implement Adam+LBFGS optimizer
            self.optimizer = enhancements.HybridOptimizer(self.net.parameters(),
                                                           self.criterion,
                                                           optim_adam=torch.optim.Adam(self.net.parameters(), 
                                                                                       lr=1e-3),
                                                           optim_lbfgs=torch.optim.LBFGS(self.net.parameters(), 
                                                                                         lr=1e-3),
                                                           switch_epoch=2000,
                                                           switch_threshold=1e-3,
                                                           early_stopping=True)        

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
        self.save_loss = net_params.save_loss

        # Loss weights
        self.weight_eq = 1000
        self.weight_bc = 1000
        self.weight_ic = 1000

        # Weights adjustment
        if net_params.use_weights_adjuster:
            self.adjuster = enhancements.LossWeightAdjuster(1e6, 1e-6, 1e-3, 10)
        else:
            self.adjuster = None

        # Save plots
        self.output_path = net_params.output_path
        utils.create_or_clear_folder(self.output_path)

        # Initial weights for fine tuning
        if net_params.initial_weights_path:
            self.load_weights = net_params.initial_weights_path

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
        if self.activation is siren.Sin():
            # SIREN
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
        else:
            # FCN
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

    def function(self, x, t, is_equation=False):
        u_pred = self.net(torch.stack((x, t)).T)

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
            sampled_x, sampled_t, sampled_u = self.randomize_data(self.x_initial, 
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
            sampled_x, sampled_t, sampled_u = self.randomize_data(self.x_boundary, 
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
            sampled_x, sampled_t = self.randomize_data(self.x_equation,
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
        self.loss.backward(retain_graph=True) # WHERE MUST BE BACKWARD?
        self.optimizer.step()

        self.iter += 1

        # Weighs calibration
        if self.iter % 10 == 0 and self.adjuster is not None:
            print("Adjusting weights...")
            [self.weight_ic, self.weight_bc, self.weight_eq] = self.adjuster.adjust_weights([self.weight_ic, self.weight_bc, self.weight_eq], 
                                                                                          [initial_loss, boundary_loss, equation_loss])
            
        if self.iter % 20 == 0:
            print('Iteration: {:}, Loss: {:0.6f}'.format(self.iter, self.loss))
            # print('Weights: {:0.6f}, {:0.6f}, {:0.6f}'.format(self.weight_ic, self.weight_bc, self.weight_eq))

        # if self.iter % 100 == 0:
        #     # Write to file
        #     with open(f'{self.output_path}/HL_{len(self.hidden_layers)}_A_{self.activation}_N_{self.hidden_layers[-1]}.csv', 'a') as f:
        #         f.write(f"{self.iter}, {self.loss}\n")

        #     # print('Iteration: {:}, Loss: {:0.6f}'.format(self.iter, self.loss))

        #     test = set_test(self.size, self.equation_points, 0.5)
        #     u_test = self.predict(test.x, test.t)

        #     # Plot numerical vs analytical solution
        #     self.plot_test(test.x, u_test, self.iter)

        return self.loss
    
    def get_loss_history(self):
        if self.save_loss:
            return f'{self.output_path}/HL_{len(self.hidden_layers)}_A_{self.activation}_N_{self.hidden_layers[-1]}.csv'
        else:
            return None
    def train(self):
        self.net.train()
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
        self.net.eval()

        with torch.no_grad():
            u_pred = self.function(x, t)

        return u_pred