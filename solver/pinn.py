import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from solver.siren import SineLayer
from solver.conditions import set_test
from solver.utils import create_or_clear_folder

class MeshParams:
    def __init__(self, size, time, alpha, initial_points, 
                 boundary_points, equation_points):
        self.size = size
        self.time = time
        self.initial_points = initial_points
        self.boundary_points = boundary_points
        self.equation_points = equation_points
        self.alpha = alpha

class NetParams:
    def __init__(self, input, output, hidden_layers, activation, training_mode, optimizer, siren_params):
        self.input = input
        self.output = output
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.training_mode = training_mode
        self.optimizer = optimizer

        if siren_params == None:
            self.siren_params = None
        else:
            self.siren_params = siren_params
        

class EarlyStopping:
    def __init__(self, patience=10000, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, current_loss):
        if self.best_loss is None:
            self.best_loss = current_loss
        elif current_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = current_loss
            self.counter = 0

class PINN():
    def __init__(self, mesh_params, net_params, initial_conditions, boundary_conditions, equation, device='cuda:0'):
        # Initial points
        self.x_initial = initial_conditions.x[:, None].to(torch.float32).requires_grad_(True)
        self.t_initial = initial_conditions.t[:, None].to(torch.float32).requires_grad_(True)
        self.u_initial = initial_conditions.u[:, None].to(torch.float32).requires_grad_(True)
        
        # Boundary points
        self.x_boundary = boundary_conditions.x[:, None].to(torch.float32).requires_grad_(True)
        self.t_boundary = boundary_conditions.t[:, None].to(torch.float32).requires_grad_(True)
        self.u_boundary = boundary_conditions.u[:, None].to(torch.float32).requires_grad_(True)
        
        # Equation points
        self.x_equation = equation.x[:, None].to(torch.float32).requires_grad_(True)
        self.t_equation = equation.t[:, None].to(torch.float32).requires_grad_(True)

        # Parameters of computational field and equation
        self.alpha = mesh_params.alpha # Coefficient of temperature conductivity
        self.size = mesh_params.size # Size of computational field
        self.initial_points = mesh_params.initial_points # Number of points in initial conditions
        self.boundary_points = mesh_params.boundary_points # Number of points in boundary conditions
        self.equation_points = mesh_params.equation_points # Number of points in equation
        self.time = mesh_params.time # Time

        # Device
        self.device = device

        # Siren parameters
        if net_params.siren_params != None:
            print("Siren parameters: ", net_params.siren_params)
            self.first_omega_0 = net_params.siren_params.first_omega_0
            self.hidden_omega_0 = net_params.siren_params.hidden_omega_0
            self.outermost_linear = net_params.siren_params.outermost_linear

        # Initialize network:
        self.activation = net_params.activation
        self.hidden_layers = net_params.hidden_layers
        self.input = net_params.input
        self.output = net_params.output
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

        # From: https://github.com/omniscientoctopus/Physics-Informed-Neural-Networks/blob/main/PyTorch/Burgers'%20Equation/Burgers.ipynb
        # optimizer = torch.optim.LBFGS(PINN.parameters(), lr=0.1, 
        #                       max_iter = 250, 
        #                       max_eval = None, 
        #                       tolerance_grad = 1e-05, 
        #                       tolerance_change = 1e-09, 
        #                       history_size = 100, 
        #                       line_search_fn = 'strong_wolfe')

        # Number of epochs (for Adam optimizer)
        self.num_epochs = 3000

        # Training mode (train on sample or full data)
        self.training_mode = net_params.training_mode

        # Number of samples for training
        self.num_samples = 100
        
        # Null vector is needed in equation loss
        self.null = torch.zeros((len(self.x_equation), 1), device=self.device)

        # Loss function
        self.mse = nn.MSELoss()

        # Loss
        self.loss = 0

        # Iteration number
        self.iter = 0

        # Save plots
        self.output_folder = f'logs/HL_{len(self.hidden_layers)}_A_{self.activation}_N_{self.hidden_layers[-1]}'
        create_or_clear_folder(self.output_folder)

    def sample_training_data(self, x, t, u=None, num_samples=None):
        # Get the total number of examples
        total_examples = x.shape[0]

        # Generate random indices for sampling
        random_indices = torch.randperm(total_examples)[:num_samples].to(self.device)

        # Sample the data
        sampled_x = x[random_indices, :]
        sampled_t = t[random_indices, :]

        if u is not None:
            sampled_u = u[random_indices, :]
            return sampled_x, sampled_t, sampled_u
        else:
            return sampled_x, sampled_t
        
    def network(self):
        if self.activation == "sin":
            # First linear layer
            layers = [SineLayer(self.input, self.hidden_layers[0], is_first=True, omega_0=self.first_omega_0)]

            # Hidden layers
            for i in range(1, len(self.hidden_layers)):
                layers.append(SineLayer(self.hidden_layers[i-1], self.hidden_layers[i], 
                                        is_first=False, omega_0=self.hidden_omega_0))

            # Last linear layer
            if self.outermost_linear:
                final_linear = nn.Linear(self.hidden_layers[-1], self.output)
                with torch.no_grad():
                    final_linear.weight.uniform_(-np.sqrt(6 / self.hidden_layers[-1]) / self.hidden_omega_0, 
                                                np.sqrt(6 / self.hidden_layers[-1]) / self.hidden_omega_0)
                layers.append(final_linear)
            else:
                layers.append(SineLayer(self.hidden_layers[-1], self.output, 
                                        is_first=False, omega_0=self.hidden_omega_0))
            
            self.net = nn.Sequential(*layers)
            self.net.to(self.device)

        else:
            layers = [nn.Linear(self.input, self.hidden_layers[0]), self.activation]

            for i in range(1, len(self.hidden_layers)):
                layers.append(nn.Linear(self.hidden_layers[i-1], self.hidden_layers[i]))
                layers.append(self.activation)
            
            layers.append(nn.Linear(self.hidden_layers[-1], self.output))

            self.net = nn.Sequential(*layers)
            self.net.to(self.device)

    def print_network(self):
        print("Activation Function:", self.activation.__class__.__name__)
        print("Hidden Dimensions:", len(self.hidden_layers))
        print("Number of neurons:", self.hidden_layers[-1])

        print("----------")
        layer_num = 0
        for i, layer in enumerate(self.net):
            if isinstance(layer, nn.Linear) or isinstance(layer, SineLayer):
                next_layer = self.net[i + 1] if i + 1 < len(self.net) else None
            
                if next_layer and isinstance(next_layer, nn.Module):
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
                                                                        self.num_samples)
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
                                                                        self.num_samples)
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
                                                             self.num_samples)
            _, dudt, d2udx2 = self.function(sampled_x, sampled_t, is_equation=True)
        else:
            _, dudt, d2udx2 = self.function(self.x_equation, self.t_equation, is_equation=True)
        heat_eq_prediction = dudt - self.alpha**2 * d2udx2
        equation_loss = self.mse(heat_eq_prediction, self.null)

        # Total loss
        self.loss = 10 * initial_loss + 1000 * boundary_loss + 10 * equation_loss

        # Derivative with respect to weights
        self.loss.backward()
        self.optimizer.step()

        self.iter += 1

        if self.iter % 100 == 0:
            # Write to file
            with open(f'{self.output_folder}/HL_{len(self.hidden_layers)}_A_{self.activation}_N_{self.hidden_layers[-1]}.csv', 'a') as f:
                f.write(f"{self.iter}, {self.loss}\n")

            # print('Iteration: {:}, Loss: {:0.6f}'.format(self.iter, self.loss))

            test = set_test(self.size, self.equation_points, 0.5)
            u_test = self.predict(test.x, test.t)

            # Plot numerical vs analytical solution
            self.plot_test(test.x, u_test, self.iter)

        return self.loss
    
    def get_loss_history(self):
        return f'{self.output_folder}/HL_{len(self.hidden_layers)}_A_{self.activation}_N_{self.hidden_layers[-1]}.csv'

    def train(self):
        # Training loop
        self.net.train()
        # self.optimizer.step(self.closure)

        # For Adam
        # early_stopping = EarlyStopping(patience=10, min_delta=0.0001)

        for epoch in range(self.num_epochs):
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

    def plot_test(self, x_test, u_test, iter):
        fig = plt.figure()

        x_test_cpu = x_test.cpu().detach().numpy()
        u_test_cpu = u_test.cpu().detach().numpy()

        plt.plot(x_test_cpu, u_test_cpu)
        plt.xlim(0, 1)
        plt.ylim(-0.1, 1.1)
        plt.grid()   
        plt.title(f'PINN Solution {iter}')
        plt.xlabel('x')
        plt.ylabel('u')
        plt.savefig(os.path.join(self.output_folder, f'{str(self.iter).zfill(5)}.png'))
        plt.close(fig)