import numpy as np
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from solver.conditions import set_test
from solver.utils import create_or_clear_folder

class Params:
    def __init__(self, size, time, alpha, initial_points, 
                 boundary_points, equation_points, hidden_dims, activation):
        self.size = size
        self.time = time
        self.initial_points = initial_points
        self.boundary_points = boundary_points
        self.equation_points = equation_points
        self.alpha = alpha
        self.hidden_dims = hidden_dims
        self.activation = activation

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0001):
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
    def __init__(self, params, initial_conditions, boundary_conditions, equation, device='cuda:0'):
        # Set data
        self.x_initial = initial_conditions.x[:, None].to(torch.float32).requires_grad_(True)
        self.t_initial = initial_conditions.t[:, None].to(torch.float32).requires_grad_(True)
        self.u_initial = initial_conditions.u[:, None].to(torch.float32).requires_grad_(True)
        
        self.x_boundary = boundary_conditions.x[:, None].to(torch.float32).requires_grad_(True)
        self.t_boundary = boundary_conditions.t[:, None].to(torch.float32).requires_grad_(True)
        self.u_boundary = boundary_conditions.u[:, None].to(torch.float32).requires_grad_(True)
        
        self.x_equation = equation.x[:, None].to(torch.float32).requires_grad_(True)
        self.t_equation = equation.t[:, None].to(torch.float32).requires_grad_(True)

        # Coefficient of termal conductivity
        self.alpha = params.alpha

        # Size of computational field
        self.size = params.size

        # Number of points in initial conditions
        self.initial_points = params.initial_points

        # Number of points in boundary conditions
        self.boundary_points = params.boundary_points

        # Number of points in equation
        self.equation_points = params.equation_points

        # Time 
        self.time = params.time

        # Device
        self.device = device

        # Initialize network:
        self.activation = params.activation
        self.hidden_dims = params.hidden_dims
        self.input_dim = 2
        self.output_dim = 1
        self.network()
        self.print_network()

        # Optimizer
        # self.optimizer = torch.optim.LBFGS(self.net.parameters(), 
        #                                    lr=1,
        #                                    max_iter=10000, 
        #                                    max_eval=100000,
        #                                    history_size=100, 
        #                                    tolerance_grad=1e-15, 
        #                                    tolerance_change=0.5 * np.finfo(float).eps,
        #                                    line_search_fn="strong_wolfe")

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)

        # Number of epochs (for Adam optimizer)
        self.num_epochs = 1000

        # Number of samples for training
        self.num_samples = 100
        
        # Null vector is needed in equation loss
        # self.null = torch.zeros((self.num_samples, 1))
        self.null = torch.zeros((len(self.x_equation), 1), device=self.device)

        # Loss function
        self.mse = nn.MSELoss()

        # Loss
        self.loss = 0

        # Iteration number
        self.iter = 0

        # Save plots
        self.output_folder = f'logs/HL_{len(self.hidden_dims)}_A_{self.activation}_N_{self.hidden_dims[-1]}'
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
        layers = [nn.Linear(self.input_dim, self.hidden_dims[0]), self.activation]

        for i in range(1, len(self.hidden_dims)):
            # print(f"Adding layer {i}: {self.hidden_dims[i-1]} -> {self.hidden_dims[i]}")
            layers.append(nn.Linear(self.hidden_dims[i-1], self.hidden_dims[i]))
            layers.append(self.activation)
        
        layers.append(nn.Linear(self.hidden_dims[-1], self.output_dim))

        self.net = nn.Sequential(*layers)
        self.net.to(self.device)

    def print_network(self):
        print("Activation Function:", self.activation.__class__.__name__)
        print("Hidden Dimensions:", len(self.hidden_dims))
        print("Number of neurons:", self.hidden_dims[-1])

        print("----------")
        layer_num = 0
        for i, layer in enumerate(self.net):
            if isinstance(layer, nn.Linear):
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

        # Initial loss (sample data)
        # sampled_x, sampled_t, sampled_u = self.sample_training_data(self.x_initial, 
        #                                                             self.t_initial, 
        #                                                             self.u_initial, 
        #                                                             self.num_samples)
        # u_prediction = self.function(sampled_x, sampled_t)
        # initial_loss = self.mse(u_prediction, sampled_u)

        # Initial loss
        u_prediction = self.function(self.x_initial, self.t_initial)
        initial_loss = self.mse(u_prediction, self.u_initial)

        # Boundary loss
        u_prediction = self.function(self.x_boundary, self.t_boundary)
        boundary_loss = self.mse(u_prediction, self.u_boundary)

        # Equation loss (sample data)
        # sampled_x, sampled_t = self.sample_training_data(self.x_equation,
        #                                                  self.t_equation,
        #                                                  None,
        #                                                  self.num_samples)
        # _, dudt, d2udx2 = self.function(sampled_x, sampled_t, is_equation=True)

        # Equation loss
        _, dudt, d2udx2 = self.function(self.x_equation, self.t_equation, is_equation=True)
        heat_eq_prediction = dudt - self.alpha**2 * d2udx2
        equation_loss = self.mse(heat_eq_prediction, self.null)

        self.loss = initial_loss + boundary_loss + equation_loss

        # Derivative with respect to weights
        self.loss.backward()

        # self.optimizer.step() # For Adam

        self.iter += 1

        if self.iter % 10 == 0:
            # Write to file
            with open(f'{self.output_folder}/HL_{len(self.hidden_dims)}_A_{self.activation}_N_{self.hidden_dims[-1]}.csv', 'a') as f:
                f.write(f"{self.iter}; {self.loss}\n")

            # Print loss
            print('Iteration: {:}, Loss: {:0.6f}'.format(self.iter, self.loss))


            test = set_test(self.size, self.equation_points, 0.5)
            x_test = torch.tensor(test.x[:, np.newaxis], dtype=torch.float32, requires_grad=True)
            t_test = torch.tensor(test.t[:, np.newaxis], dtype=torch.float32, requires_grad=True)

            u_test = self.function(x_test, t_test)

            # Plot numerical vs analytical solution
            # self.plot_test(x_test, u_test, self.iter)

        return self.loss

    def train(self):
        # Training loop
        self.net.train()
        # self.optimizer.step(self.closure)

        # For Adam
        early_stopping = EarlyStopping(patience=10, min_delta=0.0001)

        for epoch in range(self.num_epochs):
            current_loss = self.closure()
            early_stopping(current_loss)
            
            if early_stopping.early_stop:
                print("Early stopping")
                break

    def predict(self, x, t):
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
        self.output_folder
        plt.savefig(os.path.join(self.output_folder, f'{str(self.iter).zfill(5)}.png'))
        plt.close(fig)