"""Physics-Informed Neural Network implementation for solving PDEs."""

import os
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from . import callbacks
from . import conditions
from . import metrics
from . import enhancements
from . import siren
from . import utils
from .constants import (
    DEFAULT_WEIGHT_IC, DEFAULT_WEIGHT_BC, DEFAULT_WEIGHT_EQ,
    DEFAULT_DEVICE
)
from .exceptions import PINNError, TrainingError


def pde_1d(dudt: torch.Tensor, d2udx2: torch.Tensor, 
           alpha: torch.Tensor) -> torch.Tensor:
    """Calculate residual of 1D heat equation.
    
    Args:
        dudt: First derivative with respect to time.
        d2udx2: Second derivative with respect to space.
        alpha: Thermal diffusivity coefficient.
        
    Returns:
        PDE residual.
    """
    return dudt - alpha**2 * d2udx2


def pde_2d(dudt: torch.Tensor, d2udx2: torch.Tensor, d2udy2: torch.Tensor,
           alpha: torch.Tensor) -> torch.Tensor:
    """Calculate residual of 2D heat equation.
    
    Args:
        dudt: First derivative with respect to time.
        d2udx2: Second derivative with respect to space (x).
        d2udy2: Second derivative with respect to space (y).
        alpha: Thermal diffusivity coefficient.
        
    Returns:
        PDE residual.
    """
    return dudt - alpha**2 * (d2udx2 + d2udy2)


class PINN:
    """Physics-Informed Neural Network for solving PDEs.
    
    This class implements a PINN that can solve various partial differential
    equations by incorporating physics constraints into the loss function.
    """
    
    def __init__(self, problem: conditions.Problem, net_params: utils.NetParams,
                 device: str = DEFAULT_DEVICE):
        """Initialize PINN.
        
        Args:
            problem: Problem definition containing geometry, conditions, and PDE.
            net_params: Network parameters and training configuration.
            device: Device to run computations on.
            
        Raises:
            PINNError: If configuration is invalid.
        """
        self.problem = problem
        self.device = device
        
        # Validate problem dimensions
        self.dims = problem.geom.get_dimension()
        if self.dims not in [1, 2]:
            raise PINNError(f"Unsupported dimension: {self.dims}")
        
        # Get problem data
        self._setup_problem_data()
        
        # Setup network
        self._setup_network(net_params)
        
        # Setup training components
        self._setup_training(net_params)
        
        # Initialize loss components
        self._setup_loss()
        
        # Generate model name
        self.model_name = self._generate_name()
        
    def _setup_problem_data(self) -> None:
        """Setup problem data from conditions."""
        # Initial conditions
        self.x_initial, self.t_initial, self.u_initial = (
            self.problem.initial_conditions.get_initial_conditions()
        )
        
        # Boundary conditions
        self.x_boundary, self.t_boundary, self.u_boundary = (
            self.problem.boundary_conditions.get_boundary_conditions()
        )
        
        # Equation points
        self.x_equation, self.t_equation = (
            self.problem.equation.get_equation_points()
        )
        
        # PDE and coefficient
        self.pde = self.problem.equation.pde
        self.alpha = self.problem.alpha
        
    def _setup_network(self, net_params: utils.NetParams) -> None:
        """Setup neural network architecture."""
        self.input_dim = net_params.input
        self.output_dim = net_params.output
        self.hidden_layers = net_params.hidden_layers
        
        # Setup activation function
        self._setup_activation(net_params)
        
        # Build network
        self._build_network()
        
    def _setup_activation(self, net_params: utils.NetParams) -> None:
        """Setup activation function."""
        activations = {
            'tanh': torch.nn.Tanh(),
            'sigmoid': torch.nn.Sigmoid(),
            'sin': siren.Sin()
        }
        
        if net_params.activation not in activations:
            raise PINNError(f"Unsupported activation: {net_params.activation}")
        
        self.activation = activations[net_params.activation]
        
        # SIREN parameters
        if net_params.siren_params is not None:
            self.first_omega_0 = net_params.siren_params.first_omega_0
            self.hidden_omega_0 = net_params.siren_params.hidden_omega_0
            self.outermost_linear = net_params.siren_params.outermost_linear
        else:
            self.first_omega_0 = 30.0
            self.hidden_omega_0 = 30.0
            self.outermost_linear = True
            
    def _build_network(self) -> None:
        """Build the neural network."""
        if isinstance(self.activation, siren.Sin):
            self._build_siren_network()
        else:
            self._build_fcn_network()
            
        self.net.to(self.device)
        
    def _build_siren_network(self) -> None:
        """Build SIREN network."""
        layers = []
        
        # First layer
        layers.append(siren.SineLayer(
            self.input_dim, self.hidden_layers[0],
            is_first=True, omega_0=self.first_omega_0
        ))
        
        # Hidden layers
        for i in range(1, len(self.hidden_layers)):
            layers.append(siren.SineLayer(
                self.hidden_layers[i-1], self.hidden_layers[i],
                is_first=False, omega_0=self.hidden_omega_0
            ))
        
        # Output layer
        if self.outermost_linear:
            final_linear = torch.nn.Linear(self.hidden_layers[-1], self.output_dim)
            with torch.no_grad():
                final_linear.weight.uniform_(
                    -np.sqrt(6 / self.hidden_layers[-1]) / self.hidden_omega_0,
                    np.sqrt(6 / self.hidden_layers[-1]) / self.hidden_omega_0
                )
            layers.append(final_linear)
        else:
            layers.append(siren.SineLayer(
                self.hidden_layers[-1], self.output_dim,
                is_first=False, omega_0=self.hidden_omega_0
            ))
            
        self.net = torch.nn.Sequential(*layers)
        
    def _build_fcn_network(self) -> None:
        """Build fully connected network."""
        layers = []
        
        # First layer
        layers.append(torch.nn.Linear(self.input_dim, self.hidden_layers[0]))
        layers.append(self.activation)
        
        # Hidden layers
        for i in range(1, len(self.hidden_layers)):
            layers.append(torch.nn.Linear(
                self.hidden_layers[i-1], self.hidden_layers[i]
            ))
            layers.append(self.activation)
        
        # Output layer
        layers.append(torch.nn.Linear(self.hidden_layers[-1], self.output_dim))
        
        self.net = torch.nn.Sequential(*layers)
        
    def _setup_training(self, net_params: utils.NetParams) -> None:
        """Setup training components."""
        self.epochs = net_params.epochs
        self.batch_size = net_params.batch_size
        self.training_mode = net_params.training_mode
        
        # Setup optimizer
        self._setup_optimizer(net_params)
        
        # Setup scheduler
        self._setup_scheduler(net_params)
        
        # Setup other training components
        self._setup_enhancements(net_params)
        
    def _setup_optimizer(self, net_params: utils.NetParams) -> None:
        """Setup optimizer."""
        if net_params.optimizer == 'LBFGS':
            self.optimizer = torch.optim.LBFGS(
                self.net.parameters(),
                lr=net_params.lr,
                max_iter=10000,
                max_eval=100000,
                tolerance_grad=1e-05,
                tolerance_change=1e-09,
                history_size=100,
                line_search_fn='strong_wolfe'
            )
        elif net_params.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.net.parameters(),
                lr=net_params.lr
            )
        elif net_params.optimizer == 'Hybrid':
            self.optimizer = enhancements.HybridOptimizer(
                self.net.parameters(),
                switch_epoch=1000,
                switch_threshold=0.01
            )
            self.optimizer.set_optimizer_adam(
                torch.optim.Adam(self.net.parameters())
            )
            self.optimizer.set_optimizer_lbfgs(
                torch.optim.LBFGS(self.net.parameters())
            )
            self.optimizer.use_optimizer_adam()
        else:
            raise PINNError(f"Unsupported optimizer: {net_params.optimizer}")
            
    def _setup_scheduler(self, net_params: utils.NetParams) -> None:
        """Setup learning rate scheduler."""
        if net_params.optimizer == 'Hybrid':
            self.scheduler = None
        else:
            schedulers = {
                'StepLR': torch.optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=1000, gamma=0.5
                ),
                'ExponentialLR': torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizer, gamma=0.9
                ),
                'ReduceLROnPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, 'min', patience=10
                ),
                None: None
            }
            
            if net_params.scheduler not in schedulers:
                raise PINNError(f"Unsupported scheduler: {net_params.scheduler}")
                
            self.scheduler = schedulers[net_params.scheduler]
            
    def _setup_enhancements(self, net_params: utils.NetParams) -> None:
        """Setup training enhancements."""
        # Early stopping
        if net_params.early_stopping:
            self.early_stopping = enhancements.EarlyStopping(100)
        else:
            self.early_stopping = None
            
        # RAR
        if net_params.use_rar:
            self.use_rar = True
            self.rar_epsilon = 0.05
            self.rar_num_points = 10
            self.rar_interval = 50
            self.rar_random = False
        else:
            self.use_rar = False
            
        # Weight adjuster
        if net_params.use_weights_adjuster:
            self.adjuster = enhancements.LossWeightAdjuster(1e6, 1, 1e-3, 10)
        else:
            self.adjuster = None
            
        # Other parameters
        self.display_interval = net_params.display_interval
        self.model_save_path = net_params.model_save_path
        self.output_path = net_params.output_path
        self.save_loss = net_params.save_loss
        
        # Load initial weights if specified
        if net_params.initial_weights_path:
            self.load_weights(net_params.initial_weights_path)
            
    def _setup_loss(self) -> None:
        """Setup loss function components."""
        self.loss = 0.0
        self.mse = torch.nn.MSELoss()
        self.weight_eq = DEFAULT_WEIGHT_EQ
        self.weight_bc = DEFAULT_WEIGHT_BC
        self.weight_ic = DEFAULT_WEIGHT_IC
        self.null = torch.zeros((len(self.x_equation), 1), device=self.device)
        
    def _generate_name(self) -> str:
        """Generate unique model name."""
        base_name = (f"HL_{len(self.hidden_layers)}_A_{self.activation}_"
                    f"N_{self.hidden_layers[-1]}")
        
        # Find existing models
        existing_models = []
        if os.path.exists(self.output_path):
            existing_models.extend(os.listdir(self.output_path))
        if os.path.exists(self.model_save_path):
            existing_models.extend(os.listdir(self.model_save_path))
            
        # Extract numbers
        existing_numbers = []
        for name in existing_models:
            if name.startswith(base_name):
                try:
                    num = int(name.split('_')[-1].split('.')[0])
                    existing_numbers.append(num)
                except (ValueError, IndexError):
                    continue
                    
        next_number = 0 if not existing_numbers else max(existing_numbers) + 1
        return f"{base_name}_{next_number}"
        
    def load_weights(self, file_path: str) -> None:
        """Load model weights from file.
        
        Args:
            file_path: Path to the weights file.
        """
        model_weights = torch.load(file_path, map_location=self.device)
        self.net.load_state_dict(model_weights)
        print(f"Model weights loaded from {file_path}")
        
    def save_weights(self) -> None:
        """Save model weights to file."""
        utils.create_folder(self.model_save_path)
        model_weights = {}
        for name, param in self.net.named_parameters():
            model_weights[name] = param.data
        torch.save(model_weights, f'{self.model_save_path}/{self.model_name}.pth')
        print(f"Model weights saved to {self.model_save_path}")
        
    def train(self) -> None:
        """Train the PINN."""
        self.net.train()
        print(f"Optimizer: {self.optimizer.__class__.__name__}")
        
        self.start_time = time.time()
        
        if isinstance(self.optimizer, torch.optim.LBFGS):
            self.optimizer.zero_grad()
            self.loss = self.optimizer.step(self.closure)
            self._post_optimization_actions(0)
        elif isinstance(self.optimizer, torch.optim.Adam):
            for epoch in range(self.epochs):
                self.optimizer.zero_grad()
                self.loss = self.closure()
                self.optimizer.step()
                self._post_optimization_actions(epoch)
                
                if self.early_stopping is not None:
                    self.early_stopping.step(self.loss.item())
                    if self.early_stopping.early_stop:
                        print(f"Early stopping triggered at epoch {epoch}")
                        break
        else:
            # Hybrid optimizer
            local_epoch = 0
            while isinstance(self.optimizer.get_current_optimizer(), 
                           torch.optim.Adam):
                self.optimizer.zero_grad()
                self.loss = self.closure()
                self.optimizer.step(local_epoch, self.closure)
                self._post_optimization_actions(local_epoch)
                local_epoch += 1
                
                if self.early_stopping is not None:
                    self.early_stopping.step(self.loss.item())
                    if self.early_stopping.early_stop:
                        print(f"Early stopping triggered at epoch {local_epoch}")
                        break
                        
            if isinstance(self.optimizer.get_current_optimizer(), 
                         torch.optim.LBFGS):
                self.optimizer.zero_grad()
                self.loss = self.closure()
                self.optimizer.step(local_epoch, self.closure)
                
    def closure(self) -> torch.Tensor:
        """Compute loss and gradients for optimization step."""
        if hasattr(self.optimizer, 'step_closure'):
            return self.optimizer.step_closure(self)
        else:
            self.optimizer.zero_grad()
            
            # Compute losses
            self._compute_losses()
            
            # Compute total loss
            self.loss = (self.weight_ic * self.initial_loss + 
                        self.weight_bc * self.boundary_loss + 
                        self.weight_eq * self.equation_loss)
            
            # Add regularization if specified
            self._add_regularization()
            
            # Backward pass
            self.loss.backward(retain_graph=True)
            
            # Print progress
            self._print_progress()
            
            return self.loss
            
    def _compute_losses(self) -> None:
        """Compute individual loss components."""
        # Initial conditions loss
        u_pred_ic = self._forward_pass(self.x_initial, self.t_initial)
        self.initial_loss = self.mse(u_pred_ic, self.u_initial)
        
        # Boundary conditions loss
        u_pred_bc = self._forward_pass(self.x_boundary, self.t_boundary)
        self.boundary_loss = self.mse(u_pred_bc, self.u_boundary)
        
        # Equation loss
        derivatives_names = ['dudt', 'd2udx2', 'd2udy2']
        _, derivatives = self._forward_pass_with_derivatives(
            self.x_equation, self.t_equation, derivatives_names
        )
        
        heat_eq_prediction = self.pde.substitute_into_equation(derivatives)
        self.equation_loss = self.mse(heat_eq_prediction, self.null)
        
    def _add_regularization(self) -> None:
        """Add regularization terms to loss."""
        # This would be implemented based on net_params.regularization
        pass
        
    def _print_progress(self) -> None:
        """Print training progress."""
        if hasattr(self, 'iter'):
            self.iter += 1
        else:
            self.iter = 1
            
        if self.iter % self.display_interval == 0:
            exact_loss = (self.initial_loss + self.boundary_loss + 
                         self.equation_loss)
            elapsed_time = time.time() - self.start_time
            print(f'Iteration {self.iter}: Loss {exact_loss:.6f}, '
                  f'Weighted Loss {self.loss.item():.6f}, '
                  f'Time {elapsed_time:.2f}s')
                  
    def _forward_pass(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        if self.dims == 1:
            return self.net(torch.cat((x, t), dim=1))
        elif self.dims == 2:
            x_component = x[:, 0].reshape(-1, 1)
            y_component = x[:, 1].reshape(-1, 1)
            return self.net(torch.cat((x_component, y_component, t), dim=1))
        else:
            raise PINNError(f"Unsupported dimension: {self.dims}")
            
    def _forward_pass_with_derivatives(self, x: torch.Tensor, t: torch.Tensor,
                                     derivatives: List[str]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass with derivative computation."""
        results = {}
        
        if self.dims == 1:
            u_pred = self._forward_pass(x, t)
            if 'dudx' in derivatives:
                dudx = torch.autograd.grad(
                    u_pred, x, grad_outputs=torch.ones_like(u_pred),
                    create_graph=True, allow_unused=True
                )[0]
                results['dudx'] = dudx
                
            if 'd2udx2' in derivatives:
                dudx = torch.autograd.grad(
                    u_pred, x, grad_outputs=torch.ones_like(u_pred),
                    create_graph=True, allow_unused=True
                )[0]
                d2udx2 = torch.autograd.grad(
                    dudx, x, grad_outputs=torch.ones_like(dudx),
                    create_graph=True, allow_unused=True
                )[0]
                results['d2udx2'] = d2udx2
                
        elif self.dims == 2:
            x_component = x[:, 0].reshape(-1, 1)
            y_component = x[:, 1].reshape(-1, 1)
            u_pred = self.net(torch.cat((x_component, y_component, t), dim=1))
            
            if 'dudx' in derivatives:
                dudx = torch.autograd.grad(
                    u_pred, x_component, grad_outputs=torch.ones_like(u_pred),
                    create_graph=True, allow_unused=True
                )[0]
                results['dudx'] = dudx
                
            if 'dudy' in derivatives:
                dudy = torch.autograd.grad(
                    u_pred, y_component, grad_outputs=torch.ones_like(u_pred),
                    create_graph=True, allow_unused=True
                )[0]
                results['dudy'] = dudy
                
            if 'd2udx2' in derivatives:
                dudx = torch.autograd.grad(
                    u_pred, x_component, grad_outputs=torch.ones_like(u_pred),
                    create_graph=True, allow_unused=True
                )[0]
                d2udx2 = torch.autograd.grad(
                    dudx, x_component, grad_outputs=torch.ones_like(dudx),
                    create_graph=True, allow_unused=True
                )[0]
                results['d2udx2'] = d2udx2
                
            if 'd2udy2' in derivatives:
                dudy = torch.autograd.grad(
                    u_pred, y_component, grad_outputs=torch.ones_like(u_pred),
                    create_graph=True, allow_unused=True
                )[0]
                d2udy2 = torch.autograd.grad(
                    dudy, y_component, grad_outputs=torch.ones_like(dudy),
                    create_graph=True, allow_unused=True
                )[0]
                results['d2udy2'] = d2udy2
                
        if 'dudt' in derivatives:
            dudt = torch.autograd.grad(
                u_pred, t, grad_outputs=torch.ones_like(u_pred),
                create_graph=True, allow_unused=True
            )[0]
            results['dudt'] = dudt
            
        return u_pred, results
        
    def _post_optimization_actions(self, epoch: int) -> None:
        """Actions to perform after each optimization step."""
        # Update scheduler
        if self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(self.loss.item())
            else:
                self.scheduler.step()
                
        # Weight calibration
        if epoch % 10 == 0 and self.adjuster is not None:
            weights = [self.weight_ic, self.weight_bc, self.weight_eq]
            losses = [self.initial_loss.item(), 
                    self.boundary_loss.item(), 
                    self.equation_loss.item()]
            self.weight_ic, self.weight_bc, self.weight_eq = (
                self.adjuster.adjust_weights(weights, losses)
            )
            print(f"Adjusted weights: {self.weight_ic}, {self.weight_bc}, {self.weight_eq}")
            
    def predict(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Make predictions with the trained model.
        
        Args:
            x: Spatial coordinates.
            t: Time coordinates.
            
        Returns:
            Predicted values.
        """
        self.net.eval()
        with torch.no_grad():
            u_pred = self._forward_pass(x, t)
        return u_pred
        
    def get_loss_history(self) -> Optional[str]:
        """Get path to loss history file if available."""
        if self.save_loss:
            return f'{self.output_path}/{self.model_name}.csv'
        return None
