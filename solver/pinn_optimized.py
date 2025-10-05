"""Optimized PINN implementation with GPU support and efficient derivative computation."""

import os
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from . import callbacks
from . import conditions
from . import metrics
from . import enhancements
from . import siren
from . import utils
from .constants import (
    DEFAULT_WEIGHT_IC, DEFAULT_WEIGHT_BC, DEFAULT_WEIGHT_EQ,
    DEFAULT_DEVICE, CPU_DEVICE
)
from .exceptions import PINNError, TrainingError, DeviceError


class OptimizedPINN:
    """Optimized Physics-Informed Neural Network with GPU support and efficient computations.
    
    This class provides an optimized implementation of PINN with:
    - Efficient derivative computation using vectorized operations
    - Full GPU support for all operations
    - Batching for large-scale problems
    - Memory-efficient training
    """
    
    def __init__(self, problem: conditions.Problem, net_params: utils.NetParams,
                 device: str = DEFAULT_DEVICE, batch_size: Optional[int] = None):
        """Initialize optimized PINN.
        
        Args:
            problem: Problem definition containing geometry, conditions, and PDE.
            net_params: Network parameters and training configuration.
            device: Device to run computations on.
            batch_size: Batch size for training. If None, uses net_params.batch_size.
            
        Raises:
            PINNError: If configuration is invalid.
            DeviceError: If device is not available.
        """
        self.problem = problem
        self.device = self._setup_device(device)
        self.batch_size = batch_size or net_params.batch_size
        
        # Validate problem dimensions
        self.dims = problem.geom.get_dimension()
        if self.dims not in [1, 2]:
            raise PINNError(f"Unsupported dimension: {self.dims}")
        
        # Get problem data and move to device
        self._setup_problem_data()
        
        # Setup network
        self._setup_network(net_params)
        
        # Setup training components
        self._setup_training(net_params)
        
        # Initialize loss components
        self._setup_loss()
        
        # Setup batching
        self._setup_batching()
        
        # Generate model name
        self.model_name = self._generate_name()
        
    def _setup_device(self, device: str) -> torch.device:
        """Setup and validate device.
        
        Args:
            device: Device string ('cuda', 'cpu', etc.)
            
        Returns:
            PyTorch device object.
            
        Raises:
            DeviceError: If requested device is not available.
        """
        if device.startswith('cuda') and not torch.cuda.is_available():
            print(f"CUDA not available, falling back to CPU")
            device = CPU_DEVICE
            
        device_obj = torch.device(device)
        
        if device_obj.type == 'cuda':
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        return device_obj
        
    def _setup_problem_data(self) -> None:
        """Setup problem data and move to device."""
        # Initial conditions
        self.x_initial, self.t_initial, self.u_initial = (
            self.problem.initial_conditions.get_initial_conditions()
        )
        self.x_initial = self.x_initial.to(self.device)
        self.t_initial = self.t_initial.to(self.device)
        self.u_initial = self.u_initial.to(self.device)
        
        # Boundary conditions
        self.x_boundary, self.t_boundary, self.u_boundary = (
            self.problem.boundary_conditions.get_boundary_conditions()
        )
        self.x_boundary = self.x_boundary.to(self.device)
        self.t_boundary = self.t_boundary.to(self.device)
        self.u_boundary = self.u_boundary.to(self.device)
        
        # Equation points
        self.x_equation, self.t_equation = (
            self.problem.equation.get_equation_points()
        )
        self.x_equation = self.x_equation.to(self.device)
        self.t_equation = self.t_equation.to(self.device)
        
        # PDE and coefficient
        self.pde = self.problem.equation.pde
        self.alpha = torch.tensor(self.problem.alpha, device=self.device, dtype=torch.float32)
        
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
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
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
        
        # Initialize weights
        self._initialize_weights()
        
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
            final_linear = nn.Linear(self.hidden_layers[-1], self.output_dim)
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
            
        self.net = nn.Sequential(*layers)
        
    def _build_fcn_network(self) -> None:
        """Build fully connected network with optimizations."""
        layers = []
        
        # First layer
        layers.append(nn.Linear(self.input_dim, self.hidden_layers[0]))
        layers.append(self.activation)
        
        # Hidden layers with batch normalization and dropout
        for i in range(1, len(self.hidden_layers)):
            layers.append(nn.Linear(self.hidden_layers[i-1], self.hidden_layers[i]))
            layers.append(nn.BatchNorm1d(self.hidden_layers[i]))
            layers.append(self.activation)
            layers.append(nn.Dropout(0.1))  # Small dropout for regularization
        
        # Output layer
        layers.append(nn.Linear(self.hidden_layers[-1], self.output_dim))
        
        self.net = nn.Sequential(*layers)
        
    def _initialize_weights(self) -> None:
        """Initialize network weights using Xavier/He initialization."""
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                if isinstance(self.activation, (nn.ReLU, nn.GELU)):
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                else:
                    nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def _setup_training(self, net_params: utils.NetParams) -> None:
        """Setup training components."""
        self.epochs = net_params.epochs
        self.training_mode = net_params.training_mode
        
        # Setup optimizer
        self._setup_optimizer(net_params)
        
        # Setup scheduler
        self._setup_scheduler(net_params)
        
        # Setup other training components
        self._setup_enhancements(net_params)
        
    def _setup_optimizer(self, net_params: utils.NetParams) -> None:
        """Setup optimizer with optimizations."""
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
                lr=net_params.lr,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=1e-4
            )
        elif net_params.optimizer == 'AdamW':
            self.optimizer = torch.optim.AdamW(
                self.net.parameters(),
                lr=net_params.lr,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=1e-4
            )
        elif net_params.optimizer == 'Hybrid':
            self.optimizer = enhancements.HybridOptimizer(
                self.net.parameters(),
                switch_epoch=1000,
                switch_threshold=0.01
            )
            self.optimizer.set_optimizer_adam(
                torch.optim.Adam(self.net.parameters(), lr=net_params.lr)
            )
            self.optimizer.set_optimizer_lbfgs(
                torch.optim.LBFGS(self.net.parameters(), lr=net_params.lr)
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
                    self.optimizer, 'min', patience=10, factor=0.5
                ),
                'CosineAnnealingLR': torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=self.epochs, eta_min=1e-6
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
        self.mse = nn.MSELoss()
        self.weight_eq = DEFAULT_WEIGHT_EQ
        self.weight_bc = DEFAULT_WEIGHT_BC
        self.weight_ic = DEFAULT_WEIGHT_IC
        
        # Pre-allocate tensors for efficiency
        self.null = torch.zeros((len(self.x_equation), 1), device=self.device)
        
    def _setup_batching(self) -> None:
        """Setup batching for large-scale problems."""
        # Calculate number of batches
        self.num_ic_batches = max(1, len(self.x_initial) // self.batch_size)
        self.num_bc_batches = max(1, len(self.x_boundary) // self.batch_size)
        self.num_eq_batches = max(1, len(self.x_equation) // self.batch_size)
        
        # Create batch indices
        self.ic_batch_indices = self._create_batch_indices(len(self.x_initial), self.num_ic_batches)
        self.bc_batch_indices = self._create_batch_indices(len(self.x_boundary), self.num_bc_batches)
        self.eq_batch_indices = self._create_batch_indices(len(self.x_equation), self.num_eq_batches)
        
    def _create_batch_indices(self, total_size: int, num_batches: int) -> List[torch.Tensor]:
        """Create batch indices for efficient batching."""
        indices = torch.randperm(total_size, device=self.device)
        batch_size = total_size // num_batches
        batch_indices = []
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size if i < num_batches - 1 else total_size
            batch_indices.append(indices[start_idx:end_idx])
            
        return batch_indices
        
    def _generate_name(self) -> str:
        """Generate unique model name."""
        base_name = (f"OptPINN_HL_{len(self.hidden_layers)}_A_{self.activation.__class__.__name__}_"
                    f"N_{self.hidden_layers[-1]}_D_{self.device.type.upper()}")
        
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
            model_weights[name] = param.data.cpu()  # Move to CPU for saving
        torch.save(model_weights, f'{self.model_save_path}/{self.model_name}.pth')
        print(f"Model weights saved to {self.model_save_path}")
        
    def train(self) -> None:
        """Train the optimized PINN."""
        self.net.train()
        print(f"Optimizer: {self.optimizer.__class__.__name__}")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.batch_size}")
        
        self.start_time = time.time()
        
        if isinstance(self.optimizer, torch.optim.LBFGS):
            self.optimizer.zero_grad()
            self.loss = self.optimizer.step(self.closure)
            self._post_optimization_actions(0)
        elif isinstance(self.optimizer, torch.optim.Adam) or isinstance(self.optimizer, torch.optim.AdamW):
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
                           (torch.optim.Adam, torch.optim.AdamW)):
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
                        
            if isinstance(self.optimizer.get_current_optimizer(), torch.optim.LBFGS):
                self.optimizer.zero_grad()
                self.loss = self.closure()
                self.optimizer.step(local_epoch, self.closure)
                
    def closure(self) -> torch.Tensor:
        """Compute loss and gradients for optimization step with batching."""
        if hasattr(self.optimizer, 'step_closure'):
            return self.optimizer.step_closure(self)
        else:
            self.optimizer.zero_grad()
            
            # Compute losses with batching
            self._compute_losses_batched()
            
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
            
    def _compute_losses_batched(self) -> None:
        """Compute individual loss components using batching."""
        # Initial conditions loss (batched)
        ic_losses = []
        for batch_idx in self.ic_batch_indices:
            x_batch = self.x_initial[batch_idx]
            t_batch = self.t_initial[batch_idx]
            u_batch = self.u_initial[batch_idx]
            
            u_pred = self._forward_pass(x_batch, t_batch)
            ic_losses.append(self.mse(u_pred, u_batch))
        self.initial_loss = torch.stack(ic_losses).mean()
        
        # Boundary conditions loss (batched)
        bc_losses = []
        for batch_idx in self.bc_batch_indices:
            x_batch = self.x_boundary[batch_idx]
            t_batch = self.t_boundary[batch_idx]
            u_batch = self.u_boundary[batch_idx]
            
            u_pred = self._forward_pass(x_batch, t_batch)
            bc_losses.append(self.mse(u_pred, u_batch))
        self.boundary_loss = torch.stack(bc_losses).mean()
        
        # Equation loss (batched)
        eq_losses = []
        for batch_idx in self.eq_batch_indices:
            x_batch = self.x_equation[batch_idx]
            t_batch = self.t_equation[batch_idx]
            
            derivatives_names = ['dudt', 'd2udx2', 'd2udy2']
            _, derivatives = self._forward_pass_with_derivatives_optimized(
                x_batch, t_batch, derivatives_names
            )
            
            heat_eq_prediction = self.pde.substitute_into_equation(derivatives)
            null_batch = torch.zeros_like(heat_eq_prediction)
            eq_losses.append(self.mse(heat_eq_prediction, null_batch))
        self.equation_loss = torch.stack(eq_losses).mean()
        
    def _add_regularization(self) -> None:
        """Add regularization terms to loss."""
        # L2 regularization
        l2_reg = 0.0
        for param in self.net.parameters():
            l2_reg += torch.norm(param, p=2)
        self.loss += 1e-6 * l2_reg
        
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
            
            # GPU memory usage
            if self.device.type == 'cuda':
                memory_used = torch.cuda.memory_allocated() / 1e9
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                memory_info = f", GPU Memory: {memory_used:.2f}/{memory_total:.2f} GB"
            else:
                memory_info = ""
                
            print(f'Iteration {self.iter}: Loss {exact_loss:.6f}, '
                  f'Weighted Loss {self.loss.item():.6f}, '
                  f'Time {elapsed_time:.2f}s{memory_info}')
                  
    def _forward_pass(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Optimized forward pass through the network."""
        if self.dims == 1:
            return self.net(torch.cat((x, t), dim=1))
        elif self.dims == 2:
            x_component = x[:, 0].reshape(-1, 1)
            y_component = x[:, 1].reshape(-1, 1)
            return self.net(torch.cat((x_component, y_component, t), dim=1))
        else:
            raise PINNError(f"Unsupported dimension: {self.dims}")
            
    def _forward_pass_with_derivatives_optimized(self, x: torch.Tensor, t: torch.Tensor,
                                               derivatives: List[str]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Optimized forward pass with derivative computation using vectorized operations."""
        results = {}
        
        # Enable gradient computation
        x.requires_grad_(True)
        t.requires_grad_(True)
        
        if self.dims == 1:
            u_pred = self._forward_pass(x, t)
            
            # Compute derivatives efficiently
            if 'dudx' in derivatives:
                dudx = torch.autograd.grad(
                    u_pred, x, grad_outputs=torch.ones_like(u_pred),
                    create_graph=True, allow_unused=True, retain_graph=True
                )[0]
                results['dudx'] = dudx
                
            if 'd2udx2' in derivatives:
                if 'dudx' not in results:
                    dudx = torch.autograd.grad(
                        u_pred, x, grad_outputs=torch.ones_like(u_pred),
                        create_graph=True, allow_unused=True, retain_graph=True
                    )[0]
                else:
                    dudx = results['dudx']
                    
                d2udx2 = torch.autograd.grad(
                    dudx, x, grad_outputs=torch.ones_like(dudx),
                    create_graph=True, allow_unused=True, retain_graph=True
                )[0]
                results['d2udx2'] = d2udx2
                
        elif self.dims == 2:
            x_component = x[:, 0].reshape(-1, 1)
            y_component = x[:, 1].reshape(-1, 1)
            u_pred = self.net(torch.cat((x_component, y_component, t), dim=1))
            
            # Compute derivatives efficiently
            if 'dudx' in derivatives:
                dudx = torch.autograd.grad(
                    u_pred, x_component, grad_outputs=torch.ones_like(u_pred),
                    create_graph=True, allow_unused=True, retain_graph=True
                )[0]
                results['dudx'] = dudx
                
            if 'dudy' in derivatives:
                dudy = torch.autograd.grad(
                    u_pred, y_component, grad_outputs=torch.ones_like(u_pred),
                    create_graph=True, allow_unused=True, retain_graph=True
                )[0]
                results['dudy'] = dudy
                
            if 'd2udx2' in derivatives:
                if 'dudx' not in results:
                    dudx = torch.autograd.grad(
                        u_pred, x_component, grad_outputs=torch.ones_like(u_pred),
                        create_graph=True, allow_unused=True, retain_graph=True
                    )[0]
                else:
                    dudx = results['dudx']
                    
                d2udx2 = torch.autograd.grad(
                    dudx, x_component, grad_outputs=torch.ones_like(dudx),
                    create_graph=True, allow_unused=True, retain_graph=True
                )[0]
                results['d2udx2'] = d2udx2
                
            if 'd2udy2' in derivatives:
                if 'dudy' not in results:
                    dudy = torch.autograd.grad(
                        u_pred, y_component, grad_outputs=torch.ones_like(u_pred),
                        create_graph=True, allow_unused=True, retain_graph=True
                    )[0]
                else:
                    dudy = results['dudy']
                    
                d2udy2 = torch.autograd.grad(
                    dudy, y_component, grad_outputs=torch.ones_like(dudy),
                    create_graph=True, allow_unused=True, retain_graph=True
                )[0]
                results['d2udy2'] = d2udy2
                
        if 'dudt' in derivatives:
            dudt = torch.autograd.grad(
                u_pred, t, grad_outputs=torch.ones_like(u_pred),
                create_graph=True, allow_unused=True, retain_graph=True
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
            print(f"Adjusted weights: {self.weight_ic:.2e}, {self.weight_bc:.2e}, {self.weight_eq:.2e}")
            
    def predict(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Make predictions with the trained model.
        
        Args:
            x: Spatial coordinates.
            t: Time coordinates.
            
        Returns:
            Predicted values.
        """
        self.net.eval()
        
        # Move inputs to device
        x = x.to(self.device)
        t = t.to(self.device)
        
        with torch.no_grad():
            u_pred = self._forward_pass(x, t)
        return u_pred
        
    def predict_batch(self, x: torch.Tensor, t: torch.Tensor, 
                     batch_size: Optional[int] = None) -> torch.Tensor:
        """Make predictions in batches for large datasets.
        
        Args:
            x: Spatial coordinates.
            t: Time coordinates.
            batch_size: Batch size for prediction. If None, uses self.batch_size.
            
        Returns:
            Predicted values.
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        self.net.eval()
        
        # Move inputs to device
        x = x.to(self.device)
        t = t.to(self.device)
        
        predictions = []
        num_points = len(x)
        
        with torch.no_grad():
            for i in range(0, num_points, batch_size):
                end_idx = min(i + batch_size, num_points)
                x_batch = x[i:end_idx]
                t_batch = t[i:end_idx]
                
                u_pred_batch = self._forward_pass(x_batch, t_batch)
                predictions.append(u_pred_batch)
                
        return torch.cat(predictions, dim=0)
        
    def get_loss_history(self) -> Optional[str]:
        """Get path to loss history file if available."""
        if self.save_loss:
            return f'{self.output_path}/{self.model_name}.csv'
        return None
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage information.
        
        Returns:
            Dictionary with memory usage information.
        """
        if self.device.type == 'cuda':
            return {
                'allocated': torch.cuda.memory_allocated() / 1e9,
                'cached': torch.cuda.memory_reserved() / 1e9,
                'total': torch.cuda.get_device_properties(0).total_memory / 1e9
            }
        else:
            import psutil
            process = psutil.Process()
            return {
                'rss': process.memory_info().rss / 1e9,
                'vms': process.memory_info().vms / 1e9
            }
