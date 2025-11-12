__all__ = ["get", "is_external_optimizer", "HybridMuonAdam", "GeneticAlgorithm"]

import torch

from ..config import LBFGS_options, Muon_options, GA_options
from .genetic import GeneticAlgorithm


class HybridMuonAdam(torch.optim.Optimizer):
    """Hybrid optimizer that applies Muon to 2D parameters and Adam to 1D parameters.
    
    This is necessary because Muon only supports 2D parameters (weight matrices),
    while neural networks often have 1D parameters (biases).
    """
    
    def __init__(self, params, lr, weight_decay, muon_options):
        """
        Args:
            params: List of parameters to optimize
            lr: Learning rate
            weight_decay: Weight decay coefficient
            muon_options: Dictionary of Muon-specific options
        """
        # Separate parameters by dimensionality BEFORE calling super().__init__
        muon_params = []
        adam_params = []
        
        # Convert params to list if it's a generator
        params_list = list(params)
        
        for p in params_list:
            if p.ndim >= 2:
                muon_params.append(p)
            else:
                adam_params.append(p)
        
        # Initialize parent with all params
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params_list, defaults)
        
        # Create separate optimizers
        if len(muon_params) > 0:
            self.muon = torch.optim.Muon(
                muon_params,
                lr=lr,
                weight_decay=weight_decay,
                momentum=muon_options["momentum"],
                nesterov=muon_options["nesterov"],
                ns_coefficients=muon_options["ns_coefficients"],
                eps=muon_options["eps"],
                ns_steps=muon_options["ns_steps"],
            )
        else:
            self.muon = None
            
        if len(adam_params) > 0:
            self.adam = torch.optim.Adam(adam_params, lr=lr, weight_decay=weight_decay)
        else:
            self.adam = None
    
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if self.muon is not None:
            loss = self.muon.step(closure)
        if self.adam is not None:
            # For Adam, we only call closure once (already called by Muon)
            loss = self.adam.step(None)
        return loss
    
    def zero_grad(self, set_to_none=False):
        """Zero out the gradients."""
        if self.muon is not None:
            self.muon.zero_grad(set_to_none=set_to_none)
        if self.adam is not None:
            self.adam.zero_grad(set_to_none=set_to_none)


# NOTE: edited
def is_external_optimizer(optimizer):
    """Check if optimizer requires special handling (closure-based)."""
    if isinstance(optimizer, torch.optim.Optimizer):
        return isinstance(optimizer, (torch.optim.LBFGS, GeneticAlgorithm))
    return optimizer in ["L-BFGS", "L-BFGS-B", "genetic", "ga"]


def get(params, optimizer, learning_rate=None, decay=None, weight_decay=0):
    """Retrieves an Optimizer instance."""
    # Custom Optimizer
    if isinstance(optimizer, torch.optim.Optimizer):
        optim = optimizer
    elif optimizer in ["L-BFGS", "L-BFGS-B"]:
        if weight_decay > 0:
            raise ValueError("L-BFGS optimizer doesn't support weight_decay > 0")
        if learning_rate is not None or decay is not None:
            print("Warning: learning rate is ignored for {}".format(optimizer))
        optim = torch.optim.LBFGS(
            params,
            lr=1,
            max_iter=LBFGS_options["iter_per_step"],
            max_eval=LBFGS_options["fun_per_step"],
            tolerance_grad=LBFGS_options["gtol"],
            tolerance_change=LBFGS_options["ftol"],
            history_size=LBFGS_options["maxcor"],
            line_search_fn=None,
        )
    else:
        if learning_rate is None:
            raise ValueError("No learning rate for {}.".format(optimizer))
        if optimizer == "sgd":
            optim = torch.optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif optimizer == "rmsprop":
            optim = torch.optim.RMSprop(params, lr=learning_rate, weight_decay=weight_decay)
        elif optimizer == "adam":
            optim = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif optimizer == "adamw":
            if weight_decay == 0:
                raise ValueError("AdamW optimizer requires non-zero weight decay")
            optim = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
        elif optimizer == "muon":
            # Muon optimizer with Newton-Schulz preconditioner
            # NOTE: Muon only supports 2D parameters (weight matrices)
            # Use HybridMuonAdam to handle both 2D (weights) and 1D (biases) parameters
            muon_params = [p for p in params if p.ndim >= 2]
            adam_params = [p for p in params if p.ndim < 2]
            
            if len(muon_params) == 0:
                raise ValueError(
                    "Muon optimizer requires at least one 2D parameter (weight matrix). "
                    "Consider using a network with weight matrices or switch to another optimizer."
                )
            
            if len(adam_params) > 0:
                print(f"Note: Using hybrid optimizer - Muon for {len(muon_params)} 2D parameters (weights), "
                      f"Adam for {len(adam_params)} 1D parameters (biases)")
                optim = HybridMuonAdam(params, learning_rate, weight_decay, Muon_options)
            else:
                # All parameters are 2D, use pure Muon
                print(f"Using Muon for all {len(muon_params)} 2D parameters")
                optim = torch.optim.Muon(
                    params,
                    lr=learning_rate,
                    weight_decay=weight_decay,
                    momentum=Muon_options["momentum"],
                    nesterov=Muon_options["nesterov"],
                    ns_coefficients=Muon_options["ns_coefficients"],
                    eps=Muon_options["eps"],
                    ns_steps=Muon_options["ns_steps"],
                )
        elif optimizer in ["genetic", "ga"]:
            # Genetic Algorithm optimizer (gradient-free evolutionary optimization)
            print(f"Using Genetic Algorithm optimizer (population={GA_options['population_size']})")
            optim = GeneticAlgorithm(
                params,
                lr=learning_rate,  # Used as mutation scale multiplier
                population_size=GA_options["population_size"],
                mutation_rate=GA_options["mutation_rate"],
                mutation_scale=GA_options["mutation_scale"],
                crossover_rate=GA_options["crossover_rate"],
                selection_method=GA_options["selection_method"],
                tournament_size=GA_options["tournament_size"],
                elitism=GA_options["elitism"],
                generations_per_step=GA_options["generations_per_step"],
            )
        else:
            raise NotImplementedError(f"{optimizer} to be implemented for backend pytorch.")
    lr_scheduler = _get_learningrate_scheduler(optim, decay)
    return optim, lr_scheduler


def _get_learningrate_scheduler(optim, decay):
    if decay is None:
        return None

    # NOTE: edited
    if isinstance(decay, torch.optim.lr_scheduler._LRScheduler) or decay.__class__.__name__ == "ReduceLROnPlateau":
        return decay

    if decay[0] == "step":
        return torch.optim.lr_scheduler.StepLR(optim, step_size=decay[1], gamma=decay[2])

    # TODO: More learning rate scheduler
    raise NotImplementedError(f"{decay[0]} learning rate scheduler to be implemented for backend pytorch.")
