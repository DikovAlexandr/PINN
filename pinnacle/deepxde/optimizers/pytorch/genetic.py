"""
Genetic Algorithm Optimizer for PyTorch Neural Networks.

This module implements a genetic algorithm (GA) optimizer compatible with PyTorch's
optimization API. It uses evolutionary strategies to optimize neural network weights
without requiring gradient information.
"""

import torch
import numpy as np
from typing import List, Tuple, Callable, Optional


class GeneticAlgorithm(torch.optim.Optimizer):
    """Genetic Algorithm optimizer for neural networks.
    
    This optimizer uses evolutionary strategies to optimize network weights:
    1. Maintains a population of candidate solutions (weight sets)
    2. Evaluates fitness (inverse of loss) for each individual
    3. Selects parents based on fitness
    4. Creates offspring through crossover and mutation
    5. Replaces old population with new generation
    
    Unlike gradient-based optimizers, GA is:
    - Gradient-free (works with non-differentiable objectives)
    - Global search (less prone to local minima)
    - Robust to noisy/discontinuous loss landscapes
    - Computationally expensive (evaluates loss for entire population)
    
    Best used for:
    - Small to medium networks
    - Non-differentiable or noisy objectives
    - Initial exploration before fine-tuning with gradient methods
    - Reinforcement learning (policy optimization)
    """
    
    def __init__(
        self,
        params,
        lr=0.01,  # Used as mutation scale multiplier
        population_size=50,
        mutation_rate=0.1,
        mutation_scale=0.01,
        crossover_rate=0.7,
        selection_method="tournament",
        tournament_size=5,
        elitism=2,
        generations_per_step=1,
    ):
        """Initialize Genetic Algorithm optimizer.
        
        Args:
            params: Iterable of parameters to optimize
            lr: Learning rate (used as mutation scale multiplier)
            population_size: Number of individuals in population
            mutation_rate: Probability of mutating each weight
            mutation_scale: Standard deviation of Gaussian mutation
            crossover_rate: Probability of crossover between parents
            selection_method: Selection strategy ("tournament", "roulette", "rank", "sus")
            tournament_size: Size of tournament for tournament selection
            elitism: Number of best individuals to preserve
            generations_per_step: Number of generations to evolve per step()
        """
        defaults = dict(
            lr=lr,
            population_size=population_size,
            mutation_rate=mutation_rate,
            mutation_scale=mutation_scale,
            crossover_rate=crossover_rate,
            selection_method=selection_method,
            tournament_size=tournament_size,
            elitism=elitism,
            generations_per_step=generations_per_step,
        )
        super().__init__(params, defaults)
        
        # Store flat parameter vector for efficient operations
        self.param_shapes = []
        self.param_sizes = []
        self.total_params = 0
        
        for group in self.param_groups:
            for p in group['params']:
                self.param_shapes.append(p.shape)
                size = p.numel()
                self.param_sizes.append(size)
                self.total_params += size
        
        # Initialize population
        self.population_size = population_size
        self.population = None
        self.fitness = None
        self.generation = 0
        self.best_individual = None
        self.best_fitness = float('-inf')
        
        # Statistics
        self.history = {
            'best_fitness': [],
            'mean_fitness': [],
            'worst_fitness': [],
        }
        
        print(f"Genetic Algorithm Optimizer initialized:")
        print(f"  Population size: {population_size}")
        print(f"  Total parameters: {self.total_params:,}")
        print(f"  Selection method: {selection_method}")
        print(f"  Mutation rate: {mutation_rate}")
        print(f"  Crossover rate: {crossover_rate}")
        print(f"  Elitism: {elitism}")
    
    def _flatten_params(self) -> torch.Tensor:
        """Flatten all parameters into a single vector."""
        params = []
        for group in self.param_groups:
            for p in group['params']:
                params.append(p.data.flatten())
        return torch.cat(params)
    
    def _unflatten_params(self, flat_params: torch.Tensor):
        """Restore parameters from flattened vector.
        
        Note: For PDE problems, parameters must participate in computational graph.
        We use copy_ and ensure requires_grad is maintained.
        """
        offset = 0
        for group in self.param_groups:
            for p, shape, size in zip(group['params'], self.param_shapes, self.param_sizes):
                # Copy new values to parameter
                p.data.copy_(flat_params[offset:offset + size].reshape(shape))
                # Ensure requires_grad is True (needed for PDE derivatives)
                # Note: nn.Parameter should have requires_grad=True by default,
                # but we explicitly ensure it after copy
                if not p.requires_grad:
                    p.requires_grad = True
                offset += size
    
    def _initialize_population(self):
        """Initialize population around current parameters."""
        current_params = self._flatten_params()
        
        # Create population with small perturbations around current params
        self.population = []
        for i in range(self.population_size):
            if i == 0:
                # First individual is current parameters
                individual = current_params.clone()
            else:
                # Others are perturbed versions
                noise = torch.randn_like(current_params) * 0.1
                individual = current_params + noise
            self.population.append(individual)
        
        self.fitness = torch.zeros(self.population_size)
        print(f"Population initialized with {self.population_size} individuals")
    
    def _evaluate_fitness(self, closure: Callable) -> torch.Tensor:
        """Evaluate fitness for all individuals in population.
        
        Args:
            closure: Function that computes loss (lower is better)
            
        Returns:
            Fitness values (higher is better)
            
        Note:
            For PDE problems, we need gradients to compute derivatives (jacobian, hessian),
            but GA doesn't use backpropagation for weight updates - it evolves weights directly.
            We evaluate with gradients enabled, but skip backward pass.
        """
        # Import grad module for cache clearing
        from ... import gradients as grad
        
        fitness_values = []
        
        for i, individual in enumerate(self.population):
            # Set parameters to this individual
            self._unflatten_params(individual)
            
            # Clear gradient cache to ensure fresh computation for PDE
            grad.clear()
            
            # Evaluate loss with gradients enabled (needed for PDE)
            # but skip backward pass (GA doesn't use gradient descent)
            try:
                # Explicitly ensure gradients are enabled
                with torch.enable_grad():
                    loss = closure(skip_backward=True)
                fitness = -float(loss)
            except RuntimeError as e:
                # If gradient computation fails, assign worst fitness
                print(f"Warning: Fitness evaluation failed for individual {i}: {e}")
                fitness = float('-inf')
            
            fitness_values.append(fitness)
        
        return torch.tensor(fitness_values)
    
    def _selection_tournament(self, fitness: torch.Tensor, group: dict) -> torch.Tensor:
        """Tournament selection."""
        tournament_size = group['tournament_size']
        selected_indices = []
        
        for _ in range(self.population_size):
            # Random tournament
            tournament_idx = torch.randint(0, self.population_size, (tournament_size,))
            tournament_fitness = fitness[tournament_idx]
            winner_idx = tournament_idx[torch.argmax(tournament_fitness)]
            selected_indices.append(winner_idx)
        
        return torch.tensor(selected_indices)
    
    def _selection_roulette(self, fitness: torch.Tensor) -> torch.Tensor:
        """Roulette wheel selection (fitness-proportional)."""
        # Shift fitness to positive
        min_fitness = fitness.min()
        shifted_fitness = fitness - min_fitness + 1e-10
        
        # Probabilities proportional to fitness
        probabilities = shifted_fitness / shifted_fitness.sum()
        
        # Sample with replacement
        selected_indices = torch.multinomial(
            probabilities, 
            self.population_size, 
            replacement=True
        )
        return selected_indices
    
    def _selection_rank(self, fitness: torch.Tensor) -> torch.Tensor:
        """Rank-based selection."""
        # Sort by fitness and assign ranks
        sorted_indices = torch.argsort(fitness)
        ranks = torch.zeros_like(fitness)
        ranks[sorted_indices] = torch.arange(1, self.population_size + 1, dtype=fitness.dtype)
        
        # Probabilities based on rank
        probabilities = ranks / ranks.sum()
        
        selected_indices = torch.multinomial(
            probabilities,
            self.population_size,
            replacement=True
        )
        return selected_indices
    
    def _selection_sus(self, fitness: torch.Tensor) -> torch.Tensor:
        """Stochastic Universal Sampling (low variance)."""
        # Shift fitness to positive
        min_fitness = fitness.min()
        shifted_fitness = fitness - min_fitness + 1e-10
        
        total_fitness = shifted_fitness.sum()
        point_distance = total_fitness / self.population_size
        start_point = torch.rand(1) * point_distance
        
        selected_indices = []
        current_point = start_point
        cumulative_fitness = 0
        
        for i in range(self.population_size):
            cumulative_fitness += shifted_fitness[i]
            while current_point < cumulative_fitness:
                selected_indices.append(i)
                current_point += point_distance
        
        return torch.tensor(selected_indices[:self.population_size])
    
    def _select_parents(self, fitness: torch.Tensor, group: dict) -> torch.Tensor:
        """Select parents for next generation."""
        method = group['selection_method']
        
        if method == "tournament":
            return self._selection_tournament(fitness, group)
        elif method == "roulette":
            return self._selection_roulette(fitness)
        elif method == "rank":
            return self._selection_rank(fitness)
        elif method == "sus":
            return self._selection_sus(fitness)
        else:
            raise ValueError(f"Unknown selection method: {method}")
    
    def _crossover(
        self, 
        parent1: torch.Tensor, 
        parent2: torch.Tensor, 
        group: dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform crossover between two parents."""
        crossover_rate = group['crossover_rate']
        
        if torch.rand(1) < crossover_rate:
            # Uniform crossover: each gene randomly chosen from either parent
            mask = torch.rand(parent1.shape) < 0.5
            child1 = torch.where(mask, parent1, parent2)
            child2 = torch.where(mask, parent2, parent1)
        else:
            # No crossover
            child1 = parent1.clone()
            child2 = parent2.clone()
        
        return child1, child2
    
    def _mutate(self, individual: torch.Tensor, group: dict) -> torch.Tensor:
        """Apply mutation to an individual."""
        mutation_rate = group['mutation_rate']
        mutation_scale = group['mutation_scale'] * group['lr']
        
        # Each gene has mutation_rate probability of being mutated
        mutation_mask = torch.rand(individual.shape) < mutation_rate
        mutations = torch.randn(individual.shape) * mutation_scale
        
        mutated = individual.clone()
        mutated[mutation_mask] += mutations[mutation_mask]
        
        return mutated
    
    def _evolve_generation(self, group: dict):
        """Evolve population for one generation."""
        # Selection
        parent_indices = self._select_parents(self.fitness, group)
        
        # Elitism: preserve best individuals
        elitism = group['elitism']
        best_indices = torch.topk(self.fitness, elitism).indices
        elite = [self.population[idx].clone() for idx in best_indices]
        
        # Create offspring
        new_population = []
        
        # Add elite first
        new_population.extend(elite)
        
        # Generate rest through crossover and mutation
        i = 0
        while len(new_population) < self.population_size:
            # Select two parents
            parent1 = self.population[parent_indices[i]]
            parent2 = self.population[parent_indices[i + 1]]
            i += 2
            if i >= len(parent_indices):
                i = 0
            
            # Crossover
            child1, child2 = self._crossover(parent1, parent2, group)
            
            # Mutation
            child1 = self._mutate(child1, group)
            child2 = self._mutate(child2, group)
            
            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)
        
        self.population = new_population[:self.population_size]
        self.generation += 1
    
    @torch.no_grad()
    def step(self, closure: Callable):
        """Perform one optimization step (multiple GA generations).
        
        Args:
            closure: A closure that reevaluates the model and returns the loss.
                     Required for GA to evaluate fitness.
        
        Returns:
            Loss of the best individual
        """
        if closure is None:
            raise RuntimeError("GA optimizer requires closure to evaluate fitness")
        
        # Initialize population if first step
        if self.population is None:
            self._initialize_population()
        
        # Get parameters from first group (assume all groups have same config)
        group = self.param_groups[0]
        generations = group['generations_per_step']
        
        # Evolve for multiple generations
        for gen in range(generations):
            # Evaluate fitness
            self.fitness = self._evaluate_fitness(closure)
            
            # Track best
            best_idx = torch.argmax(self.fitness)
            best_fitness = self.fitness[best_idx].item()
            
            if best_fitness > self.best_fitness:
                self.best_fitness = best_fitness
                self.best_individual = self.population[best_idx].clone()
            
            # Statistics
            self.history['best_fitness'].append(best_fitness)
            self.history['mean_fitness'].append(self.fitness.mean().item())
            self.history['worst_fitness'].append(self.fitness.min().item())
            
            # Evolve to next generation (except last iteration)
            if gen < generations - 1:
                self._evolve_generation(group)
        
        # Set parameters to best individual
        if self.best_individual is not None:
            self._unflatten_params(self.best_individual)
        else:
            # If all individuals failed, keep current parameters
            print("Warning: All individuals failed fitness evaluation. Keeping current parameters.")
        
        # Return best loss (negative of fitness)
        return torch.tensor(-self.best_fitness)
    
    def get_statistics(self) -> dict:
        """Get optimization statistics."""
        return {
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'population_size': self.population_size,
            'history': self.history,
        }

