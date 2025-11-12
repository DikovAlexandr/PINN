__all__ = ["set_LBFGS_options", "set_Muon_options", "set_GA_options"]

from ..backend import backend_name


LBFGS_options = {}
Muon_options = {}
GA_options = {}


def set_LBFGS_options(
    maxcor=100,
    ftol=0,
    gtol=1e-8,
    maxiter=15000,
    maxfun=None,
    maxls=50,
):
    """Sets the hyperparameters of L-BFGS.

    Reference:
        PyTorch L-BFGS optimizer: `torch.optim.LBFGS <https://pytorch.org/docs/stable/generated/torch.optim.LBFGS.html>`_
    
    Args:
        maxcor (int): `maxcor` (scipy), `num_correction_pairs` (tfp), `history_size` (torch), `history_size` (paddle).
            The maximum number of variable metric corrections used to define the limited
            memory matrix. (The limited memory BFGS method does not store the full
            hessian but uses this many terms in an approximation to it.)
        ftol (float): `ftol` (scipy), `f_relative_tolerance` (tfp), `tolerance_change` (torch), `tolerance_change` (paddle).
            The iteration stops when `(f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol`.
        gtol (float): `gtol` (scipy), `tolerance` (tfp), `tolerance_grad` (torch), `tolerance_grad` (paddle).
            The iteration will stop when `max{|proj g_i | i = 1, ..., n} <= gtol` where
            `pg_i` is the i-th component of the projected gradient.
        maxiter (int): `maxiter` (scipy), `max_iterations` (tfp), `max_iter` (torch), `max_iter` (paddle).
            Maximum number of iterations.
        maxfun (int): `maxfun` (scipy), `max_eval` (torch), `max_eval` (paddle).
            Maximum number of function evaluations. If ``None``, `maxiter` * 1.25.
        maxls (int): `maxls` (scipy), `max_line_search_iterations` (tfp).
            Maximum number of line search steps (per iteration).

    Warning:
        If L-BFGS stops earlier than expected, set the default float type to 'float64':

        .. code-block:: python

            dde.config.set_default_float("float64")
    """
    global LBFGS_options
    LBFGS_options["maxcor"] = maxcor
    LBFGS_options["ftol"] = ftol
    LBFGS_options["gtol"] = gtol
    LBFGS_options["maxiter"] = maxiter
    LBFGS_options["maxfun"] = maxfun if maxfun is not None else int(maxiter * 1.25)
    LBFGS_options["maxls"] = maxls


set_LBFGS_options()


def set_Muon_options(
    momentum=0.95,
    nesterov=True,
    ns_coefficients=(3.4445, -4.775, 2.0315),
    eps=1e-07,
    ns_steps=5,
):
    """Sets the hyperparameters of Muon optimizer.

    Muon is a momentum-based optimizer with Newton-Schulz preconditioner,
    designed for efficient training of neural networks.

    Reference:
        PyTorch Muon optimizer: `torch.optim.Muon <https://pytorch.org/docs/stable/generated/torch.optim.Muon.html>`_

    Args:
        momentum (float): Momentum factor (default: 0.95).
            Controls the exponential moving average of gradients.
        nesterov (bool): Whether to use Nesterov momentum (default: True).
            Nesterov momentum provides better convergence in many cases.
        ns_coefficients (tuple): Coefficients for Newton-Schulz iteration (default: (3.4445, -4.775, 2.0315)).
            These coefficients control the Newton-Schulz preconditioner computation.
        eps (float): Small value for numerical stability (default: 1e-07).
            Prevents division by zero in computations.
        ns_steps (int): Number of Newton-Schulz iteration steps (default: 5).
            More steps can improve convergence but increase computational cost.

    Example:
        .. code-block:: python

            import deepxde as dde
            
            # Use default Muon settings
            model.compile("muon", lr=0.001)
            
            # Or customize Muon settings
            dde.optimizers.set_Muon_options(momentum=0.9, ns_steps=3)
            model.compile("muon", lr=0.001)

    Note:
        - Muon is particularly effective for large-scale neural network training
        - The learning rate (lr) and weight_decay are set during model.compile()
        - Higher ns_steps may improve convergence but slow down training
        - **IMPORTANT**: Muon only works with 2D parameters (weight matrices).
          For neural networks with bias terms (1D parameters), the system will
          automatically apply Muon to weights and Adam to biases in a hybrid approach.
          For best performance with Muon, consider using networks without bias terms.
    """
    global Muon_options
    Muon_options["momentum"] = momentum
    Muon_options["nesterov"] = nesterov
    Muon_options["ns_coefficients"] = ns_coefficients
    Muon_options["eps"] = eps
    Muon_options["ns_steps"] = ns_steps


set_Muon_options()


def set_GA_options(
    population_size=50,
    mutation_rate=0.1,
    mutation_scale=0.01,
    crossover_rate=0.7,
    selection_method="tournament",
    tournament_size=5,
    elitism=2,
    generations_per_step=1,
):
    """Sets the hyperparameters of Genetic Algorithm optimizer.

    Genetic Algorithm (GA) is an evolutionary optimization method inspired by natural selection.
    It maintains a population of candidate solutions (neural network weights) and evolves them
    through selection, crossover, and mutation operations.

    Reference:
        - Genetic Algorithms: `wikipedia.org/wiki/Genetic_algorithm <https://en.wikipedia.org/wiki/Genetic_algorithm>`_
        - Neuroevolution: `wikipedia.org/wiki/Neuroevolution <https://en.wikipedia.org/wiki/Neuroevolution>`_

    Args:
        population_size (int): Size of the population (default: 50).
            Larger populations explore more solutions but are computationally expensive.
            Recommended: 20-100 depending on problem complexity.
        
        mutation_rate (float): Probability of mutating each gene/weight (default: 0.1).
            Controls exploration vs exploitation trade-off.
            - Low (0.01-0.05): More exploitation, slower convergence
            - Medium (0.1-0.2): Balanced exploration/exploitation
            - High (0.3-0.5): More exploration, may be unstable
        
        mutation_scale (float): Standard deviation of Gaussian mutation (default: 0.01).
            Controls the magnitude of weight changes during mutation.
            - Small (0.001-0.01): Fine-tuning, local search
            - Medium (0.01-0.1): Moderate changes
            - Large (0.1-1.0): Large jumps, global search
        
        crossover_rate (float): Probability of performing crossover (default: 0.7).
            Higher values promote combination of good solutions.
            Recommended: 0.5-0.9 for most problems.
        
        selection_method (str): Method for parent selection (default: "tournament").
            Available methods:
            - "tournament": Tournament selection (good balance, recommended)
            - "roulette": Roulette wheel selection (fitness-proportional)
            - "rank": Rank-based selection (prevents premature convergence)
            - "sus": Stochastic Universal Sampling (low variance)
        
        tournament_size (int): Size of tournament for tournament selection (default: 5).
            Only used when selection_method="tournament".
            - Smaller (2-3): Less selection pressure
            - Larger (5-10): More selection pressure
        
        elitism (int): Number of best individuals to preserve unchanged (default: 2).
            Ensures best solutions are not lost between generations.
            Recommended: 1-5 (1-10% of population_size).
        
        generations_per_step (int): Number of GA generations per training step (default: 1).
            Higher values allow more evolution but are computationally expensive.

    Example:
        .. code-block:: python

            import deepxde as dde
            
            # Use default GA settings
            model.compile("genetic", lr=0.01)
            
            # Or customize GA settings
            dde.optimizers.set_GA_options(
                population_size=30,
                mutation_rate=0.15,
                selection_method="tournament"
            )
            model.compile("genetic", lr=0.01)

    Note:
        - GA is **gradient-free** and doesn't use learning rate in traditional sense
        - The 'lr' parameter in compile() is used as mutation_scale multiplier
        - GA is best for:
          * Non-differentiable objectives
          * Avoiding local minima
          * Small to medium networks (large networks are computationally expensive)
        - GA is slower than gradient-based methods but more robust to:
          * Noisy gradients
          * Discontinuous loss landscapes
          * Vanishing/exploding gradients
        - Consider hybrid approaches: use GA for initialization, then fine-tune with Adam/SGD

    Performance Tips:
        - Start with small population (20-30) for faster iterations
        - Increase mutation_rate if stuck in local minima
        - Decrease mutation_rate for fine-tuning near optimum
        - Use elitism=1-2 to preserve best solutions
        - For large networks, consider only evolving last layer or small subsets
    """
    global GA_options
    GA_options["population_size"] = population_size
    GA_options["mutation_rate"] = mutation_rate
    GA_options["mutation_scale"] = mutation_scale
    GA_options["crossover_rate"] = crossover_rate
    GA_options["selection_method"] = selection_method
    GA_options["tournament_size"] = tournament_size
    GA_options["elitism"] = elitism
    GA_options["generations_per_step"] = generations_per_step


set_GA_options()


# Backend-dependent options
if backend_name in ["pytorch"]:
    # number of iterations per optimization call
    LBFGS_options["iter_per_step"] = min(1000, LBFGS_options["maxiter"])
    LBFGS_options["fun_per_step"] = (
        LBFGS_options["maxfun"]
        * LBFGS_options["iter_per_step"]
        // LBFGS_options["maxiter"]
    )
