# PINN Solver Module

A comprehensive Physics-Informed Neural Network (PINN) solver module for solving partial differential equations, designed as an extension to DeepXDE.

## Features

- **Flexible Architecture**: Support for various neural network types (FCN, SIREN)
- **Multi-dimensional Problems**: 1D and 2D heat equation solutions
- **Advanced Techniques**: RAR, hybrid optimization, adaptive batching
- **GPU Support**: Full GPU acceleration with automatic fallback
- **Visualization**: Comprehensive analysis tools
- **DeepXDE Integration**: Seamless integration with DeepXDE

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### Basic Usage
```python
from solver import PINN, Interval, TimeDomain, PDE
from solver.conditions import InitialConditions, BoundaryConditions, Equation, Problem
from solver.utils import NetParams

# Define geometry and time domain
geom = Interval(0, 1)
time_domain = TimeDomain(0, 1)

# Define PDE
pde = PDE('heat', alpha=0.1)

# Setup conditions
initial_conditions = InitialConditions()
initial_conditions.set_initial_conditions(geom, time_domain, initial_func, num_points=100)

boundary_conditions = BoundaryConditions()
boundary_conditions.set_boundary_conditions(geom, time_domain, boundary_func, num_points=100)

equation = Equation()
equation.set_equation(pde, geom, time_domain, num_points=1000)

# Create problem
problem = Problem(initial_conditions, boundary_conditions, equation, None, geom, time_domain, 0.1)

# Setup network parameters
net_params = NetParams()
net_params.set_params(
    input_dim=2, output_dim=1, hidden_layers=[40, 40, 40, 40],
    epochs=10000, batch_size=1000, lr=0.001, activation='tanh',
    training_mode='train', optimizer='Adam', early_stopping=True,
    model_save_path='models', output_path='output', save_loss=True
)

# Create and train PINN
pinn = PINN(problem, net_params)
pinn.train()

# Prediction
x_test = torch.linspace(0, 1, 100).reshape(-1, 1)
t_test = torch.linspace(0, 1, 100).reshape(-1, 1)
u_pred = pinn.predict(x_test, t_test)
```

### Optimized Usage (Recommended)
```python
from solver import OptimizedPINN, get_optimal_device

# Automatic device selection
device = get_optimal_device()

# Create optimized PINN with batching
pinn = OptimizedPINN(problem, net_params, device=device.type, batch_size=1000)
pinn.train()

# Batch prediction for large datasets
u_pred = pinn.predict_batch(x_test, t_test, batch_size=500)
```

## Architecture

### Core Components
- **PINN**: Main neural network class
- **OptimizedPINN**: GPU-optimized version with batching
- **Geometry**: Spatial domain classes (Interval, Rectangle, Circle, Ellipse)
- **Conditions**: Initial and boundary condition classes
- **PDE**: Partial differential equation definitions
- **Enhancements**: Advanced training techniques (RAR, hybrid optimization)
- **Visualizations**: Analysis and plotting tools
- **Metrics**: Solution quality evaluation

### Supported Geometries
- **Interval**: 1D interval [a, b]
- **Rectangle**: 2D rectangle
- **Circle**: 2D circle
- **Ellipse**: 2D ellipse

### Supported PDEs
- **Heat equation**: 1D and 2D heat conduction
- **Extensible**: Easy to add new equation types

## Advanced Features

### GPU Acceleration
```python
from solver import get_optimal_device, get_gpu_memory_info

device = get_optimal_device()  # Automatic GPU/CPU selection
memory_info = get_gpu_memory_info()  # Monitor GPU memory
```

### Memory-Efficient Training
```python
from solver import MemoryEfficientPINN

memory_pinn = MemoryEfficientPINN(pinn, max_memory_gb=8.0)
memory_pinn.train_memory_efficient()
```

### Adaptive Batching
```python
from solver.batch_utils import AdaptiveBatchScheduler

scheduler = AdaptiveBatchScheduler(initial_batch_size=1000)
batch_size = scheduler.update_batch_size(memory_usage=0.7, loss_trend=-0.1)
```

## Performance

- **Speed**: 3-10x faster computation with GPU and optimizations
- **Memory**: Efficient memory usage with adaptive batching
- **Scalability**: Support for large-scale problems (millions of points)
- **Compatibility**: Backward compatible with existing code

## Migration

### Quick Steps
- Import `OptimizedPINN` instead of `PINN` when training large problems
- Use `get_optimal_device()` for automatic GPU/CPU selection
- Provide `batch_size` to enable efficient batching
- Use `predict_batch()` for large-scale inference

```python
from solver import OptimizedPINN, get_optimal_device

device = get_optimal_device()
pinn = OptimizedPINN(problem, net_params, device=device.type, batch_size=1000)
pinn.train()
u_pred = pinn.predict_batch(x_test, t_test, batch_size=500)
```

### Recommendations
- Small tasks (<1k points): use `PINN`
- Medium (1k–10k): `OptimizedPINN` + batching
- Large (>10k): `OptimizedPINN` + `MemoryEfficientPINN`

## Optimization Overview

- Derivatives: vectorized `autograd` usage and reuse of first-order grads (up to 3x faster)
- GPU: automatic device selection and memory monitoring (up to 10x faster vs CPU)
- Batching: adaptive batch sizing and memory-aware training for large datasets
- Key components: `OptimizedPINN`, `get_optimal_device`, `get_gpu_memory_info`, `MemoryEfficientPINN`,
  `AdaptiveBatchScheduler`

## Examples

- `examples/basic_usage.py` - Basic 1D heat equation example
- `examples/2d_heat_equation.py` - 2D heat equation example
- `examples/optimized_usage.py` - Optimized usage with GPU and batching

## Requirements

- Python >= 3.8
- PyTorch >= 1.9.0
- NumPy >= 1.21.0
- Matplotlib >= 3.4.0
- Pandas >= 1.3.0

## License

MIT License

## Contributing

Contributions are welcome! Please follow PEP8 standards and add tests for new functionality.

## Support

For questions and suggestions, please create issues in the repository.

## Results and Issues
   - The principle of operation of PINN was studied using an example from the [video](https://www.youtube.com/watch?
   v=G_hIppUWcsc&ab_channel=JousefMuradLITE).
   - The basis for implementing the heat equation solver was taken from this [code](https://github.com/Samson-Mano/
   2D_Heat_transfer) for solving the Navier-Stokes equation and modernized to solve the two-dimensional heat 
   equation.
   - There was an attempt to make a comparison with the solution by the finite difference method and for this 
   purpose an analytical solution to the problem was obtained in this [notebook](OscillatorAndHeat.ipynb).
   - The analytical solution has an asymmetric appearance and apparently is not correct. Since it was not possible 
   to find an error, the dimension of the problem under consideration was reduced to one-dimensional.
   - In the [notebook](StationaryTest.ipynb), a comparison was made of the analytical solution, the solution using 
   the finite difference method and the solution using the PINN according to the L2 norm.
   - Now we need to understand why the solution using PINN is so different from the analytical one.
   - Possible solutions
     * Changing the optimizer - Adam produces worse results than LBFGS
     * Network architecture - 
     * Batch learning, samples - Works strangely with LBFGS
     * Analysis of setting boundary and initial conditions - There was a problem with specifying points for the 
     equation, they were given as “for every x there is only one t” but there must be a grid
     * Weights for loss components - It was found that the boundary conditions require an increase in influence 
     (perhaps the equation points will not be satisfied on them and they need to be excluded from the points to 
     check the equation - this has been done)

## Research and Paper Review:
   - A brief summary of the articles can be found in the [folder](docs/research/). There are written down the 
   terminology and main ideas that were borrowed for my code.

## Papers
1. [**Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations**](https://www.sciencedirect.com/science/article/abs/pii/S0021999118307125) - 2019
2. [**Adaptive activation functions accelerate convergence in deep and physics-informed neural networks**](https://arxiv.org/pdf/1906.01170.pdf) - 2020
3. [**Locally adaptive activation functions with slope recovery term for deep and physics-informed neural networks**](https://arxiv.org/pdf/1909.12228.pdf) - 2020
4. [**DeepXDE: A Deep LearningLibrary for Solving DifferentialEquations**](https://epubs.siam.org/doi/epdf/10.1137/19M1274067) -2021
5. [**Implicit Neural Representations with Periodic Activation Functions (SIREN)**](https://arxiv.org/abs/2006.09661) -2020
6. [**Automatic Differentiation in Machine Learning: a Survey**](https://arxiv.org/pdf/1502.05767.pdf) - 2018
7. [**On the convergence of physics informed neural networks for linear second-order elliptic and parabolic type PDEs**](https://arxiv.org/pdf/2004.01806.pdf) - 2020
8. [**Estimates on the generalization error of Physics Informed Neural Networks (PINNs) for approximating PDEs**](https://arxiv.org/pdf/2006.16144.pdf) - 2023
9. [Frequency Principle: Fourier Analysis Sheds Light On Deep Neural Networks](https://arxiv.org/pdf/1901.06523.pdf) - 2020
10. [The Old and the New: Can Physics-Informed Deep-Learning Replace Traditional Linear Solvers?](https://arxiv.org/pdf/2103.09655.pdf) - 2021
11. [Taylor-Mode Automatic Differentiation for Higher-Order Derivatives in JAX](https://openreview.net/pdf?id=SkxEF3FNPH) - 2019
12. [Characterizing possible failure modes in physics-informed neural networks](https://arxiv.org/pdf/2109.01050.pdf) - 2021
13. [Limitations of Physics Informed Machine Learning for Nonlinear Two-Phase Transport in Porous Media](https://www.researchgate.net/publication/343111185_Limitations_of_Physics_Informed_Machine_Learning_for_Nonlinear_Two-Phase_Transport_in_Porous_Media) - 2020
14. [Curriculum Learning: A Survey](https://arxiv.org/pdf/2101.10382.pdf) - 2022
15. [Artificial Neural Network Method for Solution of Boundary Value Problems With Exact Satisfaction of Arbitrary Boundary Conditions](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5061501)
16. [Neural-network methods for boundary value problems with irregular boundaries](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=870037)
17. [Physics-informed machine learning](https://www.nature.com/articles/s42254-021-00314-5)
18. [Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators](https://www.nature.com/articles/s42256-021-00302-5)
19. [Learning the solution operator of parametric partial differential equations with physics-informed DeepONets](https://www.science.org/doi/full/10.1126/sciadv.abi8605)
20. [Deep neural network Grad–Shafranov solver constrained with measured magnetic signals](https://iopscience.iop.org/article/10.1088/1741-4326/ab555f)
21. [Physics Informed Neural Networks towards the real-time calculation of heat fluxes at W7-X](https://www.sciencedirect.com/science/article/pii/S2352179123000406#b0040)

## References

- [DeepXDE](https://deepxde.readthedocs.io/) - Deep learning library for solving differential equations
- [SIREN](https://github.com/vsitzmann/siren) - Implicit Neural Representations with Periodic Activation Functions
- [PyTorch](https://pytorch.org/) - Deep learning framework