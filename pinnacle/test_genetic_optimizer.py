"""
Test script for Genetic Algorithm optimizer integration in DeepXDE.
"""
import sys
import os
import torch
import numpy as np

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add the pinnacle directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import deepxde as dde
from deepxde.backend import backend_name

print("=" * 60)
print("GENETIC ALGORITHM OPTIMIZER TEST")
print("=" * 60)
print(f"Backend: {backend_name}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
print("=" * 60)
print()

# Test 1: Check if GA options are available
print("Test 1: Checking GA configuration...")
try:
    from deepxde.optimizers import GA_options, set_GA_options
    print(f"✓ GA_options imported successfully")
    print(f"  Default GA options:")
    for key, value in GA_options.items():
        print(f"    {key}: {value}")
    
    # Test custom configuration
    set_GA_options(population_size=30, mutation_rate=0.15, selection_method="rank")
    print(f"✓ set_GA_options() works correctly")
    print(f"  Updated population_size: {GA_options['population_size']}")
    print(f"  Updated selection_method: {GA_options['selection_method']}")
    
    # Reset to defaults
    set_GA_options()
    print(f"✓ Reset to default options")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 2: Simple optimization problem (quadratic function)
print("Test 2: Testing GA with simple PDE...")
try:
    # Define a simple 1D PDE: du/dx = 1, u(0) = 0
    # Solution: u(x) = x
    def pde(x, y):
        dy_x = dde.grad.jacobian(y, x)
        return dy_x - 1
    
    def boundary(x, on_boundary):
        return on_boundary and np.isclose(x[0], 0)
    
    def boundary_value(x):
        return 0
    
    # Geometry and BC
    geom = dde.geometry.Interval(0, 1)
    bc = dde.icbc.DirichletBC(geom, boundary_value, boundary)
    
    # PDE problem
    data = dde.data.PDE(geom, pde, bc, num_domain=20, num_boundary=2, num_test=50)
    
    # Small network for GA (large networks are expensive for GA)
    net = dde.nn.FNN([1] + [10] * 1 + [1], "tanh", "Glorot uniform")
    
    # Model
    model = dde.Model(data, net)
    
    print("✓ Model created successfully")
    
    # Test 3: Compile with GA (small population for faster test)
    print()
    print("Test 3: Compiling with GA (small population)...")
    dde.optimizers.set_GA_options(
        population_size=10,  # Small for fast testing
        mutation_rate=0.15,
        selection_method="tournament",
        elitism=1
    )
    model.compile("genetic", lr=0.02)  # lr used as mutation scale multiplier
    print("✓ Model compiled with GA optimizer")
    
    # Short training test (10 generations)
    print()
    print("Test 4: Running short training test (10 generations)...")
    losshistory, train_state = model.train(iterations=10, display_every=5)
    print(f"✓ Training completed successfully")
    final_loss = train_state.loss_train if isinstance(train_state.loss_train, (int, float)) else sum(train_state.loss_train)
    print(f"  Final loss: {final_loss:.6f}")
    
    # Test 5: Different selection methods
    print()
    print("Test 5: Testing different selection methods...")
    selection_methods = ["tournament", "roulette", "rank", "sus"]
    
    for method in selection_methods:
        print(f"\n  Testing selection method: {method}")
        dde.optimizers.set_GA_options(
            population_size=10,
            selection_method=method,
            elitism=1
        )
        
        # Reset network
        net = dde.nn.FNN([1] + [10] * 1 + [1], "tanh", "Glorot uniform")
        model = dde.Model(data, net)
        model.compile("genetic", lr=0.02)
        
        # Train for 5 generations
        losshistory, train_state = model.train(iterations=5, display_every=10)
        final_loss = train_state.loss_train if isinstance(train_state.loss_train, (int, float)) else sum(train_state.loss_train)
        print(f"    ✓ {method} selection completed, final loss: {final_loss:.6f}")
    
    # Test 6: Larger population for better results
    print()
    print("Test 6: Testing with larger population...")
    dde.optimizers.set_GA_options(
        population_size=20,
        mutation_rate=0.1,
        selection_method="tournament",
        elitism=2
    )
    
    net = dde.nn.FNN([1] + [10] * 1 + [1], "tanh", "Glorot uniform")
    model = dde.Model(data, net)
    model.compile("genetic", lr=0.01)
    print("✓ Model compiled with larger population")
    
    losshistory, train_state = model.train(iterations=20, display_every=10)
    print(f"✓ Training with larger population completed")
    final_loss = train_state.loss_train if isinstance(train_state.loss_train, (int, float)) else sum(train_state.loss_train)
    print(f"  Final loss: {final_loss:.6f}")
    
    # Test 7: Test GA statistics
    print()
    print("Test 7: Checking GA statistics...")
    stats = model.opt.get_statistics()
    print(f"✓ Statistics retrieved:")
    print(f"  Generation: {stats['generation']}")
    print(f"  Best fitness: {stats['best_fitness']:.6e}")
    print(f"  Population size: {stats['population_size']}")
    print(f"  History length: {len(stats['history']['best_fitness'])}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 60)
print("ALL TESTS PASSED! ✓")
print("=" * 60)
print()
print("Genetic Algorithm optimizer is successfully integrated!")
print()
print("Usage examples:")
print("  1. Basic usage:")
print("     model.compile('genetic', lr=0.01)")
print()
print("  2. Custom settings:")
print("     dde.optimizers.set_GA_options(")
print("         population_size=50,")
print("         mutation_rate=0.1,")
print("         selection_method='tournament'")
print("     )")
print("     model.compile('genetic', lr=0.01)")
print()
print("  3. Training (iterations = generations):")
print("     model.train(iterations=100, display_every=10)")
print()
print("Key features:")
print("  ✓ Gradient-free optimization")
print("  ✓ Multiple selection methods (tournament, roulette, rank, SUS)")
print("  ✓ Configurable mutation and crossover")
print("  ✓ Elitism to preserve best solutions")
print("  ✓ Population evolution tracking")
print()
print("Best use cases:")
print("  • Non-differentiable objectives")
print("  • Avoiding local minima")
print("  • Small to medium networks")
print("  • Initial exploration + fine-tuning with gradient methods")
print("=" * 60)

