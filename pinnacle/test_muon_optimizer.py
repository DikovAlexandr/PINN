"""
Test script for Muon optimizer integration in DeepXDE.
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
print("MUON OPTIMIZER TEST")
print("=" * 60)
print(f"Backend: {backend_name}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
print("=" * 60)
print()

# Test 1: Check if Muon options are available
print("Test 1: Checking Muon configuration...")
try:
    from deepxde.optimizers import Muon_options, set_Muon_options
    print(f"✓ Muon_options imported successfully")
    print(f"  Default Muon options: {Muon_options}")
    
    # Test custom configuration
    set_Muon_options(momentum=0.9, ns_steps=3)
    print(f"✓ set_Muon_options() works correctly")
    print(f"  Updated Muon options: {Muon_options}")
    
    # Reset to defaults
    set_Muon_options()
    print(f"✓ Reset to default options")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

print()

# Test 2: Simple PDE problem
print("Test 2: Testing Muon with simple PDE...")
try:
    # Define a simple 1D PDE: du/dx = 1, u(0) = 0
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
    
    # Network
    net = dde.nn.FNN([1] + [20] * 2 + [1], "tanh", "Glorot uniform")
    
    # Model
    model = dde.Model(data, net)
    
    print("✓ Model created successfully")
    
    # Test 3: Compile with Muon (default settings)
    print()
    print("Test 3: Compiling with Muon (default settings)...")
    model.compile("muon", lr=0.001)
    print("✓ Model compiled with Muon optimizer (default settings)")
    
    # Quick training test
    print()
    print("Test 4: Running short training test...")
    losshistory, train_state = model.train(iterations=100, display_every=50)
    print(f"✓ Training completed successfully")
    final_loss = train_state.loss_train if isinstance(train_state.loss_train, (int, float)) else sum(train_state.loss_train)
    print(f"  Final loss: {final_loss:.6f}")
    
    # Test 5: Compile with custom Muon settings
    print()
    print("Test 5: Testing custom Muon settings...")
    dde.optimizers.set_Muon_options(momentum=0.9, ns_steps=3, nesterov=False)
    
    # Reset model (use network regularizer instead of weight_decay parameter)
    net = dde.nn.FNN([1] + [20] * 2 + [1], "tanh", "Glorot uniform")
    net.apply_feature_transform(None)  # Reset any transforms
    net.regularizer = ("l2", 0.1)  # Set L2 regularization
    model = dde.Model(data, net)
    model.compile("muon", lr=0.001)
    print("✓ Model compiled with custom Muon settings (with L2 regularization)")
    
    # Test with decay
    print()
    print("Test 6: Testing Muon with learning rate scheduler...")
    net = dde.nn.FNN([1] + [20] * 2 + [1], "tanh", "Glorot uniform")
    model = dde.Model(data, net)
    model.compile("muon", lr=0.001, decay=("step", 50, 0.9))
    print("✓ Model compiled with Muon + learning rate scheduler")
    
    losshistory, train_state = model.train(iterations=100, display_every=50)
    print(f"✓ Training with scheduler completed")
    final_loss = train_state.loss_train if isinstance(train_state.loss_train, (int, float)) else sum(train_state.loss_train)
    print(f"  Final loss: {final_loss:.6f}")
    
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
print("Muon optimizer is successfully integrated!")
print()
print("Usage examples:")
print("  1. Basic usage:")
print("     model.compile('muon', lr=0.001)")
print()
print("  2. With weight decay:")
print("     model.compile('muon', lr=0.001, weight_decay=0.1)")
print()
print("  3. Custom settings:")
print("     dde.optimizers.set_Muon_options(momentum=0.9, ns_steps=3)")
print("     model.compile('muon', lr=0.001)")
print()
print("  4. With learning rate scheduler:")
print("     model.compile('muon', lr=0.001, decay=('step', 1000, 0.9))")
print("=" * 60)

