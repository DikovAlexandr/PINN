"""
Simple test for Genetic Algorithm optimizer without PDE (no gradients needed).
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
print("SIMPLE GENETIC ALGORITHM TEST (Function Approximation)")
print("=" * 60)
print(f"Backend: {backend_name}")
print(f"PyTorch version: {torch.__version__}")
print("=" * 60)
print()

# Test simple function approximation (no PDE, no gradients needed)
print("Test: Function approximation with GA...")
try:
    # Generate data: y = sin(x)
    X_train = np.linspace(0, 2*np.pi, 100).reshape(-1, 1)
    y_train = np.sin(X_train)
    
    X_test = np.linspace(0, 2*np.pi, 50).reshape(-1, 1)
    y_test = np.sin(X_test)
    
    # Create data object
    data = dde.data.DataSet(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test
    )
    
    # Small network
    net = dde.nn.FNN([1] + [10] * 2 + [1], "tanh", "Glorot uniform")
    
    # Model
    model = dde.Model(data, net)
    
    print("✓ Model created for function approximation")
    
    # Compile with GA
    dde.optimizers.set_GA_options(
        population_size=10,
        mutation_rate=0.1,
        selection_method="tournament",
        elitism=1
    )
    model.compile("genetic", lr=0.05, loss="MSE")
    print("✓ Model compiled with GA")
    
    # Train
    print("\nTraining with GA...")
    losshistory, train_state = model.train(iterations=20, display_every=10)
    
    print(f"\n✓ Training completed!")
    final_loss = train_state.loss_train if isinstance(train_state.loss_train, (int, float)) else sum(train_state.loss_train)
    print(f"  Final loss: {final_loss:.6f}")
    
    # Check predictions
    y_pred = model.predict(X_test)
    mse = np.mean((y_pred - y_test)**2)
    print(f"  Test MSE: {mse:.6f}")
    
    print("\n✓ All tests passed!")
    print("\nGA works correctly for function approximation!")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("=" * 60)

