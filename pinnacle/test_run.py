import argparse
import time
import os
from trainer import Trainer

os.environ["DDEBACKEND"] = "pytorch"

import numpy as np
import torch
import deepxde as dde
from src.pde.burgers import Burgers1D
from src.utils.callbacks import TesterCallback, PlotCallback, LossCallback

print("="*60)
print("PINNACLE BENCHMARK TEST RUN")
print("="*60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PINNacle Test Run')
    parser.add_argument('--device', type=str, default="0")
    parser.add_argument('--iter', type=int, default=1000)
    parser.add_argument('--log-every', type=int, default=100)
    
    command_args = parser.parse_args()
    
    # Установка seed для воспроизводимости
    seed = 42
    dde.config.set_random_seed(seed)
    
    date_str = time.strftime('%m.%d-%H.%M.%S', time.localtime())
    trainer = Trainer(f"{date_str}-test_run", command_args.device)
    
    print("\nRunning test case: Burgers1D")
    print(f"   Iterations: {command_args.iter}")
    print(f"   Device: {command_args.device}")
    print(f"   Seed: {seed}\n")
    
    def get_model_dde():
        pde = Burgers1D()
        
        # Standard architecture: [2, 100, 100, 100, 100, 100, 1]
        net = dde.nn.FNN([pde.input_dim, 100, 100, 100, 100, 100, pde.output_dim], 
                         "tanh", "Glorot normal")
        net = net.float()
        
        loss_weights = np.ones(pde.num_loss)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        
        model = pde.create_model(net)
        model.compile(opt, loss_weights=loss_weights)
        
        return model
    
    trainer.add_task(
        get_model_dde, {
            "iterations": command_args.iter,
            "display_every": command_args.log_every,
            "callbacks": [
                TesterCallback(log_every=command_args.log_every),
                LossCallback(verbose=True),
            ]
        }
    )
    
    trainer.setup(__file__, seed)
    
    print("Starting training...")
    trainer.train_all()
    
    print("\nTraining completed!")
    print(f"Results saved to: runs/{date_str}-test_run/")
    
    trainer.summary()
    
    print("\n" + "="*60)
    print("TEST COMPLETED SUCCESSFULLY")
    print("="*60)