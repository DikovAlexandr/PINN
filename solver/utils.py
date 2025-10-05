"""Utility functions and classes for PINN solver module."""

import os
import json
import shutil
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt



class NetParams:
    """Network parameters configuration class."""
    
    def __init__(self):
        """Initialize NetParams with default None values."""
        self.input: Optional[int] = None
        self.output: Optional[int] = None
        self.hidden_layers: Optional[List[int]] = None

        self.epochs: Optional[int] = None
        self.batch_size: Optional[int] = None
        self.lr: Optional[float] = None
        self.activation: Optional[str] = None
        self.training_mode: Optional[str] = None
        self.optimizer: Optional[str] = None

        self.early_stopping: Optional[bool] = None
        self.start_weights: Optional[str] = None
        self.use_rar: Optional[bool] = None
        self.use_loss_weight_adjuster: Optional[bool] = None

        self.display_interval: Optional[int] = None
        self.model_save_path: Optional[str] = None
        self.output_path: Optional[str] = None
        self.initial_weights_path: Optional[str] = None

        self.siren_params: Optional[object] = None

    def set_params(self, input_dim: int, output_dim: int, 
                   hidden_layers: List[int], epochs: int, batch_size: int,
                   lr: float, activation: str, training_mode: str,
                   regularization: str, lambda_reg: float,
                   optimizer: str, scheduler: Optional[str],
                   early_stopping: bool, use_rar: bool, 
                   use_weights_adjuster: bool, display_interval: int,
                   model_save_path: str, output_path: str, save_loss: bool,
                   initial_weights_path: Optional[str], 
                   siren_params: Optional[object]) -> None:
        """Set network parameters.
        
        Args:
            input_dim: The input dimension of the model.
            output_dim: The output dimension of the model.
            hidden_layers: A list of integers representing the number of 
                neurons in each hidden layer.
            epochs: The number of epochs for training.
            batch_size: The number of samples per batch.
            lr: The learning rate for the model.
            activation: The activation function for the hidden layers.
            training_mode: The mode of training, e.g., 'train' or 'test'.
            regularization: The type of regularization to use. 
                Options: 'Lasso', 'Ridge' or 'Elastic'.
            lambda_reg: The coefficient of the regularization.
            optimizer: The optimizer for model training. 
                Options: 'LBFGS', 'Adam' or 'Hybrid'.
            scheduler: The scheduler for model training. 
                Options: 'StepLR', 'ExponentialLR', 'ReduceLROnPlateau'.
            early_stopping: Boolean indicating whether to use early stopping.
            use_rar: Boolean indicating whether to use RAR (Residual-based 
                Adaptive Refinement).
            use_weights_adjuster: Boolean indicating whether to use the 
                loss weight adjuster.
            display_interval: The interval for displaying training progress.
            model_save_path: The path to save the trained model.
            output_path: The path to save the output.
            save_loss: Boolean indicating whether to save the loss.
            initial_weights_path: The path to load initial weights for the model.
            siren_params: Additional parameters for the SIREN model.
        """
        self.input = input_dim
        self.output = output_dim
        self.hidden_layers = hidden_layers

        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.activation = activation
        self.training_mode = training_mode

        self.regularization = regularization
        self.lambda_reg = lambda_reg

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.early_stopping = early_stopping
        self.use_rar = use_rar
        self.use_weights_adjuster = use_weights_adjuster

        self.display_interval = display_interval
        self.model_save_path = model_save_path
        self.output_path = output_path
        self.save_loss = save_loss
        self.initial_weights_path = initial_weights_path

        self.siren_params = siren_params

    @classmethod
    def from_json_file(cls, file_path: str) -> 'NetParams':
        """Create NetParams instance from JSON configuration file.
        
        Args:
            file_path: Path to the JSON configuration file.
            
        Returns:
            NetParams instance with parameters loaded from file.
        """
        with open(file_path, "r") as json_file:
            config_data = json.load(json_file)

        instance = cls()
        instance.set_params(
            input_dim=config_data["input"],
            output_dim=config_data["output"],
            hidden_layers=config_data["hidden_layers"],
            epochs=config_data["epochs"],
            batch_size=config_data["batch_size"],
            lr=config_data["lr"],
            activation=config_data["activation"],
            training_mode=config_data["training_mode"],
            regularization=config_data.get("regularization", "None"),
            lambda_reg=config_data.get("lambda_reg", 0.0),
            optimizer=config_data["optimizer"],
            scheduler=config_data.get("scheduler", None),
            early_stopping=config_data["early_stopping"],
            use_rar=config_data["use_rar"],
            use_weights_adjuster=config_data["use_weights_adjuster"],
            display_interval=config_data["display_interval"],
            model_save_path=config_data["model_save_path"],
            output_path=config_data["output_path"],
            save_loss=config_data["save_loss"],
            initial_weights_path=config_data.get("initial_weights_path", None),
            siren_params=config_data.get("siren_params", None)
        )
        return instance


def create_or_clear_folder(folder_path: str) -> None:
    """Delete all files and subdirectories in the specified folder path if it exists, 
    excluding .md files. If the folder does not exist, creates a new folder at the 
    specified path.

    Args:
        folder_path: The path of the folder to delete or create.
    """
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                # Check if the file is a regular file or a symbolic link
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    # Skip deletion of .md files
                    if not filename.endswith('.md'):
                        os.unlink(file_path)
                # Check if the file is a directory
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        os.makedirs(folder_path)

def create_folder(folder_path: str) -> None:
    """Create a folder at the specified path if it does not already exist.

    Args:
        folder_path: The path of the folder to be created.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def split_number(number: int) -> tuple[int, int, int, int]:
    """Split a number into four parts using a normal distribution.
    
    Args:
        number: The number to split.
    
    Returns:
        A tuple of four integers representing the four parts of the number.
    """
    half = number // 2

    part1 = int(np.random.normal(half / 2, half / 20))
    part2 = half - part1
    part3 = int(np.random.normal(half / 2, half / 20))
    part4 = half - part3

    return part1, part2, part3, part4


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a PyTorch tensor to a NumPy array.

    Args:
        tensor: The input PyTorch tensor.

    Returns:
        A NumPy array converted from the input tensor.
    """
    return tensor.cpu().detach().numpy()