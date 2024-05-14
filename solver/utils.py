import os
import json
import torch
import shutil
import numpy as np
import pandas as pd
from typing import List
import matplotlib.pyplot as plt



class NetParams:
    def __init__(self):
        self.input = None
        self.output = None
        self.hidden_layers = None

        self.epochs = None
        self.batch_size = None
        self.lr = None
        self.activation = None
        self.training_mode = None
        self.optimizer = None

        self.early_stopping = None
        self.start_weights = None
        self.use_rar = None
        self.use_loss_weight_adjuster = None

        self.display_interval = None
        self.model_save_path = None
        self.output_path = None
        self.initial_weights_path = None

        self.siren_params = None

    def set_params(self, input: int, output: int, hidden_layers: List[int],
                   epochs: int, batch_size: int,
                   lr: float, activation: str, training_mode: str,
                   regularization: str, lambda_reg: float,
                   optimizer, scheduler,
                   early_stopping, use_rar, use_weights_adjuster,
                   display_interval, model_save_path, output_path, save_loss,
                   initial_weights_path, siren_params):
        """
        Parameters:
            input (int): The input dimension of the model.
            output (int): The output dimension of the model.
            hidden_layers (list): A list of integers representing the number of neurons in each hidden layer.
            epochs: The number of epochs for training.
            batch_size: The number of samples per batch.
            lr: The learning rate for the model.
            activation: The activation function for the hidden layers.
            training_mode: The mode of training, e.g., 'train' or 'test'.
            regularization: The type of regularization to use. Might be 'Lasso', 'Ridge' or 'Elastic'.
            lambda_reg: The coefficient of the regularization.
            optimizer: The optimizer for model training. Might be 'LBFGS', 'Adam' or 'Hybrid'.
            scheduler: The scheduler for model training. Might be 'StepLR', 'ExponentialLR', 'ReduceLROnPlateau'.
            early_stopping: Boolean indicating whether to use early stopping.
            start_weights: Path to initial weights for the model.
            use_rar: Boolean indicating whether to use relative angular representations.
            use_weights_adjuster: Boolean indicating whether to use the loss weight adjuster.
            display_interval: The interval for displaying training progress.
            model_save_path: The path to save the trained model.
            output_path: The path to save the output.
            save_loss: Boolean indicating whether to save the loss.
            initial_weights_path: The path to load initial weights for the model.
            siren_params: Additional parameters for the SIREN model.
        """
        # TODO: add constructor as "[2] + [20] * 3 + [1]"
        self.input = input
        self.output = output
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
        self.save_loss = save_loss,
        self.initial_weights_path = initial_weights_path

        if siren_params == None:
            self.siren_params = None
        else:
            self.siren_params = siren_params

    @classmethod
    def from_json_file(cls, file_path):
        with open(file_path, "r") as json_file:
            config_data = json.load(json_file)

        return cls(
            input=config_data["input"],
            output=config_data["output"],
            hidden_layers=config_data["hidden_layers"],
            epochs=config_data["epochs"],
            batch_size=config_data["batch_size"],
            learning_rate=config_data["learning_rate"],
            activation=config_data["activation"],
            training_mode=config_data["training_mode"],
            regularization=config_data["regularization"],
            lambda_reg=config_data["lambda_reg"],
            optimizer=config_data["optimizer"],
            scheduler=config_data["scheduler"],
            early_stopping=config_data["early_stopping"],
            use_rar=config_data["use_rar"],
            use_weights_adjuster=config_data["use_weights_adjuster"],
            display_interval=config_data["display_interval"],
            model_save_path=config_data["model_save_path"],
            output_path=config_data["output_path"],
            save_loss=config_data["save_loss"],
            initial_weights_path=config_data["initial_weights_path"],
            siren_params=config_data["siren_params"]
        )


def create_or_clear_folder(folder_path: str):
    """
    Deletes all files and subdirectories in the specified folder path if it exists, 
    excluding .md files. If the folder does not exist, creates a new folder at the specified path.

    Parameters:
        folder_path (str): The path of the folder to delete or create.
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

def create_folder(folder_path: str):
    """
	Create a folder at the specified path if it does not already exist.

	Parameters:
        folder_path (str): the path of the folder to be created
	"""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def split_number(number):
    """
    Splits a number into four parts using a normal distribution.
    
    Parameters:
        number (int): The number to split.
    
    Returns:
        part1, part2, part3, part4 (int): The four parts of the number.
    """
    half = number // 2

    part1 = int(np.random.normal(half / 2, half / 20))
    part2 = half - part1
    part3 = int(np.random.normal(half / 2, half / 20))
    part4 = half - part3

    return part1, part2, part3, part4


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a PyTorch tensor to a NumPy array.

    Parameters:
        tensor (torch.Tensor): The input PyTorch tensor.

    Returns:
        numpy.ndarray: A NumPy array converted from the input tensor.
    """
    return tensor.cpu().detach().numpy()