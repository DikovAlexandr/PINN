import os
import json
import shutil
import numpy as np


class NetParams:
    def __init__(self, input, output, hidden_layers, iterations, batch_size, 
                 learning_rate, activation, training_mode, optimizer, 
                 display_interval, model_save_path, initial_weights_path, siren_params):
        """
        Args:
            input: The input dimension of the model.
            output: The output dimension of the model.
            hidden_layers: A list of integers representing the number of neurons in each hidden layer.
            iterations: The number of iterations for training.
            batch_size: The number of samples per batch.
            learning_rate: The learning rate for the model.
            activation: The activation function for the hidden layers.
            training_mode: The mode of training, e.g., 'train' or 'test'.
            optimizer: The optimizer for model training.
            display_interval: The interval for displaying training progress.
            model_save_path: The path to save the trained model.
            initial_weights_path: The path to load initial weights for the model.
            siren_params: Additional parameters for the SIREN model.
        """

        # TODO: add constructor as "[2] + [20] * 3 + [1]"
        self.input = input
        self.output = output
        self.hidden_layers = hidden_layers

        self.iterations = iterations
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.activation = activation
        self.training_mode = training_mode
        self.optimizer = optimizer
        
        self.display_interval = display_interval
        self.model_save_path = model_save_path
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
            iterations=config_data["iterations"],
            batch_size=config_data["batch_size"],
            learning_rate=config_data["learning_rate"],
            activation=config_data["activation"],
            training_mode=config_data["training_mode"],
            optimizer=config_data["optimizer"],
            display_interval=config_data["display_interval"],
            model_save_path=config_data["model_save_path"],
            initial_weights_path=config_data["initial_weights_path"],
            siren_params=config_data["siren_params"]
        )


def create_or_clear_folder(folder_path):
    """
    Deletes all files and subdirectories in the specified folder path if it exists, 
    excluding .md files. If the folder does not exist, creates a new folder at the specified path.

    Parameters:
        folder_path (str): The path of the folder to delete or create.

    Returns:
        None
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