import os
import json
import shutil
import numpy as np
import pandas as pd
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

    def set_params(self, input, output, hidden_layers, 
                   epochs, batch_size, 
                   lr, activation, training_mode, optimizer, scheduler,
                   early_stopping, use_rar, use_weights_adjuster,
                   display_interval, model_save_path, output_path, save_loss,
                   initial_weights_path, siren_params):
        """
        Args:
            input: The input dimension of the model.
            output: The output dimension of the model.
            hidden_layers: A list of integers representing the number of neurons in each hidden layer.
            epochs: The number of epochs for training.
            batch_size: The number of samples per batch.
            lr: The learning rate for the model.
            activation: The activation function for the hidden layers.
            training_mode: The mode of training, e.g., 'train' or 'test'.
            optimizer: The optimizer for model training.
            scheduler: The scheduler for model training.
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


def to_numpy(tensor):
        return tensor.cpu().detach().numpy()


def comparison_plot(x, u_analytical, u_pinn, 
                    output_folder, title='Comparison'):
    """
    Plots the comparison of analytical and PINN solutions.

    Parameters:
        x (torch.Tensor): X data for the plot
        u_analytical (torch.Tensor): Analytical solution data
        u_pinn (torch.Tensor): PINN solution data
        output_folder (str): Path to the output folder
        title (str, optional): Title of the plot. Defaults to 'Comparison'.

    Returns:
        None
    """
    plt.figure(figsize=(10, 10))
    if len(x.shape) == 1:
        plt.plot(to_numpy(x), to_numpy(u_analytical), label="Analytical")
        plt.plot(to_numpy(x), to_numpy(u_pinn), label="PINN")
        plt.xlabel('x')
    else:
        # TODO: make more comparable plot
        plt.scatter(to_numpy(x[:, 0]), to_numpy(x[:, 1]), c=u_analytical, 
                    marker='o', label="Analytical", cmap='viridis')
        plt.scatter(to_numpy(x[:, 0]), to_numpy(x[:, 1]), c=u_pinn, 
                    marker='x', label="PINN", cmap='viridis')
        plt.colorbar(label='u')
    plt.ylabel('u')
    plt.legend()
    plt.title(title)
    if output_folder:
        plt.savefig(os.path.join(output_folder, title + '.png'))
        plt.show()
    else:
        plt.show()

def loss_history_plot(data_path, output_folder, is_log=False, title='LossHistory'):
    """
    Generate a plot of loss history from the data.

    Parameters:
        data_path (str): The path to the CSV file containing the loss history data.
        output_folder (str): The folder where the plot will be saved.
        is_log (bool, optional): If True, plot the data using a logarithmic scale.
        title (str, optional): The title of the plot.

    Returns:
        None
    """
    if os.path.exists(data_path):
        loss_history = pd.read_csv(data_path, header=None)
        if is_log:
            plt.semilogy(loss_history[0], loss_history[1])
            plt.ylabel('Logarithmic Loss')
        else:
            plt.plot(loss_history[0], loss_history[1])
            plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.title(title)
        plt.savefig(os.path.join(output_folder, title + '.png'))
        plt.show()