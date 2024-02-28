import os
import time
import numpy as np


class ModelCheckpoint:
    def __init__(self, filepath, verbose=0, 
                 save_better_only=False, period=1, monitor="train loss"):
        """
        Parameters:
            filepath (str): File path where the checkpoints will be saved.
            verbose (int, optional): Verbosity mode, 0 or 1.
            save_better_only (bool, optional): If True, only the best model will be saved.
            period (int, optional): Interval (number of epochs) between checkpoints.
            monitor (str, optional): Quantity to monitor for saving the best model.

        """
        self.filepath = filepath
        self.verbose = verbose
        self.save_better_only = save_better_only
        self.period = period
        self.monitor = monitor

        self.epochs_since_last_save = 0
        self.monitor_op = np.less
        self.best = np.Inf

    def on_epoch_end(self):
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save < self.period:
            return
        else :
            self.epochs_since_last_save = 0
            if self.save_better_only:
                current = self.get_monitor_value()
                if current < self.best:
                    self.best = current
                    save_path = self.model.save(self.filepath, verbose=0)
                    if self.verbose:
                        print(f"Epoch {self.model.train_state.epoch}: ",
                            f"{self.monitor} improved from {self.best} ",
                            f"to {current}, saving model to {save_path} ...\n")
            else:
                self.model.save(self.filepath, verbose=self.verbose)

    def get_monitor_value(self, loss):
        if self.monitor == "train loss":
            result = sum(loss)
        elif self.monitor == "test loss":
            result = sum(loss)
        else:
            raise ValueError("The specified monitor function is incorrect.")
        return result
    

class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()

    def elapsed_time(self):
        if self.start_time is None:
            raise ValueError("Timer has not been started")
        if self.end_time is None:
            raise ValueError("Timer has not been stopped")
        return self.end_time - self.start_time

    def reset(self):
        self.start_time = None
        self.end_time = None