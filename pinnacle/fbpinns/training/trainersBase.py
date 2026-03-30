"""Base trainer utilities for FBPINN / PINN experiments.

This module defines the generic trainer base class used by both FBPINN and
standard PINN trainers. It also contains helpers for running multiple
independent training jobs in parallel.
"""

from __future__ import annotations

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import IPython.display
from tensorboardX import SummaryWriter


class _Trainer:
    """Generic model trainer base class.

    Concrete trainers should inherit from this class and implement the
    :meth:`train` method (and any additional helpers they require).
    """

    def __init__(self, c) -> None:
        """Initialise Torch, output directories and logging.

        :param c: Configuration object (see :class:`config.Constants`). It
            must provide directory helpers and training hyperparameters.
        """
        # set seed
        if c.SEED is None:
            c.SEED = torch.initial_seed()
        else:
            # independent of numpy
            torch.manual_seed(c.SEED)
        np.random.seed(c.SEED)

        # clear directories
        # constantsBase::get_outdirs(), creates/clears the model and output dirs
        c.get_outdirs()
        # constantsBase::save_constants_file(), saves info of Constants to txt
        c.save_constants_file()
        print(c)

        # get device / set threads
        if c.DEVICE != "cpu" and torch.cuda.is_available():
            # stops weird memory being allocated on cuda:0 even if c.DEVICE != 0
            device = torch.device(f"cuda:{c.DEVICE}")
            torch.cuda.set_device(c.DEVICE)
        else:
            device = torch.device("cpu")
        print(f"Device: {device}")
        # let cudnn find the best algorithm to use for your hardware
        # (not good for dynamic nets)
        torch.backends.cudnn.benchmark = False
        # for main inference
        torch.set_num_threads(1)
        print(f"Main thread ID: {os.getpid()}")
        print("Torch seed: ", torch.initial_seed())

        # initialise summary writer
        writer = SummaryWriter(c.SUMMARY_OUT_DIR)
        # uses markdown
        writer.add_text("constants", str(c).replace("\n", "  \n"))

        self.c, self.device, self.writer = c, device, writer

        # set problem properties
        self.need_mask = hasattr(self.c.P, "mask_x")
        self.need_bd = hasattr(self.c.P, "sample_bd")
        self.need_od = hasattr(self.c.P, "sample_data")

    def _print_summary(self, i: int, loss: float, rate: float, start: float) -> None:
        """Print a one-line training summary and log the iteration rate.

        :param i: Current iteration (0-based).
        :param loss: Current loss value.
        :param rate: Iterations per second.
        :param start: Training start time (``time.time()``).
        """
        print(
            "[i: %i/%i] loss: %.4f rate: %.1f elapsed: %.2f hr %s %s\n"
            % (
                i + 1,
                self.c.N_STEPS,
                loss,
                rate,
                (time.time() - start) / (60 * 60),
                time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                self.c.RUN,
            )
        )
        self.writer.add_scalar("rate/", rate, i + 1)

    def _save_figs(self, i: int, fs) -> None:
        """Save figures to disk and log them to TensorBoard.

        This project prefers IEEE-friendly PDF outputs for publication-quality
        figures. We also keep PNG exports for quick previews and for legacy
        consumers that expect raster images.

        :param i: Current iteration (0-based).
        :param fs: Iterable of ``(name, figure)`` pairs.
        """
        if self.c.CLEAR_OUTPUT:
            IPython.display.clear_output(wait=True)
        for name, f in fs:
            if self.c.SAVE_FIGURES:
                step = i + 1
                base = f"{self.c.SUMMARY_OUT_DIR}{name}_{step:08d}"
                # Vector (publication-quality)
                f.savefig(base + ".pdf", bbox_inches="tight", pad_inches=0.02)
                # Raster (fast preview)
                f.savefig(base + ".png", bbox_inches="tight", pad_inches=0.02, dpi=200)
            self.writer.add_figure(name, f, i + 1, close=False)
        if self.c.SHOW_FIGURES:
            plt.show()
        else:
            plt.close("all")

    def _save_model(self, i: int, model: torch.nn.Module, im: int | None = None) -> None:
        """Save a model checkpoint to disk.

        The model is temporarily moved to CPU to avoid potential CUDA
        out-of-memory issues when serialising.

        :param i: Current iteration (0-based).
        :param model: PyTorch model to save.
        :param im: Optional extra index (e.g. inner loop counter).
        """
        tag = f"model_{i + 1:08d}_{im:08d}.torch" if im is not None else f"model_{i + 1:08d}.torch"
        model.eval()
        # put model on cpu before saving to avoid out-of-memory error
        model.to(torch.device("cpu"))
        torch.save(
            {
                "i": i + 1,
                "model_state_dict": model.state_dict(),
            },
            os.path.join(self.c.MODEL_OUT_DIR, tag),
        )
        model.to(self.device)

    def train(self) -> None:
        """Run the training loop (to be implemented by subclasses)."""
        raise NotImplementedError


## HELPER FUNCTIONS


def train_models_multiprocess(ip: int, devices, c, Trainer, wait: float = 0) -> None:
    """Helper for training multiple runs at once (for use with ``multiprocessing.Pool``).

    Each worker process trains one model on a specified device and logs to its
    own screenlog file.

    :param ip: Process index.
    :param devices: Sequence of device identifiers (CUDA indices or ``\"cpu\"``).
    :param c: Configuration object.
    :param Trainer: Trainer class to instantiate (subclass of :class:`_Trainer`).
    :param wait: Optional delay before starting (helps order TensorBoard logs).
    """
    # small hack so that tensorboard summaries appear in order
    time.sleep(wait)
    # grab socket name if using screen
    tag = os.environ["STY"].split(".")[-1] if "STY" in os.environ else "main"
    logfile = f"screenlog.{tag}.{ip}.log"
    # line buffering
    sys.stdout = open(logfile, "a", buffering=1)
    sys.stderr = open(logfile, "a", buffering=1)
    print("tag: " + str(tag))
    print("ip: " + str(ip))
    # set device to run on, based on process id
    c.DEVICE = devices[ip]
    # make sure plots are not shown
    c.SHOW_FIGURES = c.CLEAR_OUTPUT = False
    run = Trainer(c)
    run.train()
