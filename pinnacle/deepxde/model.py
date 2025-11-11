__all__ = ["LossHistory", "Model", "TrainState"]

import pickle
import logging
from collections import OrderedDict

import numpy as np

from . import config
from . import display
from . import gradients as grad
from . import losses as losses_module
from . import metrics as metrics_module
from . import optimizers
from . import utils
from .backend import backend_name, torch
# Note: Only PyTorch backend is supported
from .callbacks import CallbackList
from .utils import list_to_str


class Model:
    """A ``Model`` trains a ``NN`` on a ``Data``.

    Args:
        data: ``deepxde.data.Data`` instance.
        net: ``deepxde.nn.NN`` instance.
    """

    def __init__(self, data, net):
        self.data = data
        self.net = net

        self.opt_name = None
        self.batch_size = None
        self.callbacks = None
        self.metrics = None
        self.external_trainable_variables = []
        self.train_state = TrainState()
        self.losshistory = LossHistory()
        self.stop_training = False

        # Backend-dependent attributes (PyTorch only)
        self.opt = None
        # Tensor or callable
        self.outputs = None
        self.outputs_losses_train = None
        self.outputs_losses_test = None
        self.train_step = None
        self.lr_scheduler = None

    @utils.timing
    def compile(
        self,
        optimizer,
        lr=None,
        loss="MSE",
        metrics=None,
        decay=None,
        loss_weights=None,
        external_trainable_variables=None,
    ):
        """Configures the model for training (PyTorch backend only).

        Args:
            optimizer: String name of an optimizer, or a PyTorch optimizer class instance.
            lr (float): The learning rate. For L-BFGS, use
                ``dde.optimizers.set_LBFGS_options`` to set the hyperparameters.
            loss: If the same loss is used for all errors, then `loss` is a String name
                of a loss function or a loss function. If different errors use
                different losses, then `loss` is a list whose size is equal to the
                number of errors.
            metrics: List of metrics to be evaluated by the model during training.
            decay (tuple): Name and parameters of decay to the initial learning rate.
                For PyTorch backend:
                - `StepLR <https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html>`_:
                  ("step", step_size, gamma)
            loss_weights: A list specifying scalar coefficients (Python floats) to
                weight the loss contributions. The loss value that will be minimized by
                the model will then be the weighted sum of all individual losses,
                weighted by the `loss_weights` coefficients.
            external_trainable_variables: A trainable ``dde.Variable`` object or a list
                of trainable ``dde.Variable`` objects. The unknown parameters in the
                physics systems that need to be recovered.
        """
        print("Compiling model...")
        self.opt_name = optimizer
        loss_fn = losses_module.get(loss)
        self.losshistory.set_loss_weights(loss_weights)
        if external_trainable_variables is None:
            self.external_trainable_variables = []
        else:
            if not isinstance(external_trainable_variables, list):
                external_trainable_variables = [external_trainable_variables]
            self.external_trainable_variables = external_trainable_variables

        # Only PyTorch backend is supported
        assert backend_name == "pytorch", f"Only PyTorch backend is supported, got: {backend_name}"
        self._compile_pytorch(lr, loss_fn, decay, loss_weights)
        # metrics may use model variables such as self.net, and thus are instantiated
        # after backend compile.
        metrics = metrics or []
        self.metrics = [metrics_module.get(m) for m in metrics]

    def _compile_pytorch(self, lr, loss_fn, decay, loss_weights):
        """pytorch"""

        def outputs(training, inputs):
            self.net.train(mode=training)
            with torch.no_grad():
                if isinstance(inputs, tuple):
                    inputs = tuple(map(lambda x: torch.as_tensor(x).requires_grad_(), inputs))
                else:
                    inputs = torch.as_tensor(inputs)
                    inputs.requires_grad_()
            # Clear cached Jacobians and Hessians.
            grad.clear()
            return self.net(inputs)

        def outputs_losses(training, inputs, targets, losses_fn):
            self.net.train(mode=training)
            if isinstance(inputs, tuple):
                inputs = tuple(map(lambda x: torch.as_tensor(x).requires_grad_(), inputs))
            else:
                inputs = torch.as_tensor(inputs)
                inputs.requires_grad_()
            outputs_ = self.net(inputs)
            # Data losses
            if targets is not None:
                targets = torch.as_tensor(targets)
            # NOTE: edited
            losses = losses_fn(targets, outputs_, loss_fn, inputs, self, self.net.auxiliary_vars)
            if not isinstance(losses, list):
                losses = [losses]
            losses = torch.stack(losses)
            # Weighted losses
            if loss_weights is not None:
                losses *= torch.as_tensor(loss_weights)
            # Clear cached Jacobians and Hessians.
            grad.clear()
            return outputs_, losses

        def outputs_losses_train(inputs, targets):
            return outputs_losses(True, inputs, targets, self.data.losses_train)

        def outputs_losses_test(inputs, targets):
            return outputs_losses(False, inputs, targets, self.data.losses_test)

        # Another way is using per-parameter options
        # https://pytorch.org/docs/stable/optim.html#per-parameter-options,
        # but not all optimizers (such as L-BFGS) support this.
        trainable_variables = (list(self.net.parameters()) + self.external_trainable_variables)
        if self.net.regularizer is None:
            self.opt, self.lr_scheduler = optimizers.get(trainable_variables, self.opt_name, learning_rate=lr, decay=decay)
        else:
            if self.net.regularizer[0] == "l2":
                self.opt, self.lr_scheduler = optimizers.get(
                    trainable_variables,
                    self.opt_name,
                    learning_rate=lr,
                    decay=decay,
                    weight_decay=self.net.regularizer[1],
                )
            else:
                raise NotImplementedError(f"{self.net.regularizer[0]} regularizaiton to be implemented for "
                                          "backend pytorch.")

        def train_step(inputs, targets):
            # NOTE: edited
            def closure(*, skip_backward=False):
                losses = outputs_losses_train(inputs, targets)[1]
                self.opt.losses = losses
                total_loss = torch.sum(losses)
                if not skip_backward:
                    self.opt.zero_grad()
                    total_loss.backward()
                return total_loss

            loss = self.opt.step(closure)
            # NOTE: edited
            if self.lr_scheduler is not None:
                if decay.__class__.__name__ == "ReduceLROnPlateau":
                    self.lr_scheduler.step(loss)
                else:
                    self.lr_scheduler.step()

        # Callables
        self.outputs = outputs
        self.outputs_losses_train = outputs_losses_train
        self.outputs_losses_test = outputs_losses_test
        self.train_step = train_step

    def _outputs(self, training, inputs):
        """Get network outputs (PyTorch only)."""
        assert backend_name == "pytorch"
        outs = self.outputs(training, inputs)
        return utils.to_numpy(outs)

    def _outputs_losses(self, training, inputs, targets, auxiliary_vars):
        """Get network outputs and losses (PyTorch only)."""
        assert backend_name == "pytorch"
        if training:
            outputs_losses = self.outputs_losses_train
        else:
            outputs_losses = self.outputs_losses_test
        # TODO: auxiliary_vars
        # NOTE: edited
        self.net.auxiliary_vars = auxiliary_vars
        self.net.requires_grad_(requires_grad=False)
        outs = outputs_losses(inputs, targets)
        self.net.requires_grad_()
        self.net.auxiliary_vars = None
        return utils.to_numpy(outs[0]), utils.to_numpy(outs[1])

    def _train_step(self, inputs, targets, auxiliary_vars):
        """Perform a single training step (PyTorch only)."""
        assert backend_name == "pytorch"
        # TODO: auxiliary_vars
        self.train_step(inputs, targets)

    @utils.timing
    def train(
        self,
        iterations=None,
        batch_size=None,
        display_every=100,
        disregard_previous_best=False,
        callbacks=None,
        model_restore_path=None,
        model_save_path=None,
        save_model=True,
        epochs=None,
    ):
        """Trains the model.

        Args:
            iterations (Integer): Number of iterations to train the model, i.e., number
                of times the network weights are updated.
            batch_size: Integer, tuple, or ``None``.

                - If you solve PDEs via ``dde.data.PDE`` or ``dde.data.TimePDE``, do not use `batch_size`, and instead use
                  `dde.callbacks.PDEPointResampler
                  <https://deepxde.readthedocs.io/en/latest/modules/deepxde.html#deepxde.callbacks.PDEPointResampler>`_,
                  see an `example <https://github.com/lululxvi/deepxde/blob/master/examples/diffusion_1d_resample.py>`_.
                - For DeepONet in the format of Cartesian product, if `batch_size` is an Integer,
                  then it is the batch size for the branch input; if you want to also use mini-batch for the trunk net input,
                  set `batch_size` as a tuple, where the fist number is the batch size for the branch net input
                  and the second number is the batch size for the trunk net input.
            display_every (Integer): Print the loss and metrics every this steps.
            disregard_previous_best: If ``True``, disregard the previous saved best
                model.
            callbacks: List of ``dde.callbacks.Callback`` instances. List of callbacks
                to apply during training.
            model_restore_path (String): Path where parameters were previously saved.
            model_save_path (String): Prefix of filenames created for the checkpoint.
            epochs (Integer): Deprecated alias to `iterations`. This will be removed in
                a future version.
        """
        if iterations is None and epochs is not None:
            print("Warning: epochs is deprecated and will be removed in a future version."
                  " Use iterations instead.")
            iterations = epochs
        self.batch_size = batch_size
        self.callbacks = CallbackList(callbacks=callbacks)
        self.callbacks.set_model(self)
        if disregard_previous_best:
            self.train_state.disregard_best()
        # NOTE: edited
        self.model_save_path = model_save_path
        self.display_every = display_every
        print(f"PDE Class Name: {type(self.pde).__name__}")

        if model_restore_path is not None:
            self.restore(model_restore_path, verbose=1)

        print("Training model...\n")
        self.stop_training = False
        self.train_state.set_data_train(*self.data.train_next_batch(self.batch_size))
        self.train_state.set_data_test(*self.data.test())
        self._test()
        self.callbacks.on_train_begin()
        if optimizers.is_external_optimizer(self.opt_name):
            # NOTE: edited
            logger = logging.getLogger(__name__)
            if iterations is not None:
                logger.warning("The number of iterations is ignored for external optimizer.")
            if batch_size is not None:
                logger.warning("The batch size is ignored for external optimizer.")

            # Only PyTorch backend is supported
            assert backend_name == "pytorch"
            self._train_pytorch_lbfgs()
        else:
            if iterations is None:
                raise ValueError("No iterations for {}.".format(self.opt_name))
            self._train_sgd(iterations, display_every)
        self.callbacks.on_train_end()

        print("")
        display.training_display.summary(self.train_state)
        # NOTE: edited
        if model_save_path is not None and save_model:
            self.save(model_save_path, verbose=1)
        return self.losshistory, self.train_state

    def _train_sgd(self, iterations, display_every):
        for i in range(iterations):
            self.callbacks.on_epoch_begin()
            self.callbacks.on_batch_begin()

            self.train_state.set_data_train(*self.data.train_next_batch(self.batch_size))
            self._train_step(
                self.train_state.X_train,
                self.train_state.y_train,
                self.train_state.train_aux_vars,
            )

            self.train_state.epoch += 1
            self.train_state.step += 1
            if self.train_state.step % display_every == 0 or i + 1 == iterations:
                self._test()

            self.callbacks.on_batch_end()
            self.callbacks.on_epoch_end()

            if self.stop_training:
                break

    def _train_pytorch_lbfgs(self):
        prev_n_iter = 0
        while prev_n_iter < optimizers.LBFGS_options["maxiter"]:
            self.callbacks.on_epoch_begin()
            self.callbacks.on_batch_begin()

            self.train_state.set_data_train(*self.data.train_next_batch(self.batch_size))
            self._train_step(
                self.train_state.X_train,
                self.train_state.y_train,
                self.train_state.train_aux_vars,
            )

            n_iter = self.opt.state_dict()["state"][0]["n_iter"]
            if prev_n_iter == n_iter:
                # Converged
                break

            self.train_state.epoch += n_iter - prev_n_iter
            self.train_state.step += n_iter - prev_n_iter
            prev_n_iter = n_iter
            self._test()

            self.callbacks.on_batch_end()
            self.callbacks.on_epoch_end()

            if self.stop_training:
                break

    def _test(self):
        (
            self.train_state.y_pred_train,
            self.train_state.loss_train,
        ) = self._outputs_losses(
            True,
            self.train_state.X_train,
            self.train_state.y_train,
            self.train_state.train_aux_vars,
        )
        self.train_state.y_pred_test, self.train_state.loss_test = self._outputs_losses(
            False,
            self.train_state.X_test,
            self.train_state.y_test,
            self.train_state.test_aux_vars,
        )

        if isinstance(self.train_state.y_test, (list, tuple)):
            self.train_state.metrics_test = [
                m(self.train_state.y_test[i], self.train_state.y_pred_test[i]) for m in self.metrics for i in range(len(self.train_state.y_test))
            ]
        else:
            self.train_state.metrics_test = [m(self.train_state.y_test, self.train_state.y_pred_test) for m in self.metrics]

        self.train_state.update_best()
        self.losshistory.append(
            self.train_state.step,
            self.train_state.loss_train,
            self.train_state.loss_test,
            self.train_state.metrics_test,
        )

        if (np.isnan(self.train_state.loss_train).any() or np.isnan(self.train_state.loss_test).any()):
            self.stop_training = True
        display.training_display(self.train_state)

    def predict(self, x, operator=None, callbacks=None):
        """Generates predictions for the input samples. If `operator` is ``None``,
        returns the network output, otherwise returns the output of the `operator`.

        Args:
            x: The network inputs. A Numpy array or a tuple of Numpy arrays.
            operator: A function takes arguments (`inputs`, `outputs`) or (`inputs`,
                `outputs`, `auxiliary_variables`) and outputs a tensor. `inputs` and
                `outputs` are the network input and output tensors, respectively.
                `auxiliary_variables` is the output of `auxiliary_var_function(x)`
                in `dde.data.PDE`. `operator` is typically chosen as the PDE (used to
                define `dde.data.PDE`) to predict the PDE residual.
            callbacks: List of ``dde.callbacks.Callback`` instances. List of callbacks
                to apply during prediction.
        """
        if isinstance(x, tuple):
            x = tuple(np.asarray(xi, dtype=config.real(np)) for xi in x)
        else:
            x = np.asarray(x, dtype=config.real(np))
        callbacks = CallbackList(callbacks=callbacks)
        callbacks.set_model(self)
        callbacks.on_predict_begin()

        if operator is None:
            y = self._outputs(False, x)
            callbacks.on_predict_end()
            return y

        # operator is not None (PyTorch only)
        assert backend_name == "pytorch"
        if utils.get_num_args(operator) == 3:
            aux_vars = self.data.auxiliary_var_fn(x).astype(config.real(np))
        
        self.net.eval()
        inputs = torch.as_tensor(x)
        inputs.requires_grad_()
        outputs = self.net(inputs)
        if utils.get_num_args(operator) == 2:
            y = operator(inputs, outputs)
        elif utils.get_num_args(operator) == 3:
            # TODO: PyTorch backend Implementation of Auxiliary variables.
            # y = operator(inputs, outputs, torch.as_tensor(aux_vars))
            raise NotImplementedError("Model.predict() with auxiliary variable hasn't been implemented "
                                      "for backend pytorch.")
        # Clear cached Jacobians and Hessians.
        grad.clear()
        y = utils.to_numpy(y)
        callbacks.on_predict_end()
        return y

    # def evaluate(self, x, y, callbacks=None):
    #     """Returns the loss values & metrics values for the model in test mode."""
    #     raise NotImplementedError(
    #         "Model.evaluate to be implemented. Alternatively, use Model.predict."
    #     )

    def state_dict(self):
        """Returns a dictionary containing all variables (PyTorch only)."""
        assert backend_name == "pytorch"
        return self.net.state_dict()

    def save(self, save_path, protocol="backend", verbose=0):
        """Saves all variables to a disk file (PyTorch only).

        Args:
            save_path (string): Prefix of filenames to save the model file.
            protocol (string): If `protocol` is "backend", save using PyTorch's
                `torch.save <https://pytorch.org/docs/stable/generated/torch.save.html>`_.
                If `protocol` is "pickle", save using the Python pickle module.
                Only the protocol "backend" supports ``restore()``.

        Returns:
            string: Path where model is saved.
        """
        # NOTE: edited
        save_path = f"{save_path}/{self.train_state.epoch}"
        if protocol == "pickle":
            save_path += ".pkl"
            with open(save_path, "wb") as f:
                pickle.dump(self.state_dict(), f)
        elif protocol == "backend":
            # Only PyTorch backend is supported
            assert backend_name == "pytorch"
            save_path += ".pt"
            checkpoint = {
                "model_state_dict": self.net.state_dict(),
                "optimizer_state_dict": self.opt.state_dict(),
            }
            torch.save(checkpoint, save_path)
        if verbose > 0:
            print("Epoch {}: saving model to {} ...\n".format(self.train_state.epoch, save_path))
        return save_path

    def restore(self, save_path, verbose=0):
        """Restore all variables from a disk file (PyTorch only).

        Args:
            save_path (string): Path where model was previously saved.
        """
        if verbose > 0:
            print("Restoring model from {} ...\n".format(save_path))
        # Only PyTorch backend is supported
        assert backend_name == "pytorch"
        checkpoint = torch.load(save_path)
        self.net.load_state_dict(checkpoint["model_state_dict"])
        self.opt.load_state_dict(checkpoint["optimizer_state_dict"])

    def print_model(self):
        """Prints all trainable variables (PyTorch only)."""
        assert backend_name == "pytorch"
        for name, param in self.net.named_parameters():
            if param.requires_grad:
                print(f"Variable: {name}, Shape: {param.shape}")
                print(param.data)


class TrainState:

    def __init__(self):
        self.epoch = 0
        self.step = 0

        # Current data
        self.X_train = None
        self.y_train = None
        self.train_aux_vars = None
        self.X_test = None
        self.y_test = None
        self.test_aux_vars = None

        # Results of current step
        # Train results
        self.loss_train = None
        self.y_pred_train = None
        # Test results
        self.loss_test = None
        self.y_pred_test = None
        self.y_std_test = None
        self.metrics_test = None

        # The best results correspond to the min train loss
        self.best_step = 0
        self.best_loss_train = np.inf
        self.best_loss_test = np.inf
        self.best_y = None
        self.best_ystd = None
        self.best_metrics = None

    def set_data_train(self, X_train, y_train, train_aux_vars=None):
        self.X_train = X_train
        self.y_train = y_train
        self.train_aux_vars = train_aux_vars

    def set_data_test(self, X_test, y_test, test_aux_vars=None):
        self.X_test = X_test
        self.y_test = y_test
        self.test_aux_vars = test_aux_vars

    def update_best(self):
        if self.best_loss_train > np.sum(self.loss_train):
            self.best_step = self.step
            self.best_loss_train = np.sum(self.loss_train)
            self.best_loss_test = np.sum(self.loss_test)
            self.best_y = self.y_pred_test
            self.best_ystd = self.y_std_test
            self.best_metrics = self.metrics_test

    def disregard_best(self):
        self.best_loss_train = np.inf


class LossHistory:

    def __init__(self):
        self.steps = []
        self.loss_train = []
        self.loss_test = []
        self.metrics_test = []
        self.loss_weights = None

    def set_loss_weights(self, loss_weights):
        self.loss_weights = loss_weights

    def append(self, step, loss_train, loss_test, metrics_test):
        self.steps.append(step)
        self.loss_train.append(loss_train)
        if loss_test is None:
            loss_test = self.loss_test[-1]
        if metrics_test is None:
            metrics_test = self.metrics_test[-1]
        self.loss_test.append(loss_test)
        self.metrics_test.append(metrics_test)
