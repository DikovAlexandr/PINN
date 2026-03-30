"""Module for the Residual-Based Attention PINN solver."""

import torch

from .pinn import PINN
from ...utils import check_consistency


class RBAPINN(PINN):
    r"""
    Residual-based Attention Physics-Informed Neural Network (RBAPINN) solver
    class. This class implements the Residual-based Attention Physics-Informed
    Neural Network solver, using a user specified ``model`` to solve a specific
    ``problem``. It can be used to solve both forward and inverse problems.

    The Residual-based Attention Physics-Informed Neural Network solver aims to
    find the solution :math:`\mathbf{u}:\Omega\rightarrow\mathbb{R}^m` of a
    differential problem:

    .. math::

        \begin{cases}
        \mathcal{A}[\mathbf{u}](\mathbf{x})=0\quad,\mathbf{x}\in\Omega\\
        \mathcal{B}[\mathbf{u}](\mathbf{x})=0\quad,
        \mathbf{x}\in\partial\Omega
        \end{cases}
    
    minimizing the loss function:

    .. math::

        \mathcal{L}_{\rm{problem}} = \frac{1}{N} \sum_{i=1}^{N_\Omega} 
        \lambda_{\Omega}^{i} \mathcal{L} \left( \mathcal{A}
        [\mathbf{u}](\mathbf{x}) \right) + \frac{1}{N} 
        \sum_{i=1}^{N_{\partial\Omega}}
        \lambda_{\partial\Omega}^{i} \mathcal{L} 
        \left( \mathcal{B}[\mathbf{u}](\mathbf{x})
        \right),
    
    denoting the weights as:
    :math:`\lambda_{\Omega}^1, \dots, \lambda_{\Omega}^{N_\Omega}` and
    :math:`\lambda_{\partial \Omega}^1, \dots, 
    \lambda_{\Omega}^{N_\partial \Omega}`
    for :math:`\Omega` and :math:`\partial \Omega`, respectively.

    Residual-based Attention Physics-Informed Neural Network updates the weights
    of the residuals at every epoch as follows:

    .. math::

        \lambda_i^{k+1} \leftarrow \gamma\lambda_i^{k} + 
        \eta\frac{\lvert r_i\rvert}{\max_j \lvert r_j\rvert},

    where :math:`r_i` denotes the residual at point :math:`i`, :math:`\gamma`
    denotes the decay rate, and :math:`\eta` is the learning rate for the
    weights' update.

    .. seealso::
        **Original reference**: Sokratis J. Anagnostopoulos, Juan D. Toscano,
        Nikolaos Stergiopulos, and George E. Karniadakis.
        *Residual-based attention and connection to information 
        bottleneck theory in PINNs.*
        Computer Methods in Applied Mechanics and Engineering 421 (2024): 116805
        DOI: `10.1016/j.cma.2024.116805
        <https://doi.org/10.1016/j.cma.2024.116805>`_.
    """

    def __init__(
        self,
        problem,
        model,
        optimizer=None,
        scheduler=None,
        weighting=None,
        loss=None,
        eta=0.001,
        gamma=0.999,
    ):
        """
        Initialization of the :class:`RBAPINN` class.

        :param AbstractProblem problem: The problem to be solved.
        :param torch.nn.Module model: The neural network model to be used.
        :param Optimizer optimizer: The optimizer to be used.
            If ``None``, the :class:`torch.optim.Adam` optimizer is used.
            Default is ``None``.
        :param Scheduler scheduler: Learning rate scheduler.
            If ``None``, the :class:`torch.optim.lr_scheduler.ConstantLR`
            scheduler is used. Default is ``None``.
        :param WeightingInterface weighting: The weighting schema to be used.
            If ``None``, no weighting schema is used. Default is ``None``.
        :param torch.nn.Module loss: The loss function to be minimized.
            If ``None``, the :class:`torch.nn.MSELoss` loss is used.
            Default is `None`.
        :param float | int eta: The learning rate for the weights of the
            residuals. Default is ``0.001``.
        :param float gamma: The decay parameter in the update of the weights
            of the residuals. Must be between ``0`` and ``1``.
            Default is ``0.999``.
        :raises: ValueError if `gamma` is not in the range (0, 1).
        :raises: ValueError if `eta` is not greater than 0.
        """
        super().__init__(
            model=model,
            problem=problem,
            optimizer=optimizer,
            scheduler=scheduler,
            weighting=weighting,
            loss=loss,
        )

        # check consistency
        check_consistency(eta, (float, int))
        check_consistency(gamma, float)

        # Validate range for gamma
        if not 0 < gamma < 1:
            raise ValueError(
                f"Invalid range: expected 0 < gamma < 1, but got {gamma}"
            )

        # Validate range for eta
        if eta <= 0:
            raise ValueError(f"Invalid range: expected eta > 0, but got {eta}")

        # Initialize parameters
        self.eta = eta
        self.gamma = gamma

        # Initialize the weight of each point to 0
        self.weights = {}
        for cond, data in self.problem.input_pts.items():
            buffer_tensor = torch.zeros((len(data), 1), device=self.device)
            self.register_buffer(f"weight_{cond}", buffer_tensor)
            self.weights[cond] = getattr(self, f"weight_{cond}")

        # Extract the reduction method from the loss function
        self._reduction = self._loss_fn.reduction

        # Set the loss function to return non-aggregated losses
        self._loss_fn = type(self._loss_fn)(reduction="none")

    def on_train_start(self):
        """
        Ensure that all residual weight buffers registered during initialization
        are moved to the correct computation device.
        """
        # Move all weight buffers to the correct device
        for cond in self.problem.input_pts:

            # Get the buffer for the current condition
            weight_buf = getattr(self, f"weight_{cond}")

            # Move the buffer to the correct device
            weight_buf.data = weight_buf.data.to(self.device)
            self.weights[cond] = weight_buf

    def training_step(self, batch, batch_idx, **kwargs):
        """
        Solver training step. It computes the optimization cycle and aggregates
        the losses using the ``weighting`` attribute.

        :param list[tuple[str, dict]] batch: A batch of data. Each element is a
            tuple containing a condition name and a dictionary of points.
        :param int batch_idx: The index of the current batch.
        :param dict kwargs: Additional keyword arguments passed to
            ``optimization_cycle``.
        :return: The loss of the training step.
        :rtype: torch.Tensor
        """
        loss = self._optimization_cycle(
            batch=batch, batch_idx=batch_idx, **kwargs
        )
        self.store_log("train_loss", loss, self.get_batch_size(batch))
        return loss

    @torch.set_grad_enabled(True)
    def validation_step(self, batch, **kwargs):
        """
        The validation step for the PINN solver. It returns the average residual
        computed with the ``loss`` function not aggregated.

        :param list[tuple[str, dict]] batch: A batch of data. Each element is a
            tuple containing a condition name and a dictionary of points.
        :param dict kwargs: Additional keyword arguments passed to
            ``optimization_cycle``.
        :return: The loss of the validation step.
        :rtype: torch.Tensor
        """
        losses = self.optimization_cycle(batch=batch, **kwargs)

        # Aggregate losses for each condition
        for cond, loss in losses.items():
            losses[cond] = self._apply_reduction(loss=losses[cond])

        loss = (sum(losses.values()) / len(losses)).as_subclass(torch.Tensor)
        self.store_log("val_loss", loss, self.get_batch_size(batch))
        return loss

    @torch.set_grad_enabled(True)
    def test_step(self, batch, **kwargs):
        """
        The test step for the PINN solver. It returns the average residual
        computed with the ``loss`` function not aggregated.

        :param list[tuple[str, dict]] batch: A batch of data. Each element is a
            tuple containing a condition name and a dictionary of points.
        :param dict kwargs: Additional keyword arguments passed to
            ``optimization_cycle``.
        :return: The loss of the test step.
        :rtype: torch.Tensor
        """
        losses = self.optimization_cycle(batch=batch, **kwargs)

        # Aggregate losses for each condition
        for cond, loss in losses.items():
            losses[cond] = self._apply_reduction(loss=losses[cond])

        loss = (sum(losses.values()) / len(losses)).as_subclass(torch.Tensor)
        self.store_log("test_loss", loss, self.get_batch_size(batch))
        return loss

    def _optimization_cycle(self, batch, batch_idx, **kwargs):
        """
        Aggregate the loss for each condition in the batch.

        :param list[tuple[str, dict]] batch: A batch of data. Each element is a
            tuple containing a condition name and a dictionary of points.
        :param int batch_idx: The index of the current batch.
        :param dict kwargs: Additional keyword arguments passed to
            ``optimization_cycle``.
        :return: The losses computed for all conditions in the batch, casted
            to a subclass of :class:`torch.Tensor`. It should return a dict
            containing the condition name and the associated scalar loss.
        :rtype: dict
        """
        # compute non-aggregated residuals
        residuals = self.optimization_cycle(batch)

        # update weights based on residuals
        self._update_weights(batch, batch_idx, residuals)

        # compute losses
        losses = {}
        for cond, res in residuals.items():

            # Get the correct indices for the weights. Modulus is used according
            # to the number of points in the condition, as in the PinaDataset.
            len_res = len(res)
            idx = torch.arange(
                batch_idx * len_res,
                (batch_idx + 1) * len_res,
                device=self.weights[cond].device,
            ) % len(self.problem.input_pts[cond])

            losses[cond] = self._apply_reduction(
                loss=(res * self.weights[cond][idx])
            )

            # store log
            self.store_log(
                f"{cond}_loss", losses[cond].item(), self.get_batch_size(batch)
            )

        # clamp unknown parameters in InverseProblem (if needed)
        self._clamp_params()

        # aggregate
        loss = self.weighting.aggregate(losses).as_subclass(torch.Tensor)

        return loss

    def _update_weights(self, batch, batch_idx, residuals):
        """
        Update weights based on residuals.

        :param list[tuple[str, dict]] batch: A batch of data. Each element is a
            tuple containing a condition name and a dictionary of points.
        :param int batch_idx: The index of the current batch.
        :param dict residuals: A dictionary containing the residuals for each
            condition. The keys are the condition names and the values are the
            residuals as tensors.
        """
        # Iterate over each condition in the batch
        for cond, data in batch:

            # Compute normalized residuals
            res = residuals[cond]
            res_abs = torch.linalg.vector_norm(res, ord=2, dim=1, keepdim=True)
            r_norm = (self.eta * res_abs) / (res_abs.max() + 1e-12)

            # Get the correct indices for the weights. Modulus is used according
            # to the number of points in the condition, as in the PinaDataset.
            len_pts = len(data["input"])
            idx = torch.arange(
                batch_idx * len_pts,
                (batch_idx + 1) * len_pts,
                device=self.weights[cond].device,
            ) % len(self.problem.input_pts[cond])

            # Update weights
            weights = self.weights[cond]
            update = self.gamma * weights[idx] + r_norm
            weights[idx] = update.detach()

    def _apply_reduction(self, loss):
        """
        Apply the specified reduction to the loss. The reduction is deferred
        until the end of the optimization cycle to allow residual-based weights
        to be applied to each point beforehand.

        :param torch.Tensor loss: The loss tensor to be reduced.
        :return: The reduced loss tensor.
        :rtype: torch.Tensor
        :raises ValueError: If the reduction method is neither "mean" nor "sum".
        """
        # Apply the specified reduction method
        if self._reduction == "mean":
            return loss.mean()
        if self._reduction == "sum":
            return loss.sum()

        # Raise an error if the reduction method is not recognized
        raise ValueError(
            f"Unknown reduction: {self._reduction}."
            " Supported reductions are 'mean' and 'sum'."
        )
