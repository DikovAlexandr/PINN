"""Module for the GAROM solver."""

import torch
from torch.nn.modules.loss import _Loss
from .solver import MultiSolverInterface
from ..condition import InputTargetCondition
from ..utils import check_consistency
from ..loss import LossInterface, PowerLoss


class GAROM(MultiSolverInterface):
    """
    GAROM solver class. This class implements Generative Adversarial Reduced
    Order Model solver, using user specified ``models`` to solve a specific
    order reduction ``problem``.

    .. seealso::

        **Original reference**: Coscia, D., Demo, N., & Rozza, G. (2023).
        *Generative Adversarial Reduced Order Modelling*.
        DOI: `arXiv preprint arXiv:2305.15881.
        <https://doi.org/10.48550/arXiv.2305.15881>`_.
    """

    accepted_conditions_types = InputTargetCondition

    def __init__(
        self,
        problem,
        generator,
        discriminator,
        loss=None,
        optimizer_generator=None,
        optimizer_discriminator=None,
        scheduler_generator=None,
        scheduler_discriminator=None,
        gamma=0.3,
        lambda_k=0.001,
        regularizer=False,
    ):
        """
        Initialization of the :class:`GAROM` class.

        :param AbstractProblem problem: The formulation of the problem.
        :param torch.nn.Module generator: The generator model.
        :param torch.nn.Module discriminator: The discriminator model.
        :param torch.nn.Module loss: The loss function to be minimized.
            If ``None``, :class:`~pina.loss.power_loss.PowerLoss` with ``p=1``
            is used. Default is ``None``.
        :param Optimizer optimizer_generator: The optimizer for the generator.
            If ``None``, the :class:`torch.optim.Adam` optimizer is used.
            Default is ``None``.
        :param Optimizer optimizer_discriminator: The optimizer for the
            discriminator. If ``None``, the :class:`torch.optim.Adam`
            optimizer is used. Default is ``None``.
        :param Scheduler scheduler_generator: The learning rate scheduler for
            the generator.
            If ``None``, the :class:`torch.optim.lr_scheduler.ConstantLR`
            scheduler is used. Default is ``None``.
        :param Scheduler scheduler_discriminator: The learning rate scheduler
            for the discriminator.
            If ``None``, the :class:`torch.optim.lr_scheduler.ConstantLR`
            scheduler is used. Default is ``None``.
        :param float gamma: Ratio of expected loss for generator and
            discriminator. Default is ``0.3``.
        :param float lambda_k: Learning rate for control theory optimization.
            Default is ``0.001``.
        :param bool regularizer: If ``True``, uses a regularization term in the
            GAROM loss. Default is ``False``.
        """

        # set loss
        if loss is None:
            loss = PowerLoss(p=1)

        super().__init__(
            models=[generator, discriminator],
            problem=problem,
            optimizers=[optimizer_generator, optimizer_discriminator],
            schedulers=[
                scheduler_generator,
                scheduler_discriminator,
            ],
            use_lt=False,
        )

        # check consistency
        check_consistency(
            loss, (LossInterface, _Loss, torch.nn.Module), subclass=False
        )
        self._loss_fn = loss

        # set automatic optimization for GANs
        self.automatic_optimization = False

        # check consistency
        check_consistency(gamma, float)
        check_consistency(lambda_k, float)
        check_consistency(regularizer, bool)

        # began hyperparameters
        self.k = 0
        self.gamma = gamma
        self.lambda_k = lambda_k
        self.regularizer = float(regularizer)

    def forward(self, x, mc_steps=20, variance=False):
        """
        Forward pass implementation.

        :param torch.Tensor x: The input tensor.
        :param int mc_steps: Number of Montecarlo samples to approximate the
            expected value. Default is ``20``.
        :param bool variance: If ``True``, the method returns also the variance
            of the solution. Default is ``False``.
        :return: The expected value of the generator distribution. If
            ``variance=True``, the method returns also the variance.
        :rtype: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
        """

        # sampling
        field_sample = [self.sample(x) for _ in range(mc_steps)]
        field_sample = torch.stack(field_sample)

        # extract mean
        mean = field_sample.mean(dim=0)

        if variance:
            var = field_sample.var(dim=0)
            return mean, var

        return mean

    def sample(self, x):
        """
        Sample from the generator distribution.

        :param torch.Tensor x: The input tensor.
        :return: The generated sample.
        :rtype: torch.Tensor
        """
        # sampling
        return self.generator(x)

    def _train_generator(self, parameters, snapshots):
        """
        Train the generator model.

        :param torch.Tensor parameters: The input tensor.
        :param torch.Tensor snapshots: The target tensor.
        :return: The residual loss and the generator loss.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        self.optimizer_generator.instance.zero_grad()

        # Generate a batch of images
        generated_snapshots = self.sample(parameters)

        # generator loss
        r_loss = self._loss_fn(snapshots, generated_snapshots)
        d_fake = self.discriminator([generated_snapshots, parameters])
        g_loss = (
            self._loss_fn(d_fake, generated_snapshots)
            + self.regularizer * r_loss
        )

        # backward step
        g_loss.backward()
        self.optimizer_generator.instance.step()
        self.scheduler_generator.instance.step()

        return r_loss, g_loss

    def _train_discriminator(self, parameters, snapshots):
        """
        Train the discriminator model.

        :param torch.Tensor parameters: The input tensor.
        :param torch.Tensor snapshots: The target tensor.
        :return: The residual loss and the generator loss.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        self.optimizer_discriminator.instance.zero_grad()

        # Generate a batch of images
        generated_snapshots = self.sample(parameters)

        # Discriminator pass
        d_real = self.discriminator([snapshots, parameters])
        d_fake = self.discriminator([generated_snapshots, parameters])

        # evaluate loss
        d_loss_real = self._loss_fn(d_real, snapshots)
        d_loss_fake = self._loss_fn(d_fake, generated_snapshots.detach())
        d_loss = d_loss_real - self.k * d_loss_fake

        # backward step
        d_loss.backward()
        self.optimizer_discriminator.instance.step()
        self.scheduler_discriminator.instance.step()

        return d_loss_real, d_loss_fake, d_loss

    def _update_weights(self, d_loss_real, d_loss_fake):
        """
        Update the weights of the generator and discriminator models.

        :param torch.Tensor d_loss_real: The discriminator loss computed on
            dataset samples.
        :param torch.Tensor d_loss_fake: The discriminator loss computed on
            generated samples.
        :return: The difference between the loss computed on the dataset samples
            and the loss computed on the generated samples.
        :rtype: torch.Tensor
        """

        diff = torch.mean(self.gamma * d_loss_real - d_loss_fake)

        # Update weight term for fake samples
        self.k += self.lambda_k * diff.item()
        self.k = min(max(self.k, 0), 1)  # Constraint to interval [0, 1]
        return diff

    def optimization_cycle(self, batch):
        """
        The optimization cycle for the GAROM solver.

        :param list[tuple[str, dict]] batch: A batch of data. Each element is a
            tuple containing a condition name and a dictionary of points.
        :return: The losses computed for all conditions in the batch, casted
            to a subclass of :class:`torch.Tensor`. It should return a dict
            containing the condition name and the associated scalar loss.
        :rtype: dict
        """
        condition_loss = {}
        for condition_name, points in batch:
            parameters, snapshots = (
                points["input"],
                points["target"],
            )
            d_loss_real, d_loss_fake, d_loss = self._train_discriminator(
                parameters, snapshots
            )
            r_loss, g_loss = self._train_generator(parameters, snapshots)
            diff = self._update_weights(d_loss_real, d_loss_fake)
            condition_loss[condition_name] = r_loss

        # some extra logging
        self.store_log("d_loss", float(d_loss), self.get_batch_size(batch))
        self.store_log("g_loss", float(g_loss), self.get_batch_size(batch))
        self.store_log(
            "stability_metric",
            float(d_loss_real + torch.abs(diff)),
            self.get_batch_size(batch),
        )
        return condition_loss

    def validation_step(self, batch):
        """
        The validation step for the PINN solver.

        :param list[tuple[str, dict]] batch: A batch of data. Each element is a
            tuple containing a condition name and a dictionary of points.
        :return: The loss of the validation step.
        :rtype: torch.Tensor
        """
        condition_loss = {}
        for condition_name, points in batch:
            parameters, snapshots = (
                points["input"],
                points["target"],
            )
            snapshots_gen = self.generator(parameters)
            condition_loss[condition_name] = self._loss_fn(
                snapshots, snapshots_gen
            )
        loss = self.weighting.aggregate(condition_loss)
        self.store_log("val_loss", loss, self.get_batch_size(batch))
        return loss

    def test_step(self, batch):
        """
        The test step for the PINN solver.

        :param list[tuple[str, dict]] batch: A batch of data. Each element is a
            tuple containing a condition name and a dictionary of points.
        :return: The loss of the test step.
        :rtype: torch.Tensor
        """
        condition_loss = {}
        for condition_name, points in batch:
            parameters, snapshots = (
                points["input"],
                points["target"],
            )
            snapshots_gen = self.generator(parameters)
            condition_loss[condition_name] = self._loss_fn(
                snapshots, snapshots_gen
            )
        loss = self.weighting.aggregate(condition_loss)
        self.store_log("test_loss", loss, self.get_batch_size(batch))
        return loss

    @property
    def generator(self):
        """
        The generator model.

        :return: The generator model.
        :rtype: torch.nn.Module
        """
        return self.models[0]

    @property
    def discriminator(self):
        """
        The discriminator model.

        :return: The discriminator model.
        :rtype: torch.nn.Module
        """
        return self.models[1]

    @property
    def optimizer_generator(self):
        """
        The optimizer for the generator.

        :return: The optimizer for the generator.
        :rtype: Optimizer
        """
        return self.optimizers[0]

    @property
    def optimizer_discriminator(self):
        """
        The optimizer for the discriminator.

        :return: The optimizer for the discriminator.
        :rtype: Optimizer
        """
        return self.optimizers[1]

    @property
    def scheduler_generator(self):
        """
        The scheduler for the generator.

        :return: The scheduler for the generator.
        :rtype: Scheduler
        """
        return self.schedulers[0]

    @property
    def scheduler_discriminator(self):
        """
        The scheduler for the discriminator.

        :return: The scheduler for the discriminator.
        :rtype: Scheduler
        """
        return self.schedulers[1]
