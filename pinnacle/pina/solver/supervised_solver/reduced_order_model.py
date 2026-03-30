"""Module for the Reduced Order Model solver"""

import torch
from .supervised_solver_interface import SupervisedSolverInterface
from ..solver import SingleSolverInterface


class ReducedOrderModelSolver(SupervisedSolverInterface, SingleSolverInterface):
    r"""
    Reduced Order Model solver class. This class implements the Reduced Order
    Model solver, using user specified ``reduction_network`` and
    ``interpolation_network`` to solve a specific ``problem``.

    The Reduced Order Model solver aims to find the solution
    :math:`\mathbf{u}:\Omega\rightarrow\mathbb{R}^m` of a differential problem:

    .. math::

        \begin{cases}
        \mathcal{A}[\mathbf{u}(\mu)](\mathbf{x})=0\quad,\mathbf{x}\in\Omega\\
        \mathcal{B}[\mathbf{u}(\mu)](\mathbf{x})=0\quad,
        \mathbf{x}\in\partial\Omega
        \end{cases}

    This is done by means of two neural networks: the ``reduction_network``,
    which defines an encoder :math:`\mathcal{E}_{\rm{net}}`, and a decoder
    :math:`\mathcal{D}_{\rm{net}}`; and the ``interpolation_network``
    :math:`\mathcal{I}_{\rm{net}}`. The input is assumed to be discretised in
    the spatial dimensions.

    The following loss function is minimized during training:

    .. math::
        \mathcal{L}_{\rm{problem}} = \frac{1}{N}\sum_{i=1}^N
        \mathcal{L}(\mathcal{E}_{\rm{net}}[\mathbf{u}(\mu_i)] -
        \mathcal{I}_{\rm{net}}[\mu_i]) + 
        \mathcal{L}(
            \mathcal{D}_{\rm{net}}[\mathcal{E}_{\rm{net}}[\mathbf{u}(\mu_i)]] -
            \mathbf{u}(\mu_i))

    where :math:`\mathcal{L}` is a specific loss function, typically the MSE:

    .. math::
        \mathcal{L}(v) = \| v \|^2_2.

    .. seealso::

        **Original reference**: Hesthaven, Jan S., and Stefano Ubbiali.
        *Non-intrusive reduced order modeling of nonlinear problems using
        neural networks.*
        Journal of Computational Physics 363 (2018): 55-78.
        DOI `10.1016/j.jcp.2018.02.037
        <https://doi.org/10.1016/j.jcp.2018.02.037>`_.

        Pichi, Federico, Beatriz Moya, and Jan S.
        Hesthaven. 
        *A graph convolutional autoencoder approach to model order reduction
        for parametrized PDEs.*
        Journal of Computational Physics 501 (2024): 112762.
        DOI `10.1016/j.jcp.2024.112762
        <https://doi.org/10.1016/j.jcp.2024.112762>`_.
        
    .. note::
        The specified ``reduction_network`` must contain two methods, namely
        ``encode`` for input encoding, and ``decode`` for decoding the former
        result. The ``interpolation_network`` network ``forward`` output
        represents the interpolation of the latent space obtained with
        ``reduction_network.encode``.

    .. note::
        This solver uses the end-to-end training strategy, i.e. the
        ``reduction_network`` and ``interpolation_network`` are trained
        simultaneously. For reference on this trainig strategy look at the
        following:

    .. warning::
        This solver works only for data-driven model. Hence in the ``problem``
        definition the codition must only contain ``input``
        (e.g. coefficient parameters, time parameters), and ``target``.
    """

    def __init__(
        self,
        problem,
        reduction_network,
        interpolation_network,
        loss=None,
        optimizer=None,
        scheduler=None,
        weighting=None,
        use_lt=True,
    ):
        """
        Initialization of the :class:`ReducedOrderModelSolver` class.

        :param AbstractProblem problem: The formualation of the problem.
        :param torch.nn.Module reduction_network: The reduction network used
            for reducing the input space. It must contain two methods, namely
            ``encode`` for input encoding, and ``decode`` for decoding the
            former result.
        :param torch.nn.Module interpolation_network: The interpolation network
            for interpolating the control parameters to latent space obtained by
            the ``reduction_network`` encoding.
        :param torch.nn.Module loss: The loss function to be minimized.
            If ``None``, the :class:`torch.nn.MSELoss` loss is used.
            Default is `None`.
        :param Optimizer optimizer: The optimizer to be used.
            If ``None``, the :class:`torch.optim.Adam` optimizer is used.
            Default is ``None``.
        :param Scheduler scheduler: Learning rate scheduler.
            If ``None``, the :class:`torch.optim.lr_scheduler.ConstantLR`
            scheduler is used. Default is ``None``.
        :param WeightingInterface weighting: The weighting schema to be used.
            If ``None``, no weighting schema is used. Default is ``None``.
        :param bool use_lt: If ``True``, the solver uses LabelTensors as input.
            Default is ``True``.
        """
        model = torch.nn.ModuleDict(
            {
                "reduction_network": reduction_network,
                "interpolation_network": interpolation_network,
            }
        )

        super().__init__(
            model=model,
            problem=problem,
            loss=loss,
            optimizer=optimizer,
            scheduler=scheduler,
            weighting=weighting,
            use_lt=use_lt,
        )

        # assert reduction object contains encode/ decode
        if not hasattr(self.model["reduction_network"], "encode"):
            raise SyntaxError(
                "reduction_network must have encode method. "
                "The encode method should return a lower "
                "dimensional representation of the input."
            )
        if not hasattr(self.model["reduction_network"], "decode"):
            raise SyntaxError(
                "reduction_network must have decode method. "
                "The decode method should return a high "
                "dimensional representation of the encoding."
            )

    def forward(self, x):
        """
        Forward pass implementation.
        It computes the encoder representation by calling the forward method
        of the ``interpolation_network`` on the input, and maps it to output
        space by calling the decode methode of the ``reduction_network``.

        :param x: The input to the neural network.
        :type x: LabelTensor | torch.Tensor | Graph | Data
        :return: The solver solution.
        :rtype: LabelTensor | torch.Tensor | Graph | Data
        """
        reduction_network = self.model["reduction_network"]
        interpolation_network = self.model["interpolation_network"]
        return reduction_network.decode(interpolation_network(x))

    def loss_data(self, input, target):
        """
        Compute the data loss by evaluating the loss between the network's
        output and the true solution. This method should not be overridden, if
        not intentionally.

        :param input: The input to the neural network.
        :type input: LabelTensor | torch.Tensor | Graph | Data
        :param target: The target to compare with the network's output.
        :type target: LabelTensor | torch.Tensor | Graph | Data
        :return: The supervised loss, averaged over the number of observations.
        :rtype: LabelTensor | torch.Tensor | Graph | Data
        """
        # extract networks
        reduction_network = self.model["reduction_network"]
        interpolation_network = self.model["interpolation_network"]
        # encoded representations loss
        encode_repr_inter_net = interpolation_network(input)
        encode_repr_reduction_network = reduction_network.encode(target)
        loss_encode = self._loss_fn(
            encode_repr_inter_net, encode_repr_reduction_network
        )
        # reconstruction loss
        decode = reduction_network.decode(encode_repr_reduction_network)
        loss_reconstruction = self._loss_fn(decode, target)
        return loss_encode + loss_reconstruction
