"""Module for the DeepONet and MIONet model classes."""

from functools import partial
import torch
from torch import nn
from ..utils import check_consistency, is_function


class MIONet(torch.nn.Module):
    """
    MIONet model class.

    The MIONet is a general architecture for learning operators, which map
    functions to functions. It can be trained with both Supervised and
    Physics-Informed learning strategies.

    .. seealso::

        **Original reference**: Jin, P., Meng, S., and Lu L. (2022).
        *MIONet: Learning multiple-input operators via tensor product.*
        SIAM Journal on Scientific Computing 44.6 (2022): A3490-A351
        DOI: `10.1137/22M1477751 <https://doi.org/10.1137/22M1477751>`_
    """

    def __init__(
        self,
        networks,
        aggregator="*",
        reduction="+",
        scale=True,
        translation=True,
    ):
        """
        Initialization of the :class:`MIONet` class.

        :param dict networks: The neural networks to use as models. The ``dict``
            takes as key a neural network, and as value the list of indeces to
            extract from the input variable in the forward pass of the neural
            network. If a ``list[int]`` is passed, the corresponding columns of
            the inner most entries are extracted. If a ``list[str]`` is passed
            the variables of the corresponding
            :class:`~pina.label_tensor.LabelTensor` are extracted.
            Each :class:`torch.nn.Module` model has to take as input either a
            :class:`~pina.label_tensor.LabelTensor` or a :class:`torch.Tensor`.
            Default implementation consists of several branch nets and one
            trunk nets.
        :param aggregator: The aggregator to be used to aggregate component-wise
            partial results from the modules in ``networks``. Available
            aggregators include: sum: ``+``, product: ``*``, mean: ``mean``,
            min: ``min``, max: ``max``. Default is ``*``.
        :type aggregator: str or Callable
        :param reduction: The reduction to be used to reduce the aggregated
            result of the modules in ``networks`` to the desired output
            dimension. Available reductions include: sum: ``+``, product: ``*``,
            mean: ``mean``, min: ``min``, max: ``max``, identity: "id".
            Default is ``+``.
        :type reduction: str or Callable
        :param bool scale: If ``True``, the final output is scaled before being
            returned in the forward pass. Default is ``True``.
        :param bool translation: If ``True``, the final output is translated
            before being returned in the forward pass.  Default is ``True``.
        :raises ValueError: If the passed networks have not the same output
            dimension.

        .. warning::
            No checks are performed in the forward pass to verify if the input
            is instance of either :class:`~pina.label_tensor.LabelTensor` or
            :class:`torch.Tensor`. In general, in case of a
            :class:`~pina.label_tensor.LabelTensor`, both a ``list[int]`` or a
            ``list[str]`` can be passed as ``networks`` dict values.
            Differently, in case of a :class:`torch.Tensor`, only a
            ``list[int]`` can be passed as ``networks`` dict values.

        :Example:
            >>> branch_net1 = FeedForward(input_dimensons=1,
            ... output_dimensions=10)
            >>> branch_net2 = FeedForward(input_dimensons=2,
            ... output_dimensions=10)
            >>> trunk_net = FeedForward(input_dimensons=1, output_dimensions=10)
            >>> networks = {branch_net1 : ['x'],
                            branch_net2 : ['x', 'y'],
            ...             trunk_net : ['z']}
            >>> model = MIONet(networks=networks,
            ...                reduction='+',
            ...                aggregator='*')
            >>> model
            MIONet(
            (models): ModuleList(
                (0): FeedForward(
                (model): Sequential(
                    (0): Linear(in_features=1, out_features=20, bias=True)
                    (1): Tanh()
                    (2): Linear(in_features=20, out_features=20, bias=True)
                    (3): Tanh()
                    (4): Linear(in_features=20, out_features=10, bias=True)
                )
                )
                (1): FeedForward(
                (model): Sequential(
                    (0): Linear(in_features=2, out_features=20, bias=True)
                    (1): Tanh()
                    (2): Linear(in_features=20, out_features=20, bias=True)
                    (3): Tanh()
                    (4): Linear(in_features=20, out_features=10, bias=True)
                )
                )
                (2): FeedForward(
                (model): Sequential(
                    (0): Linear(in_features=1, out_features=20, bias=True)
                    (1): Tanh()
                    (2): Linear(in_features=20, out_features=20, bias=True)
                    (3): Tanh()
                    (4): Linear(in_features=20, out_features=10, bias=True)
                )
                )
            )
            )
        """
        super().__init__()

        # check type consistency
        check_consistency(networks, dict)
        check_consistency(scale, bool)
        check_consistency(translation, bool)

        for value in networks.values():
            check_consistency(value, (str, int))

        # assign trunk and branch net with their input indeces
        self.models = torch.nn.ModuleList(networks.keys())
        self._indeces = networks.values()

        # initializie aggregation
        self._init_aggregator(aggregator=aggregator)
        self._init_reduction(reduction=reduction)

        # scale and translation
        self._scale = (
            torch.nn.Parameter(torch.tensor([1.0]))
            if scale
            else torch.tensor([1.0])
        )
        self._trasl = (
            torch.nn.Parameter(torch.tensor([1.0]))
            if translation
            else torch.tensor([1.0])
        )

    @staticmethod
    def _symbol_functions(**kwargs):
        """
        Return a dictionary of functions that can be used as aggregators or
        reductions.

        :param dict kwargs: Additional parameters.
        :return: A dictionary of functions.
        :rtype: dict
        """
        return {
            "+": partial(torch.sum, **kwargs),
            "*": partial(torch.prod, **kwargs),
            "mean": partial(torch.mean, **kwargs),
            "min": lambda x: torch.min(x, **kwargs).values,
            "max": lambda x: torch.max(x, **kwargs).values,
            "id": lambda x: x,
        }

    def _init_aggregator(self, aggregator):
        """
        Initialize the aggregator.

        :param aggregator: The aggregator to be used to aggregate.
        :type aggregator: str or Callable
        :raises ValueError: If the aggregator is not supported.
        """
        aggregator_funcs = self._symbol_functions(dim=-1)
        if aggregator in aggregator_funcs:
            aggregator_func = aggregator_funcs[aggregator]
        elif isinstance(aggregator, nn.Module) or is_function(aggregator):
            aggregator_func = aggregator
        else:
            raise ValueError(f"Unsupported aggregation: {str(aggregator)}")

        self._aggregator = aggregator_func
        self._aggregator_type = aggregator

    def _init_reduction(self, reduction):
        """
        Initialize the reduction.

        :param reduction: The reduction to be used.
        :type reduction: str or Callable
        :raises ValueError: If the reduction is not supported.
        """
        reduction_funcs = self._symbol_functions(dim=-1)
        if reduction in reduction_funcs:
            reduction_func = reduction_funcs[reduction]
        elif isinstance(reduction, nn.Module) or is_function(reduction):
            reduction_func = reduction
        else:
            raise ValueError(f"Unsupported reduction: {reduction}")

        self._reduction = reduction_func
        self._reduction_type = reduction

    def _get_vars(self, x, indeces):
        """
        Extract the variables from the input tensor.

        :param x: The input tensor.
        :type x: LabelTensor | torch.Tensor
        :param indeces: The indeces to extract.
        :type indeces: list[int] | list[str]
        :raises RuntimeError: If failing to extract the variables.
        :raises RuntimeError: If failing to extract the right indeces.
        :return: The extracted variables.
        :rtype: LabelTensor | torch.Tensor
        """
        if isinstance(indeces[0], str):
            try:
                return x.extract(indeces)
            except AttributeError as e:
                raise RuntimeError(
                    "Not possible to extract input variables from tensor."
                    " Ensure that the passed tensor is a LabelTensor or"
                    " pass list of integers to extract variables. For"
                    " more information refer to warning in the documentation."
                ) from e
        elif isinstance(indeces[0], int):
            return x[..., indeces]
        else:
            raise RuntimeError(
                "Not able to extract right indeces for tensor."
                " For more information refer to warning in the documentation."
            )

    def forward(self, x):
        """
        Forward pass for the :class:`MIONet` model.

        :param x: The input tensor.
        :type x: LabelTensor | torch.Tensor
        :return: The output tensor.
        :rtype: LabelTensor | torch.Tensor
        """

        # forward pass
        output_ = [
            model(self._get_vars(x, indeces))
            for model, indeces in zip(self.models, self._indeces)
        ]

        # aggregation
        aggregated = self._aggregator(torch.dstack(output_))

        # reduce
        output_ = self._reduction(aggregated)
        if self._reduction_type in self._symbol_functions(dim=-1):
            output_ = output_.reshape(*output_.shape, 1)

        return self._scale * output_ + self._trasl

    @property
    def aggregator(self):
        """
        The aggregator function.

        :return: The aggregator function.
        :rtype: str or Callable
        """
        return self._aggregator

    @property
    def reduction(self):
        """
        The reduction function.

        :return: The reduction function.
        :rtype: str or Callable
        """
        return self._reduction

    @property
    def scale(self):
        """
        The scale factor.

        :return: The scale factor.
        :rtype: torch.Tensor
        """
        return self._scale

    @property
    def translation(self):
        """
        The translation factor.

        :return: The translation factor.
        :rtype: torch.Tensor
        """
        return self._trasl

    @property
    def indeces_variables_extracted(self):
        """
        The input indeces for each model in form of list.

        :return: The indeces for each model.
        :rtype: list
        """
        return self._indeces

    @property
    def model(self):
        """
        The models in form of list.

        :return: The models.
        :rtype: list[torch.nn.Module]
        """
        return self._indeces


class DeepONet(MIONet):
    """
    DeepONet model class.

    The MIONet is a general architecture for learning operators, which map
    functions to functions. It can be trained with both Supervised and
    Physics-Informed learning strategies.

    .. seealso::

        **Original reference**: Lu, L., Jin, P., Pang, G. et al.
        *Learning nonlinear operators via DeepONet based on the universal
        approximation theorem of operator*.
        Nat Mach Intell 3, 218-229 (2021).
        DOI: `10.1038/s42256-021-00302-5
        <https://doi.org/10.1038/s42256-021-00302-5>`_

    """

    def __init__(
        self,
        branch_net,
        trunk_net,
        input_indeces_branch_net,
        input_indeces_trunk_net,
        aggregator="*",
        reduction="+",
        scale=True,
        translation=True,
    ):
        """
        Initialization of the :class:`DeepONet` class.

        :param torch.nn.Module branch_net: The neural network to use as branch
            model. It has to take as input either a
            :class:`~pina.label_tensor.LabelTensor` or a :class:`torch.Tensor`.
            The output dimension has to be the same as that of ``trunk_net``.
        :param torch.nn.Module trunk_net: The neural network to use as trunk
            model. It has to take as input either a
            :class:`~pina.label_tensor.LabelTensor` or a :class:`torch.Tensor`.
            The output dimension has to be the same as that of ``branch_net``.
        :param input_indeces_branch_net: List of indeces to extract from the
            input variable of the ``branch_net``.
            If a list of ``int`` is passed, the corresponding columns of the
            inner most entries are extracted. If a list of ``str`` is passed the
            variables of the corresponding
            :class:`~pina.label_tensor.LabelTensor` are extracted.
        :type input_indeces_branch_net: list[int] | list[str]
        :param input_indeces_trunk_net: List of indeces to extract from the
            input variable of the ``trunk_net``.
            If a list of ``int`` is passed, the corresponding columns of the
            inner most entries are extracted. If a list of ``str`` is passed the
            variables of the corresponding
            :class:`~pina.label_tensor.LabelTensor` are extracted.
        :type input_indeces_trunk_net: list[int] | list[str]
        :param aggregator: The aggregator to be used to aggregate component-wise
            partial results from the modules in ``networks``. Available
            aggregators include: sum: ``+``, product: ``*``, mean: ``mean``,
            min: ``min``, max: ``max``. Default is ``*``.
        :type aggregator: str or Callable
        :param reduction: The reduction to be used to reduce the aggregated
            result of the modules in ``networks`` to the desired output
            dimension. Available reductions include: sum: ``+``, product: ``*``,
            mean: ``mean``, min: ``min``, max: ``max``. Default is ``+``.
        :type reduction: str or Callable
        :param bool scale: If ``True``, the final output is scaled before being
            returned in the forward pass. Default is ``True``.
        :param bool translation: If ``True``, the final output is translated
            before being returned in the forward pass.  Default is ``True``.

        .. warning::
            In the forward pass we do not check if the input is instance of
            :py:obj:`pina.label_tensor.LabelTensor` or :class:`torch.Tensor`.
            A general rule is that for a :py:obj:`pina.label_tensor.LabelTensor`
            input both list of integers and list of strings can be passed for
            ``input_indeces_branch_net`` and ``input_indeces_trunk_net``.
            Differently, for a :class:`torch.Tensor` only a list of integers can
            be passed for ``input_indeces_branch_net`` and
            ``input_indeces_trunk_net``.

        .. warning::
            No checks are performed in the forward pass to verify if the input
            is instance of either :class:`~pina.label_tensor.LabelTensor` or
            :class:`torch.Tensor`. In general, in case of a
            :class:`~pina.label_tensor.LabelTensor`, both a ``list[int]`` or a
            ``list[str]`` can be passed as ``input_indeces_branch_net`` and
            ``input_indeces_trunk_net``. Differently, in case of a
            :class:`torch.Tensor`, only a ``list[int]`` can be passed.

        :Example:
            >>> branch_net = FeedForward(input_dimensons=1,
            ... output_dimensions=10)
            >>> trunk_net = FeedForward(input_dimensons=1, output_dimensions=10)
            >>> model = DeepONet(branch_net=branch_net,
            ...                  trunk_net=trunk_net,
            ...                  input_indeces_branch_net=['x'],
            ...                  input_indeces_trunk_net=['t'],
            ...                  reduction='+',
            ...                  aggregator='*')
            >>> model
            DeepONet(
            (trunk_net): FeedForward(
                (model): Sequential(
                (0): Linear(in_features=1, out_features=20, bias=True)
                (1): Tanh()
                (2): Linear(in_features=20, out_features=20, bias=True)
                (3): Tanh()
                (4): Linear(in_features=20, out_features=10, bias=True)
                )
            )
            (branch_net): FeedForward(
                (model): Sequential(
                (0): Linear(in_features=1, out_features=20, bias=True)
                (1): Tanh()
                (2): Linear(in_features=20, out_features=20, bias=True)
                (3): Tanh()
                (4): Linear(in_features=20, out_features=10, bias=True)
                )
            )
            )
        """
        networks = {
            branch_net: input_indeces_branch_net,
            trunk_net: input_indeces_trunk_net,
        }
        super().__init__(
            networks=networks,
            aggregator=aggregator,
            reduction=reduction,
            scale=scale,
            translation=translation,
        )

    def forward(self, x):
        """
        Forward pass for the :class:`DeepONet` model.

        :param x: The input tensor.
        :type x: LabelTensor | torch.Tensor
        :return: The output tensor.
        :rtype: LabelTensor | torch.Tensor
        """
        return super().forward(x)

    @property
    def branch_net(self):
        """
        The branch net of the DeepONet.

        :return: The branch net.
        :rtype: torch.nn.Module
        """
        return self.models[0]

    @property
    def trunk_net(self):
        """
        The trunk net of the DeepONet.

        :return: The trunk net.
        :rtype: torch.nn.Module
        """
        return self.models[1]
