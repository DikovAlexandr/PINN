"""Module for the Adaptive Function interface."""

from abc import ABCMeta
import torch
from ..utils import check_consistency, is_function


class AdaptiveActivationFunctionInterface(torch.nn.Module, metaclass=ABCMeta):
    r"""
    The :class:`AdaptiveActivationFunctionInterface`
    class makes a :class:`torch.nn.Module` activation function into an adaptive
    trainable activation function. If one wants to create an adpative activation
    function, this class must be use as base class.

    Given a function :math:`f:\mathbb{R}^n\rightarrow\mathbb{R}^m`, the adaptive
    function :math:`f_{\text{adaptive}}:\mathbb{R}^n\rightarrow\mathbb{R}^m`
    is defined as:

    .. math::
        f_{\text{adaptive}}(\mathbf{x}) = \alpha\,f(\beta\mathbf{x}+\gamma),

    where :math:`\alpha,\,\beta,\,\gamma` are trainable parameters.

    .. seealso::

        **Original reference**: Godfrey, Luke B., and Michael S. Gashler.
        *A continuum among logarithmic, linear, and exponential functions,
        and its potential to improve generalization in neural networks.*
        2015 7th international joint conference on knowledge discovery,
        knowledge engineering and knowledge management (IC3K).
        Vol. 1. IEEE, 2015. DOI: `arXiv preprint arXiv:1602.01321.
        <https://arxiv.org/abs/1602.01321>`_.

        Jagtap, Ameya D., Kenji Kawaguchi, and George Em Karniadakis. *Adaptive
        activation functions accelerate convergence in deep and
        physics-informed neural networks*. Journal of
        Computational Physics 404 (2020): 109136.
        DOI: `JCP 10.1016
        <https://doi.org/10.1016/j.jcp.2019.109136>`_.
    """

    def __init__(self, alpha=None, beta=None, gamma=None, fixed=None):
        """
        Initializes the Adaptive Function.

        :param float | complex alpha: Scaling parameter alpha.
            Defaults to ``None``. When ``None`` is passed,
            the variable is initialized to 1.
        :param float | complex beta: Scaling parameter beta.
            Defaults to ``None``. When ``None`` is passed,
            the variable is initialized to 1.
        :param float | complex gamma: Shifting parameter gamma.
            Defaults to ``None``. When ``None`` is passed,
            the variable is initialized to 1.
        :param list fixed: List of parameters to fix during training,
            i.e. not optimized (``requires_grad`` set to ``False``).
            Options are ``alpha``, ``beta``, ``gamma``. Defaults to None.
        """
        super().__init__()

        # see if there are fixed variables
        if fixed is not None:
            check_consistency(fixed, str)
            if not all(key in ["alpha", "beta", "gamma"] for key in fixed):
                raise TypeError(
                    "Fixed keys must be in [`alpha`, `beta`, `gamma`]."
                )

        # initialize alpha, beta, gamma if they are None
        if alpha is None:
            alpha = 1.0
        if beta is None:
            beta = 1.0
        if gamma is None:
            gamma = 0.0

        # checking consistency
        check_consistency(alpha, (float, complex))
        check_consistency(beta, (float, complex))
        check_consistency(gamma, (float, complex))

        # registering as tensors
        alpha = torch.tensor(alpha, requires_grad=False)
        beta = torch.tensor(beta, requires_grad=False)
        gamma = torch.tensor(gamma, requires_grad=False)

        # setting not fixed variables as torch.nn.Parameter with gradient
        # registering the buffer for the one which are fixed, buffers by
        # default are saved alongside trainable parameters
        if "alpha" not in (fixed or []):
            self._alpha = torch.nn.Parameter(alpha, requires_grad=True)
        else:
            self.register_buffer("alpha", alpha)

        if "beta" not in (fixed or []):
            self._beta = torch.nn.Parameter(beta, requires_grad=True)
        else:
            self.register_buffer("beta", beta)

        if "gamma" not in (fixed or []):
            self._gamma = torch.nn.Parameter(gamma, requires_grad=True)
        else:
            self.register_buffer("gamma", gamma)

    def forward(self, x):
        """
        Define the computation performed at every call.
        The function to the input elementwise.

        :param x: The input tensor to evaluate the activation function.
        :type x: torch.Tensor | LabelTensor
        """
        return self.alpha * (self._func(self.beta * x + self.gamma))

    @property
    def alpha(self):
        """
        The alpha variable.
        """
        return self._alpha

    @property
    def beta(self):
        """
        The beta variable.
        """
        return self._beta

    @property
    def gamma(self):
        """
        The gamma variable.
        """
        return self._gamma

    @property
    def func(self):
        """
        The callable activation function.
        """
        return self._func

    @func.setter
    def func(self, value):
        """
        Set the activation function.
        """
        if not is_function(value):
            raise TypeError("The function must be callable.")
        self._func = value
        return self._func
