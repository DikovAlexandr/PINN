"""Module for the Adaptive Functions."""

import torch
from ..utils import check_consistency
from .adaptive_function_interface import AdaptiveActivationFunctionInterface


class AdaptiveReLU(AdaptiveActivationFunctionInterface):
    r"""
    Adaptive trainable :class:`~torch.nn.ReLU` activation function.

    Given the function :math:`\text{ReLU}:\mathbb{R}^n\rightarrow\mathbb{R}^n`,
    the adaptive function
    :math:`\text{ReLU}_{\text{adaptive}}:\mathbb{R}^n\rightarrow\mathbb{R}^n`
    is defined as:

    .. math::
        \text{ReLU}_{\text{adaptive}}({x})=\alpha\,\text{ReLU}(\beta{x}+\gamma),

    where :math:`\alpha,\,\beta,\,\gamma` are trainable parameters, and the
    ReLU function is defined as:

    .. math::
        \text{ReLU}(x)  = \max(0, x)

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
        super().__init__(alpha, beta, gamma, fixed)
        self._func = torch.nn.ReLU()


class AdaptiveSigmoid(AdaptiveActivationFunctionInterface):
    r"""
    Adaptive trainable :class:`~torch.nn.Sigmoid` activation function.

    Given the function
    :math:`\text{Sigmoid}:\mathbb{R}^n\rightarrow\mathbb{R}^n`,
    the adaptive function
    :math:`\text{Sigmoid}_{\text{adaptive}}:\mathbb{R}^n\rightarrow\mathbb{R}^n`
    is defined as:

    .. math::
        \text{Sigmoid}_{\text{adaptive}}({x})=
        \alpha\,\text{Sigmoid}(\beta{x}+\gamma),

    where :math:`\alpha,\,\beta,\,\gamma` are trainable parameters, and the
    Sigmoid function is defined as:

    .. math::
        \text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}

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
        super().__init__(alpha, beta, gamma, fixed)
        self._func = torch.nn.Sigmoid()


class AdaptiveTanh(AdaptiveActivationFunctionInterface):
    r"""
    Adaptive trainable :class:`~torch.nn.Tanh` activation function.

    Given the function :math:`\text{Tanh}:\mathbb{R}^n\rightarrow\mathbb{R}^n`,
    the adaptive function
    :math:`\text{Tanh}_{\text{adaptive}}:\mathbb{R}^n\rightarrow\mathbb{R}^n`
    is defined as:

    .. math::
        \text{Tanh}_{\text{adaptive}}({x})=\alpha\,\text{Tanh}(\beta{x}+\gamma),

    where :math:`\alpha,\,\beta,\,\gamma` are trainable parameters, and the
    Tanh function is defined as:

    .. math::
        \text{Tanh}(x) = \frac{\exp(x) - \exp(-x)} {\exp(x) + \exp(-x)}

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
        super().__init__(alpha, beta, gamma, fixed)
        self._func = torch.nn.Tanh()


class AdaptiveSiLU(AdaptiveActivationFunctionInterface):
    r"""
    Adaptive trainable :class:`~torch.nn.SiLU` activation function.

    Given the function :math:`\text{SiLU}:\mathbb{R}^n\rightarrow\mathbb{R}^n`,
    the adaptive function
    :math:`\text{SiLU}_{\text{adaptive}}:\mathbb{R}^n\rightarrow\mathbb{R}^n`
    is defined as:

    .. math::
        \text{SiLU}_{\text{adaptive}}({x})=\alpha\,\text{SiLU}(\beta{x}+\gamma),

    where :math:`\alpha,\,\beta,\,\gamma` are trainable parameters, and the
    SiLU function is defined as:

    .. math::
        \text{SiLU}(x) = x * \sigma(x), \text{where }\sigma(x)
        \text{ is the logistic sigmoid.}

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
        super().__init__(alpha, beta, gamma, fixed)
        self._func = torch.nn.SiLU()


class AdaptiveMish(AdaptiveActivationFunctionInterface):
    r"""
    Adaptive trainable :class:`~torch.nn.Mish` activation function.

    Given the function :math:`\text{Mish}:\mathbb{R}^n\rightarrow\mathbb{R}^n`,
    the adaptive function
    :math:`\text{Mish}_{\text{adaptive}}:\mathbb{R}^n\rightarrow\mathbb{R}^n`
    is defined as:

    .. math::
        \text{Mish}_{\text{adaptive}}({x})=\alpha\,\text{Mish}(\beta{x}+\gamma),

    where :math:`\alpha,\,\beta,\,\gamma` are trainable parameters, and the
    Mish function is defined as:

    .. math::
        \text{Mish}(x) = x * \text{Tanh}(x)

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
        super().__init__(alpha, beta, gamma, fixed)
        self._func = torch.nn.Mish()


class AdaptiveELU(AdaptiveActivationFunctionInterface):
    r"""
    Adaptive trainable :class:`~torch.nn.ELU` activation function.

    Given the function :math:`\text{ELU}:\mathbb{R}^n\rightarrow\mathbb{R}^n`,
    the adaptive function
    :math:`\text{ELU}_{\text{adaptive}}:\mathbb{R}^n\rightarrow\mathbb{R}^n`
    is defined as:

    .. math::
        \text{ELU}_{\text{adaptive}}({x}) = \alpha\,\text{ELU}(\beta{x}+\gamma),

    where :math:`\alpha,\,\beta,\,\gamma` are trainable parameters, and the
    ELU function is defined as:
    
    .. math::
        \text{ELU}(x) = \begin{cases}
        x, & \text{ if  }x > 0\\
        \exp(x) - 1, & \text{ if  }x \leq 0
        \end{cases}

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
        super().__init__(alpha, beta, gamma, fixed)
        self._func = torch.nn.ELU()


class AdaptiveCELU(AdaptiveActivationFunctionInterface):
    r"""
    Adaptive trainable :class:`~torch.nn.CELU` activation function.

    Given the function :math:`\text{CELU}:\mathbb{R}^n\rightarrow\mathbb{R}^n`,
    the adaptive function
    :math:`\text{CELU}_{\text{adaptive}}:\mathbb{R}^n\rightarrow\mathbb{R}^n`
    is defined as:

    .. math::
        \text{CELU}_{\text{adaptive}}({x})=\alpha\,\text{CELU}(\beta{x}+\gamma),

    where :math:`\alpha,\,\beta,\,\gamma` are trainable parameters, and the
    CELU function is defined as:

    .. math::
        \text{CELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x) - 1))

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
        super().__init__(alpha, beta, gamma, fixed)
        self._func = torch.nn.CELU()


class AdaptiveGELU(AdaptiveActivationFunctionInterface):
    r"""
    Adaptive trainable :class:`~torch.nn.GELU` activation function.

    Given the function :math:`\text{GELU}:\mathbb{R}^n\rightarrow\mathbb{R}^n`,
    the adaptive function
    :math:`\text{GELU}_{\text{adaptive}}:\mathbb{R}^n\rightarrow\mathbb{R}^n`
    is defined as:

    .. math::
        \text{GELU}_{\text{adaptive}}({x})=\alpha\,\text{GELU}(\beta{x}+\gamma),

    where :math:`\alpha,\,\beta,\,\gamma` are trainable parameters, and the
    GELU function is defined as:

    .. math::
        \text{GELU}(x)=0.5*x*(1+\text{Tanh}(\sqrt{2 / \pi}*(x+0.044715*x^3)))


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
        super().__init__(alpha, beta, gamma, fixed)
        self._func = torch.nn.GELU()


class AdaptiveSoftmin(AdaptiveActivationFunctionInterface):
    r"""
    Adaptive trainable :class:`~torch.nn.Softmin` activation function.

    Given the function
    :math:`\text{Softmin}:\mathbb{R}^n\rightarrow\mathbb{R}^n`,
    the adaptive function
    :math:`\text{Softmin}_{\text{adaptive}}:\mathbb{R}^n\rightarrow\mathbb{R}^n`
    is defined as:

    .. math::
        \text{Softmin}_{\text{adaptive}}({x})=\alpha\,
        \text{Softmin}(\beta{x}+\gamma),

    where :math:`\alpha,\,\beta,\,\gamma` are trainable parameters, and the
    Softmin function is defined as:

    .. math::
        \text{Softmin}(x_{i}) = \frac{\exp(-x_i)}{\sum_j \exp(-x_j)}

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
        super().__init__(alpha, beta, gamma, fixed)
        self._func = torch.nn.Softmin()


class AdaptiveSoftmax(AdaptiveActivationFunctionInterface):
    r"""
    Adaptive trainable :class:`~torch.nn.Softmax` activation function.

    Given the function
    :math:`\text{Softmax}:\mathbb{R}^n\rightarrow\mathbb{R}^n`,
    the adaptive function
    :math:`\text{Softmax}_{\text{adaptive}}:\mathbb{R}^n\rightarrow\mathbb{R}^n`
    is defined as:

    .. math::
        \text{Softmax}_{\text{adaptive}}({x})=\alpha\,
        \text{Softmax}(\beta{x}+\gamma),

    where :math:`\alpha,\,\beta,\,\gamma` are trainable parameters, and the
    Softmax function is defined as:

    .. math::
        \text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}

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
        super().__init__(alpha, beta, gamma, fixed)
        self._func = torch.nn.Softmax()


class AdaptiveSIREN(AdaptiveActivationFunctionInterface):
    r"""
    Adaptive trainable :obj:`~torch.sin` function.

    Given the function :math:`\text{sin}:\mathbb{R}^n\rightarrow\mathbb{R}^n`,
    the adaptive function
    :math:`\text{sin}_{\text{adaptive}}:\mathbb{R}^n\rightarrow\mathbb{R}^n`
    is defined as:

    .. math::
        \text{sin}_{\text{adaptive}}({x}) = \alpha\,\text{sin}(\beta{x}+\gamma),

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
        super().__init__(alpha, beta, gamma, fixed)
        self._func = torch.sin


class AdaptiveExp(AdaptiveActivationFunctionInterface):
    r"""
    Adaptive trainable :obj:`~torch.exp` function.

    Given the function :math:`\text{exp}:\mathbb{R}^n\rightarrow\mathbb{R}^n`,
    the adaptive function
    :math:`\text{exp}_{\text{adaptive}}:\mathbb{R}^n\rightarrow\mathbb{R}^n`
    is defined as:

    .. math::
        \text{exp}_{\text{adaptive}}({x}) = \alpha\,\text{exp}(\beta{x}),

    where :math:`\alpha,\,\beta` are trainable parameters.

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

    def __init__(self, alpha=None, beta=None, fixed=None):

        # only alpha, and beta parameters (gamma=0 fixed)
        if fixed is None:
            fixed = ["gamma"]
        else:
            check_consistency(fixed, str)
            fixed = list(fixed) + ["gamma"]

        # calling super
        super().__init__(alpha, beta, 0.0, fixed)
        self._func = torch.exp
