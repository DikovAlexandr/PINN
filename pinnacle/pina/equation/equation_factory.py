"""Module for defining various general equations."""

from typing import Callable
import torch
from .equation import Equation
from ..operator import grad, div, laplacian
from ..utils import check_consistency


class FixedValue(Equation):  # pylint: disable=R0903
    """
    Equation to enforce a fixed value. Can be used to enforce Dirichlet Boundary
    conditions.
    """

    def __init__(self, value, components=None):
        """
        Initialization of the :class:`FixedValue` class.

        :param float value: The fixed value to be enforced.
        :param list[str] components: The name of the output variables for which
            the fixed value condition is applied. It should be a subset of the
            output labels. If ``None``, all output variables are considered.
            Default is ``None``.
        """

        def equation(_, output_):
            """
            Definition of the equation to enforce a fixed value.

            :param LabelTensor input_: Input points where the equation is
                evaluated.
            :param LabelTensor output_: Output tensor, eventually produced by a
                :class:`torch.nn.Module` instance.
            :return: The computed residual of the equation.
            :rtype: LabelTensor
            """
            if components is None:
                return output_ - value
            return output_.extract(components) - value

        super().__init__(equation)


class FixedGradient(Equation):  # pylint: disable=R0903
    """
    Equation to enforce a fixed gradient for a specific condition.
    """

    def __init__(self, value, components=None, d=None):
        """
        Initialization of the :class:`FixedGradient` class.

        :param float value: The fixed value to be enforced to the gradient.
        :param list[str] components: The name of the output variables for which
            the fixed gradient condition is applied. It should be a subset of
            the output labels. If ``None``, all output variables are considered.
            Default is ``None``.
        :param list[str] d: The name of the input variables on which the
            gradient is computed. It should be a subset of the input labels.
            If ``None``, all the input variables are considered.
            Default is ``None``.
        """

        def equation(input_, output_):
            """
            Definition of the equation to enforce a fixed gradient.

            :param LabelTensor input_: Input points where the equation is
                evaluated.
            :param LabelTensor output_: Output tensor, eventually produced by a
                :class:`torch.nn.Module` instance.
            :return: The computed residual of the equation.
            :rtype: LabelTensor
            """
            return grad(output_, input_, components=components, d=d) - value

        super().__init__(equation)


class FixedFlux(Equation):  # pylint: disable=R0903
    """
    Equation to enforce a fixed flux, or divergence, for a specific condition.
    """

    def __init__(self, value, components=None, d=None):
        """
        Initialization of the :class:`FixedFlux` class.

        :param float value: The fixed value to be enforced to the flux.
        :param list[str] components: The name of the output variables for which
            the fixed flux condition is applied. It should be a subset of the
            output labels. If ``None``, all output variables are considered.
            Default is ``None``.
        :param list[str] d: The name of the input variables on which the flux
            is computed. It should be a subset of the input labels. If ``None``,
            all the input variables are considered. Default is ``None``.
        """

        def equation(input_, output_):
            """
            Definition of the equation to enforce a fixed flux.

            :param LabelTensor input_: Input points where the equation is
                evaluated.
            :param LabelTensor output_: Output tensor, eventually produced by a
                :class:`torch.nn.Module` instance.
            :return: The computed residual of the equation.
            :rtype: LabelTensor
            """
            return div(output_, input_, components=components, d=d) - value

        super().__init__(equation)


class FixedLaplacian(Equation):  # pylint: disable=R0903
    """
    Equation to enforce a fixed laplacian for a specific condition.
    """

    def __init__(self, value, components=None, d=None):
        """
        Initialization of the :class:`FixedLaplacian` class.

        :param float value: The fixed value to be enforced to the laplacian.
        :param list[str] components: The name of the output variables for which
            the fixed laplace condition is applied. It should be a subset of the
            output labels. If ``None``, all output variables are considered.
            Default is ``None``.
        :param list[str] d: The name of the input variables on which the
            laplacian is computed. It should be a subset of the input labels.
            If ``None``, all the input variables are considered.
            Default is ``None``.
        """

        def equation(input_, output_):
            """
            Definition of the equation to enforce a fixed laplacian.

            :param LabelTensor input_: Input points where the equation is
                evaluated.
            :param LabelTensor output_: Output tensor, eventually produced by a
                :class:`torch.nn.Module` instance.
            :return: The computed residual of the equation.
            :rtype: LabelTensor
            """
            return (
                laplacian(output_, input_, components=components, d=d) - value
            )

        super().__init__(equation)


class Laplace(FixedLaplacian):  # pylint: disable=R0903
    r"""
    Equation to enforce a null laplacian for a specific condition.
    The equation is defined as follows:

    .. math::

        \delta u = 0

    """

    def __init__(self, components=None, d=None):
        """
        Initialization of the :class:`Laplace` class.

        :param list[str] components: The name of the output variables for which
            the null laplace condition is applied. It should be a subset of the
            output labels. If ``None``, all output variables are considered.
            Default is ``None``.
        :param list[str] d: The name of the input variables on which the
            laplacian is computed. It should be a subset of the input labels.
            If ``None``, all the input variables are considered.
            Default is ``None``.
        """
        super().__init__(0.0, components=components, d=d)


class Advection(Equation):  # pylint: disable=R0903
    r"""
    Implementation of the N-dimensional advection equation with constant
    velocity parameter. The equation is defined as follows:

    .. math::

        \frac{\partial u}{\partial t} + c \cdot \nabla u = 0

    Here, :math:`c` is the advection velocity parameter.
    """

    def __init__(self, c):
        """
        Initialization of the :class:`Advection` class.

        :param c: The advection velocity. If a scalar is provided, the same
            velocity is applied to all spatial dimensions. If a list is
            provided, it must contain one value per spatial dimension.
        :type c: float | int | List[float] | List[int]
        :raises ValueError: If ``c`` is an empty list.
        """
        # Check consistency
        check_consistency(c, (float, int, list))
        if isinstance(c, list):
            all(check_consistency(ci, (float, int)) for ci in c)
            if len(c) < 1:
                raise ValueError("'c' cannot be an empty list.")
        else:
            c = [c]

        # Store advection velocity parameter
        self.c = torch.tensor(c).unsqueeze(0)

        def equation(input_, output_):
            """
            Implementation of the advection equation.

            :param LabelTensor input_: The input data of the problem.
            :param LabelTensor output_: The output data of the problem.
            :return: The residual of the advection equation.
            :rtype: LabelTensor
            :raises ValueError: If the ``input_`` labels do not contain the time
                variable 't'.
            :raises ValueError: If ``c`` is a list and its length is not
                consistent with the number of spatial dimensions.
            """
            # Store labels
            input_lbl = input_.labels
            spatial_d = [di for di in input_lbl if di != "t"]

            # Ensure time is passed as input
            if "t" not in input_lbl:
                raise ValueError(
                    "The ``input_`` labels must contain the time 't' variable."
                )

            # Ensure consistency of c length
            if self.c.shape[-1] != len(input_lbl) - 1 and self.c.shape[-1] > 1:
                raise ValueError(
                    "If 'c' is passed as a list, its length must be equal to "
                    "the number of spatial dimensions."
                )

            # Repeat c to ensure consistent shape for advection
            c = self.c.repeat(output_.shape[0], 1)
            if c.shape[1] != (len(input_lbl) - 1):
                c = c.repeat(1, len(input_lbl) - 1)

            # Add a dimension to c for the following operations
            c = c.unsqueeze(-1)

            # Compute the time derivative and the spatial gradient
            time_der = grad(output_, input_, components=None, d="t")
            grads = grad(output_=output_, input_=input_, d=spatial_d)

            # Reshape and transpose
            tmp = grads.reshape(*output_.shape, len(spatial_d))
            tmp = tmp.transpose(-1, -2)

            # Compute advection term
            adv = (tmp * c).sum(dim=tmp.tensor.ndim - 2)

            return time_der + adv

        super().__init__(equation)


class AllenCahn(Equation):  # pylint: disable=R0903
    r"""
    Implementation of the N-dimensional Allen-Cahn equation, defined as follows:

    .. math::

        \frac{\partial u}{\partial t} - \alpha \Delta u + \beta(u^3 - u) = 0

    Here, :math:`\alpha` and :math:`\beta` are parameters of the equation.
    """

    def __init__(self, alpha, beta):
        """
        Initialization of the :class:`AllenCahn` class.

        :param alpha: The diffusion coefficient.
        :type alpha: float | int
        :param beta: The reaction coefficient.
        :type beta: float | int
        """
        check_consistency(alpha, (float, int))
        check_consistency(beta, (float, int))
        self.alpha = alpha
        self.beta = beta

        def equation(input_, output_):
            """
            Implementation of the Allen-Cahn equation.

            :param LabelTensor input_: The input data of the problem.
            :param LabelTensor output_: The output data of the problem.
            :return: The residual of the Allen-Cahn equation.
            :rtype: LabelTensor
            :raises ValueError: If the ``input_`` labels do not contain the time
                variable 't'.
            """
            # Ensure time is passed as input
            if "t" not in input_.labels:
                raise ValueError(
                    "The ``input_`` labels must contain the time 't' variable."
                )

            # Compute the time derivative and the spatial laplacian
            u_t = grad(output_, input_, d=["t"])
            u_xx = laplacian(
                output_, input_, d=[di for di in input_.labels if di != "t"]
            )

            return u_t - self.alpha * u_xx + self.beta * (output_**3 - output_)

        super().__init__(equation)


class DiffusionReaction(Equation):  # pylint: disable=R0903
    r"""
    Implementation of the N-dimensional Diffusion-Reaction equation,
    defined as follows:

    .. math::

        \frac{\partial u}{\partial t} - \alpha \Delta u - f = 0

    Here, :math:`\alpha` is a parameter of the equation, while :math:`f` is the
    reaction term.
    """

    def __init__(self, alpha, forcing_term):
        """
        Initialization of the :class:`DiffusionReaction` class.

        :param alpha: The diffusion coefficient.
        :type alpha: float | int
        :param Callable forcing_term: The forcing field function, taking as
            input the points on which evaluation is required.
        """
        check_consistency(alpha, (float, int))
        check_consistency(forcing_term, (Callable))
        self.alpha = alpha
        self.forcing_term = forcing_term

        def equation(input_, output_):
            """
            Implementation of the Diffusion-Reaction equation.

            :param LabelTensor input_: The input data of the problem.
            :param LabelTensor output_: The output data of the problem.
            :return: The residual of the Diffusion-Reaction equation.
            :rtype: LabelTensor
            :raises ValueError: If the ``input_`` labels do not contain the time
                variable 't'.
            """
            # Ensure time is passed as input
            if "t" not in input_.labels:
                raise ValueError(
                    "The ``input_`` labels must contain the time 't' variable."
                )

            # Compute the time derivative and the spatial laplacian
            u_t = grad(output_, input_, d=["t"])
            u_xx = laplacian(
                output_, input_, d=[di for di in input_.labels if di != "t"]
            )

            return u_t - self.alpha * u_xx - self.forcing_term(input_)

        super().__init__(equation)


class Helmholtz(Equation):  # pylint: disable=R0903
    r"""
    Implementation of the Helmholtz equation, defined as follows:

    .. math::

            \Delta u + k u - f = 0

    Here, :math:`k` is a parameter of the equation, while :math:`f` is the
    forcing term.
    """

    def __init__(self, k, forcing_term):
        """
        Initialization of the :class:`Helmholtz` class.

        :param k: The parameter of the equation.
        :type k: float | int
        :param Callable forcing_term: The forcing field function, taking as
            input the points on which evaluation is required.
        """
        check_consistency(k, (int, float))
        check_consistency(forcing_term, (Callable))
        self.k = k
        self.forcing_term = forcing_term

        def equation(input_, output_):
            """
            Implementation of the Helmholtz equation.

            :param LabelTensor input_: The input data of the problem.
            :param LabelTensor output_: The output data of the problem.
            :return: The residual of the Helmholtz equation.
            :rtype: LabelTensor
            """
            lap = laplacian(output_, input_)
            return lap + self.k * output_ - self.forcing_term(input_)

        super().__init__(equation)


class Poisson(Equation):  # pylint: disable=R0903
    r"""
    Implementation of the Poisson equation, defined as follows:

    .. math::

            \Delta u - f = 0

    Here, :math:`f` is the forcing term.
    """

    def __init__(self, forcing_term):
        """
        Initialization of the :class:`Poisson` class.

        :param Callable forcing_term: The forcing field function, taking as
            input the points on which evaluation is required.
        """
        check_consistency(forcing_term, (Callable))
        self.forcing_term = forcing_term

        def equation(input_, output_):
            """
            Implementation of the Poisson equation.

            :param LabelTensor input_: The input data of the problem.
            :param LabelTensor output_: The output data of the problem.
            :return: The residual of the Poisson equation.
            :rtype: LabelTensor
            """
            lap = laplacian(output_, input_)
            return lap - self.forcing_term(input_)

        super().__init__(equation)


class AcousticWave(Equation):  # pylint: disable=R0903
    r"""
    Implementation of the N-dimensional isotropic acoustic wave equation.
    The equation is defined as follows:

    .. math::

        \frac{\partial^2 u}{\partial t^2} - c^2 \Delta u = 0

    or alternatively:

    .. math::

        \Box u = 0

    Here, :math:`c` is the wave propagation speed, and :math:`\Box` is the
    d'Alembert operator.
    """

    def __init__(self, c):
        """
        Initialization of the :class:`AcousticWaveEquation` class.

        :param c: The wave propagation speed.
        :type c: float | int
        """
        check_consistency(c, (float, int))
        self.c = c

        def equation(input_, output_):
            """
            Implementation of the acoustic wave equation.

            :param LabelTensor input_: The input data of the problem.
            :param LabelTensor output_: The output data of the problem.
            :return: The residual of the acoustic wave equation.
            :rtype: LabelTensor
            :raises ValueError: If the ``input_`` labels do not contain the time
                variable 't'.
            """
            # Ensure time is passed as input
            if "t" not in input_.labels:
                raise ValueError(
                    "The ``input_`` labels must contain the time 't' variable."
                )

            # Compute the time second derivative and the spatial laplacian
            u_tt = laplacian(output_, input_, d=["t"])
            u_xx = laplacian(
                output_, input_, d=[di for di in input_.labels if di != "t"]
            )

            return u_tt - self.c**2 * u_xx

        super().__init__(equation)
