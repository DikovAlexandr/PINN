"""Module for the Equation."""

import inspect
from .equation_interface import EquationInterface


class Equation(EquationInterface):
    """
    Implementation of the Equation class. Every ``equation`` passed to a
    :class:`~pina.condition.condition.Condition` object must be either an
    instance of :class:`Equation` or
    :class:`~pina.equation.system_equation.SystemEquation`.
    """

    def __init__(self, equation):
        """
        Initialization of the :class:`Equation` class.

        :param Callable equation: A ``torch`` callable function used to compute
            the residual of a mathematical equation.
        :raises ValueError: If the equation is not a callable function.
        """
        if not callable(equation):
            raise ValueError(
                "equation must be a callable function."
                "Expected a callable function, got "
                f"{equation}"
            )
        # compute the signature
        sig = inspect.signature(equation)
        self.__len_sig = len(sig.parameters)
        self.__equation = equation

    def residual(self, input_, output_, params_=None):
        """
        Compute the residual of the equation.

        :param LabelTensor input_: Input points where the equation is evaluated.
        :param LabelTensor output_: Output tensor, eventually produced by a
            :class:`torch.nn.Module` instance.
        :param dict params_: Dictionary of unknown parameters, associated with a
            :class:`~pina.problem.inverse_problem.InverseProblem` instance.
            If the equation is not related to a
            :class:`~pina.problem.inverse_problem.InverseProblem` instance, the
            parameters must be initialized to ``None``. Default is ``None``.
        :return: The computed residual of the equation.
        :rtype: LabelTensor
        :raises RuntimeError: If the underlying equation signature length is not
            2 (direct problem) or 3 (inverse problem).
        """
        # Move the equation to the input_ device
        self.to(input_.device)

        # Call the underlying equation based on its signature length
        if self.__len_sig == 2:
            return self.__equation(input_, output_)
        if self.__len_sig == 3:
            return self.__equation(input_, output_, params_)
        raise RuntimeError(
            f"Unexpected number of arguments in equation: {self.__len_sig}. "
            "Expected either 2 (direct problem) or 3 (inverse problem)."
        )
