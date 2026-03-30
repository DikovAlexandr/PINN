"""Module for the Equation Interface."""

from abc import ABCMeta, abstractmethod
import torch


class EquationInterface(metaclass=ABCMeta):
    """
    Abstract base class for equations.

    Equations in PINA simplify the training process. When defining a problem,
    each equation passed to a :class:`~pina.condition.condition.Condition`
    object must be either an :class:`~pina.equation.equation.Equation` or a
    :class:`~pina.equation.system_equation.SystemEquation` instance.

    An :class:`~pina.equation.equation.Equation` is a wrapper for a callable
    function, while :class:`~pina.equation.system_equation.SystemEquation`
    wraps a list of callable functions. To streamline code writing, PINA
    provides a diverse set of pre-implemented equations, such as
    :class:`~pina.equation.equation_factory.FixedValue`,
    :class:`~pina.equation.equation_factory.FixedGradient`, and many others.
    """

    @abstractmethod
    def residual(self, input_, output_, params_):
        """
        Abstract method to compute the residual of an equation.

        :param LabelTensor input_: Input points where the equation is evaluated.
        :param LabelTensor output_: Output tensor, eventually produced by a
            :class:`torch.nn.Module` instance.
        :param dict params_: Dictionary of unknown parameters, associated with a
            :class:`~pina.problem.inverse_problem.InverseProblem` instance.
        :return: The computed residual of the equation.
        :rtype: LabelTensor
        """

    def to(self, device):
        """
        Move all tensor attributes to the specified device.

        :param torch.device device: The target device to move the tensors to.
        :return: The instance moved to the specified device.
        :rtype: EquationInterface
        """
        # Iterate over all attributes of the Equation
        for key, val in self.__dict__.items():

            # Move tensors in dictionaries to the specified device
            if isinstance(val, dict):
                self.__dict__[key] = {
                    k: v.to(device) if torch.is_tensor(v) else v
                    for k, v in val.items()
                }

            # Move tensors in lists to the specified device
            elif isinstance(val, list):
                self.__dict__[key] = [
                    v.to(device) if torch.is_tensor(v) else v for v in val
                ]

            # Move tensor attributes to the specified device
            elif torch.is_tensor(val):
                self.__dict__[key] = val.to(device)

        return self
