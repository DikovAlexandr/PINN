"""Module for PINA Conditions classes."""

__all__ = [
    "Condition",
    "ConditionInterface",
    "DomainEquationCondition",
    "InputTargetCondition",
    "TensorInputTensorTargetCondition",
    "TensorInputGraphTargetCondition",
    "GraphInputTensorTargetCondition",
    "GraphInputGraphTargetCondition",
    "InputEquationCondition",
    "InputTensorEquationCondition",
    "InputGraphEquationCondition",
    "DataCondition",
    "GraphDataCondition",
    "TensorDataCondition",
]

from .condition_interface import ConditionInterface
from .condition import Condition
from .domain_equation_condition import DomainEquationCondition
from .input_target_condition import (
    InputTargetCondition,
    TensorInputTensorTargetCondition,
    TensorInputGraphTargetCondition,
    GraphInputTensorTargetCondition,
    GraphInputGraphTargetCondition,
)
from .input_equation_condition import (
    InputEquationCondition,
    InputTensorEquationCondition,
    InputGraphEquationCondition,
)
from .data_condition import (
    DataCondition,
    GraphDataCondition,
    TensorDataCondition,
)
