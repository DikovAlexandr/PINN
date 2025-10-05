"""Custom exceptions for PINN solver module."""


class PINNError(Exception):
    """Base exception class for PINN solver errors."""
    pass


class GeometryError(PINNError):
    """Exception raised for geometry-related errors."""
    pass


class ConfigurationError(PINNError):
    """Exception raised for configuration-related errors."""
    pass


class TrainingError(PINNError):
    """Exception raised for training-related errors."""
    pass


class ValidationError(PINNError):
    """Exception raised for validation-related errors."""
    pass


class ConvergenceError(TrainingError):
    """Exception raised when training fails to converge."""
    pass


class DeviceError(PINNError):
    """Exception raised for device-related errors."""
    pass
