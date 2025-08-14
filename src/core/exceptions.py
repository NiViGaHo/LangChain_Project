class ApplicationError(Exception):
    """Base application exception."""


class ExternalServiceError(ApplicationError):
    """Raised when an external dependency fails (e.g., API quota)."""


class ConfigurationError(ApplicationError):
    """Raised when required configuration is missing or invalid."""


