"""
AESG Exception Hierarchy.

Provides granular, subsystem-specific exceptions for all AESG operations.
All exceptions inherit from AESGError, enabling both fine-grained and
broad exception handling via standard polymorphism.

Exception chaining is supported via the standard ``raise X from original``
mechanism, preserving the original cause in ``__cause__``.
"""


class AESGError(Exception):
    """Base exception for all AESG errors.

    Parameters
    ----------
    message : str
        Human-readable error description. Truncated to 500 characters.
    subsystem : str
        Name of the subsystem that originated the error.
    """

    def __init__(self, message: str, subsystem: str):
        self.message = message[:500]
        self.subsystem = subsystem
        super().__init__(self.message)


class AESGConfigError(AESGError):
    """Raised when configuration validation fails.

    Subsystem: config
    """

    def __init__(self, message: str):
        super().__init__(message, subsystem="config")


class AESGMemoryError(AESGError):
    """Raised when a memory operation fails.

    Subsystem: memory
    """

    def __init__(self, message: str):
        super().__init__(message, subsystem="memory")


class AESGNavigationError(AESGError):
    """Raised when a navigation/spreading activation operation fails.

    Subsystem: navigation
    """

    def __init__(self, message: str):
        super().__init__(message, subsystem="navigation")


class AESGTrainingError(AESGError):
    """Raised when a training operation fails.

    Subsystem: trainer
    """

    def __init__(self, message: str):
        super().__init__(message, subsystem="trainer")


class AESGStorageError(AESGError):
    """Raised when a storage/persistence operation fails.

    Subsystem: storage
    """

    def __init__(self, message: str):
        super().__init__(message, subsystem="storage")
