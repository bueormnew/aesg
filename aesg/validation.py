"""
AESG Input Validation.

Validates all configuration parameters and model inputs before any disk
or memory operations occur. Collects multiple errors and reports them in
a single exception for efficient debugging.
"""

from typing import List

from aesg.exceptions import AESGConfigError


class ValidationError:
    """Represents a single validation error for a specific parameter.

    Parameters
    ----------
    param : str
        Name of the parameter that failed validation.
    value : object
        The invalid value that was provided.
    expected : str
        Human-readable description of what was expected.
    """

    def __init__(self, param: str, value, expected: str):
        self.param = param
        self.value = value
        self.expected = expected

    def __str__(self) -> str:
        return f"Parameter '{self.param}': got {self.value!r}, expected {self.expected}"


# Valid values for string-based config fields
_VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
_VALID_MEMORY_MODES = ("TRAIN", "FINETUNE", "INFERENCE", "ONLINE")

# Integer fields that must be >= 1
_POSITIVE_INT_FIELDS = (
    "vector_dim",
    "max_concepts",
    "max_edges_per_node",
    "spreading_activation_steps",
    "consolidation_epoch_interval",
    "coactivation_threshold_create",
    "split_density_threshold",
    "novelty_birth_threshold",
    "survival_threshold_frequency",
    "region_detection_interval",
    "batch_size",
)

# Float fields that must be in [0.0, 1.0]
_UNIT_FLOAT_FIELDS = (
    "spreading_activation_decay",
    "prune_confidence_threshold",
    "survival_threshold_relevance",
    "merge_similarity_threshold",
    "novelty_explanation_threshold",
)


class Validator:
    """Validates AESG configuration parameters and model inputs.

    All validation methods are static. Errors are collected and reported
    as a single AESGConfigError containing up to 20 individual messages.
    """

    @staticmethod
    def validate_config(config: "AESGConfig") -> None:
        """Validate all parameters in an AESGConfig instance.

        Checks integer positives (>= 1), float thresholds ([0.0, 1.0]),
        vector_dim range ([1, 8192]), log_level, and memory_mode.

        Parameters
        ----------
        config : AESGConfig
            The configuration instance to validate.

        Raises
        ------
        AESGConfigError
            If any validation errors are found. The message lists all
            errors (up to 20).
        """
        errors: List[ValidationError] = []

        # Integer positive checks (>= 1)
        for param in _POSITIVE_INT_FIELDS:
            val = getattr(config, param)
            if not isinstance(val, int) or val < 1:
                errors.append(ValidationError(param, val, "integer >= 1"))

        # vector_dim upper bound check [1, 8192]
        if isinstance(config.vector_dim, int) and config.vector_dim > 8192:
            # Remove the generic "integer >= 1" error if present since we
            # provide a more specific range message for vector_dim
            errors = [e for e in errors if e.param != "vector_dim"]
            errors.append(
                ValidationError(
                    "vector_dim", config.vector_dim, "integer in [1, 8192]"
                )
            )

        # Float threshold checks [0.0, 1.0]
        for param in _UNIT_FLOAT_FIELDS:
            val = getattr(config, param)
            if not isinstance(val, (int, float)) or val < 0.0 or val > 1.0:
                errors.append(
                    ValidationError(param, val, "float in [0.0, 1.0]")
                )

        # log_level validation
        if config.log_level not in _VALID_LOG_LEVELS:
            errors.append(
                ValidationError(
                    "log_level",
                    config.log_level,
                    f"one of {list(_VALID_LOG_LEVELS)}",
                )
            )

        # memory_mode validation
        if config.memory_mode not in _VALID_MEMORY_MODES:
            errors.append(
                ValidationError(
                    "memory_mode",
                    config.memory_mode,
                    f"one of {list(_VALID_MEMORY_MODES)}",
                )
            )

        if errors:
            capped = errors[:20]
            msg = "Configuration validation failed:\n" + "\n".join(
                str(e) for e in capped
            )
            raise AESGConfigError(msg)

    @staticmethod
    def validate_vector_dim_match(config_dim: int, tensor_dim: int) -> None:
        """Validate that a tensor's dimension matches the configured vector_dim.

        Parameters
        ----------
        config_dim : int
            The vector_dim from AESGConfig.
        tensor_dim : int
            The actual dimension of the tensor being checked.

        Raises
        ------
        AESGConfigError
            If the dimensions do not match.
        """
        if config_dim != tensor_dim:
            raise AESGConfigError(
                f"vector_dim mismatch: config expects {config_dim}, "
                f"got tensor with dim {tensor_dim}"
            )

    @staticmethod
    def validate_memory_mode(mode: str) -> None:
        """Validate that a memory mode string is one of the allowed values.

        Parameters
        ----------
        mode : str
            The mode string to validate.

        Raises
        ------
        AESGConfigError
            If the mode is not one of TRAIN, FINETUNE, INFERENCE, ONLINE.
        """
        if mode not in _VALID_MEMORY_MODES:
            raise AESGConfigError(
                f"Invalid memory_mode '{mode}'. "
                f"Valid modes: {list(_VALID_MEMORY_MODES)}"
            )
