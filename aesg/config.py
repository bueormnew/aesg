"""
AESG Unified Configuration.

Provides a single frozen dataclass centralizing all AESG parameters
with factory methods for common domains and JSON serialization.

All subsystems (memory, navigation, consolidation, evolutionary pressure,
training, storage, abstraction, regions, logging) are configured through
this single AESGConfig instance.
"""

from dataclasses import dataclass, asdict, fields
from typing import Optional
import json

from aesg.exceptions import AESGConfigError


@dataclass(frozen=True)
class AESGConfig:
    """Unified configuration for AESG V3.

    A frozen (immutable) dataclass that centralizes all parameters for
    every AESG subsystem. Once created, attributes cannot be reassigned.

    Parameters
    ----------
    vector_dim : int
        Dimensionality of concept vectors. Must be in [1, 8192].
    max_concepts : int
        Maximum number of concepts allowed in the semantic graph.
    max_edges_per_node : int
        Maximum edges per node before considering split.

    spreading_activation_decay : float
        Decay factor per hop during spreading activation. In [0.0, 1.0].
    spreading_activation_steps : int
        Number of hops for spreading activation traversal.
    region_facilitation_multiplier : float
        Multiplier for intra-region navigation weighting.

    consolidation_epoch_interval : int
        Number of epochs between consolidation passes.
    prune_confidence_threshold : float
        Minimum confidence for an edge to survive pruning. In [0.0, 1.0].

    survival_threshold_relevance : float
        Minimum relevance for a concept to survive evolutionary pressure.
    survival_threshold_frequency : int
        Minimum activation frequency for concept survival.

    coactivation_threshold_create : int
        Coactivation count required to create an abstract node.
    merge_similarity_threshold : float
        Neighborhood similarity threshold for merging concepts.
    split_density_threshold : int
        Maximum edges before considering node partition.
    novelty_explanation_threshold : float
        Minimum graph explanation score for novelty detection.
    novelty_birth_threshold : int
        Persistence count before novelty becomes a concept.

    learning_rate : float
        Default learning rate for training.
    batch_size : int
        Default batch size for data loading.

    storage_directory : Optional[str]
        Default storage directory path. None means caller must provide.

    abstraction_enabled : bool
        Whether abstraction layer is active.

    region_detection_interval : int
        Steps between region detection passes.

    log_level : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).

    memory_mode : str
        Default memory mode (TRAIN, FINETUNE, INFERENCE, ONLINE).

    Examples
    --------
    >>> config = AESGConfig()
    >>> config.vector_dim
    256
    >>> config = AESGConfig.for_text()
    >>> config.vector_dim
    128
    >>> json_str = config.to_json()
    >>> restored = AESGConfig.from_json(json_str)
    >>> restored == config
    True
    """

    # --- Memory ---
    vector_dim: int = 256
    max_concepts: int = 1_000_000
    max_edges_per_node: int = 1000

    # --- Navigation (Spreading Activation) ---
    spreading_activation_decay: float = 0.8
    spreading_activation_steps: int = 3
    region_facilitation_multiplier: float = 1.5

    # --- Consolidation ---
    consolidation_epoch_interval: int = 10
    prune_confidence_threshold: float = 0.1

    # --- Evolutionary Pressure ---
    survival_threshold_relevance: float = 0.05
    survival_threshold_frequency: int = 5

    # --- Cognitive Thresholds ---
    coactivation_threshold_create: int = 50
    merge_similarity_threshold: float = 0.95
    split_density_threshold: int = 800
    novelty_explanation_threshold: float = 0.6
    novelty_birth_threshold: int = 3

    # --- Training ---
    learning_rate: float = 1e-3
    batch_size: int = 32

    # --- Storage ---
    storage_directory: Optional[str] = None

    # --- Abstraction ---
    abstraction_enabled: bool = True

    # --- Regions ---
    region_detection_interval: int = 50

    # --- Logging ---
    log_level: str = "INFO"

    # --- Memory Mode ---
    memory_mode: str = "TRAIN"

    # -----------------------------------------------------------------
    # Factory methods
    # -----------------------------------------------------------------

    @classmethod
    def for_text(cls) -> "AESGConfig":
        """Create a configuration optimized for text tasks.

        Returns an AESGConfig with vector_dim=128 and max_concepts=500_000.
        """
        return cls(vector_dim=128, max_concepts=500_000)

    @classmethod
    def for_image(cls) -> "AESGConfig":
        """Create a configuration optimized for image tasks.

        Returns an AESGConfig with vector_dim=256 and max_concepts=200_000.
        """
        return cls(vector_dim=256, max_concepts=200_000)

    @classmethod
    def for_classification(cls) -> "AESGConfig":
        """Create a configuration optimized for classification tasks.

        Returns an AESGConfig with vector_dim=64 and max_concepts=100_000.
        """
        return cls(vector_dim=64, max_concepts=100_000)

    # -----------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------

    def to_json(self) -> str:
        """Serialize this configuration to a JSON string.

        Returns
        -------
        str
            JSON representation of all configuration parameters.
        """
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "AESGConfig":
        """Deserialize a JSON string into an AESGConfig instance.

        Parameters
        ----------
        json_str : str
            JSON string previously produced by ``to_json()`` or manually
            constructed with valid field names and values.

        Returns
        -------
        AESGConfig
            A new frozen configuration instance.

        Raises
        ------
        AESGConfigError
            If the JSON is malformed or contains unknown fields.
        """
        try:
            data = json.loads(json_str)
        except (json.JSONDecodeError, TypeError) as e:
            raise AESGConfigError(
                f"Malformed JSON: {e}"
            ) from e

        if not isinstance(data, dict):
            raise AESGConfigError(
                "JSON must be an object (dict), got "
                f"{type(data).__name__}"
            )

        # Reject unknown fields
        valid_fields = {f.name for f in fields(cls)}
        unknown = set(data.keys()) - valid_fields
        if unknown:
            raise AESGConfigError(
                f"Unknown configuration fields: {sorted(unknown)}"
            )

        try:
            return cls(**data)
        except TypeError as e:
            raise AESGConfigError(
                f"Error creating AESGConfig from JSON data: {e}"
            ) from e
