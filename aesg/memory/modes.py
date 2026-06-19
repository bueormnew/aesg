"""
AESG Memory Modes.

Defines the MemoryMode enum and the permission matrix that controls
which operations are permitted in each operational phase of the memory
system (training, fine-tuning, inference, online learning).

The permission matrix is consulted by the AESGMemory controller to gate
operations like concept creation, consolidation, reorganization, and
evolutionary pressure based on the current mode.
"""

from enum import Enum
from typing import Dict


class MemoryMode(Enum):
    """Operational mode for the AESG memory controller.

    Each mode defines a set of permitted operations and a learning rate
    scale factor that modulates the base learning rate.

    Attributes
    ----------
    TRAIN : str
        Full training mode. All operations enabled, learning_rate_scale=1.0.
    FINETUNE : str
        Fine-tuning mode. Concept creation and consolidation enabled,
        reorganization and evolutionary pressure disabled,
        learning_rate_scale=0.1.
    INFERENCE : str
        Read-only mode. All write operations disabled,
        learning_rate_scale=0.0.
    ONLINE : str
        Online/continual learning mode. Concept creation and consolidation
        enabled per forward call, reorganization and evolutionary pressure
        disabled, learning_rate_scale=1.0.
    """

    TRAIN = "TRAIN"
    FINETUNE = "FINETUNE"
    INFERENCE = "INFERENCE"
    ONLINE = "ONLINE"


# Permission matrix: mode -> allowed operations and learning rate scale.
#
# Keys:
#   create_concepts (bool): Whether new sensory/abstract concepts can be created.
#   consolidation (bool): Whether consolidation (age increment, relevance decay) runs.
#   reorganization (bool): Whether topological ops (merge, split, find_semantic_regions) run.
#   evolutionary_pressure (bool): Whether pruning by relevance/frequency is active.
#   learning_rate_scale (float): Multiplier applied to the base learning rate.
MODE_PERMISSIONS: Dict[MemoryMode, Dict] = {
    MemoryMode.TRAIN: {
        "create_concepts": True,
        "consolidation": True,
        "reorganization": True,
        "evolutionary_pressure": True,
        "learning_rate_scale": 1.0,
    },
    MemoryMode.FINETUNE: {
        "create_concepts": True,
        "consolidation": True,
        "reorganization": False,
        "evolutionary_pressure": False,
        "learning_rate_scale": 0.1,
    },
    MemoryMode.INFERENCE: {
        "create_concepts": False,
        "consolidation": False,
        "reorganization": False,
        "evolutionary_pressure": False,
        "learning_rate_scale": 0.0,
    },
    MemoryMode.ONLINE: {
        "create_concepts": True,
        "consolidation": True,
        "reorganization": False,
        "evolutionary_pressure": False,
        "learning_rate_scale": 1.0,
    },
}
