"""
AESG Pack Manager.

Manages the lifecycle of memory packs: loading from disk, attaching to active
memory with priority-based weighting, and detaching without affecting base memory.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, TYPE_CHECKING

import numpy as np

from aesg.exceptions import AESGMemoryError, AESGStorageError
from aesg.packs.format import deserialize_pack

if TYPE_CHECKING:
    from aesg.memory.controller import AESGMemory


class PackState(Enum):
    """Lifecycle state of a memory pack."""
    LOADED = "loaded"
    ATTACHED = "attached"


@dataclass
class MemoryPack:
    """A deserialized memory pack with metadata and state.

    Parameters
    ----------
    name : str
        Unique identifier for this pack (derived from filename).
    version : str
        Version string for the pack.
    vector_dim : int
        Dimensionality of node vectors in this pack.
    nodes : np.ndarray
        Structured numpy array of graph nodes.
    edges : np.ndarray
        Structured numpy array of graph edges.
    checksum : str
        SHA-256 hex digest of the pack payload.
    priority : int
        Priority weight for spreading activation (0-100). Default 50.
    state : PackState
        Current lifecycle state of the pack.
    """
    name: str
    version: str
    vector_dim: int
    nodes: np.ndarray
    edges: np.ndarray
    checksum: str
    priority: int = 50
    state: PackState = field(default=PackState.LOADED)


MAX_ATTACHED_PACKS = 16


class PackManager:
    """Manages loading, attaching, and detaching of memory packs.

    Parameters
    ----------
    memory : AESGMemory
        Reference to the active memory controller for dimension validation.
    """

    def __init__(self, memory: "AESGMemory"):
        self.memory = memory
        self._loaded_packs: Dict[str, MemoryPack] = {}
        self._attached_packs: Dict[str, MemoryPack] = {}

    def load_pack(self, path: str) -> MemoryPack:
        """Load a .aesgpack file into LOADED state.

        Parameters
        ----------
        path : str
            Path to the .aesgpack binary file.

        Returns
        -------
        MemoryPack
            The deserialized pack in LOADED state.

        Raises
        ------
        AESGStorageError
            If the file is invalid, corrupt, or vector_dim doesn't match memory.
        """
        data = deserialize_pack(path)

        # Validate vector_dim compatibility
        if data["vector_dim"] != self.memory.vector_dim:
            raise AESGStorageError(
                f"Pack vector_dim mismatch: pack has {data['vector_dim']}, "
                f"memory expects {self.memory.vector_dim}"
            )

        pack = MemoryPack(
            name=data["name"],
            version="1.0",
            vector_dim=data["vector_dim"],
            nodes=data["nodes"],
            edges=data["edges"],
            checksum=data["checksum"],
            priority=50,
            state=PackState.LOADED,
        )

        self._loaded_packs[pack.name] = pack
        return pack

    def attach_pack(self, pack: MemoryPack, priority: int = 50) -> None:
        """Attach a loaded pack to active memory with given priority.

        Parameters
        ----------
        pack : MemoryPack
            The pack to attach (must be in LOADED state).
        priority : int
            Priority weight for spreading activation weighting (0-100).

        Raises
        ------
        AESGMemoryError
            If the maximum number of attached packs (16) would be exceeded.
        """
        if len(self._attached_packs) >= MAX_ATTACHED_PACKS:
            raise AESGMemoryError(
                f"Maximum {MAX_ATTACHED_PACKS} packs can be attached simultaneously"
            )

        pack.priority = priority
        pack.state = PackState.ATTACHED

        # Remove from loaded if present, add to attached
        self._loaded_packs.pop(pack.name, None)
        self._attached_packs[pack.name] = pack

    def detach_pack(self, pack_name: str) -> None:
        """Remove pack from active memory, returning it to LOADED state.

        Parameters
        ----------
        pack_name : str
            Name of the pack to detach.

        Raises
        ------
        AESGMemoryError
            If the pack is not currently attached.
        """
        if pack_name not in self._attached_packs:
            raise AESGMemoryError(f"Pack '{pack_name}' is not attached")

        pack = self._attached_packs.pop(pack_name)
        pack.state = PackState.LOADED
        self._loaded_packs[pack_name] = pack

    def get_pack_weights(self) -> Dict[str, float]:
        """Return normalized priority weights for all attached packs.

        Returns
        -------
        Dict[str, float]
            Mapping of pack name to normalized weight (priority_i / sum_all).
            Empty dict if no packs are attached.
        """
        if not self._attached_packs:
            return {}

        total_priority = sum(p.priority for p in self._attached_packs.values())
        if total_priority == 0:
            return {}

        return {
            name: pack.priority / total_priority
            for name, pack in self._attached_packs.items()
        }

    def get_attached_packs(self) -> Dict[str, MemoryPack]:
        """Return a copy of the currently attached packs dictionary.

        Returns
        -------
        Dict[str, MemoryPack]
            Copy of the attached packs mapping.
        """
        return dict(self._attached_packs)
