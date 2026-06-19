import torch
import torch.nn as nn
from typing import Optional, List
import uuid
import os

from aesg.storage.engine import AESGStorage
from aesg.storage.logger import EvolutionLogger
from aesg.graph.core import SemanticGraph
from aesg.graph.cognitive_engine import CognitiveEngine
from aesg.memory.navigation import Navigator
from aesg.memory.context import RetrievedContext
from aesg.memory.state import BatchActiveContext
from aesg.memory.modes import MemoryMode, MODE_PERMISSIONS
from aesg.config import AESGConfig
from aesg.exceptions import AESGMemoryError
from aesg.validation import Validator
from aesg.packs.manager import PackManager
from aesg.packs.export import export_pack as _export_pack

class AESGMemory(nn.Module):
    """
    AESGMemory V2.1 - Ecosistema Conceptual
    """
    def __init__(self, directory: str, config: AESGConfig = AESGConfig(), device: str = 'cpu'):
        super().__init__()
        
        # Validate config before any disk operations
        Validator.validate_config(config)
        
        self.mode = MemoryMode(config.memory_mode)
        
        self.directory = directory
        self.config = config
        self.vector_dim = config.vector_dim
        self.device = device
        
        self.storage = AESGStorage(
            directory, 
            self.vector_dim, 
            initial_capacity_nodes=config.max_concepts,
            initial_capacity_edges=config.max_concepts * 5
        )
        self.logger = EvolutionLogger(directory)
        self.graph = SemanticGraph(self.storage)
        
        # El CognitiveEngine es un nn.Module para que `abstraction_projection` reciba gradientes
        self.cognitive_engine = CognitiveEngine(self.graph, self.config, self.logger)
        
        self.navigator = Navigator(self.graph, self.config)
        
        self.pack_manager = PackManager(self)
        
        self.active_state = BatchActiveContext(batch_size=1)
        self.current_batch_size = 1

    def _ensure_batch_state(self, batch_size: int):
        if batch_size != self.current_batch_size:
            self.active_state = BatchActiveContext(batch_size)
            self.current_batch_size = batch_size

    def set_mode(self, mode_name: str) -> None:
        """Set the operational mode of the memory controller.

        Parameters
        ----------
        mode_name : str
            One of TRAIN, FINETUNE, INFERENCE, ONLINE.

        Raises
        ------
        AESGMemoryError
            If mode_name is not a valid MemoryMode value.
        """
        valid_modes = [m.value for m in MemoryMode]
        if mode_name not in valid_modes:
            raise AESGMemoryError(
                f"Invalid memory mode '{mode_name}'. "
                f"Valid modes: {valid_modes}"
            )
        self.mode = MemoryMode(mode_name)

    def _can(self, operation: str) -> bool:
        """Check if the current mode permits the given operation.

        Parameters
        ----------
        operation : str
            Operation key from MODE_PERMISSIONS (e.g. 'create_concepts',
            'consolidation', 'reorganization', 'evolutionary_pressure').

        Returns
        -------
        bool
            True if the operation is permitted in the current mode.
        """
        return MODE_PERMISSIONS[self.mode][operation]

    def _permissions(self) -> dict:
        """Return the full permission dict for the current mode.

        Returns
        -------
        dict
            Permission mapping for the active MemoryMode.
        """
        return MODE_PERMISSIONS[self.mode]

    def load_pack(self, path: str):
        """Load a .aesgpack file into LOADED state.

        Parameters
        ----------
        path : str
            Path to the .aesgpack binary file.

        Returns
        -------
        MemoryPack
            The deserialized pack in LOADED state.
        """
        return self.pack_manager.load_pack(path)

    def attach_pack(self, pack, priority: int = 50) -> None:
        """Attach a loaded pack to active memory with given priority.

        Parameters
        ----------
        pack : MemoryPack
            The pack to attach (must be in LOADED state).
        priority : int
            Priority weight for spreading activation weighting (0-100).
        """
        self.pack_manager.attach_pack(pack, priority)

    def detach_pack(self, pack_name: str) -> None:
        """Remove pack from active memory, returning it to LOADED state.

        Parameters
        ----------
        pack_name : str
            Name of the pack to detach.
        """
        self.pack_manager.detach_pack(pack_name)

    def export_pack(self, path: str, region_id=None, min_relevance=None) -> None:
        """Export a filtered subgraph from memory as a .aesgpack file.

        Parameters
        ----------
        path : str
            Output file path for the .aesgpack file.
        region_id : int, optional
            If provided, only export nodes belonging to this region.
        min_relevance : float, optional
            If provided, only export nodes with relevance >= this threshold.
        """
        _export_pack(self, path, region_id=region_id, min_relevance=min_relevance)

    def retrieve(self, query: torch.Tensor, batch_idx: int = 0) -> RetrievedContext:
        """
        Navega la memoria y maneja la detección de novedad.
        """
        if query.dim() > 1:
            q_flat = query.view(-1)[0:self.vector_dim]
        else:
            q_flat = query
        
        # Validate tensor dimension matches configured vector_dim
        Validator.validate_vector_dim_match(self.vector_dim, q_flat.shape[-1])
            
        # 1. Si la memoria está literalmente vacía, boostrap forzado
        if self.storage.node_count == 0:
            if self._can("create_concepts"):
                self.cognitive_engine.create_sensory_concept(q_flat)
            
        # 2. Navegar
        context, explanation_score = self.navigator.retrieve(q_flat, self.active_state, batch_idx=batch_idx)
        
        # 3. Curiosidad Adaptativa (Novelty Detection)
        if explanation_score < self.config.novelty_explanation_threshold:
            self.active_state.novelty_buffer[batch_idx] += 1
            
            # Si el contador supera el umbral, nace el concepto
            if self.active_state.novelty_buffer[batch_idx] >= self.config.novelty_birth_threshold:
                if self._can("create_concepts"):
                    self.cognitive_engine.create_sensory_concept(q_flat)
                self.active_state.novelty_buffer[batch_idx] = 0 # Reset
                # Forzar recálculo del contexto después del nacimiento
                context, _ = self.navigator.retrieve(q_flat, self.active_state, batch_idx=batch_idx)
        else:
            # Si el grafo lo explica bien, reducimos el buffer (ruido aislado se disipa)
            self.active_state.novelty_buffer[batch_idx] = max(0, self.active_state.novelty_buffer[batch_idx] - 1)
            
        return context
        
    def forward(self, query: torch.Tensor) -> RetrievedContext:
        batch_size = query.size(0) if query.dim() > 1 else 1
        self._ensure_batch_state(batch_size)
        return self.retrieve(query, batch_idx=0)

    def update_topology(self):
        if self._can("consolidation"):
            self.cognitive_engine.consolidate()
        self.storage.flush()

    def reset_state(self):
        self.active_state.clear()

    def save(self):
        self.storage.flush()
