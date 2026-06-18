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
from aesg.config import AESGConfig

class AESGMemory(nn.Module):
    """
    AESGMemory V2.1 - Ecosistema Conceptual
    """
    def __init__(self, directory: str, config: AESGConfig = AESGConfig(), device: str = 'cpu'):
        super().__init__()
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
        
        self.active_state = BatchActiveContext(batch_size=1)
        self.current_batch_size = 1

    def _ensure_batch_state(self, batch_size: int):
        if batch_size != self.current_batch_size:
            self.active_state = BatchActiveContext(batch_size)
            self.current_batch_size = batch_size

    def retrieve(self, query: torch.Tensor, batch_idx: int = 0) -> RetrievedContext:
        """
        Navega la memoria y maneja la detección de novedad.
        """
        if query.dim() > 1:
            q_flat = query.view(-1)[0:self.vector_dim]
        else:
            q_flat = query
            
        # 1. Si la memoria está literalmente vacía, boostrap forzado
        if self.storage.node_count == 0:
            self.cognitive_engine.create_sensory_concept(q_flat)
            
        # 2. Navegar
        context, explanation_score = self.navigator.retrieve(q_flat, self.active_state, batch_idx=batch_idx)
        
        # 3. Curiosidad Adaptativa (Novelty Detection)
        if explanation_score < self.config.novelty_explanation_threshold:
            self.active_state.novelty_buffer[batch_idx] += 1
            
            # Si el contador supera el umbral, nace el concepto
            if self.active_state.novelty_buffer[batch_idx] >= self.config.novelty_birth_threshold:
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
        self.cognitive_engine.consolidate()
        self.storage.flush()

    def reset_state(self):
        self.active_state.clear()

    def save(self):
        self.storage.flush()
