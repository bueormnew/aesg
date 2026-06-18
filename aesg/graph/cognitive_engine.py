import torch
import torch.nn as nn
from typing import List, Set
import numpy as np
import uuid

from aesg.graph.core import SemanticGraph
from aesg.storage.logger import EvolutionLogger
from aesg.config import AESGConfig

class CognitiveEngine(nn.Module):
    """
    Motor Cognitivo (AESG V2.1).
    Se encarga de la evolución, consolidación, creación (novedad) y eliminación conceptual.
    Incorpora proyecciones aprendibles.
    """
    def __init__(self, graph: SemanticGraph, config: AESGConfig, logger: EvolutionLogger):
        super().__init__()
        self.graph = graph
        self.storage = graph.storage
        self.config = config
        self.logger = logger
        self.epoch = 0
        
        # Abstracciones aprendibles (permite que A+B genere un concepto en otra región del espacio)
        self.abstraction_projection = nn.Linear(config.vector_dim, config.vector_dim)

    def create_abstract_concept(self, source_indices: List[int]) -> int:
        """Crea un concepto abstracto superior pasando por la capa aprendible."""
        vectors = self.storage.nodes['vector'][source_indices]
        mean_vector = torch.tensor(np.mean(vectors, axis=0), dtype=torch.float32)
        
        # Proyección (genera el nuevo embedding abstracto)
        abstract_vector = self.abstraction_projection(mean_vector)
        abstract_np = abstract_vector.detach().numpy()
        
        new_id = hash(str(uuid.uuid4())) % (10**10)
        new_idx = self.storage.add_node(new_id, abstract_np)
        
        # Conectar con los hijos
        parent1 = self.storage.nodes[source_indices[0]]['id'] if len(source_indices) > 0 else 0
        
        for s_idx in source_indices:
            s_id = self.storage.nodes[s_idx]['id']
            self.storage.add_edge(new_idx, s_id, weight=1.0, confidence=0.5)
            self.storage.add_edge(s_idx, new_id, weight=1.0, confidence=0.5)
            
        self.logger.log_event("CREATE", id1=new_id, id2=parent1, val=float(len(source_indices)))
        return new_idx
        
    def create_sensory_concept(self, query_vector: torch.Tensor) -> int:
        """Crea un concepto desde un estímulo nuevo (novelty). El vector es exactamente el input."""
        q_np = query_vector.detach().cpu().numpy()
        new_id = hash(str(uuid.uuid4())) % (10**10)
        new_idx = self.storage.add_node(new_id, q_np)
        
        self.logger.log_event("CREATE", id1=new_id, val=1.0) # val=1.0 indicates sensory
        return new_idx

    def merge_concepts(self, idx1: int, idx2: int):
        id1 = self.storage.nodes[idx1]['id']
        id2 = self.storage.nodes[idx2]['id']
        
        v1 = self.storage.nodes['vector'][idx1]
        v2 = self.storage.nodes['vector'][idx2]
        self.storage.nodes['vector'][idx1] = (v1 + v2) / 2.0
        
        edges2 = self.storage.get_edges(idx2)
        for e in edges2:
            self.storage.add_edge(idx1, e['target_id'], e['weight'], e['confidence'])
            
        self.storage.nodes['is_active'][idx2] = 0
        self.logger.log_event("MERGE", id1=id1, id2=id2)

    def split_concept(self, idx: int):
        node_id = self.storage.nodes[idx]['id']
        self.logger.log_event("SPLIT", id1=node_id)

    def apply_evolutionary_pressure(self):
        pruned_count = 0
        active_nodes = np.where(self.storage.nodes['is_active'][:self.storage.node_count] == 1)[0]
        
        if len(active_nodes) > self.config.max_concepts:
            relevance = self.storage.nodes['relevance'][active_nodes]
            to_prune_indices = np.argsort(relevance)[:(len(active_nodes) - self.config.max_concepts)]
            for idx in to_prune_indices:
                real_idx = active_nodes[idx]
                self.storage.nodes['is_active'][real_idx] = 0
                pruned_count += 1
                
        for idx in active_nodes:
            node = self.storage.nodes[idx]
            if node['age'] > 5 and node['relevance'] < self.config.survival_threshold_relevance and node['use_frequency'] < self.config.survival_threshold_frequency:
                self.storage.nodes['is_active'][idx] = 0
                pruned_count += 1
                self.logger.log_event("PRUNE", id1=int(node['id']))

    def consolidate(self):
        active_nodes = np.where(self.storage.nodes['is_active'][:self.storage.node_count] == 1)[0]
        self.storage.nodes['age'][active_nodes] += 1
        self.storage.nodes['relevance'][active_nodes] *= 0.95
        
        self.apply_evolutionary_pressure()
        self.logger.log_event("CONSOLIDATE", id1=self.epoch, id2=len(active_nodes))
        self.epoch += 1

    def find_semantic_regions(self):
        self.logger.log_event("RESTRUCTURE")
