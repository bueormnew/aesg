import torch
import numpy as np
from typing import List, Set, Tuple
from aesg.graph.core import SemanticGraph
from aesg.memory.context import RetrievedContext
from aesg.memory.state import BatchActiveContext
from aesg.config import AESGConfig

class Navigator:
    def __init__(self, graph: SemanticGraph, config: AESGConfig):
        self.graph = graph
        self.config = config

    def _graph_explanation_score(self, query_np: np.ndarray, active_indices: Set[int]) -> float:
        """
        Calcula qué tan bien el subgrafo activo puede explicar el query_vector.
        Representa la 'Capacidad Explicativa del Subgrafo Activo'.
        """
        if not active_indices:
            return 0.0
            
        vectors = self.graph.storage.nodes['vector'][list(active_indices)]
        
        # Un enfoque simple: ¿Qué tan cerca está el query del centroide de la región activa?
        # En una arquitectura real más avanzada, esto podría ser la proyección ortogonal sobre la base.
        centroid = np.mean(vectors, axis=0)
        
        norm_q = np.linalg.norm(query_np)
        norm_c = np.linalg.norm(centroid)
        if norm_q == 0 or norm_c == 0:
            return 0.0
            
        sim = np.dot(query_np, centroid) / (norm_q * norm_c)
        return float(max(0.0, sim)) # 0.0 to 1.0

    def retrieve(self, query_vector: torch.Tensor, active_state: BatchActiveContext, batch_idx: int = 0) -> Tuple[RetrievedContext, float]:
        """
        Devuelve el contexto recuperado y el 'graph_explanation_score'.
        """
        q_np = query_vector.detach().cpu().numpy()
        current_active = active_state.get_active(batch_idx)
        
        # 1. Calcular sorpresa (Novedad)
        explanation_score = self._graph_explanation_score(q_np, current_active)
        
        primary_indices = set()
        
        if not current_active:
            knn = self.graph.find_nearest_neighbors(q_np, top_k=2)
            for idx, _ in knn:
                primary_indices.add(idx)
        else:
            primary_indices = set(current_active)
            
        # Determinar región dominante del contexto activo
        dominant_region = 0
        if current_active:
            regions = [self.graph.storage.nodes['region_id'][idx] for idx in current_active]
            # Get most common region
            dominant_region = max(set(regions), key=regions.count)
            
        # 2. Spreading Activation con Multiplicador de Región
        activated_nodes = dict()
        for idx in primary_indices:
            activated_nodes[idx] = 1.0 
            
        paths = []
        
        for step in range(self.config.spreading_activation_steps):
            next_activated = dict()
            for idx, energy in activated_nodes.items():
                if energy < 0.1: continue
                
                edges = self.graph.storage.get_edges(idx)
                for e in edges:
                    t_id = e['target_id']
                    t_idx_arr = np.where(self.graph.storage.nodes['id'][:self.graph.storage.node_count] == t_id)[0]
                    if len(t_idx_arr) > 0:
                        t_idx = t_idx_arr[0]
                        if self.graph.storage.nodes['is_active'][t_idx] == 1:
                            # Facilitación de Región
                            t_region = self.graph.storage.nodes['region_id'][t_idx]
                            region_mult = self.config.region_facilitation_multiplier if t_region == dominant_region else 1.0
                            
                            transferred_energy = energy * e['weight'] * e['confidence'] * self.config.spreading_activation_decay * region_mult
                            
                            if transferred_energy > 0.05:
                                next_activated[t_idx] = next_activated.get(t_idx, 0.0) + transferred_energy
                                paths.append([idx, t_idx])
            
            for idx, energy in next_activated.items():
                activated_nodes[idx] = max(activated_nodes.get(idx, 0.0), energy)
                
        sorted_activated = sorted(activated_nodes.items(), key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, e in sorted_activated[:10]]
        
        active_state.set_active(batch_idx, set(top_indices[:3]))
        
        if not top_indices:
             return RetrievedContext([], [], torch.zeros((1, len(q_np)), device=query_vector.device), torch.ones((1,), device=query_vector.device), []), explanation_score
             
        vectors_np = self.graph.storage.nodes['vector'][top_indices]
        vectors_tensor = torch.tensor(vectors_np, device=query_vector.device, dtype=torch.float32)
        
        weights = [activated_nodes[idx] for idx in top_indices]
        weights_tensor = torch.tensor(weights, device=query_vector.device, dtype=torch.float32)
        
        context = RetrievedContext(
            primary_concepts=top_indices[:3],
            secondary_concepts=top_indices[3:],
            concept_vectors=vectors_tensor,
            relation_weights=weights_tensor,
            paths=paths
        )
        
        return context, explanation_score
