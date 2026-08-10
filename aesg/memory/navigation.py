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

    def retrieve(self, query_vector: torch.Tensor, active_state: BatchActiveContext, batch_idx: int = 0) -> Tuple[RetrievedContext, float, dict]:
        """
        Devuelve el contexto recuperado, el 'graph_explanation_score' y la
        información de activación necesaria para el aprendizaje hebbiano.

        Returns
        -------
        Tuple[RetrievedContext, float, dict]
            El contexto, la puntuación explicativa y un dict con las claves
            ``activated`` (índices activados en este paso, ordenados por
            energía) y ``previous`` (índices que estaban activos antes).
        """
        q_np = query_vector.detach().cpu().numpy()
        current_active = active_state.get_active(batch_idx)
        # Snapshot before it is overwritten: needed to learn temporal succession.
        previous_active = set(current_active)
        
        # 1. Calcular sorpresa (Novedad)
        explanation_score = self._graph_explanation_score(q_np, current_active)
        
        # Seeds: the current query always decides where we enter the graph
        # (otherwise navigation stays pinned to whatever was active first and
        # the query is ignored), while the previously active set is carried
        # over to provide contextual continuity across steps.
        query_seeds = set()
        knn = self.graph.find_nearest_neighbors(q_np, top_k=2)
        for idx, _ in knn:
            query_seeds.add(idx)

        context_seeds = {
            idx for idx in current_active
            if self.graph.storage.nodes['is_active'][idx] == 1
        } - query_seeds

        primary_indices = query_seeds | context_seeds
            
        # Determinar región dominante del contexto activo
        dominant_region = 0
        if current_active:
            regions = [self.graph.storage.nodes['region_id'][idx] for idx in current_active]
            # Get most common region
            dominant_region = max(set(regions), key=regions.count)
            
        # 2. Spreading Activation con Multiplicador de Región
        activated_nodes = dict()
        for idx in query_seeds:
            activated_nodes[idx] = 1.0
        for idx in context_seeds:
            # Residual energy: prior context biases navigation but must not
            # outweigh the concepts the current query actually points at.
            activated_nodes[idx] = self.config.context_carryover_energy
            
        paths = []
        seen_paths = set()
        
        for step in range(self.config.spreading_activation_steps):
            next_activated = dict()
            for idx, energy in activated_nodes.items():
                if energy < 0.1: continue
                
                edges = self.graph.storage.get_edges(idx)
                for e in edges:
                    t_id = e['target_id']
                    # O(1) id -> index resolution (was an O(N) scan per edge).
                    # Faded associations stop conducting activation.
                    if e['confidence'] < self.config.prune_confidence_threshold:
                        continue
                    t_idx = self.graph.storage.get_node_index(t_id)
                    if t_idx >= 0:
                        if self.graph.storage.nodes['is_active'][t_idx] == 1:
                            # Facilitación de Región
                            t_region = self.graph.storage.nodes['region_id'][t_idx]
                            region_mult = self.config.region_facilitation_multiplier if t_region == dominant_region else 1.0
                            
                            transferred_energy = energy * e['weight'] * e['confidence'] * self.config.spreading_activation_decay * region_mult

                            # Region facilitation can push the per-hop factor
                            # above 1.0, so clamp: activation must never gain
                            # energy as it spreads, or it diverges over hops.
                            transferred_energy = min(1.0, float(transferred_energy))
                            
                            if transferred_energy > 0.05:
                                next_activated[t_idx] = max(next_activated.get(t_idx, 0.0), transferred_energy)
                                if (idx, t_idx) not in seen_paths:
                                    seen_paths.add((idx, t_idx))
                                    paths.append([idx, t_idx])
            
            for idx, energy in next_activated.items():
                activated_nodes[idx] = max(activated_nodes.get(idx, 0.0), energy)
                
        sorted_activated = sorted(activated_nodes.items(), key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, e in sorted_activated[:10]]
        
        active_state.set_active(batch_idx, set(top_indices[:3]))

        activation_info = {
            "activated": top_indices,
            "previous": previous_active,
        }
        
        if not top_indices:
             return RetrievedContext([], [], torch.zeros((1, len(q_np)), device=query_vector.device), torch.ones((1,), device=query_vector.device), []), explanation_score, activation_info
             
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
        
        return context, explanation_score, activation_info
