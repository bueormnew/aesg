from typing import List, Optional, Set, Tuple
import numpy as np
from aesg.storage.engine import AESGStorage

class SemanticGraph:
    """
    Wrapper around AESGStorage to provide logical graph operations.
    Handles semantic queries and topology modifications.
    """
    def __init__(self, storage: AESGStorage):
        self.storage = storage

    def get_node_vector(self, node_idx: int) -> np.ndarray:
        return self.storage.nodes[node_idx]['vector']

    def update_relevance(self, node_idx: int, delta: float):
        """Credit a node for being retrieved: bump relevance and usage.

        Relevance saturates at 1.0 so that repeated retrieval cannot inflate a
        node without bound; it simply keeps the node pinned at full relevance
        while the per-consolidation decay pulls unused nodes down towards the
        survival threshold.
        """
        current = float(self.storage.nodes[node_idx]['relevance'])
        self.storage.nodes[node_idx]['relevance'] = min(1.0, current + delta)
        self.storage.nodes[node_idx]['use_frequency'] += 1

    def reinforce_edge(
        self,
        source_idx: int,
        target_idx: int,
        learning_rate: float,
        max_edges_per_node: int,
        initial_weight: float = 0.5,
    ) -> bool:
        """Create or strengthen a directed edge between two co-activated nodes.

        Implements the Hebbian rule "cells that fire together, wire together":
        if the edge already exists its weight/confidence are reinforced and its
        use_count incremented; otherwise a new edge is created, provided the
        source node has not exceeded ``max_edges_per_node``.

        Parameters
        ----------
        source_idx : int
            Storage index of the source node.
        target_idx : int
            Storage index of the target node.
        learning_rate : float
            Step size for the reinforcement update, in [0.0, 1.0].
        max_edges_per_node : int
            Hard cap on out-degree; prevents unbounded edge growth on hubs.
        initial_weight : float
            Weight/confidence given to a brand-new edge.

        Returns
        -------
        bool
            True if an edge was created or reinforced, False if the update was
            skipped (self-loop, inactive node, or degree cap reached).
        """
        if source_idx == target_idx:
            return False

        storage = self.storage
        if storage.nodes['is_active'][source_idx] == 0:
            return False
        if storage.nodes['is_active'][target_idx] == 0:
            return False

        target_id = int(storage.nodes[target_idx]['id'])
        edge_idx = storage.find_edge(source_idx, target_id)

        if edge_idx >= 0:
            # Existing association: reinforce it, saturating towards 1.0.
            edge = storage.edges[edge_idx]
            edge['weight'] = min(
                1.0, float(edge['weight']) + learning_rate * (1.0 - float(edge['weight']))
            )
            edge['confidence'] = min(
                1.0,
                float(edge['confidence'])
                + learning_rate * (1.0 - float(edge['confidence'])),
            )
            edge['use_count'] += 1
            return True

        # New association: respect the per-node degree cap.
        if storage.get_degree(source_idx) >= max_edges_per_node:
            return False

        storage.add_edge(
            source_idx, target_id, weight=initial_weight, confidence=initial_weight
        )
        return True

    def hebbian_update(
        self,
        co_activated: List[int],
        previous: Optional[Set[int]] = None,
        learning_rate: float = 0.1,
        max_edges_per_node: int = 1000,
        initial_weight: float = 0.5,
    ) -> int:
        """Wire together the concepts that were activated by the same query.

        Two kinds of association are learned:

        1. **Co-activation** — every pair of concepts retrieved in the same
           step is linked bidirectionally.
        2. **Temporal succession** — concepts active in the *previous* step are
           linked to the concepts active now, capturing sequence structure.

        Parameters
        ----------
        co_activated : List[int]
            Storage indices activated by the current query.
        previous : Set[int], optional
            Storage indices that were active in the previous step.
        learning_rate : float
            Hebbian step size, in [0.0, 1.0].
        max_edges_per_node : int
            Hard cap on out-degree per node.
        initial_weight : float
            Weight/confidence given to brand-new edges.

        Returns
        -------
        int
            Number of edges created or reinforced.
        """
        updates = 0

        # 1. Symmetric co-activation within the current step.
        for i, src in enumerate(co_activated):
            for tgt in co_activated[i + 1:]:
                if self.reinforce_edge(
                    src, tgt, learning_rate, max_edges_per_node, initial_weight
                ):
                    updates += 1
                if self.reinforce_edge(
                    tgt, src, learning_rate, max_edges_per_node, initial_weight
                ):
                    updates += 1

        # 2. Directed temporal succession from the previous step to this one.
        if previous:
            current = set(co_activated)
            for src in previous:
                if src in current:
                    continue
                for tgt in co_activated:
                    if self.reinforce_edge(
                        src, tgt, learning_rate, max_edges_per_node, initial_weight
                    ):
                        updates += 1

        return updates
        
    def find_nearest_neighbors(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        O(N) search for initial MVP. 
        In production, this would use a spatial index (e.g. HNSW) integrated with the storage.
        """
        if self.storage.node_count == 0:
            return []
            
        vectors = self.storage.nodes['vector'][:self.storage.node_count]
        
        # Cosine similarity
        norms_v = np.linalg.norm(vectors, axis=1)
        norm_q = np.linalg.norm(query_vector)
        
        if norm_q == 0:
            return []
            
        # Avoid division by zero
        norms_v = np.where(norms_v == 0, 1e-10, norms_v)
        
        sims = np.dot(vectors, query_vector) / (norms_v * norm_q)

        # Logically deleted nodes must never be returned as navigation seeds,
        # otherwise pruned concepts get re-wired by the Hebbian update.
        inactive = self.storage.nodes['is_active'][:self.storage.node_count] == 0
        sims = np.where(inactive, -np.inf, sims)
        
        top_indices = np.argsort(sims)[-top_k:][::-1]
        
        return [(int(idx), float(sims[idx])) for idx in top_indices if sims[idx] > 0.1] # threshold
