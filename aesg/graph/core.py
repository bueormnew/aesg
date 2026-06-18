from typing import List, Tuple
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
        self.storage.nodes[node_idx]['relevance'] += delta
        self.storage.nodes[node_idx]['use_frequency'] += 1
        
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
        norms_v[norms_v == 0] = 1e-10 
        
        sims = np.dot(vectors, query_vector) / (norms_v * norm_q)
        
        top_indices = np.argsort(sims)[-top_k:][::-1]
        
        return [(int(idx), float(sims[idx])) for idx in top_indices if sims[idx] > 0.1] # threshold
