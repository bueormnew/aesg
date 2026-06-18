from typing import List, Optional
import torch

class RetrievedContext:
    """
    AESG no pretende ser una caché de embeddings; pretende ser una memoria navegable.
    
    Por tanto, la recuperación devuelve una estructura completa:
    Preserva la estructura conceptual de los elementos recuperados.
    """
    def __init__(
        self, 
        primary_concepts: List[int], 
        secondary_concepts: List[int],
        concept_vectors: torch.Tensor,
        relation_weights: torch.Tensor,
        paths: List[List[int]]
    ):
        self.primary_concepts = primary_concepts
        self.secondary_concepts = secondary_concepts
        self.vectors = concept_vectors # (N, vector_dim)
        self.weights = relation_weights
        self.paths = paths

    def aggregate(self) -> torch.Tensor:
        """
        Genera un único vector contextual (Tensor colapsado).
        Este método es un adaptador opcional para arquitecturas como GRU o LSTM 
        que necesiten consumir la memoria como un único vector.
        """
        # Un enfoque simple de colapso: suma ponderada o media
        # Aquí tomamos la media, pero podría ponderarse por relevance o weight
        return torch.mean(self.vectors, dim=0)
