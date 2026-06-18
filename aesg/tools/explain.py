import torch
import numpy as np
from typing import List, Dict, Any
from aesg.memory.controller import AESGMemory

class Explainer:
    def __init__(self, memory: AESGMemory):
        self.memory = memory
        self.storage = memory.storage
        
    def explain(self, concept_id: int) -> Dict[str, Any]:
        """
        Devuelve información detallada de un nodo conceptual específico.
        """
        idx_array = np.where(self.storage.nodes['id'][:self.storage.node_count] == concept_id)[0]
        if len(idx_array) == 0:
            return {"error": "Concepto no encontrado"}
            
        idx = idx_array[0]
        node = self.storage.nodes[idx]
        
        edges = self.storage.get_edges(idx)
        # Sort edges by weight
        edges = sorted(edges, key=lambda x: x['weight'], reverse=True)
        
        main_connections = [
            {"target_id": int(e['target_id']), "weight": float(e['weight'])} 
            for e in edges[:10]
        ]
        
        return {
            "Concept_ID": int(node['id']),
            "Main_Connections": main_connections,
            "Use_Frequency": int(node['use_frequency']),
            "Relevance": float(node['relevance']),
            "Created_At": int(node['created_at']),
            "Modified_At": int(node['modified_at']),
            "Vector_Preview": node['vector'][:5].tolist() # First 5 dims
        }
        
    def trace(self, input_vector: torch.Tensor) -> List[int]:
        """
        Rastrea la ruta de navegación desde un input a través de la memoria.
        """
        context = self.memory.retrieve(input_vector, top_k=1)
        if not context.primary_concepts:
            return []
            
        path_ids = []
        # Translate indices to IDs
        for idx in context.primary_concepts:
            path_ids.append(int(self.storage.nodes[idx]['id']))
            
        for path in context.paths:
            for idx in path:
                path_ids.append(int(self.storage.nodes[idx]['id']))
                
        # Remove duplicates preserving order
        seen = set()
        trace_ids = [x for x in path_ids if not (x in seen or seen.add(x))]
        
        return trace_ids

    def find_hubs(self, top_n: int = 10) -> List[int]:
        """
        Encuentra los nodos más conectados y utilizados.
        """
        frequencies = self.storage.nodes['use_frequency'][:self.storage.node_count]
        top_indices = np.argsort(frequencies)[-top_n:][::-1]
        
        hubs = []
        for idx in top_indices:
            hubs.append(int(self.storage.nodes[idx]['id']))
        return hubs

    def find_dead_nodes(self, use_threshold: int = 0) -> List[int]:
        """
        Encuentra nodos que no han sido utilizados.
        """
        frequencies = self.storage.nodes['use_frequency'][:self.storage.node_count]
        dead_indices = np.where(frequencies <= use_threshold)[0]
        
        dead_ids = []
        for idx in dead_indices:
            dead_ids.append(int(self.storage.nodes[idx]['id']))
        return dead_ids
