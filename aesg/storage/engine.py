import os
import numpy as np
import time

class AESGStorage:
    """
    Storage engine for AESG using memory-mapped files (mmap) via NumPy.
    Ensures that the entire graph doesn't need to be loaded into RAM,
    enabling massive scalability.
    """
    def __init__(self, directory: str, vector_dim: int, initial_capacity_nodes: int = 100000, initial_capacity_edges: int = 1000000):
        self.directory = os.path.abspath(directory)
        self.vector_dim = vector_dim
        self.nodes_path = os.path.join(self.directory, "nodes.aesg")
        self.edges_path = os.path.join(self.directory, "edges.aesg")
        self.meta_path = os.path.join(self.directory, "meta.npy")
        
        os.makedirs(self.directory, exist_ok=True)
        
        # Define struct formats
        self.node_dtype = np.dtype([
            ('id', np.uint64),
            ('vector', np.float32, (vector_dim,)),
            ('created_at', np.uint64),
            ('modified_at', np.uint64),
            ('use_frequency', np.uint32),
            ('relevance', np.float32),
            ('age', np.uint32),         # V2: Edad del concepto en épocas
            ('stability', np.float32),  # V2: Estabilidad conceptual
            ('region_id', np.uint32),   # V2: Región semántica (comunidad)
            ('is_active', np.uint8),    # V2: 1 activo, 0 borrado lógicamente
            ('head_edge_idx', np.int64), # -1 for null
        ])
        
        self.edge_dtype = np.dtype([
            ('source_id', np.uint64),
            ('target_id', np.uint64),
            ('weight', np.float32),
            ('confidence', np.float32),
            ('use_count', np.uint32),
            ('next_edge_idx', np.int64), # -1 for null
        ])
        self._load_or_initialize(initial_capacity_nodes, initial_capacity_edges)

    def _load_or_initialize(self, cap_nodes, cap_edges):
        if os.path.exists(self.meta_path):
            meta = np.load(self.meta_path, allow_pickle=True).item()
            self.node_count = meta['node_count']
            self.edge_count = meta['edge_count']
            self.vector_dim = meta['vector_dim']
            self.cap_nodes = meta['cap_nodes']
            self.cap_edges = meta['cap_edges']
            
            self.nodes = np.memmap(self.nodes_path, dtype=self.node_dtype, mode='r+', shape=(self.cap_nodes,))
            self.edges = np.memmap(self.edges_path, dtype=self.edge_dtype, mode='r+', shape=(self.cap_edges,))
        else:
            self.node_count = 0
            self.edge_count = 0
            self.cap_nodes = cap_nodes
            self.cap_edges = cap_edges
            self.nodes = np.memmap(self.nodes_path, dtype=self.node_dtype, mode='w+', shape=(self.cap_nodes,))
            self.edges = np.memmap(self.edges_path, dtype=self.edge_dtype, mode='w+', shape=(self.cap_edges,))
            self._save_meta(self.cap_nodes, self.cap_edges)

    def _save_meta(self, cap_nodes, cap_edges):
        self.cap_nodes = cap_nodes
        self.cap_edges = cap_edges
        meta = {
            'node_count': self.node_count,
            'edge_count': self.edge_count,
            'vector_dim': self.vector_dim,
            'cap_nodes': self.cap_nodes,
            'cap_edges': self.cap_edges
        }
        np.save(self.meta_path, meta)

    def add_node(self, node_id: int, vector: np.ndarray) -> int:
        if self.node_count >= self.cap_nodes:
            new_cap = self.cap_nodes * 2
            new_nodes = np.memmap(self.nodes_path, dtype=self.node_dtype, mode='r+', shape=(new_cap,))
            new_nodes[:self.cap_nodes] = self.nodes
            self.nodes = new_nodes
            self.cap_nodes = new_cap
            self._save_meta(self.cap_nodes, self.cap_edges)
        
        idx = self.node_count
        # For this version we assume sufficient initial capacity or implement resize later
        
        self.nodes[idx]['id'] = node_id
        self.nodes[idx]['vector'] = vector
        self.nodes[idx]['created_at'] = int(time.time())
        self.nodes[idx]['modified_at'] = int(time.time())
        self.nodes[idx]['use_frequency'] = 0
        self.nodes[idx]['relevance'] = 1.0
        self.nodes[idx]['age'] = 0
        self.nodes[idx]['stability'] = 0.5
        self.nodes[idx]['region_id'] = 0
        self.nodes[idx]['is_active'] = 1
        self.nodes[idx]['head_edge_idx'] = -1
        
        self.node_count += 1
        return idx

    def add_edge(self, source_idx: int, target_id: int, weight: float, confidence: float = 1.0):
        edge_idx = self.edge_count
        source_id = self.nodes[source_idx]['id']
        
        self.edges[edge_idx]['source_id'] = source_id
        self.edges[edge_idx]['target_id'] = target_id
        self.edges[edge_idx]['weight'] = weight
        self.edges[edge_idx]['confidence'] = confidence
        self.edges[edge_idx]['use_count'] = 0
        
        # Link to source's edge list
        prev_head = self.nodes[source_idx]['head_edge_idx']
        self.edges[edge_idx]['next_edge_idx'] = prev_head
        self.nodes[source_idx]['head_edge_idx'] = edge_idx
        
        self.edge_count += 1

    def flush(self):
        self.nodes.flush()
        self.edges.flush()
        self._save_meta(len(self.nodes), len(self.edges))

    def get_node(self, idx: int):
        return self.nodes[idx]

    def get_edges(self, source_idx: int):
        edges = []
        curr_idx = self.nodes[source_idx]['head_edge_idx']
        while curr_idx != -1:
            edges.append(self.edges[curr_idx])
            curr_idx = self.edges[curr_idx]['next_edge_idx']
        return edges
