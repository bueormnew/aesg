import os
import numpy as np
import time

from aesg.exceptions import AESGStorageError

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

            # The on-disk vector_dim defines the byte layout of nodes.aesg.
            # Reopening with a different vector_dim would reinterpret the
            # file with a mismatched dtype and silently corrupt every vector,
            # so refuse instead of guessing.
            stored_dim = int(meta['vector_dim'])
            if stored_dim != self.vector_dim:
                raise AESGStorageError(
                    f"vector_dim mismatch for memory directory '{self.directory}': "
                    f"the stored graph was created with vector_dim={stored_dim}, "
                    f"but vector_dim={self.vector_dim} was requested. "
                    f"Use vector_dim={stored_dim}, or point to an empty directory "
                    f"to start a new graph."
                )

            self.node_count = meta['node_count']
            self.edge_count = meta['edge_count']
            self.vector_dim = stored_dim
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
            self.nodes['head_edge_idx'][:] = -1
            self.edges['next_edge_idx'][:] = -1
            self._save_meta(self.cap_nodes, self.cap_edges)

        # id -> index map for O(1) lookups during graph traversal.
        self._rebuild_index()

    def _rebuild_index(self):
        """(Re)build the in-memory id -> storage index mapping."""
        ids = self.nodes['id'][:self.node_count]
        self._id_to_idx = {int(node_id): idx for idx, node_id in enumerate(ids)}

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

    def _grow_nodes(self):
        """Double the node memmap capacity in place, preserving existing data."""
        new_cap = max(self.cap_nodes * 2, self.cap_nodes + 1)

        # Flush and drop the old mapping before remapping the same file with a
        # larger shape; numpy extends the file on disk for us in 'r+' mode.
        self.nodes.flush()
        del self.nodes

        self.nodes = np.memmap(
            self.nodes_path, dtype=self.node_dtype, mode='r+', shape=(new_cap,)
        )
        # Freshly appended region is zero-filled by the OS; mark it inactive
        # explicitly so it is never mistaken for real data.
        self.nodes['is_active'][self.cap_nodes:] = 0
        self.nodes['head_edge_idx'][self.cap_nodes:] = -1

        self.cap_nodes = new_cap
        self._save_meta(self.cap_nodes, self.cap_edges)

    def _grow_edges(self):
        """Double the edge memmap capacity in place, preserving existing data."""
        new_cap = max(self.cap_edges * 2, self.cap_edges + 1)

        self.edges.flush()
        del self.edges

        self.edges = np.memmap(
            self.edges_path, dtype=self.edge_dtype, mode='r+', shape=(new_cap,)
        )
        # Null out the link pointers in the newly allocated region.
        self.edges['next_edge_idx'][self.cap_edges:] = -1

        self.cap_edges = new_cap
        self._save_meta(self.cap_nodes, self.cap_edges)

    def add_node(self, node_id: int, vector: np.ndarray) -> int:
        if self.node_count >= self.cap_nodes:
            self._grow_nodes()
        
        idx = self.node_count
        
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
        self._id_to_idx[int(node_id)] = idx
        return idx

    def add_edge(self, source_idx: int, target_id: int, weight: float, confidence: float = 1.0):
        if self.edge_count >= self.cap_edges:
            self._grow_edges()

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

    def get_node_index(self, node_id: int) -> int:
        """Resolve a node id to its storage index in O(1).

        Returns -1 if the id is unknown. Backed by an in-memory index that is
        built once on load and kept in sync by ``add_node``; without it every
        edge traversal would need an O(N) scan over the id column.
        """
        return self._id_to_idx.get(int(node_id), -1)

    def find_edge(self, source_idx: int, target_id: int) -> int:
        """Return the index of the edge source_idx -> target_id, or -1.

        Walks the source node's singly-linked adjacency list.
        """
        target_id = int(target_id)
        curr_idx = self.nodes[source_idx]['head_edge_idx']
        while curr_idx != -1:
            if int(self.edges[curr_idx]['target_id']) == target_id:
                return int(curr_idx)
            curr_idx = self.edges[curr_idx]['next_edge_idx']
        return -1

    def get_degree(self, source_idx: int) -> int:
        """Return the out-degree of a node by walking its adjacency list."""
        degree = 0
        curr_idx = self.nodes[source_idx]['head_edge_idx']
        while curr_idx != -1:
            degree += 1
            curr_idx = self.edges[curr_idx]['next_edge_idx']
        return degree
