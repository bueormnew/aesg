"""
AESG Pack Export.

Provides functionality to export filtered subgraphs from an AESGMemory
instance into portable .aesgpack binary files.
"""

from typing import Optional, TYPE_CHECKING

import numpy as np

from aesg.packs.format import serialize_pack

if TYPE_CHECKING:
    from aesg.memory.controller import AESGMemory


def export_pack(
    memory: "AESGMemory",
    path: str,
    region_id: Optional[int] = None,
    min_relevance: Optional[float] = None,
) -> None:
    """Export a filtered subgraph from memory as a .aesgpack file.

    Filters active nodes from memory storage and collects all edges
    whose source is among the filtered node IDs. The result is serialized
    to the specified path using the .aesgpack binary format.

    Parameters
    ----------
    memory : AESGMemory
        The memory controller whose storage will be exported.
    path : str
        Output file path for the .aesgpack file.
    region_id : int, optional
        If provided, only export nodes belonging to this region.
    min_relevance : float, optional
        If provided, only export nodes with relevance >= this threshold.
    """
    storage = memory.storage
    nodes = storage.nodes[:storage.node_count]

    # Always filter to active nodes only
    mask = nodes['is_active'] == 1

    # Apply optional region filter
    if region_id is not None:
        mask = mask & (nodes['region_id'] == region_id)

    # Apply optional relevance filter
    if min_relevance is not None:
        mask = mask & (nodes['relevance'] >= min_relevance)

    filtered_nodes = np.array(nodes[mask])

    # Collect node IDs from filtered set
    filtered_node_ids = set(filtered_nodes['id'].tolist())

    # Collect edges whose source_id is among filtered node IDs
    edges = storage.edges[:storage.edge_count]
    if len(edges) > 0:
        edge_mask = np.isin(edges['source_id'], list(filtered_node_ids))
        filtered_edges = np.array(edges[edge_mask])
    else:
        edge_dtype = storage.edge_dtype
        filtered_edges = np.array([], dtype=edge_dtype)

    # Serialize to .aesgpack format
    serialize_pack(
        name="export",
        version="1.0",
        vector_dim=memory.vector_dim,
        nodes_array=filtered_nodes,
        edges_array=filtered_edges,
        path=path,
    )
