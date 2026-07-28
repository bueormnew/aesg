"""Tests for the AESGStorage memmap engine.

Covers capacity growth for both nodes and edges, data integrity across
reallocations, and vector_dim compatibility checks on reload.
"""

import numpy as np
import pytest

from aesg.exceptions import AESGStorageError
from aesg.storage.engine import AESGStorage


def test_edge_capacity_grows_beyond_initial(tmp_path):
    """Adding more edges than initial_capacity_edges must not raise."""
    storage = AESGStorage(
        str(tmp_path / "g"), 4,
        initial_capacity_nodes=10, initial_capacity_edges=2,
    )
    for i in range(3):
        storage.add_node(i + 1, np.zeros(4, dtype=np.float32))

    for _ in range(50):
        storage.add_edge(0, 2, 0.5, 0.5)

    assert storage.edge_count == 50
    assert storage.cap_edges >= 50


def test_node_capacity_grows_and_preserves_data(tmp_path):
    """Node reallocation must keep every previously written record intact."""
    storage = AESGStorage(
        str(tmp_path / "g"), 4,
        initial_capacity_nodes=2, initial_capacity_edges=2,
    )
    for i in range(9):
        storage.add_node(i + 100, np.full(4, i, dtype=np.float32))

    assert storage.node_count == 9
    for i in range(9):
        assert int(storage.nodes[i]["id"]) == i + 100
        assert float(storage.nodes[i]["vector"][0]) == i


def test_adjacency_list_survives_edge_growth(tmp_path):
    """The linked adjacency list must stay traversable after reallocation."""
    storage = AESGStorage(
        str(tmp_path / "g"), 4,
        initial_capacity_nodes=4, initial_capacity_edges=1,
    )
    for i in range(4):
        storage.add_node(i + 1, np.zeros(4, dtype=np.float32))

    for target in (2, 3, 4):
        storage.add_edge(0, target, 1.0, 1.0)

    targets = sorted(int(e["target_id"]) for e in storage.get_edges(0))
    assert targets == [2, 3, 4]
    assert storage.get_degree(0) == 3


def test_reload_with_mismatched_vector_dim_raises(tmp_path):
    """Reopening a graph with a different vector_dim must fail loudly."""
    directory = str(tmp_path / "g")
    storage = AESGStorage(directory, 8, initial_capacity_nodes=5, initial_capacity_edges=5)
    storage.add_node(1, np.ones(8, dtype=np.float32))
    storage.flush()
    del storage

    with pytest.raises(AESGStorageError, match="vector_dim mismatch"):
        AESGStorage(directory, 16, initial_capacity_nodes=5, initial_capacity_edges=5)


def test_reload_with_matching_vector_dim_succeeds(tmp_path):
    """The happy path must keep working after the compatibility check."""
    directory = str(tmp_path / "g")
    storage = AESGStorage(directory, 8, initial_capacity_nodes=5, initial_capacity_edges=5)
    storage.add_node(1, np.ones(8, dtype=np.float32))
    storage.flush()
    del storage

    reopened = AESGStorage(directory, 8)
    assert reopened.node_count == 1
    assert reopened.vector_dim == 8


def test_nodes_and_edges_persist_across_sessions(tmp_path):
    """Both node and edge data must survive a flush/reopen cycle."""
    directory = str(tmp_path / "g")
    storage = AESGStorage(directory, 4, initial_capacity_nodes=2, initial_capacity_edges=1)
    for i in range(5):
        storage.add_node(i + 1, np.zeros(4, dtype=np.float32))
    for _ in range(10):
        storage.add_edge(0, 3, 0.7, 0.7)
    storage.flush()
    expected = (storage.node_count, storage.edge_count)
    del storage

    reopened = AESGStorage(directory, 4)
    assert (reopened.node_count, reopened.edge_count) == expected


def test_get_node_index_resolves_ids(tmp_path):
    """The id -> index map must resolve known ids and reject unknown ones."""
    storage = AESGStorage(str(tmp_path / "g"), 4)
    idx = storage.add_node(4242, np.zeros(4, dtype=np.float32))

    assert storage.get_node_index(4242) == idx
    assert storage.get_node_index(999999) == -1


def test_find_edge(tmp_path):
    """find_edge must locate existing edges and report -1 otherwise."""
    storage = AESGStorage(str(tmp_path / "g"), 4)
    storage.add_node(1, np.zeros(4, dtype=np.float32))
    storage.add_node(2, np.zeros(4, dtype=np.float32))
    storage.add_edge(0, 2, 0.5, 0.5)

    assert storage.find_edge(0, 2) >= 0
    assert storage.find_edge(0, 99) == -1
