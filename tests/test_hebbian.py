"""Tests for Hebbian edge learning and usage crediting.

These cover the core promise of AESG: that co-activated concepts become
connected, that those connections carry activation, and that the resulting
structure survives persistence.
"""

import numpy as np
import pytest
import torch

from aesg.config import AESGConfig
from aesg.memory.controller import AESGMemory


def _memory(tmp_path, **overrides):
    params = {
        "vector_dim": 16,
        "novelty_birth_threshold": 1,
        "novelty_explanation_threshold": 0.9,
        # Keep the preallocated memmap small; the default max_concepts of 1M
        # would reserve ~117 MB of sparse file per test instance.
        "max_concepts": 2_000,
    }
    params.update(overrides)
    return AESGMemory(str(tmp_path / "mem"), AESGConfig(**params))


def test_retrieval_creates_edges(tmp_path):
    """Repeated retrieval of alternating patterns must wire the graph."""
    torch.manual_seed(0)
    memory = _memory(tmp_path)
    a, b = torch.randn(16), torch.randn(16)

    for _ in range(20):
        memory.retrieve(a + torch.randn(16) * 0.01)
        memory.retrieve(b + torch.randn(16) * 0.01)

    assert memory.storage.edge_count > 0


def test_retrieval_credits_usage(tmp_path):
    """Retrieved concepts must accumulate use_frequency and relevance."""
    torch.manual_seed(0)
    memory = _memory(tmp_path)
    pattern = torch.randn(16)

    for _ in range(20):
        memory.retrieve(pattern + torch.randn(16) * 0.01)

    storage = memory.storage
    usage = storage.nodes["use_frequency"][: storage.node_count].sum()
    assert int(usage) > 0


def test_edges_conduct_activation(tmp_path):
    """Learned edges must actually propagate spreading activation."""
    torch.manual_seed(1)
    memory = _memory(tmp_path)
    a, b = torch.randn(16), torch.randn(16)

    for _ in range(30):
        memory.retrieve(a + torch.randn(16) * 0.01)
        memory.retrieve(b + torch.randn(16) * 0.01)

    memory.reset_state()
    context = memory.retrieve(a)
    assert len(context.paths) > 0


def test_activation_energy_never_diverges(tmp_path):
    """Region facilitation must not let activation gain energy per hop."""
    torch.manual_seed(1)
    memory = _memory(tmp_path, spreading_activation_steps=5)
    pattern = torch.randn(16)

    for _ in range(40):
        memory.retrieve(pattern + torch.randn(16) * 0.05)

    memory.reset_state()
    context = memory.retrieve(pattern)
    assert bool((context.weights <= 1.0).all())


def test_edge_weights_saturate_at_one(tmp_path):
    """Hebbian reinforcement must saturate rather than grow unbounded."""
    torch.manual_seed(2)
    memory = _memory(tmp_path)
    pattern = torch.randn(16)

    for _ in range(60):
        memory.retrieve(pattern + torch.randn(16) * 0.01)

    storage = memory.storage
    if storage.edge_count:
        weights = storage.edges["weight"][: storage.edge_count]
        assert float(weights.max()) <= 1.0


def test_degree_cap_is_respected(tmp_path):
    """No node may exceed max_edges_per_node outgoing edges."""
    torch.manual_seed(3)
    memory = _memory(tmp_path, max_edges_per_node=3)

    for _ in range(60):
        memory.retrieve(torch.randn(16))

    storage = memory.storage
    degrees = [storage.get_degree(i) for i in range(storage.node_count)]
    assert max(degrees) <= 3


def test_inference_mode_is_read_only(tmp_path):
    """INFERENCE must not create nodes, edges, or usage counters."""
    torch.manual_seed(4)
    memory = _memory(tmp_path)
    for _ in range(20):
        memory.retrieve(torch.randn(16))

    storage = memory.storage
    memory.set_mode("INFERENCE")
    before = (
        storage.node_count,
        storage.edge_count,
        int(storage.nodes["use_frequency"][: storage.node_count].sum()),
    )

    for _ in range(30):
        memory.retrieve(torch.randn(16))

    after = (
        storage.node_count,
        storage.edge_count,
        int(storage.nodes["use_frequency"][: storage.node_count].sum()),
    )
    assert before == after


def test_used_concepts_survive_consolidation(tmp_path):
    """A recurring pattern must not be pruned away by relevance decay."""
    torch.manual_seed(5)
    memory = _memory(tmp_path)
    pattern = torch.randn(16)

    for _ in range(80):
        memory.retrieve(pattern + torch.randn(16) * 0.01)
        memory.update_topology()

    storage = memory.storage
    alive = int(np.sum(storage.nodes["is_active"][: storage.node_count]))
    assert alive > 0


def test_graph_stays_sparse(tmp_path):
    """Edge decay must keep the graph from collapsing into a complete one."""
    torch.manual_seed(7)
    memory = _memory(tmp_path, novelty_birth_threshold=2,
                     novelty_explanation_threshold=0.7)
    patterns = [torch.randn(16) for _ in range(5)]

    for step in range(300):
        memory.retrieve(patterns[step % 5] + torch.randn(16) * 0.05)
        if step % 10 == 0:
            memory.update_topology()

    storage = memory.storage
    n = storage.node_count
    if n > 2:
        density = storage.edge_count / (n * (n - 1))
        assert density < 0.5


def test_inactive_nodes_are_never_seeded(tmp_path):
    """Pruned concepts must not be returned as navigation seeds."""
    torch.manual_seed(8)
    memory = _memory(tmp_path)
    for _ in range(10):
        memory.retrieve(torch.randn(16))

    storage = memory.storage
    storage.nodes["is_active"][: storage.node_count] = 0
    storage.nodes["is_active"][0] = 1

    seeds = memory.graph.find_nearest_neighbors(np.random.randn(16).astype(np.float32), top_k=5)
    assert all(storage.nodes["is_active"][idx] == 1 for idx, _ in seeds)


def test_learned_graph_persists(tmp_path):
    """Edges must survive a save/reopen cycle and still conduct."""
    torch.manual_seed(9)
    directory = str(tmp_path / "mem")
    config = AESGConfig(vector_dim=16, novelty_birth_threshold=1,
                        novelty_explanation_threshold=0.9,
                        max_concepts=2_000)
    memory = AESGMemory(directory, config)
    a, b = torch.randn(16), torch.randn(16)
    for _ in range(30):
        memory.retrieve(a + torch.randn(16) * 0.01)
        memory.retrieve(b + torch.randn(16) * 0.01)
    memory.save()
    expected_edges = memory.storage.edge_count
    del memory

    reopened = AESGMemory(directory, config)
    assert reopened.storage.edge_count == expected_edges
    reopened.reset_state()
    assert len(reopened.retrieve(a).paths) > 0
