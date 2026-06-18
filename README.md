# AESG — Adaptive External Semantic Graph

**Version 0.3.0** · [PyPI](https://pypi.org/project/aesg) · [GitHub](https://github.com/bueormnew/aesg) · MIT License

AESG is a persistent external memory module for PyTorch models. It stores knowledge as a graph of concept nodes backed by memory-mapped files (`mmap`), which means the graph lives on disk and is accessed without full serialization into RAM. Retrieval works through spreading activation rather than nearest-neighbor search, propagating activation energy through the graph's edges.

It is designed to be attached to an existing architecture (Transformer, LSTM, CNN) as an `nn.Module` without modifying the base model's weights.

---

## Installation

```bash
pip install aesg
```

**Requirements:** `torch >= 1.10`, `numpy >= 1.20`

---

## How it works

When a query vector arrives, AESG:

1. Finds the most relevant seed node in the graph
2. Propagates activation energy outward through edges (spreading activation)
3. Returns the activated subgraph as context
4. Optionally creates a new node if the query is not well-explained by the current graph (novelty detection)

Periodically, a consolidation step ages all nodes, decays their relevance score, and prunes those that have not been activated above a configurable threshold.

---

## Quick Start

```python
from aesg.config import AESGConfig
from aesg.memory.controller import AESGMemory
import torch

config = AESGConfig(
    vector_dim=128,
    max_concepts=500_000,
    spreading_activation_steps=3,
    novelty_birth_threshold=3,
)

memory = AESGMemory(directory="./my_memory", config=config)
```

---

## Usage Examples

### Retrieving context from a query

```python
import torch
from aesg.config import AESGConfig
from aesg.memory.controller import AESGMemory

config = AESGConfig(vector_dim=64, max_concepts=100_000)
memory = AESGMemory(directory="./memory_store", config=config)

query = torch.randn(64)

# Reset state between unrelated sequences
memory.reset_state()

context = memory.retrieve(query)

# context.concept_vectors: Tensor of activated concept embeddings
# context.activated_nodes: list of node indices
if context.concept_vectors is not None:
    print(f"Retrieved {context.concept_vectors.shape[0]} concepts")
```

---

### Attaching AESG to an LSTM

```python
import torch
import torch.nn as nn
from aesg.config import AESGConfig
from aesg.memory.controller import AESGMemory

class LSTMWithMemory(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=64, output_dim=10):
        super().__init__()
        config = AESGConfig(vector_dim=hidden_dim, max_concepts=50_000)
        self.memory = AESGMemory(directory="./lstm_memory", config=config)
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: [B, T, input_dim]
        self.memory.reset_state()
        lstm_out, _ = self.lstm(x)          # [B, T, H]
        final_hidden = lstm_out[:, -1, :]   # [B, H]
        ctx = self.memory(final_hidden)
        return self.head(final_hidden)

model = LSTMWithMemory()
x = torch.randn(4, 10, 64)  # batch=4, seq_len=10
out = model(x)
print(out.shape)  # [4, 10]
```

---

### Attaching AESG to a Transformer encoder

```python
import torch
import torch.nn as nn
from aesg.config import AESGConfig
from aesg.memory.controller import AESGMemory

class TransformerWithMemory(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        config = AESGConfig(vector_dim=d_model, max_concepts=100_000)
        self.memory = AESGMemory(directory="./transformer_memory", config=config)
        self.projector = nn.Linear(d_model * 2, d_model)

    def forward(self, src):
        # src: [B, T, d_model]
        encoded = self.encoder(src)
        cls_token = encoded[:, 0, :]  # Use first token as query

        self.memory.reset_state()
        ctx = self.memory(cls_token)

        if ctx.concept_vectors is not None:
            ctx_mean = ctx.concept_vectors.mean(dim=0, keepdim=True).expand(cls_token.size(0), -1)
            output = self.projector(torch.cat([cls_token, ctx_mean], dim=1))
        else:
            output = cls_token

        return output

model = TransformerWithMemory()
src = torch.randn(2, 8, 128)
out = model(src)
print(out.shape)  # [2, 128]
```

---

### Saving and loading the memory

```python
from aesg.config import AESGConfig
from aesg.memory.controller import AESGMemory

# --- Session 1: build and save ---
memory = AESGMemory(directory="./persistent_memory", config=AESGConfig(vector_dim=64))

import torch
for _ in range(100):
    memory.retrieve(torch.randn(64))

memory.save()
print(f"Saved {memory.storage.node_count} concepts")

# --- Session 2: reload and continue ---
memory2 = AESGMemory(directory="./persistent_memory")
print(f"Loaded {memory2.storage.node_count} concepts")

ctx = memory2.retrieve(torch.randn(64))
```

---

### Triggering consolidation (pruning)

```python
import torch
from aesg.config import AESGConfig
from aesg.memory.controller import AESGMemory

config = AESGConfig(
    vector_dim=32,
    max_concepts=1000,
    survival_threshold_relevance=0.1,   # Nodes below this relevance are pruned
    survival_threshold_frequency=2,     # Nodes accessed fewer than N times are pruned
)
memory = AESGMemory(directory="./mem_prune", config=config)

for i in range(3000):
    memory.cognitive_engine.create_sensory_concept(torch.randn(32))

print(f"Before consolidation: {memory.storage.node_count} nodes")
memory.update_topology()  # Runs consolidation + pruning
active = (memory.storage.nodes['is_active'][:memory.storage.node_count] == 1).sum()
print(f"Active after consolidation: {active}")
```

---

### Reading the evolution log

```python
from aesg.config import AESGConfig
from aesg.memory.controller import AESGMemory
import torch

memory = AESGMemory(directory="./mem_log", config=AESGConfig(vector_dim=16))

for _ in range(20):
    memory.retrieve(torch.randn(16))

# Each event is a dict: timestamp, type, id1, id2, val
history = memory.logger.get_history()
for event in history[:5]:
    print(event)
# {'timestamp': 1718728800, 'type': 'CREATE', 'id1': 4293871, 'id2': 0, 'val': 1.0}
```

---

### Handling common errors

```python
import torch
from aesg.config import AESGConfig
from aesg.memory.controller import AESGMemory

# Wrong vector dimension
config = AESGConfig(vector_dim=64)
memory = AESGMemory(directory="./mem_err", config=config)

try:
    wrong_query = torch.randn(128)   # dim mismatch
    ctx = memory.retrieve(wrong_query)
except Exception as e:
    print(f"Dimension mismatch: {e}")

# Safe pattern: always match query dim to config.vector_dim
query = torch.randn(config.vector_dim)
ctx = memory.retrieve(query)


# Loading from a non-existent directory (will initialize fresh)
memory_new = AESGMemory(directory="./brand_new_memory", config=config)
print(f"Fresh memory: {memory_new.storage.node_count} nodes")  # 0


# Windows: always call memory.save() before deleting the directory,
# otherwise the mmap files remain locked by the process.
memory.save()
```

---

## Configuration Reference

All parameters are set via `AESGConfig`. No values are hardcoded in the core.

```python
from aesg.config import AESGConfig

config = AESGConfig(
    # Core
    vector_dim=256,                          # Embedding dimensionality
    max_concepts=1_000_000,                  # Maximum active nodes

    # Novelty detection
    novelty_explanation_threshold=0.6,       # Graph score below this triggers novelty check
    novelty_birth_threshold=3,               # How many times a novel input must persist to create a node

    # Evolutionary pressure
    survival_threshold_relevance=0.05,       # Minimum relevance score to survive pruning
    survival_threshold_frequency=5,          # Minimum activation count to survive pruning

    # Navigation
    spreading_activation_steps=3,           # Depth of activation propagation
    spreading_activation_decay=0.8,         # Energy decay per hop
    region_facilitation_multiplier=1.5,     # Bonus for intra-region traversal
)
```

---

## Storage Format

The graph is stored as binary memory-mapped files in the specified directory:

```
my_memory/
├── nodes.aesg          # Node records (mmap, struct array)
├── edges.aesg          # Edge records (mmap, linked list)
├── meta.npy            # Capacity and count metadata
└── evolution.aesglog   # Binary event log (32 bytes/event)
```

Node struct fields: `id`, `vector[D]`, `created_at`, `modified_at`, `use_frequency`, `relevance`, `age`, `stability`, `region_id`, `is_active`, `head_edge_idx`.

The graph does not need to be fully loaded into RAM — it is accessed via `numpy.memmap` with zero-copy reads.

---

## Performance Notes

Benchmarks run on a single CPU core, 24 GB RAM, standard SSD, Windows 11. `vector_dim=16` for insertion tests.

| Measurement | Result |
|:---|---:|
| Insertion throughput | ~1,600–1,940 nodes/s (pure Python loop) |
| Retrieve latency (1–8 hops) | 1.04–1.17 ms average |
| 1,000,000 nodes — RAM overhead | 81 MB |
| 1,000,000 nodes — disk size | 283 MB |
| Evolution log read (100k events) | 0.13 s |

Insertion throughput is bounded by the Python interpreter loop. The storage and retrieval layer itself has no algorithmic bottleneck for graph sizes in the millions.

---

## License

MIT License — Copyright (c) 2026 bueormnew

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

*Author: bueormnew · dalusx64@gmail.com*
