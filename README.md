# AESG — Adaptive External Semantic Graph

**Version 3.0.0** · [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) [![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org) [![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D1.10-red.svg)](https://pytorch.org)

A persistent, self-organizing external memory layer for neural networks. AESG is a semantic graph that grows, prunes, and reorganizes itself alongside your model's training — acting as long-term memory that your model can query at every forward pass.

---

## Table of Contents

1. [What is AESG?](#what-is-aesg)
2. [Quick Start](#quick-start-5-minutes)
3. [Installation](#installation)
4. [Core Concepts](#core-concepts)
5. [Architecture Deep Dive](#architecture-deep-dive)
6. [Text Models — Complete Guide](#text-models--complete-guide)
7. [Image Models — Complete Guide](#image-models--complete-guide)
8. [AESGTrainer — Complete Guide](#aesgtrainer--complete-guide)
9. [Memory Modes — Complete Guide](#memory-modes--complete-guide)
10. [Pack System — Complete Guide](#pack-system--complete-guide)
11. [Evaluation & Metrics — Complete Guide](#evaluation--metrics--complete-guide)
12. [AESGConfig — Complete Reference](#aesgconfig--complete-reference)
13. [Exception Handling](#exception-handling)
14. [Benchmark System](#benchmark-system)
15. [Storage & Persistence](#storage--persistence)
16. [Integration with PyTorch](#integration-with-pytorch)
17. [Performance Notes](#performance-notes)
18. [License](#license)

---

## What is AESG?

AESG (Adaptive External Semantic Graph) is a Python library that gives neural networks a **persistent external memory** in the form of a semantic graph. Unlike attention mechanisms that only look at the current input window, AESG maintains a growing knowledge structure on disk that persists between training runs, can be shared across models, and evolves over time.

**The core idea:** Every time your model processes data, AESG decides whether the input represents something genuinely new. If so, it creates a new "concept" in the graph. Over time, related concepts form connections. Old, unused concepts get pruned. The result is a self-organizing knowledge base that the model queries at every step.

**What makes AESG different from other memory systems:**

| Feature | AESG | Attention/KV Cache | Memory Networks |
|---------|------|-------------------|-----------------|
| Persists across sessions | ✓ | ✗ | ✗ |
| Self-organizes (prunes, merges) | ✓ | ✗ | ✗ |
| Scales to millions of concepts | ✓ | Limited | Limited |
| Lives on disk (mmap) | ✓ | RAM only | RAM only |
| Portable (packs) | ✓ | ✗ | ✗ |
| Works with any architecture | ✓ | Transformers only | Specific arch |

**Target simplicity:**

```python
from aesg import AESGTransformer, AESGTrainer

model = AESGTransformer.small(vocab_size=10000)
trainer = AESGTrainer(model)
trainer.fit(train_data, epochs=5)
trainer.evaluate(test_data)
```

That's it. Three lines to create a Transformer with persistent semantic memory, train it, and evaluate it. No manual graph configuration, no memory management, no custom training loops.

---

## Quick Start (5 minutes)

### Step 1: Install

```bash
pip install aesg
```

### Step 2: Create a model

```python
from aesg import AESGTransformer

# This creates a Transformer with:
# - 128-dim hidden layers, 1 layer, 4 attention heads
# - An internal AESGMemory with vector_dim=64 and max 100k concepts
# - Memory stored on disk at ./aesg_memory/
model = AESGTransformer.small(vocab_size=10000)
```

### Step 3: Train

```python
from aesg import AESGTrainer
import torch

# Create dummy training data: list of (input_tokens, target_tokens) tuples
train_data = [
    (torch.randint(0, 10000, (32,)), torch.randint(0, 10000, (32,)))
    for _ in range(100)
]

# AESGTrainer handles everything: optimizer, loss, memory topology updates
trainer = AESGTrainer(model)
losses = trainer.fit(train_data, epochs=5)
print(f"Final loss: {losses[-1]:.4f}")
```

### Step 4: Evaluate

```python
test_data = [
    (torch.randint(0, 10000, (32,)), torch.randint(0, 10000, (32,)))
    for _ in range(20)
]
metrics = trainer.evaluate(test_data)
print(f"Test loss: {metrics['loss']:.4f}")
```

---

## Installation

```bash
pip install aesg
```

### Requirements (installed automatically)

- Python >= 3.8
- PyTorch >= 1.10.0
- NumPy >= 1.20.0
- NetworkX >= 2.5

### Optional dependencies

```bash
pip install torchvision   # For image architectures with real datasets
pip install tensorboard    # For TensorBoardCallback logging
pip install wandb          # For Weights & Biases logging
```

---

## Core Concepts

Before diving into the API, here's what you need to understand about how AESG works under the hood.

### Concepts

A **concept** is a node in the AESG graph. Each concept is a vector (like an embedding) plus metadata: when it was created, how often it's been activated, how relevant it is. Concepts represent "things the model has learned." They're created automatically when the model encounters genuinely novel patterns.

### Spreading Activation

When your model needs to retrieve context from memory, AESG doesn't do a simple nearest-neighbor lookup. Instead, it finds the most relevant seed node and then **propagates activation energy** outward through edges. Nearby concepts that share strong connections get activated together. This gives you richer, more structured context than a flat lookup would.

### Novelty Detection

At every forward pass, AESG computes an "explanation score" — how well the current graph can explain the incoming input. If the score is too low (the input is novel), AESG increments a novelty buffer. If the buffer reaches a threshold, a new concept is born. This means the graph only grows when needed, not on every input.

### Consolidation & Evolutionary Pressure

Periodically, AESG runs a consolidation pass that:
1. Increments the age of every concept
2. Decays relevance by 5% (concepts fade if not used)
3. Prunes concepts that are old, low-relevance, and rarely activated

This prevents unbounded growth. The graph evolves: useful concepts survive, unused ones disappear.

### Persistence

The entire graph lives on disk as memory-mapped files (numpy mmap). This means:
- The graph doesn't need to fit in RAM
- It survives between program restarts
- Multiple training runs accumulate knowledge in the same graph
- You can copy the directory to share the memory

---

## Architecture Deep Dive

AESG is organized in layers:

```
┌─────────────────────────────────────────────────────────┐
│  Application Layer                                       │
│  AESGTrainer · Callbacks · Evaluator · Benchmark        │
├─────────────────────────────────────────────────────────┤
│  Architecture Layer                                      │
│  AESGTransformer · AESGGRUText · AESGCNNClassifier etc. │
├─────────────────────────────────────────────────────────┤
│  Core Layer                                              │
│  MemoryModes · PackManager · Validation                 │
├─────────────────────────────────────────────────────────┤
│  Cognitive Core (Preserved from V2)                      │
│  AESGMemory · CognitiveEngine · Navigator · Storage     │
└─────────────────────────────────────────────────────────┘
```

### How a forward pass works (step by step)

1. **Your model receives input** (e.g., a batch of token IDs)
2. **Embedding** converts tokens to continuous vectors
3. **At each layer**, the model generates a query vector from its current state
4. **AESGMemory.retrieve(query)** is called:
   - Navigator finds seed nodes via nearest-neighbor search
   - Spreading activation propagates energy through the graph
   - Top activated concepts are returned as a `RetrievedContext`
5. **Context injection**: the retrieved memory vectors are incorporated into the neural computation (via concatenation for RNNs, cross-attention for Transformers, spatial addition for CNNs)
6. **Novelty check**: if the graph couldn't explain the input well, the novelty buffer increments; after enough persistence, a new concept is created
7. **After each batch** (during training): `update_topology()` runs consolidation

### Data flow diagram

```
Input Tensor ──→ Model Layer ──→ query_proj ──→ AESGMemory.retrieve()
                     ↑                              │
                     │                              ↓
                     │                     Navigator (spreading activation)
                     │                              │
                     └── context injection ←── RetrievedContext
                                                    │
                                                    ↓
                                            Novelty Detection
                                                    │
                                                    ↓ (if novel enough)
                                            Create Sensory Concept
```

---

## Text Models — Complete Guide

AESG provides five pre-assembled text architectures. Each one integrates AESG memory directly into its forward pass, so the model queries long-term knowledge at every step.

### What `.small()`, `.medium()`, `.large()` mean

These are factory methods that create a model with preset sizes:

| Size | hidden_size | num_layers | vector_dim | nhead (Transformer) |
|------|:-----------:|:----------:|:----------:|:-------------------:|
| small | 128 | 1 | 64 | 4 |
| medium | 256 | 2 | 128 | 8 |
| large | 512 | 4 | 256 | 16 |

- **hidden_size**: The dimension of the neural network's internal states
- **num_layers**: How many stacked layers
- **vector_dim**: The dimension of concepts in the AESG memory graph
- **nhead**: Number of attention heads (Transformer only)

When you call `.small(vocab_size=10000)`, it:
1. Creates an `AESGConfig(vector_dim=64, max_concepts=100_000)`
2. Creates an `AESGMemory("./aesg_memory", config)` 
3. Builds the neural network with hidden_size=128, 1 layer
4. Wires the memory into the model's forward pass
5. Returns a ready-to-train `nn.Module`

### AESGGRUText — In Detail

**What it is:** A GRU (Gated Recurrent Unit) that queries AESG memory at every timestep. The hidden state is used to generate a memory query, and the retrieved context is concatenated with the input before being fed to the GRU cell.

**How it works internally:**
```
For each timestep t:
  1. h_prev → query_proj → query_vector
  2. AESGMemory.retrieve(query_vector) → context
  3. context.aggregate() → memory_vector (collapsed to 1D)
  4. concat(input_embedding[t], memory_vector) → gru_input
  5. GRUCell(gru_input, h_prev) → h_new
  6. output_proj(h_new) → logits[t]
```

**When to use it:** Sequential tasks where you want the model to accumulate knowledge across training runs. Good for language modeling, sequence classification, and simple text generation.

**Complete example:**

```python
from aesg import AESGGRUText, AESGTrainer
import torch

# Create model: vocabulary of 5000 tokens, small size
model = AESGGRUText.small(vocab_size=5000)

# Input: batch of 4 sequences, each 20 tokens long
x = torch.randint(0, 5000, (4, 20))

# Forward pass produces logits for each position
logits = model(x)
print(logits.shape)  # torch.Size([4, 20, 5000])
# → For each of the 4 sequences, at each of the 20 positions,
#   we get a probability distribution over 5000 tokens

# Training
trainer = AESGTrainer(model, learning_rate=0.001)
train_data = [(torch.randint(0, 5000, (20,)), torch.randint(0, 5000, (20,))) for _ in range(200)]
losses = trainer.fit(train_data, epochs=3)
```

### AESGLSTMText — In Detail

**What it is:** Same concept as AESGGRUText but uses an LSTM cell instead of GRU. The LSTM maintains both a hidden state (h) and a cell state (c), giving it more capacity for long-range dependencies.

**Key difference from GRU:** The LSTM's cell state provides a "highway" for gradient flow, making it better for longer sequences where GRU might struggle.

```python
from aesg import AESGLSTMText, AESGTrainer
import torch

model = AESGLSTMText.medium(vocab_size=8000)

# Same interface as GRU — the complexity is hidden inside
x = torch.randint(0, 8000, (4, 30))
logits = model(x)
print(logits.shape)  # torch.Size([4, 30, 8000])
```

### AESGSeq2Seq — In Detail

**What it is:** An encoder-decoder architecture. The encoder is a standard GRU (no AESG). The decoder is an AESG-enhanced GRU that generates output step by step while querying semantic memory.

**When to use it:** Translation, summarization, or any sequence-to-sequence task where the input and output are different sequences.

**How it works internally:**
```
Encoder:
  standard GRU processes input → produces enc_hidden

Decoder (with AESG):
  For each output timestep:
    1. AESG_GRU(dec_input, h_prev) → h_new
    2. output_proj(h_new) → logits[t]
    3. Next input = teacher-forced ground truth (during training)
```

```python
from aesg import AESGSeq2Seq, AESGTrainer
import torch

model = AESGSeq2Seq.medium(vocab_size=10000)

# Input: source sequence (to be "translated")
src = torch.randint(0, 10000, (4, 25))
logits = model(src)
print(logits.shape)  # torch.Size([4, 25, 10000])
```

### AESGDecoderLM — In Detail

**What it is:** A decoder-only (causal) language model. Processes tokens left-to-right, predicting the next token at each position. This is the architecture used by GPT-style models, but at smaller scale and with AESG memory.

**When to use it:** Text generation, auto-completion, causal language modeling.

```python
from aesg import AESGDecoderLM, AESGTrainer
import torch

model = AESGDecoderLM.large(vocab_size=30000)

# Generate: feed a prompt, get logits for next tokens
prompt = torch.randint(0, 30000, (1, 50))
logits = model(prompt)
# logits[0, -1, :] contains the probability distribution for the next token
next_token_probs = torch.softmax(logits[0, -1, :], dim=0)
predicted_token = next_token_probs.argmax().item()
print(f"Predicted next token ID: {predicted_token}")
```

### AESGTransformer — In Detail

**What it is:** A Transformer encoder with AESG memory integrated via **cross-attention**. At each layer, after self-attention over the sequence, the model performs cross-attention against the memory concepts retrieved from the AESG graph. This is the most powerful architecture because it doesn't collapse the memory into a single vector — it attends to individual concepts.

**How it works internally:**
```
For each layer:
  1. Self-attention(src, src, src) → attended
  2. query_proj(mean(attended)) → memory_query
  3. AESGMemory.retrieve(memory_query) → RetrievedContext
  4. Cross-attention(attended, memory_vectors, memory_vectors) → enriched
  5. FeedForward(enriched) → output
```

**When to use it:** Any task where you want maximum expressiveness. The cross-attention mechanism lets the model selectively attend to relevant past concepts rather than being forced to use a single aggregated vector.

```python
from aesg import AESGTransformer, AESGTrainer
import torch

# Small: 128 hidden, 1 layer, 4 heads, 64-dim memory
model = AESGTransformer.small(vocab_size=10000)

# Medium: 256 hidden, 2 layers, 8 heads, 128-dim memory
model = AESGTransformer.medium(vocab_size=10000)

# Large: 512 hidden, 4 layers, 16 heads, 256-dim memory
model = AESGTransformer.large(vocab_size=10000)

# Custom storage location
model = AESGTransformer.medium(vocab_size=10000, storage_dir="./my_transformer_memory")

# Forward pass
x = torch.randint(0, 10000, (4, 64))  # batch=4, seq_len=64
logits = model(x)
print(logits.shape)  # torch.Size([4, 64, 10000])
```

### Choosing the right text model

| Task | Recommended Model | Why |
|------|-------------------|-----|
| Simple sequence classification | AESGGRUText.small | Fast, lightweight |
| Language modeling (next-token) | AESGDecoderLM.medium | Causal, good capacity |
| Translation / summarization | AESGSeq2Seq.medium | Encoder-decoder |
| Best quality, any task | AESGTransformer.large | Cross-attention to memory |
| Long-range dependencies | AESGLSTMText.medium | Cell state highway |

---

## Image Models — Complete Guide

AESG provides three pre-assembled image architectures. Each one injects semantic memory at the CNN bottleneck — the point where spatial features are most compressed and abstract.

### How memory works in CNNs

In image models, AESG memory is injected at the **bottleneck** layer:

```
Input Image (B, C, H, W)
    ↓
Encoder (Conv + BN + ReLU + MaxPool) × N blocks
    ↓
Bottleneck feature maps (B, channels, small_H, small_W)
    ↓
AESG_CNNLayer:
  1. Flatten spatial features → query vector
  2. AESGMemory.retrieve(query) → context
  3. context.aggregate() → memory vector
  4. Project memory back to spatial shape
  5. Add memory spatial tensor to conv output
    ↓
Decoder (ConvTranspose + BN + ReLU) × N blocks
    ↓
Output Image
```

The memory injection happens at the most abstract level of the CNN, where the features are highest-level and most semantically meaningful. This lets the memory influence high-level decisions (like "this region should be blue because I've seen sky before") rather than low-level pixel details.

### AESGColorizationNet — In Detail

**What it is:** Takes a grayscale image (1 channel) and produces a colorized version (3 channels, RGB). The encoder compresses the image, AESG provides semantic context about what colors to use, and the decoder reconstructs the full-color image.

**Input/Output:**
- Input: `(batch_size, 1, height, width)` — grayscale image, values in [0, 1]
- Output: `(batch_size, 3, height, width)` — RGB color image, values in [0, 1]

**Architecture by size:**

| Size | Base Filters | Blocks | vector_dim | Bottleneck spatial |
|------|:-----------:|:------:|:----------:|:------------------:|
| small | 32 | 3 | 64 | 16×16 |
| medium | 64 | 4 | 128 | 8×8 |
| large | 128 | 5 | 256 | 4×4 |

```python
from aesg import AESGColorizationNet, AESGTrainer
import torch

# Create model
model = AESGColorizationNet.small()

# Input: batch of 8 grayscale images, 128×128
gray_images = torch.rand(8, 1, 128, 128)

# Output: colorized images
color_output = model(gray_images)
print(color_output.shape)   # torch.Size([8, 3, 128, 128])
print(color_output.min())   # tensor(~0.03) — sigmoid ensures [0,1]
print(color_output.max())   # tensor(~0.97)

# Train with MSE loss (pixel reconstruction)
trainer = AESGTrainer(model, criterion=torch.nn.MSELoss())
# Each sample: (grayscale_image, color_image)
data = [(torch.rand(1, 128, 128), torch.rand(3, 128, 128)) for _ in range(100)]
losses = trainer.fit(data, epochs=3)
```

### AESGCNNClassifier — In Detail

**What it is:** A CNN that classifies images into categories. The encoder extracts features, AESG provides semantic context at the bottleneck, and a classification head produces class logits.

**Input/Output:**
- Input: `(batch_size, 3, height, width)` — RGB image
- Output: `(batch_size, num_classes)` — raw logits (use softmax for probabilities)

**`num_classes` parameter:** You specify how many categories to classify into. Default is 10 (like CIFAR-10).

```python
from aesg import AESGCNNClassifier, AESGTrainer
import torch

# CIFAR-10 style: 10 classes
model = AESGCNNClassifier.small(num_classes=10)

# ImageNet style: 1000 classes
model = AESGCNNClassifier.large(num_classes=1000)

# Forward pass
images = torch.rand(8, 3, 128, 128)
logits = model(images)
print(logits.shape)  # torch.Size([8, 10])

# Get predicted classes
predictions = logits.argmax(dim=1)
print(predictions)  # tensor([3, 7, 2, 5, 1, 0, 4, 6])

# Train with CrossEntropyLoss (default)
trainer = AESGTrainer(model)
data = [(torch.rand(3, 128, 128), torch.tensor(label)) for label in range(100) for _ in range(1)]
losses = trainer.fit(data, epochs=5)
```

### AESGCNNAutoencoder — In Detail

**What it is:** A symmetric encoder-decoder that reconstructs its input. The model learns a compressed representation at the bottleneck, with AESG memory augmenting that representation. Useful for anomaly detection, denoising, and learning representations.

**Input/Output:**
- Input: `(batch_size, channels, height, width)` — any image
- Output: `(batch_size, channels, height, width)` — reconstructed image, same dimensions

```python
from aesg import AESGCNNAutoencoder, AESGTrainer
import torch

model = AESGCNNAutoencoder.small()

# Input and target are the same image (reconstruction task)
images = torch.rand(8, 3, 128, 128)
reconstructed = model(images)
print(reconstructed.shape)  # torch.Size([8, 3, 128, 128])

# Train — target is the same as input
trainer = AESGTrainer(model, criterion=torch.nn.MSELoss())
data = [(img, img) for img in [torch.rand(3, 128, 128) for _ in range(100)]]
losses = trainer.fit(data, epochs=5)
```

---

## AESGTrainer — Complete Guide

`AESGTrainer` is the professional training interface for AESG models. It handles everything: the training loop, memory topology updates, data format adaptation, checkpointing, and callbacks.

### What AESGTrainer does that a manual loop doesn't

1. **Dual training:** Updates both neural weights (backprop) AND the memory graph topology (consolidation, pruning) automatically
2. **Mode management:** Switches memory to TRAIN during fit(), INFERENCE during evaluate()/predict()
3. **Data adaptation:** Accepts DataLoaders, Datasets, or plain lists — no manual conversion needed
4. **Lifecycle callbacks:** EarlyStopping, checkpointing, logging — all composable
5. **Full persistence:** Saves model weights + optimizer state + memory graph + training state in one call

### Constructor Parameters

```python
from aesg import AESGTrainer

trainer = AESGTrainer(
    model,                    # Any nn.Module (with or without AESGMemory)
    optimizer=None,           # Default: Adam(lr=learning_rate)
    criterion=None,           # Default: CrossEntropyLoss
    callbacks=None,           # Default: [] (no callbacks)
    device="cpu",             # "cpu" or "cuda" or "cuda:0"
    batch_size=32,            # Used when adapting lists/Datasets to DataLoaders
    learning_rate=1e-3,       # Used by default Adam optimizer
)
```

**What happens if your model doesn't have AESGMemory?** The trainer still works — it just operates in "neural-only" mode and emits a warning. You can use AESGTrainer for any PyTorch model, not just AESG models.

### fit() — Training

```python
losses = trainer.fit(train_data, epochs=10)
```

**What it does step by step:**
1. Sets memory to TRAIN mode
2. Invokes `on_train_start` callbacks
3. For each epoch:
   - Invokes `on_epoch_start`
   - For each batch:
     - Resets memory navigation state
     - Forward pass (model processes input, queries memory)
     - Computes loss
     - Backward pass (gradients flow to neural weights AND memory's abstraction projection)
     - Optimizer step (updates weights)
     - Topology update (consolidation: ages nodes, decays relevance, prunes)
   - Invokes `on_epoch_end` (with loss in logs)
   - Saves memory to disk
   - Checks for early stopping signal
4. Invokes `on_train_end`
5. Returns list of average loss per epoch

**Data format flexibility:**

```python
# Option 1: PyTorch DataLoader (used directly, no conversion)
from torch.utils.data import DataLoader, TensorDataset
dataset = TensorDataset(inputs, targets)
loader = DataLoader(dataset, batch_size=64, shuffle=True)
trainer.fit(loader, epochs=5)

# Option 2: PyTorch Dataset (wrapped in DataLoader with trainer's batch_size)
trainer.fit(dataset, epochs=5)

# Option 3: Simple list of tuples (most convenient for prototyping)
data = [(input_tensor, target_tensor) for ...]
trainer.fit(data, epochs=5)
```

### evaluate() — Evaluation

```python
metrics = trainer.evaluate(test_data)
# Returns: {"loss": 0.5432}
```

**What it does:**
1. Saves current memory mode
2. Switches memory to INFERENCE (read-only, no concept creation)
3. Puts model in eval mode (disables dropout, batchnorm tracks running stats)
4. Computes average loss over all test batches with `torch.no_grad()`
5. Restores previous memory mode
6. Returns metrics dictionary

### predict() — Inference

```python
output = trainer.predict(input_tensor)
```

**What it does:**
1. Switches to INFERENCE mode
2. Runs model forward with `torch.no_grad()` (no gradient computation)
3. Returns raw output tensor
4. Restores previous mode

### save() and load() — Persistence

```python
# Save everything in one call
trainer.save("./checkpoints/epoch_10")

# What gets saved:
# ./checkpoints/epoch_10/
# ├── model_weights.pt       (PyTorch state_dict)
# ├── optimizer_state.pt     (optimizer state)
# ├── config.json            (AESGConfig + trainer metadata)
# ├── training_state.json    (epoch, global_step, best_loss)
# └── memory/                (copy of the AESG memory directory)

# Load everything back
trainer.load("./checkpoints/epoch_10")
```

### resume() — Continue Training

```python
# Resume from where you left off
start_epoch = trainer.resume("./checkpoints/epoch_10")
# start_epoch = 10 (the epoch to continue from)
losses = trainer.fit(train_data, epochs=10)
# This will train epochs 10-19
```

### Callbacks — In Detail

Callbacks let you hook into the training lifecycle without modifying the trainer. They're called in registration order.

**Available lifecycle hooks:**
- `on_train_start(logs)` — Once, before first epoch
- `on_train_end(logs)` — Once, after all epochs
- `on_epoch_start(epoch, logs)` — Start of each epoch
- `on_epoch_end(epoch, logs)` — End of each epoch (has "loss" in logs)
- `on_batch_start(batch, logs)` — Start of each batch
- `on_batch_end(batch, logs)` — End of each batch

**EarlyStopping:** Monitors a metric and stops training if it doesn't improve.

```python
from aesg import EarlyStopping

# Stop if loss doesn't improve for 5 consecutive epochs
es = EarlyStopping(patience=5, monitor="loss")
trainer = AESGTrainer(model, callbacks=[es])
losses = trainer.fit(data, epochs=100)  # Will stop early if loss plateaus
```

**Checkpoint:** Saves the model at regular intervals.

```python
from aesg import Checkpoint

# Save every 2 epochs
ckpt = Checkpoint(save_every=2, path="./checkpoints")
trainer = AESGTrainer(model, callbacks=[ckpt])
trainer.fit(data, epochs=10)
# Creates: ./checkpoints/epoch_2, ./checkpoints/epoch_4, ...
```

**Custom callbacks:** Implement the `Callback` base class.

```python
from aesg import Callback

class PrintProgress(Callback):
    def on_epoch_end(self, epoch, logs):
        loss = logs.get("loss", "?")
        print(f"Epoch {epoch}: loss = {loss:.4f}")

trainer = AESGTrainer(model, callbacks=[PrintProgress()])
```

---

## Memory Modes — Complete Guide

Memory modes control what operations AESG is allowed to perform. This prevents accidental writes during inference and enables fine-tuning without destroying existing knowledge.

### The Four Modes

**TRAIN** — Full power. All operations enabled.
- Creates new concepts from novel inputs
- Runs consolidation (ages nodes, decays relevance)
- Applies evolutionary pressure (prunes weak concepts)
- Allows reorganization (merge, split concepts)
- Learning rate scale: 1.0x

**FINETUNE** — Careful learning. Adds knowledge without disrupting structure.
- Creates new concepts (at 0.1x rate)
- Runs consolidation
- NO reorganization (won't merge/split existing concepts)
- NO evolutionary pressure (won't prune)
- Learning rate scale: 0.1x

**INFERENCE** — Read-only. Zero modification to memory.
- NO concept creation
- NO consolidation
- NO reorganization
- NO pruning
- The graph is static, only queried
- Learning rate scale: 0.0x

**ONLINE** — Continuous learning. Learns at full speed but doesn't reorganize.
- Creates new concepts
- Runs consolidation
- NO reorganization
- NO evolutionary pressure
- Learning rate scale: 1.0x
- Useful for streaming/real-time scenarios

### How to use modes

```python
from aesg import AESGTransformer

model = AESGTransformer.small(vocab_size=10000)

# Access the memory object
memory = model.memory

# Check current mode
print(memory.mode)  # MemoryMode.TRAIN (default)

# Switch modes
memory.set_mode("INFERENCE")
# Now retrieve() works but no concepts are created

memory.set_mode("FINETUNE")
# Concepts can be created slowly, but structure is preserved

memory.set_mode("TRAIN")
# Full learning restored
```

### When blocked operations are called

If you call something that the current mode doesn't allow (e.g., trying to create a concept in INFERENCE mode), it's a **silent no-op**. No exception is raised, no error logged. The operation simply doesn't execute. This makes it safe to use the same model code in both training and inference without conditional logic.

### AESGTrainer handles modes automatically

You don't need to manage modes manually when using AESGTrainer:
- `trainer.fit()` → automatically sets TRAIN
- `trainer.evaluate()` → automatically sets INFERENCE, then restores
- `trainer.predict()` → automatically sets INFERENCE, then restores

---

## Pack System — Complete Guide

Packs are portable memory snapshots. They let you:
- Export knowledge from a trained model
- Share domain expertise between models
- Compose multiple knowledge domains
- Specialize a general model without retraining

### What is a .aesgpack file?

A `.aesgpack` file contains a serialized subgraph of concepts and their connections. It includes:
- Magic bytes for identification ("AESGPACK")
- Format version
- The vector_dim of the concepts
- All node data (vectors, metadata, connections)
- A SHA-256 checksum for integrity verification

### Exporting a pack

After training a model, you can export its memory (or a filtered subset) as a pack:

```python
from aesg import AESGTransformer, AESGTrainer

# Train a model on medical text
model = AESGTransformer.medium(vocab_size=30000, storage_dir="./medical_memory")
trainer = AESGTrainer(model)
trainer.fit(medical_data, epochs=20)

# Export ALL learned concepts
model.memory.export_pack("./packs/medical_full.aesgpack")

# Export only highly-relevant concepts (relevance >= 0.5)
model.memory.export_pack("./packs/medical_core.aesgpack", min_relevance=0.5)

# Export only concepts from region 2 (if region detection has run)
model.memory.export_pack("./packs/medical_region2.aesgpack", region_id=2)
```

### Loading and attaching a pack

```python
from aesg import AESGTransformer

# Create a fresh model
model = AESGTransformer.medium(vocab_size=30000, storage_dir="./new_model_memory")

# Load the pack (validates checksum, checks vector_dim compatibility)
pack = model.memory.load_pack("./packs/medical_core.aesgpack")
print(f"Loaded: {pack.name}, {len(pack.nodes)} concepts")

# Attach with priority (higher = more influence during retrieval)
model.memory.attach_pack(pack, priority=80)

# Now the model can access medical concepts during inference!
```

### Combining multiple packs

```python
# Load specialized packs
medical = model.memory.load_pack("./packs/medical.aesgpack")
chemistry = model.memory.load_pack("./packs/chemistry.aesgpack")
general = model.memory.load_pack("./packs/general.aesgpack")

# Attach with different priorities
model.memory.attach_pack(medical, priority=90)    # Highest influence
model.memory.attach_pack(chemistry, priority=60)  # Medium influence
model.memory.attach_pack(general, priority=20)    # Background knowledge

# During spreading activation, pack node energies are weighted by
# normalized priority: medical gets 90/(90+60+20) = 53% weight
```

### Detaching packs

```python
# Remove a pack (returns it to "loaded" state, doesn't delete it)
model.memory.detach_pack("medical")

# The base memory and other packs are unaffected
```

### Constraints

- Maximum 16 packs can be attached simultaneously
- Pack's `vector_dim` MUST match the model's `vector_dim`
- If the file is corrupt (bad checksum), an `AESGStorageError` is raised

---

## Evaluation & Metrics — Complete Guide

The `Evaluator` class computes domain-specific quality metrics automatically based on what type of model you're evaluating.

### Automatic domain detection

The Evaluator looks at your model's class name to figure out which metrics to compute:

| Model class contains... | Detected domain | Metrics computed |
|-------------------------|:---------------:|-----------------|
| "Transformer", "GRUText", "LSTMText", "Seq2Seq", "DecoderLM" | text | BLEU, ROUGE, Accuracy, Perplexity |
| "Classifier" | classification | Accuracy, Precision, Recall, F1 |
| "Colorization", "Autoencoder" | image | PSNR, SSIM, MSE, MAE |

### Text Metrics — What they mean

- **BLEU** (0.0 to 1.0): Measures n-gram overlap between prediction and reference. Higher = better match. Uses 4-gram precision with brevity penalty.
- **ROUGE** (0.0 to 1.0): Measures unigram recall — what fraction of reference words appear in the prediction.
- **Accuracy** (0.0 to 1.0): Exact match ratio — how many predictions are identical to references.
- **Perplexity** (1.0 to ∞): How "surprised" the model is by the reference. Lower = better. Computed via character-level cross-entropy.

```python
from aesg import Evaluator

# Predictions and references as lists of strings
predictions = ["the cat sat on the mat", "hello world"]
references = ["the cat sat on a mat", "hello world"]

metrics = Evaluator.compute_text_metrics(predictions, references)
print(f"BLEU: {metrics['bleu']:.4f}")        # ~0.5-0.8
print(f"ROUGE: {metrics['rouge']:.4f}")       # ~0.8-1.0
print(f"Accuracy: {metrics['accuracy']:.4f}")  # 0.5 (1 exact match out of 2)
print(f"Perplexity: {metrics['perplexity']:.2f}")
```

### Image Metrics — What they mean

- **PSNR** (dB, higher is better): Peak Signal-to-Noise Ratio. Measures pixel-level reconstruction quality. 20+ dB is decent, 30+ dB is very good.
- **SSIM** (-1.0 to 1.0, higher is better): Structural Similarity. Measures perceived visual quality including luminance, contrast, and structure. 0.9+ is good.
- **MSE** (0.0 to 1.0, lower is better): Mean Squared Error between pixels.
- **MAE** (0.0 to 1.0, lower is better): Mean Absolute Error between pixels.

```python
import torch
from aesg import Evaluator

# Model predictions and ground truth (both in [0, 1] range)
predicted_images = torch.rand(16, 3, 128, 128)
target_images = torch.rand(16, 3, 128, 128)

metrics = Evaluator.compute_image_metrics(predicted_images, target_images)
print(f"PSNR: {metrics['psnr']:.2f} dB")
print(f"SSIM: {metrics['ssim']:.4f}")
print(f"MSE: {metrics['mse']:.6f}")
print(f"MAE: {metrics['mae']:.6f}")
```

### Classification Metrics — What they mean

- **Accuracy** (0.0 to 1.0): Fraction of correctly classified samples.
- **Precision** (0.0 to 1.0): Of all predicted positives, how many are correct? Macro-averaged across classes.
- **Recall** (0.0 to 1.0): Of all actual positives, how many did we find? Macro-averaged across classes.
- **F1** (0.0 to 1.0): Harmonic mean of precision and recall. Balances both.

```python
import torch
from aesg import Evaluator

# Model outputs raw logits (before softmax), targets are class indices
logits = torch.randn(200, 10)       # 200 samples, 10 classes
targets = torch.randint(0, 10, (200,))

metrics = Evaluator.compute_classification_metrics(logits, targets)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1: {metrics['f1']:.4f}")
```

### Using with Evaluator.evaluate() (auto-detect)

```python
from aesg import AESGCNNClassifier, Evaluator
import torch

model = AESGCNNClassifier.small(num_classes=5)
evaluator = Evaluator()

# The evaluator detects "Classifier" in the class name → uses classification metrics
logits = torch.randn(50, 5)
targets = torch.randint(0, 5, (50,))
metrics = evaluator.evaluate(model, logits, targets)
# Returns: {"accuracy": ..., "precision": ..., "recall": ..., "f1": ...}
```

---

## AESGConfig — Complete Reference

`AESGConfig` is the single source of truth for all AESG parameters. It's a **frozen dataclass** — once created, you cannot modify its values. This prevents accidental mid-training changes.

### Creating configurations

```python
from aesg import AESGConfig

# Default configuration (good for most cases)
config = AESGConfig()

# Domain-specific presets
config = AESGConfig.for_text()            # vector_dim=128, max_concepts=500k
config = AESGConfig.for_image()           # vector_dim=256, max_concepts=200k
config = AESGConfig.for_classification()  # vector_dim=64, max_concepts=100k

# Fully custom
config = AESGConfig(
    vector_dim=128,
    max_concepts=500_000,
    spreading_activation_steps=4,
    novelty_birth_threshold=5,
    learning_rate=5e-4,
)
```

### Why frozen?

```python
config = AESGConfig(vector_dim=128)
config.vector_dim = 256  # ← FrozenInstanceError! Cannot modify.
```

This is intentional. Changing config mid-training could corrupt the memory graph (e.g., changing `vector_dim` after nodes were already created). If you need different settings, create a new `AESGConfig`.

### Parameter groups explained

**Memory parameters:**
| Parameter | Default | What it controls |
|-----------|---------|-----------------|
| `vector_dim` | 256 | Dimension of concept vectors. Must match your model's internal dimension. Range: [1, 8192]. |
| `max_concepts` | 1,000,000 | Maximum number of concepts in the graph. When exceeded, evolutionary pressure prunes the weakest. |
| `max_edges_per_node` | 1000 | If a node has more edges than this, it may be split into subgraphs. |

**Navigation parameters:**
| Parameter | Default | What it controls |
|-----------|---------|-----------------|
| `spreading_activation_steps` | 3 | How many hops activation propagates. More hops = broader context but slower. |
| `spreading_activation_decay` | 0.8 | How much energy is lost per hop. 0.8 means 80% survives each hop. Range: [0.0, 1.0]. |
| `region_facilitation_multiplier` | 1.5 | Bonus for traversing within the same semantic region. >1.0 means intra-region paths are preferred. |

**Novelty parameters:**
| Parameter | Default | What it controls |
|-----------|---------|-----------------|
| `novelty_explanation_threshold` | 0.6 | If the graph explains the input below this score, it's considered novel. Range: [0.0, 1.0]. |
| `novelty_birth_threshold` | 3 | How many times a novel input must persist before a concept is created. Prevents single outliers from creating nodes. |

**Evolutionary pressure:**
| Parameter | Default | What it controls |
|-----------|---------|-----------------|
| `survival_threshold_relevance` | 0.05 | Concepts with relevance below this (and old enough) get pruned. |
| `survival_threshold_frequency` | 5 | Concepts activated fewer times than this (and old enough) get pruned. |

**Training parameters:**
| Parameter | Default | What it controls |
|-----------|---------|-----------------|
| `learning_rate` | 1e-3 | Default learning rate for the optimizer. |
| `batch_size` | 32 | Default batch size when AESGTrainer wraps data. |
| `memory_mode` | "TRAIN" | Initial memory mode. |

### JSON serialization

Save and restore configurations:

```python
from aesg import AESGConfig

config = AESGConfig(vector_dim=128, max_concepts=200_000)

# Save to JSON string
json_str = config.to_json()
print(json_str)
# {
#   "vector_dim": 128,
#   "max_concepts": 200000,
#   ...
# }

# Restore from JSON string
restored = AESGConfig.from_json(json_str)
assert config == restored  # True — exact round-trip
```

### Validation

If you pass invalid values, you get a clear error:

```python
from aesg import AESGConfig, AESGMemory

# This creates the config (frozen dataclass allows any values at creation)
config = AESGConfig(vector_dim=-1, spreading_activation_decay=5.0)

# Validation happens when you use it:
try:
    memory = AESGMemory("./test", config)  # ← Validator runs here
except Exception as e:
    print(e)
    # Configuration validation failed:
    # Parameter 'vector_dim': got -1, expected integer >= 1
    # Parameter 'spreading_activation_decay': got 5.0, expected float in [0.0, 1.0]
```

---

## Exception Handling

AESG has a clean exception hierarchy. Every error raised by AESG inherits from `AESGError`, so you can catch everything with one except clause, or be specific per subsystem.

### The hierarchy

```
AESGError (base)
├── AESGConfigError      → Invalid configuration
├── AESGMemoryError      → Memory operation failures
├── AESGNavigationError  → Spreading activation / retrieval issues
├── AESGTrainingError    → Training loop issues (bad data, etc.)
└── AESGStorageError     → File I/O, corrupt packs, checkpoint errors
```

### Each exception has:
- `e.message` — Human-readable description (max 500 chars)
- `e.subsystem` — Which subsystem ("config", "memory", "navigation", "trainer", "storage")

### Common error scenarios and how to handle them

```python
from aesg import (
    AESGError, AESGConfigError, AESGMemoryError,
    AESGTrainingError, AESGStorageError
)

# 1. Invalid vocab_size
from aesg import AESGTransformer
try:
    model = AESGTransformer.small(vocab_size=0)
except AESGConfigError as e:
    print(f"[{e.subsystem}] {e.message}")
    # [config] vocab_size must be >= 1

# 2. Invalid memory mode
from aesg import AESGMemory, AESGConfig
memory = AESGMemory("./test", AESGConfig(vector_dim=16, max_concepts=100))
try:
    memory.set_mode("INVALID")
except AESGMemoryError as e:
    print(f"[{e.subsystem}] {e.message}")
    # [memory] Invalid memory mode 'INVALID'. Valid modes: ['TRAIN', 'FINETUNE', 'INFERENCE', 'ONLINE']

# 3. Empty training data
from aesg import AESGTrainer, AESGGRUText
model = AESGGRUText.small(vocab_size=100)
trainer = AESGTrainer(model)
try:
    trainer.fit([], epochs=5)
except AESGTrainingError as e:
    print(f"[{e.subsystem}] {e.message}")
    # [trainer] Empty dataset provided. Training requires at least one sample.

# 4. Corrupt pack file
try:
    memory.load_pack("./not_a_real_pack.aesgpack")
except AESGStorageError as e:
    print(f"[{e.subsystem}] {e.message}")
    # [storage] Failed to read pack file './not_a_real_pack.aesgpack': ...

# 5. Catch ANY AESG error (broad handler)
try:
    # ... any AESG operation ...
    pass
except AESGError as e:
    print(f"AESG error in {e.subsystem}: {e.message}")
```

---

## Benchmark System

AESG includes a built-in benchmark that compares a CNN with AESG memory vs a plain CNN on image colorization.

### What it does

1. Downloads (or generates) 2000 color images
2. Converts them to grayscale/color pairs at 128×128 resolution
3. Splits into 80% train / 20% eval
4. Trains `AESGColorizationNet.small()` for 3 epochs
5. Trains a `BaselineCNN` (same architecture, no AESG) for 3 epochs
6. Computes PSNR and SSIM on the eval set
7. Saves visual comparison samples
8. Generates a markdown report with tables and analysis

### Running the benchmark

```python
from aesg import ColorizationBenchmark

benchmark = ColorizationBenchmark(
    dataset_name="cifar10",       # Uses CIFAR-10 if torchvision installed
    max_images=2000,              # Limit for fast execution
    image_size=128,               # All images resized to 128×128
    max_epochs=3,                 # Quick training run
    output_dir="./benchmark_results",  # Where results go
)

results = benchmark.run()

# Results contain metrics for both models
print(f"AESG — PSNR: {results['aesg'].psnr:.2f} dB, SSIM: {results['aesg'].ssim:.4f}")
print(f"Base — PSNR: {results['baseline'].psnr:.2f} dB, SSIM: {results['baseline'].ssim:.4f}")
print(f"AESG Memory: {results['aesg'].memory_nodes} concepts created")
```

### Output structure

```
./benchmark_results/
├── benchmark_report.md    # Markdown comparison report
├── results.json           # Raw metrics as JSON
└── samples/               # Visual comparison images
    ├── sample_000.png     # [original | grayscale | AESG color | baseline color]
    ├── sample_001.png
    └── ...
```

---

## Storage & Persistence

### Memory directory structure

When you create an `AESGMemory`, it writes to disk immediately:

```
./aesg_memory/
├── nodes.aesg          # All concept nodes (mmap file, structured numpy array)
├── edges.aesg          # All edges between nodes (mmap file, linked list)
├── meta.npy            # Metadata: node_count, edge_count, capacity, vector_dim
└── evolution.aesglog   # Binary event log (32 bytes per event)
```

**nodes.aesg** — Each node stores: ID, vector (D floats), created_at, modified_at, use_frequency, relevance, age, stability, region_id, is_active, head_edge_idx.

**edges.aesg** — Each edge stores: source_id, target_id, weight, confidence, use_count, next_edge_idx (linked list).

**meta.npy** — NumPy file with counts and capacities.

**evolution.aesglog** — Binary log of all evolutionary events (CREATE, MERGE, SPLIT, PRUNE, CONSOLIDATE, RESTRUCTURE). Each event is exactly 32 bytes.

### How mmap works

The files are memory-mapped (`numpy.memmap`), which means:
- Only the portions you access are loaded into RAM
- The OS manages paging automatically
- You can have a 1GB graph but only use 50MB of RAM
- Changes are written to disk when you call `memory.save()` or the trainer finishes an epoch

### Checkpoint structure

When `AESGTrainer.save()` is called:

```
./checkpoints/epoch_10/
├── model_weights.pt       # PyTorch state_dict of the full model
├── optimizer_state.pt     # Optimizer state (momentum, etc.)
├── config.json            # AESGConfig + trainer metadata
├── training_state.json    # {"epoch": 10, "global_step": 1000, "best_loss": 0.5}
└── memory/                # Copy of the entire memory directory
    ├── nodes.aesg
    ├── edges.aesg
    ├── meta.npy
    └── evolution.aesglog
```

---

## Integration with PyTorch

AESG models are standard `nn.Module` instances. They work with everything PyTorch offers.

### Using with a custom optimizer

```python
from aesg import AESGTransformer, AESGTrainer
import torch.optim as optim

model = AESGTransformer.medium(vocab_size=10000)

# Use any PyTorch optimizer
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
trainer = AESGTrainer(model, optimizer=optimizer)
```

### Using with a custom loss function

```python
import torch.nn as nn
from aesg import AESGColorizationNet, AESGTrainer

model = AESGColorizationNet.small()

# Use L1 loss instead of default CrossEntropyLoss
criterion = nn.L1Loss()
trainer = AESGTrainer(model, criterion=criterion)
```

### Using with GPU

```python
from aesg import AESGTransformer, AESGTrainer

model = AESGTransformer.large(vocab_size=30000)
trainer = AESGTrainer(model, device="cuda")
# Model is automatically moved to GPU
# Data tensors are automatically moved to GPU during training
```

### Using with a standard PyTorch DataLoader

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from aesg import AESGGRUText, AESGTrainer

model = AESGGRUText.medium(vocab_size=5000)

# Standard PyTorch Dataset
inputs = torch.randint(0, 5000, (1000, 50))
targets = torch.randint(0, 5000, (1000, 50))
dataset = TensorDataset(inputs, targets)
loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

# AESGTrainer accepts DataLoader directly
trainer = AESGTrainer(model)
losses = trainer.fit(loader, epochs=10)
```

### Manual training loop (without AESGTrainer)

If you need full control, you can use the model directly:

```python
import torch
import torch.nn as nn
from aesg import AESGGRUText

model = AESGGRUText.small(vocab_size=5000)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    model.memory.set_mode("TRAIN")
    model.memory.reset_state()
    
    x = torch.randint(0, 5000, (32, 20))
    y = torch.randint(0, 5000, (32, 20))
    
    optimizer.zero_grad()
    output = model(x)  # (32, 20, 5000)
    loss = criterion(output.reshape(-1, 5000), y.reshape(-1))
    loss.backward()
    optimizer.step()
    
    # Don't forget topology update!
    model.memory.update_topology()
    
    print(f"Epoch {epoch}: loss={loss.item():.4f}")

# Save memory to disk
model.memory.save()
```

### Accessing memory stats

```python
from aesg import AESGTransformer

model = AESGTransformer.small(vocab_size=5000)

# After training...
memory = model.memory
print(f"Total concepts: {memory.storage.node_count}")
print(f"Total edges: {memory.storage.edge_count}")
print(f"Current mode: {memory.mode.value}")
print(f"Vector dim: {memory.vector_dim}")

# Read the evolution log
history = memory.logger.get_history()
print(f"Total evolutionary events: {len(history)}")
for event in history[:5]:
    print(f"  {event['type']} at {event['timestamp']}")
```

---

## Performance Notes

### Expected throughput (CPU, single-threaded)

| Model | Size | Forward (ms/batch) | Memory Overhead | Graph Nodes (10 epochs) |
|-------|------|----:|----:|----:|
| AESGGRUText | small | ~2.1 | +12 MB | ~500 |
| AESGGRUText | large | ~8.4 | +48 MB | ~2,000 |
| AESGTransformer | small | ~3.5 | +12 MB | ~500 |
| AESGTransformer | large | ~15.2 | +48 MB | ~2,000 |
| AESGCNNClassifier | small | ~4.8 | +12 MB | ~300 |
| AESGColorizationNet | small | ~6.2 | +12 MB | ~400 |

### Storage scaling

| Concepts | RAM (vector_dim=16) | Disk |
|----------|---:|---:|
| 10,000 | ~2 MB | ~3 MB |
| 100,000 | ~17 MB | ~32 MB |
| 500,000 | ~80 MB | ~151 MB |
| 1,000,000 | ~173 MB | ~314 MB |

### Navigation latency

| Spreading activation hops | Average latency | p99 latency |
|:-------------------------:|:---------------:|:-----------:|
| 1 | ~1 ms | ~3 ms |
| 2 | ~4 ms | ~7 ms |
| 4 | ~9 ms | ~20 ms |
| 8 | ~49 ms | ~139 ms |

### Tips for performance

- Use `spreading_activation_steps=2` for real-time applications
- Use `spreading_activation_steps=4` for maximum quality (offline)
- Keep `vector_dim` small (64-128) for text tasks
- Use `max_concepts` wisely — 100k-500k is sufficient for most tasks
- The graph auto-prunes, so don't worry about setting it too high

---

## License

MIT License — Copyright (c) 2026 bueormnew

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED.

---

**Author:** [bueormnew](https://github.com/bueormnew) · **Homepage:** [github.com/bueormnew/aesg](https://github.com/bueormnew/aesg)
