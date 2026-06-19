"""
AESG Text Architecture Factories.

Pre-assembled text models with factory methods (.small/.medium/.large)
that configure appropriate hidden sizes, layers, and AESG memory dimensions.
"""

import torch
import torch.nn as nn

from aesg.nn.rnn import AESG_GRU, AESG_LSTM
from aesg.nn.transformer import AESG_TransformerLayer
from aesg.config import AESGConfig
from aesg.memory.controller import AESGMemory
from aesg.exceptions import AESGConfigError


# Size presets: (hidden_size, num_layers, vector_dim, nhead_transformer)
TEXT_SIZES = {
    "small": (128, 1, 64, 4),
    "medium": (256, 2, 128, 8),
    "large": (512, 4, 256, 16),
}


class AESGGRUText(nn.Module):
    """GRU-based text model with AESG memory integration.

    Processes sequences of token IDs through an embedding layer,
    then feeds each timestep through AESG_GRU cells that incorporate
    semantic memory retrieval at each step.

    Parameters
    ----------
    vocab_size : int
        Size of the token vocabulary.
    hidden_size : int
        Dimensionality of the GRU hidden state.
    num_layers : int
        Number of stacked GRU layers.
    memory : AESGMemory
        The AESG memory controller instance.
    """

    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int,
                 memory: AESGMemory):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.memory = memory
        self.gru = AESG_GRU(hidden_size, hidden_size, memory)
        self.output_proj = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the GRU text model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_length) containing
            integer token IDs.

        Returns
        -------
        torch.Tensor
            Logits of shape (batch_size, seq_length, vocab_size).
        """
        emb = self.embedding(x)  # (B, S, H)
        B, S, H = emb.shape
        h = torch.zeros(B, H, device=x.device)
        outputs = []
        for t in range(S):
            h = self.gru(emb[:, t, :], h)
            outputs.append(h)
        out = torch.stack(outputs, dim=1)  # (B, S, H)
        return self.output_proj(out)  # (B, S, vocab_size)

    @classmethod
    def small(cls, vocab_size: int, storage_dir: str = "./aesg_memory") -> "AESGGRUText":
        """Create a small GRU text model (hidden=128, 1 layer, vector_dim=64)."""
        return cls._from_size("small", vocab_size, storage_dir)

    @classmethod
    def medium(cls, vocab_size: int, storage_dir: str = "./aesg_memory") -> "AESGGRUText":
        """Create a medium GRU text model (hidden=256, 2 layers, vector_dim=128)."""
        return cls._from_size("medium", vocab_size, storage_dir)

    @classmethod
    def large(cls, vocab_size: int, storage_dir: str = "./aesg_memory") -> "AESGGRUText":
        """Create a large GRU text model (hidden=512, 4 layers, vector_dim=256)."""
        return cls._from_size("large", vocab_size, storage_dir)

    @classmethod
    def _from_size(cls, size: str, vocab_size: int, storage_dir: str) -> "AESGGRUText":
        """Internal factory that builds an AESGGRUText from a size preset.

        Parameters
        ----------
        size : str
            One of "small", "medium", "large".
        vocab_size : int
            Vocabulary size. Must be >= 1.
        storage_dir : str
            Directory for AESG memory storage.

        Returns
        -------
        AESGGRUText
            A fully constructed model instance.

        Raises
        ------
        AESGConfigError
            If vocab_size < 1.
        """
        if vocab_size < 1:
            raise AESGConfigError("vocab_size must be >= 1")
        hidden_size, num_layers, vector_dim, _ = TEXT_SIZES[size]
        config = AESGConfig(vector_dim=vector_dim, max_concepts=100_000)
        memory = AESGMemory(storage_dir, config)
        return cls(vocab_size, hidden_size, num_layers, memory)


class AESGLSTMText(nn.Module):
    """LSTM-based text model with AESG memory integration.

    Processes sequences of token IDs through an embedding layer,
    then feeds each timestep through an AESG_LSTM cell that incorporates
    semantic memory retrieval at each step. Maintains both hidden state
    and cell state across timesteps.

    Parameters
    ----------
    vocab_size : int
        Size of the token vocabulary.
    hidden_size : int
        Dimensionality of the LSTM hidden state.
    num_layers : int
        Number of stacked LSTM layers.
    memory : AESGMemory
        The AESG memory controller instance.
    """

    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int,
                 memory: AESGMemory):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.memory = memory
        self.lstm = AESG_LSTM(hidden_size, hidden_size, memory)
        self.output_proj = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the LSTM text model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_length) containing
            integer token IDs.

        Returns
        -------
        torch.Tensor
            Logits of shape (batch_size, seq_length, vocab_size).
        """
        emb = self.embedding(x)  # (B, S, H)
        B, S, H = emb.shape
        h = torch.zeros(B, H, device=x.device)
        c = torch.zeros(B, H, device=x.device)
        outputs = []
        for t in range(S):
            h, c = self.lstm(emb[:, t, :], (h, c))
            outputs.append(h)
        out = torch.stack(outputs, dim=1)  # (B, S, H)
        return self.output_proj(out)  # (B, S, vocab_size)

    @classmethod
    def small(cls, vocab_size: int, storage_dir: str = "./aesg_memory") -> "AESGLSTMText":
        """Create a small LSTM text model (hidden=128, 1 layer, vector_dim=64)."""
        return cls._from_size("small", vocab_size, storage_dir)

    @classmethod
    def medium(cls, vocab_size: int, storage_dir: str = "./aesg_memory") -> "AESGLSTMText":
        """Create a medium LSTM text model (hidden=256, 2 layers, vector_dim=128)."""
        return cls._from_size("medium", vocab_size, storage_dir)

    @classmethod
    def large(cls, vocab_size: int, storage_dir: str = "./aesg_memory") -> "AESGLSTMText":
        """Create a large LSTM text model (hidden=512, 4 layers, vector_dim=256)."""
        return cls._from_size("large", vocab_size, storage_dir)

    @classmethod
    def _from_size(cls, size: str, vocab_size: int, storage_dir: str) -> "AESGLSTMText":
        """Internal factory that builds an AESGLSTMText from a size preset.

        Parameters
        ----------
        size : str
            One of "small", "medium", "large".
        vocab_size : int
            Vocabulary size. Must be >= 1.
        storage_dir : str
            Directory for AESG memory storage.

        Returns
        -------
        AESGLSTMText
            A fully constructed model instance.

        Raises
        ------
        AESGConfigError
            If vocab_size < 1.
        """
        if vocab_size < 1:
            raise AESGConfigError("vocab_size must be >= 1")
        hidden_size, num_layers, vector_dim, _ = TEXT_SIZES[size]
        config = AESGConfig(vector_dim=vector_dim, max_concepts=100_000)
        memory = AESGMemory(storage_dir, config)
        return cls(vocab_size, hidden_size, num_layers, memory)


class AESGSeq2Seq(nn.Module):
    """GRU encoder-decoder text model with AESG memory in the decoder.

    The encoder is a standard nn.GRU that processes the input sequence
    and produces a final hidden state. The decoder uses an AESG_GRU cell
    that processes step-by-step using the encoder's last hidden state
    as the initial decoder state, with teacher-forcing during training.

    Parameters
    ----------
    vocab_size : int
        Size of the token vocabulary.
    hidden_size : int
        Dimensionality of the hidden states.
    num_layers : int
        Number of encoder GRU layers.
    memory : AESGMemory
        The AESG memory controller instance.
    """

    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int,
                 memory: AESGMemory):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.memory = memory
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Standard GRU encoder (no AESG)
        self.encoder = nn.GRU(hidden_size, hidden_size, num_layers=num_layers,
                              batch_first=True)

        # AESG-enhanced decoder
        self.decoder_cell = AESG_GRU(hidden_size, hidden_size, memory)

        # Output projection
        self.output_proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with teacher-forcing.

        The encoder processes the full input sequence. The decoder then
        generates outputs step-by-step, using its own projected output
        as the next input token embedding (teacher-forcing with own output).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_length) containing
            integer token IDs.

        Returns
        -------
        torch.Tensor
            Logits of shape (batch_size, seq_length, vocab_size).
        """
        emb = self.embedding(x)  # (B, S, H)
        B, S, H = emb.shape

        # Encode
        _, enc_hidden = self.encoder(emb)  # enc_hidden: (num_layers, B, H)
        # Use last layer hidden state as initial decoder state
        h = enc_hidden[-1]  # (B, H)

        # Decode step-by-step with teacher forcing (use own output as next input)
        outputs = []
        dec_input = emb[:, 0, :]  # Start with first token embedding
        for t in range(S):
            h = self.decoder_cell(dec_input, h)
            outputs.append(h)
            # Use projected hidden as next step input (teacher-forcing with own output)
            if t < S - 1:
                dec_input = emb[:, t + 1, :]  # teacher forcing uses ground truth
        out = torch.stack(outputs, dim=1)  # (B, S, H)
        return self.output_proj(out)  # (B, S, vocab_size)

    @classmethod
    def small(cls, vocab_size: int, storage_dir: str = "./aesg_memory") -> "AESGSeq2Seq":
        """Create a small Seq2Seq model (hidden=128, 1 layer, vector_dim=64)."""
        return cls._from_size("small", vocab_size, storage_dir)

    @classmethod
    def medium(cls, vocab_size: int, storage_dir: str = "./aesg_memory") -> "AESGSeq2Seq":
        """Create a medium Seq2Seq model (hidden=256, 2 layers, vector_dim=128)."""
        return cls._from_size("medium", vocab_size, storage_dir)

    @classmethod
    def large(cls, vocab_size: int, storage_dir: str = "./aesg_memory") -> "AESGSeq2Seq":
        """Create a large Seq2Seq model (hidden=512, 4 layers, vector_dim=256)."""
        return cls._from_size("large", vocab_size, storage_dir)

    @classmethod
    def _from_size(cls, size: str, vocab_size: int, storage_dir: str) -> "AESGSeq2Seq":
        """Internal factory that builds an AESGSeq2Seq from a size preset.

        Parameters
        ----------
        size : str
            One of "small", "medium", "large".
        vocab_size : int
            Vocabulary size. Must be >= 1.
        storage_dir : str
            Directory for AESG memory storage.

        Returns
        -------
        AESGSeq2Seq
            A fully constructed model instance.

        Raises
        ------
        AESGConfigError
            If vocab_size < 1.
        """
        if vocab_size < 1:
            raise AESGConfigError("vocab_size must be >= 1")
        hidden_size, num_layers, vector_dim, _ = TEXT_SIZES[size]
        config = AESGConfig(vector_dim=vector_dim, max_concepts=100_000)
        memory = AESGMemory(storage_dir, config)
        return cls(vocab_size, hidden_size, num_layers, memory)


class AESGDecoderLM(nn.Module):
    """Decoder-only language model with AESG memory.

    A single AESG_GRU cell processes tokens autoregressively from left
    to right, incorporating semantic memory retrieval at each step.
    Suitable for causal language modeling tasks.

    Parameters
    ----------
    vocab_size : int
        Size of the token vocabulary.
    hidden_size : int
        Dimensionality of the GRU hidden state.
    num_layers : int
        Number of stacked layers (stored for metadata; single cell used).
    memory : AESGMemory
        The AESG memory controller instance.
    """

    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int,
                 memory: AESGMemory):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.memory = memory
        self.gru = AESG_GRU(hidden_size, hidden_size, memory)
        self.output_proj = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the decoder-only language model.

        Processes each token left-to-right autoregressively, with the
        AESG_GRU cell retrieving relevant memory at each step.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_length) containing
            integer token IDs.

        Returns
        -------
        torch.Tensor
            Logits of shape (batch_size, seq_length, vocab_size).
        """
        emb = self.embedding(x)  # (B, S, H)
        B, S, H = emb.shape
        h = torch.zeros(B, H, device=x.device)
        outputs = []
        for t in range(S):
            h = self.gru(emb[:, t, :], h)
            outputs.append(h)
        out = torch.stack(outputs, dim=1)  # (B, S, H)
        return self.output_proj(out)  # (B, S, vocab_size)

    @classmethod
    def small(cls, vocab_size: int, storage_dir: str = "./aesg_memory") -> "AESGDecoderLM":
        """Create a small decoder LM (hidden=128, 1 layer, vector_dim=64)."""
        return cls._from_size("small", vocab_size, storage_dir)

    @classmethod
    def medium(cls, vocab_size: int, storage_dir: str = "./aesg_memory") -> "AESGDecoderLM":
        """Create a medium decoder LM (hidden=256, 2 layers, vector_dim=128)."""
        return cls._from_size("medium", vocab_size, storage_dir)

    @classmethod
    def large(cls, vocab_size: int, storage_dir: str = "./aesg_memory") -> "AESGDecoderLM":
        """Create a large decoder LM (hidden=512, 4 layers, vector_dim=256)."""
        return cls._from_size("large", vocab_size, storage_dir)

    @classmethod
    def _from_size(cls, size: str, vocab_size: int, storage_dir: str) -> "AESGDecoderLM":
        """Internal factory that builds an AESGDecoderLM from a size preset.

        Parameters
        ----------
        size : str
            One of "small", "medium", "large".
        vocab_size : int
            Vocabulary size. Must be >= 1.
        storage_dir : str
            Directory for AESG memory storage.

        Returns
        -------
        AESGDecoderLM
            A fully constructed model instance.

        Raises
        ------
        AESGConfigError
            If vocab_size < 1.
        """
        if vocab_size < 1:
            raise AESGConfigError("vocab_size must be >= 1")
        hidden_size, num_layers, vector_dim, _ = TEXT_SIZES[size]
        config = AESGConfig(vector_dim=vector_dim, max_concepts=100_000)
        memory = AESGMemory(storage_dir, config)
        return cls(vocab_size, hidden_size, num_layers, memory)


class AESGTransformer(nn.Module):
    """Transformer text model with AESG memory via cross-attention.

    Uses learned positional embeddings and a stack of AESG_TransformerLayer
    instances that integrate semantic memory through cross-attention.
    Each layer performs self-attention over the sequence, then cross-attends
    to retrieved AESG memory concepts.

    Parameters
    ----------
    vocab_size : int
        Size of the token vocabulary.
    hidden_size : int
        Model dimensionality (d_model).
    num_layers : int
        Number of stacked AESG_TransformerLayer instances.
    nhead : int
        Number of attention heads.
    memory : AESGMemory
        The AESG memory controller instance.
    max_seq_len : int, optional
        Maximum sequence length for positional encoding. Default is 512.
    """

    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int,
                 nhead: int, memory: AESGMemory, max_seq_len: int = 512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_size)
        self.memory = memory
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Stack of AESG transformer layers
        self.layers = nn.ModuleList([
            AESG_TransformerLayer(hidden_size, nhead, memory)
            for _ in range(num_layers)
        ])

        self.output_proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Transformer text model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_length) containing
            integer token IDs.

        Returns
        -------
        torch.Tensor
            Logits of shape (batch_size, seq_length, vocab_size).
        """
        B, S = x.shape
        positions = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)

        # Token embedding + positional embedding
        out = self.embedding(x) + self.pos_embedding(positions)

        # Pass through transformer layers with AESG cross-attention
        for layer in self.layers:
            out = layer(out)

        return self.output_proj(out)  # (B, S, vocab_size)

    @classmethod
    def small(cls, vocab_size: int, storage_dir: str = "./aesg_memory") -> "AESGTransformer":
        """Create a small Transformer (hidden=128, 1 layer, 4 heads, vector_dim=64)."""
        return cls._from_size("small", vocab_size, storage_dir)

    @classmethod
    def medium(cls, vocab_size: int, storage_dir: str = "./aesg_memory") -> "AESGTransformer":
        """Create a medium Transformer (hidden=256, 2 layers, 8 heads, vector_dim=128)."""
        return cls._from_size("medium", vocab_size, storage_dir)

    @classmethod
    def large(cls, vocab_size: int, storage_dir: str = "./aesg_memory") -> "AESGTransformer":
        """Create a large Transformer (hidden=512, 4 layers, 16 heads, vector_dim=256)."""
        return cls._from_size("large", vocab_size, storage_dir)

    @classmethod
    def _from_size(cls, size: str, vocab_size: int, storage_dir: str) -> "AESGTransformer":
        """Internal factory that builds an AESGTransformer from a size preset.

        Parameters
        ----------
        size : str
            One of "small", "medium", "large".
        vocab_size : int
            Vocabulary size. Must be >= 1.
        storage_dir : str
            Directory for AESG memory storage.

        Returns
        -------
        AESGTransformer
            A fully constructed model instance.

        Raises
        ------
        AESGConfigError
            If vocab_size < 1.
        """
        if vocab_size < 1:
            raise AESGConfigError("vocab_size must be >= 1")
        hidden_size, num_layers, vector_dim, nhead = TEXT_SIZES[size]
        config = AESGConfig(vector_dim=vector_dim, max_concepts=100_000)
        memory = AESGMemory(storage_dir, config)
        return cls(vocab_size, hidden_size, num_layers, nhead, memory)
