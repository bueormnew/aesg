"""
AESG Image Architecture Factories.

Provides pre-assembled CNN architectures with AESG memory injection at bottleneck:
- AESGColorizationNet: Grayscale (1ch) → Color (3ch)
- AESGCNNClassifier: Image classification with configurable num_classes
- AESGCNNAutoencoder: Symmetric encoder-decoder for image reconstruction

Each architecture exposes .small(), .medium(), .large() factory methods that
auto-create the AESGConfig and AESGMemory, returning a model ready for training.
"""

import torch
import torch.nn as nn
from typing import Tuple

from aesg.config import AESGConfig
from aesg.memory.controller import AESGMemory
from aesg.nn.cnn import AESG_CNNLayer
from aesg.exceptions import AESGConfigError

# Size presets: (base_filters, num_blocks, vector_dim, bottleneck_size)
# bottleneck_size is the spatial dimension (H=W) at the AESG injection point.
IMAGE_SIZES = {
    "small": (32, 3, 64, 16),
    "medium": (64, 4, 128, 8),
    "large": (128, 5, 256, 4),
}

_VALID_SIZES = list(IMAGE_SIZES.keys())


def _validate_size(size: str) -> Tuple[int, int, int, int]:
    """Validate size name and return preset tuple.

    Raises AESGConfigError if size is not one of 'small', 'medium', 'large'.
    """
    if size not in IMAGE_SIZES:
        raise AESGConfigError(
            f"Invalid image architecture size '{size}'. "
            f"Valid sizes: {_VALID_SIZES}"
        )
    return IMAGE_SIZES[size]


def _make_memory(vector_dim: int, storage_dir: str) -> AESGMemory:
    """Create AESGConfig and AESGMemory for image architectures."""
    config = AESGConfig(vector_dim=vector_dim, max_concepts=100_000)
    return AESGMemory(storage_dir, config)


class AESGColorizationNet(nn.Module):
    """Grayscale (1ch) → Color (3ch) with AESG memory at bottleneck.

    Architecture:
        Encoder: Conv2d+BatchNorm+ReLU+MaxPool2d blocks (1ch → base_filters*2^n)
        Bottleneck: AESG_CNNLayer for semantic memory injection
        Decoder: ConvTranspose2d+BatchNorm+ReLU blocks (→ 3ch with sigmoid)

    Parameters
    ----------
    base_filters : int
        Number of filters in the first encoder block. Doubles each block.
    num_blocks : int
        Number of encoder/decoder blocks.
    memory : AESGMemory
        Pre-configured AESG memory instance.
    bottleneck_size : int
        Spatial dimension (H=W) of feature maps at the bottleneck.
    """

    def __init__(self, base_filters: int, num_blocks: int,
                 memory: AESGMemory, bottleneck_size: int):
        super().__init__()
        self.memory = memory
        self.bottleneck_size = bottleneck_size

        # Build encoder: 1ch input → progressively deeper feature maps
        encoder_layers = []
        in_ch = 1
        for i in range(num_blocks):
            out_ch = base_filters * (2 ** i)
            encoder_layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ])
            in_ch = out_ch
        self.encoder = nn.Sequential(*encoder_layers)

        # AESG injection at bottleneck
        bottleneck_channels = base_filters * (2 ** (num_blocks - 1))
        self.aesg_layer = AESG_CNNLayer(
            bottleneck_channels, bottleneck_channels, memory, bottleneck_size
        )

        # Build decoder: upscale back to original resolution, output 3 channels
        decoder_layers = []
        in_ch = bottleneck_channels
        for i in range(num_blocks - 1, 0, -1):
            out_ch = base_filters * (2 ** (i - 1))
            decoder_layers.extend([
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ])
            in_ch = out_ch
        # Final upscale to original resolution, output 3 channels
        decoder_layers.append(
            nn.ConvTranspose2d(in_ch, 3, kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(*decoder_layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: grayscale → color.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, 1, H, W).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, 3, H, W) with values in [0, 1].
        """
        enc = self.encoder(x)
        mem_out = self.aesg_layer(enc)
        dec = self.decoder(mem_out)
        return self.sigmoid(dec)

    @classmethod
    def small(cls, storage_dir: str = "./aesg_memory") -> "AESGColorizationNet":
        """Create a small colorization model (32 base filters, 3 blocks)."""
        return cls._from_size("small", storage_dir)

    @classmethod
    def medium(cls, storage_dir: str = "./aesg_memory") -> "AESGColorizationNet":
        """Create a medium colorization model (64 base filters, 4 blocks)."""
        return cls._from_size("medium", storage_dir)

    @classmethod
    def large(cls, storage_dir: str = "./aesg_memory") -> "AESGColorizationNet":
        """Create a large colorization model (128 base filters, 5 blocks)."""
        return cls._from_size("large", storage_dir)

    @classmethod
    def _from_size(cls, size: str, storage_dir: str) -> "AESGColorizationNet":
        base_filters, num_blocks, vector_dim, bottleneck_size = _validate_size(size)
        memory = _make_memory(vector_dim, storage_dir)
        return cls(base_filters, num_blocks, memory, bottleneck_size)


class AESGCNNClassifier(nn.Module):
    """Image classification with AESG memory at bottleneck.

    Architecture:
        Encoder: Conv2d+BatchNorm+ReLU+MaxPool2d blocks
        Bottleneck: AESG_CNNLayer for semantic memory injection
        Head: AdaptiveAvgPool2d → Flatten → Linear → Logits

    Parameters
    ----------
    base_filters : int
        Number of filters in the first encoder block. Doubles each block.
    num_blocks : int
        Number of encoder blocks.
    memory : AESGMemory
        Pre-configured AESG memory instance.
    bottleneck_size : int
        Spatial dimension (H=W) of feature maps at the bottleneck.
    num_classes : int
        Number of output classes.
    """

    def __init__(self, base_filters: int, num_blocks: int,
                 memory: AESGMemory, bottleneck_size: int,
                 num_classes: int = 10):
        super().__init__()
        self.memory = memory
        self.bottleneck_size = bottleneck_size
        self.num_classes = num_classes

        # Build encoder: 3ch input → progressively deeper feature maps
        encoder_layers = []
        in_ch = 3
        for i in range(num_blocks):
            out_ch = base_filters * (2 ** i)
            encoder_layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ])
            in_ch = out_ch
        self.encoder = nn.Sequential(*encoder_layers)

        # AESG injection at bottleneck
        bottleneck_channels = base_filters * (2 ** (num_blocks - 1))
        self.aesg_layer = AESG_CNNLayer(
            bottleneck_channels, bottleneck_channels, memory, bottleneck_size
        )

        # Classification head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(bottleneck_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: image → class logits.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, 3, H, W).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, num_classes) with raw logits.
        """
        enc = self.encoder(x)
        mem_out = self.aesg_layer(enc)
        pooled = self.pool(mem_out)
        flat = self.flatten(pooled)
        return self.fc(flat)

    @classmethod
    def small(cls, num_classes: int = 10,
              storage_dir: str = "./aesg_memory") -> "AESGCNNClassifier":
        """Create a small classifier (32 base filters, 3 blocks)."""
        return cls._from_size("small", num_classes, storage_dir)

    @classmethod
    def medium(cls, num_classes: int = 10,
               storage_dir: str = "./aesg_memory") -> "AESGCNNClassifier":
        """Create a medium classifier (64 base filters, 4 blocks)."""
        return cls._from_size("medium", num_classes, storage_dir)

    @classmethod
    def large(cls, num_classes: int = 10,
              storage_dir: str = "./aesg_memory") -> "AESGCNNClassifier":
        """Create a large classifier (128 base filters, 5 blocks)."""
        return cls._from_size("large", num_classes, storage_dir)

    @classmethod
    def _from_size(cls, size: str, num_classes: int,
                   storage_dir: str) -> "AESGCNNClassifier":
        base_filters, num_blocks, vector_dim, bottleneck_size = _validate_size(size)
        memory = _make_memory(vector_dim, storage_dir)
        return cls(base_filters, num_blocks, memory, bottleneck_size, num_classes)


class AESGCNNAutoencoder(nn.Module):
    """Symmetric encoder-decoder with AESG memory at bottleneck.

    Architecture:
        Encoder: Conv2d+BatchNorm+ReLU+MaxPool2d blocks
        Bottleneck: AESG_CNNLayer for semantic memory injection
        Decoder: ConvTranspose2d+BatchNorm+ReLU blocks, sigmoid output

    Parameters
    ----------
    base_filters : int
        Number of filters in the first encoder block. Doubles each block.
    num_blocks : int
        Number of encoder/decoder blocks.
    memory : AESGMemory
        Pre-configured AESG memory instance.
    bottleneck_size : int
        Spatial dimension (H=W) of feature maps at the bottleneck.
    in_channels : int
        Number of input image channels (default 3 for RGB).
    """

    def __init__(self, base_filters: int, num_blocks: int,
                 memory: AESGMemory, bottleneck_size: int,
                 in_channels: int = 3):
        super().__init__()
        self.memory = memory
        self.bottleneck_size = bottleneck_size
        self.in_channels = in_channels

        # Build encoder
        encoder_layers = []
        in_ch = in_channels
        for i in range(num_blocks):
            out_ch = base_filters * (2 ** i)
            encoder_layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ])
            in_ch = out_ch
        self.encoder = nn.Sequential(*encoder_layers)

        # AESG injection at bottleneck
        bottleneck_channels = base_filters * (2 ** (num_blocks - 1))
        self.aesg_layer = AESG_CNNLayer(
            bottleneck_channels, bottleneck_channels, memory, bottleneck_size
        )

        # Build decoder: symmetric to encoder
        decoder_layers = []
        in_ch = bottleneck_channels
        for i in range(num_blocks - 1, 0, -1):
            out_ch = base_filters * (2 ** (i - 1))
            decoder_layers.extend([
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ])
            in_ch = out_ch
        # Final upscale to original resolution, output same number of channels
        decoder_layers.append(
            nn.ConvTranspose2d(in_ch, in_channels, kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(*decoder_layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: image → reconstructed image.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, C, H, W) with values in [0, 1].
        """
        enc = self.encoder(x)
        mem_out = self.aesg_layer(enc)
        dec = self.decoder(mem_out)
        return self.sigmoid(dec)

    @classmethod
    def small(cls, storage_dir: str = "./aesg_memory") -> "AESGCNNAutoencoder":
        """Create a small autoencoder (32 base filters, 3 blocks)."""
        return cls._from_size("small", storage_dir)

    @classmethod
    def medium(cls, storage_dir: str = "./aesg_memory") -> "AESGCNNAutoencoder":
        """Create a medium autoencoder (64 base filters, 4 blocks)."""
        return cls._from_size("medium", storage_dir)

    @classmethod
    def large(cls, storage_dir: str = "./aesg_memory") -> "AESGCNNAutoencoder":
        """Create a large autoencoder (128 base filters, 5 blocks)."""
        return cls._from_size("large", storage_dir)

    @classmethod
    def _from_size(cls, size: str, storage_dir: str) -> "AESGCNNAutoencoder":
        base_filters, num_blocks, vector_dim, bottleneck_size = _validate_size(size)
        memory = _make_memory(vector_dim, storage_dir)
        return cls(base_filters, num_blocks, memory, bottleneck_size)
