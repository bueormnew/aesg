"""
AESG: Adaptive External Semantic Graph — V3 Public API.

Usage::

    from aesg import AESGTransformer, AESGTrainer
    model = AESGTransformer.small(vocab_size=1000)
    trainer = AESGTrainer(model)
    trainer.fit(train_data, epochs=5)
"""

__version__ = "3.0.0"
__author__ = "bueormnew"

# Core
from aesg.config import AESGConfig
from aesg.memory.controller import AESGMemory
from aesg.memory.modes import MemoryMode

# Exceptions
from aesg.exceptions import (
    AESGError,
    AESGConfigError,
    AESGMemoryError,
    AESGNavigationError,
    AESGTrainingError,
    AESGStorageError,
)

# Validation
from aesg.validation import Validator

# Text Architectures
from aesg.architectures.text import (
    AESGGRUText,
    AESGLSTMText,
    AESGSeq2Seq,
    AESGDecoderLM,
    AESGTransformer,
)

# Image Architectures
from aesg.architectures.image import (
    AESGCNNClassifier,
    AESGColorizationNet,
    AESGCNNAutoencoder,
)

# Trainer
from aesg.trainer.trainer import AESGTrainer

# Callbacks
from aesg.trainer.callbacks import (
    Callback,
    EarlyStopping,
    Checkpoint,
    TensorBoardCallback,
    WandBCallback,
)

# Evaluation
from aesg.evaluation import Evaluator

# Packs
from aesg.packs import PackManager, MemoryPack

# Data
from aesg.data import adapt_data

# Benchmarks
from aesg.benchmarks import ColorizationBenchmark

__all__ = [
    # Metadata
    "__version__",
    "__author__",
    # Core
    "AESGConfig",
    "AESGMemory",
    "MemoryMode",
    # Exceptions
    "AESGError",
    "AESGConfigError",
    "AESGMemoryError",
    "AESGNavigationError",
    "AESGTrainingError",
    "AESGStorageError",
    # Validation
    "Validator",
    # Text Architectures
    "AESGGRUText",
    "AESGLSTMText",
    "AESGSeq2Seq",
    "AESGDecoderLM",
    "AESGTransformer",
    # Image Architectures
    "AESGCNNClassifier",
    "AESGColorizationNet",
    "AESGCNNAutoencoder",
    # Trainer
    "AESGTrainer",
    # Callbacks
    "Callback",
    "EarlyStopping",
    "Checkpoint",
    "TensorBoardCallback",
    "WandBCallback",
    # Evaluation
    "Evaluator",
    # Packs
    "PackManager",
    "MemoryPack",
    # Data
    "adapt_data",
    # Benchmarks
    "ColorizationBenchmark",
]
