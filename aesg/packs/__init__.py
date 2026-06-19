"""
AESG Pack System.

Provides modular memory packs with the .aesgpack binary format
for domain specialization and portable knowledge transfer.
"""

from aesg.packs.format import (
    AESGPACK_MAGIC,
    AESGPACK_VERSION,
    serialize_pack,
    deserialize_pack,
)
from aesg.packs.manager import (
    PackState,
    MemoryPack,
    PackManager,
    MAX_ATTACHED_PACKS,
)
from aesg.packs.export import export_pack

__all__ = [
    "AESGPACK_MAGIC",
    "AESGPACK_VERSION",
    "serialize_pack",
    "deserialize_pack",
    "PackState",
    "MemoryPack",
    "PackManager",
    "MAX_ATTACHED_PACKS",
    "export_pack",
]
