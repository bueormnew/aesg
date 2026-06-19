"""
AESG Pack Binary Format (.aesgpack)

Binary layout:
  Offset  Size     Field
  0       8        Magic bytes: "AESGPACK"
  8       4        Format version (uint32)
  12      4        vector_dim (uint32)
  16      4        node_count (uint32)
  20      4        edge_count (uint32)
  24      32       SHA-256 checksum of payload
  56      variable Node data (node_count * node_dtype_size)
  ...     variable Edge data (edge_count * edge_dtype_size)
"""

import struct
import hashlib
import os
from typing import Dict, Any

import numpy as np

from aesg.exceptions import AESGStorageError

# Format constants
AESGPACK_MAGIC = b"AESGPACK"  # 8 bytes
AESGPACK_VERSION = 1          # uint32, 4 bytes

# Header size: magic(8) + version(4) + vector_dim(4) + node_count(4) + edge_count(4) + checksum(32) = 56 bytes
HEADER_SIZE = 56
HEADER_STRUCT = struct.Struct(">8sIIII32s")  # big-endian: 8s magic, I version, I vector_dim, I node_count, I edge_count, 32s checksum


def _make_node_dtype(vector_dim: int) -> np.dtype:
    """Construct the node structured array dtype for a given vector_dim.

    Matches AESGStorage.node_dtype exactly.
    """
    return np.dtype([
        ('id', np.uint64),
        ('vector', np.float32, (vector_dim,)),
        ('created_at', np.uint64),
        ('modified_at', np.uint64),
        ('use_frequency', np.uint32),
        ('relevance', np.float32),
        ('age', np.uint32),
        ('stability', np.float32),
        ('region_id', np.uint32),
        ('is_active', np.uint8),
        ('head_edge_idx', np.int64),
    ])


def _make_edge_dtype() -> np.dtype:
    """Construct the edge structured array dtype.

    Matches AESGStorage.edge_dtype exactly.
    """
    return np.dtype([
        ('source_id', np.uint64),
        ('target_id', np.uint64),
        ('weight', np.float32),
        ('confidence', np.float32),
        ('use_count', np.uint32),
        ('next_edge_idx', np.int64),
    ])


def serialize_pack(
    name: str,
    version: str,
    vector_dim: int,
    nodes_array: np.ndarray,
    edges_array: np.ndarray,
    path: str,
) -> None:
    """Serialize a memory pack to a .aesgpack binary file.

    Parameters
    ----------
    name : str
        Pack name (stored in filename, not in binary).
    version : str
        Pack version string (metadata only, not in binary header).
    vector_dim : int
        Dimensionality of node vectors.
    nodes_array : np.ndarray
        Structured numpy array with node_dtype fields.
    edges_array : np.ndarray
        Structured numpy array with edge_dtype fields.
    path : str
        Output file path for the .aesgpack file.

    Raises
    ------
    AESGStorageError
        If serialization fails due to I/O or data issues.
    """
    try:
        node_count = len(nodes_array)
        edge_count = len(edges_array)

        # Serialize payload (nodes bytes + edges bytes)
        nodes_bytes = nodes_array.tobytes()
        edges_bytes = edges_array.tobytes()
        payload = nodes_bytes + edges_bytes

        # Compute SHA-256 checksum of payload
        checksum = hashlib.sha256(payload).digest()

        # Pack header
        header = HEADER_STRUCT.pack(
            AESGPACK_MAGIC,
            AESGPACK_VERSION,
            vector_dim,
            node_count,
            edge_count,
            checksum,
        )

        # Ensure output directory exists
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # Write file
        with open(path, "wb") as f:
            f.write(header)
            f.write(payload)

    except (OSError, IOError) as e:
        raise AESGStorageError(f"Failed to serialize pack to '{path}': {e}") from e


def deserialize_pack(path: str) -> Dict[str, Any]:
    """Deserialize a .aesgpack binary file into its components.

    Parameters
    ----------
    path : str
        Path to the .aesgpack file.

    Returns
    -------
    dict
        Keys: name (str, from filename), vector_dim (int), nodes (np.ndarray),
        edges (np.ndarray), checksum (str, hex digest).

    Raises
    ------
    AESGStorageError
        If magic bytes are invalid, checksum verification fails, or file is unreadable.
    """
    try:
        with open(path, "rb") as f:
            # Read header
            header_bytes = f.read(HEADER_SIZE)
            if len(header_bytes) < HEADER_SIZE:
                raise AESGStorageError(
                    f"Pack file '{path}' is too small to contain a valid header"
                )

            magic, fmt_version, vector_dim, node_count, edge_count, stored_checksum = (
                HEADER_STRUCT.unpack(header_bytes)
            )

            # Validate magic bytes
            if magic != AESGPACK_MAGIC:
                raise AESGStorageError(
                    f"Invalid magic bytes in '{path}': expected {AESGPACK_MAGIC!r}, got {magic!r}"
                )

            # Read payload
            payload = f.read()

    except AESGStorageError:
        raise
    except (OSError, IOError) as e:
        raise AESGStorageError(f"Failed to read pack file '{path}': {e}") from e

    # Verify SHA-256 checksum
    computed_checksum = hashlib.sha256(payload).digest()
    if computed_checksum != stored_checksum:
        raise AESGStorageError(
            f"Checksum verification failed for '{path}': file is corrupt"
        )

    # Reconstruct dtypes from vector_dim
    node_dtype = _make_node_dtype(vector_dim)
    edge_dtype = _make_edge_dtype()

    # Calculate expected payload size
    expected_size = (node_count * node_dtype.itemsize) + (edge_count * edge_dtype.itemsize)
    if len(payload) < expected_size:
        raise AESGStorageError(
            f"Payload size mismatch in '{path}': expected {expected_size} bytes, got {len(payload)}"
        )

    # Split payload into nodes and edges
    nodes_size = node_count * node_dtype.itemsize
    nodes_bytes = payload[:nodes_size]
    edges_bytes = payload[nodes_size:nodes_size + (edge_count * edge_dtype.itemsize)]

    # Reconstruct structured arrays
    nodes = np.frombuffer(nodes_bytes, dtype=node_dtype).copy()
    edges = np.frombuffer(edges_bytes, dtype=edge_dtype).copy()

    # Derive name from filename (without extension)
    name = os.path.splitext(os.path.basename(path))[0]

    return {
        "name": name,
        "vector_dim": int(vector_dim),
        "nodes": nodes,
        "edges": edges,
        "checksum": computed_checksum.hex(),
    }
