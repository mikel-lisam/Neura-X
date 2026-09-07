# ==========================================================
# Neura-X: Intelligence Without Limits.
# Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
# Model Conversion
# ==========================================================

"""Model conversion utilities for Neura-X.

Implements dependency-free readers for several common tensor formats. The
converters read the source artifact, decode any metadata they can, and
return a ``dict`` describing the converted representation. They do not
require network access, ONNX runtime, or PyTorch.
"""

import gzip
import os
import struct
from typing import Any, Dict, Optional


def _read_safetensors_header(path: str) -> Dict[str, Any]:
    """Parse the leading 8-byte little-endian length, then JSON header of a
    safetensors file. Returns the parsed JSON dictionary or an error dict.
    """
    try:
        with open(path, "rb") as f:
            header_len_bytes = f.read(8)
            if len(header_len_bytes) != 8:
                return {"error": "truncated safetensors header length"}
            (header_len,) = struct.unpack("<Q", header_len_bytes)
            header_bytes = f.read(header_len)
            if len(header_bytes) != header_len:
                return {"error": "truncated safetensors header"}
            import json
            return json.loads(header_bytes.decode("utf-8"))
    except Exception as e:
        return {"error": f"safetensors parse failed: {e}"}


class _ConvertNamespace:
    """Namespace for nx.convert.* access."""

    @staticmethod
    def from_pytorch(path: str) -> Optional[Dict[str, Any]]:
        """Inspect a PyTorch ``.pt/.pth/.bin`` file.

        PyTorch's pickle format requires the ``torch`` package to load
        fully; without it, we report what we can learn from the file
        header (magic bytes, file size, presence of ZIP members for
        ``.bin`` archives) and return a description.
        """
        if not os.path.exists(path):
            print(f"[Neura-X Convert] PyTorch file not found: {path}")
            return None
        size = os.path.getsize(path)
        info: Dict[str, Any] = {
            "source": path,
            "format": "pytorch",
            "size_bytes": size,
        }
        with open(path, "rb") as f:
            head = f.read(8)
        info["magic_hex"] = head.hex()
        # ZIP archive (.bin zip-packaged weights) starts with "PK\x03\x04".
        if head.startswith(b"PK\x03\x04"):
            info["packaging"] = "zip"
        print(f"[Neura-X Convert] Inspected PyTorch model: {path}")
        return info

    @staticmethod
    def from_gguf(path: str) -> Optional[Dict[str, Any]]:
        """Inspect a GGUF model file and return its header fields.

        GGUF starts with the magic bytes ``GGUF`` followed by a version
        uint32, tensor count uint64, and metadata kv count uint64. We
        parse the header without materialising any tensor data.
        """
        if not os.path.exists(path):
            print(f"[Neura-X Convert] GGUF file not found: {path}")
            return None
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                return {"source": path, "error": "missing GGUF magic"}
            (version,) = struct.unpack("<I", f.read(4))
            (tensor_count,) = struct.unpack("<Q", f.read(8))
            (metadata_kv_count,) = struct.unpack("<Q", f.read(8))
        info = {
            "source": path,
            "format": "gguf",
            "version": version,
            "tensor_count": tensor_count,
            "metadata_kv_count": metadata_kv_count,
        }
        print(f"[Neura-X Convert] Inspected GGUF model: {path}")
        return info

    @staticmethod
    def from_huggingface(model_id: str) -> Optional[Dict[str, Any]]:
        """Inspect a local HuggingFace cache snapshot.

        Looks for ``<cache>/models--<id>/snapshots/<rev>/`` in the
        standard HF cache directory. If found, lists the snapshot
        contents. No network call is made.
        """
        if "/" not in model_id:
            print(f"[Neura-X Convert] HF model_id must be 'org/name': {model_id}")
            return None
        cache = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
        target_dirname = "models--" + model_id.replace("/", "--")
        target = os.path.join(cache, target_dirname, "snapshots")
        info: Dict[str, Any] = {"source": model_id, "format": "huggingface"}
        if os.path.isdir(target):
            snapshots = sorted(os.listdir(target))
            if snapshots:
                snap = os.path.join(target, snapshots[-1])
                files = sorted(os.listdir(snap))
                info["snapshot"] = snap
                info["files"] = files
        print(f"[Neura-X Convert] Inspected HuggingFace model: {model_id}")
        return info

    @staticmethod
    def from_safetensors(path: str) -> Optional[Dict[str, Any]]:
        """Parse the header of a safetensors file and return its tensor map."""
        if not os.path.exists(path):
            print(f"[Neura-X Convert] safetensors file not found: {path}")
            return None
        info = _read_safetensors_header(path)
        info["source"] = path
        info["format"] = "safetensors"
        print(f"[Neura-X Convert] Inspected safetensors model: {path}")
        return info


convert = _ConvertNamespace()