"""Safetensors-based weight serialization (no pickle).

Safetensors ``save()`` returns raw bytes and ``load()`` reads raw bytes, which
makes them a natural fit for the HTTP wire format (FR-PROTO-4).

This module is duplicated verbatim in ``client/app/weights.py``.

Reference: https://huggingface.co/docs/safetensors/main/en/api/torch
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Union

import torch
from safetensors.torch import load as st_load
from safetensors.torch import load_file, save as st_save, save_file


def state_dict_to_bytes(state_dict: Dict[str, torch.Tensor]) -> bytes:
    """Serialize a state_dict to safetensors bytes (CPU, contiguous, detached)."""
    clean = {k: v.detach().to("cpu").contiguous() for k, v in state_dict.items()}
    return st_save(clean)


def bytes_to_state_dict(data: bytes) -> Dict[str, torch.Tensor]:
    """Deserialize safetensors bytes into a CPU state_dict."""
    return st_load(data)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_state_dict(state_dict: Dict[str, torch.Tensor], path: Union[str, Path]) -> None:
    clean = {k: v.detach().to("cpu").contiguous() for k, v in state_dict.items()}
    save_file(clean, str(path))


def load_state_dict(path: Union[str, Path]) -> Dict[str, torch.Tensor]:
    return load_file(str(path))
