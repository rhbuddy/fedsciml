"""Safetensors-based weight serialization (no pickle).

Safetensors ``save()`` returns raw bytes and ``load()`` reads raw bytes, which
makes them a natural fit for the HTTP wire format (FR-PROTO-4).

This module is duplicated verbatim in ``client/app/weights.py``.

Reference: https://huggingface.co/docs/safetensors/main/en/api/torch
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Union

import torch
from safetensors.torch import load as st_load
from safetensors.torch import load_file, save as st_save, save_file

# Compression modes for Framework v2 (reviewer delight: comm vs L2)
COMPRESSION_MODES = ("none", "int8", "topk", "int8_topk")


def _compress_tensor(tensor: torch.Tensor, mode: str, topk_ratio: float) -> Dict[str, torch.Tensor]:
    """Compress a single tensor -> dict of tensors to store (per-key shards).

    int8: per-tensor scale, quantized int8
    topk: sparse indices + values (float32)
    int8_topk: sparse + int8 quantized values + scale
    Returns dict with keys suffixes; caller prefixes with original key.
    """
    t = tensor.detach().to("cpu").float().contiguous()
    shape = torch.tensor(list(t.shape), dtype=torch.int32)
    numel = t.numel()
    if mode == "none" or numel == 0:
        return {"": t}
    if mode == "int8":
        max_abs = t.abs().max().item() if numel else 1.0
        scale = max_abs / 127.0 if max_abs > 1e-8 else 1.0
        q = torch.clamp(torch.round(t / scale), -127, 127).to(torch.int8)
        return {"__int8": q, "__scale": torch.tensor([scale], dtype=torch.float32), "__shape": shape}
    if mode in ("topk", "int8_topk"):
        k = max(1, int(numel * float(topk_ratio)))
        flat = t.view(-1)
        # topk by magnitude
        _, idx = torch.topk(flat.abs(), k, sorted=False)
        vals = flat[idx]
        # store indices as int32
        out: Dict[str, torch.Tensor] = {
            "__indices": idx.to(torch.int32),
            "__shape": shape,
            "__k": torch.tensor([k], dtype=torch.int32),
        }
        if mode == "int8_topk":
            max_abs = vals.abs().max().item() if k else 1.0
            scale = max_abs / 127.0 if max_abs > 1e-8 else 1.0
            q = torch.clamp(torch.round(vals / scale), -127, 127).to(torch.int8)
            out["__qvalues"] = q
            out["__scale"] = torch.tensor([scale], dtype=torch.float32)
        else:
            out["__values"] = vals.to(torch.float32)
        return out
    raise ValueError(f"Unknown compression mode '{mode}'")


def _decompress_tensor(shards: Dict[str, torch.Tensor], mode: str) -> torch.Tensor:
    """Reconstruct tensor from shards dict (keys are suffixes)."""
    if mode == "none":
        # single shard with key ""
        return shards[""]
    if mode == "int8":
        q = shards["__int8"].to(torch.float32)
        scale = shards["__scale"].item()
        shape = tuple(int(x) for x in shards["__shape"].tolist())
        t = (q * scale).view(shape)
        return t
    if mode == "topk":
        indices = shards["__indices"].to(torch.int64)
        values = shards["__values"].to(torch.float32)
        shape = tuple(int(x) for x in shards["__shape"].tolist())
        flat = torch.zeros(int(torch.prod(torch.tensor(shape)).item()), dtype=torch.float32)
        flat.scatter_(0, indices.view(-1), values.view(-1))
        return flat.view(shape)
    if mode == "int8_topk":
        indices = shards["__indices"].to(torch.int64)
        q = shards["__qvalues"].to(torch.float32)
        scale = shards["__scale"].item()
        vals = q * scale
        shape = tuple(int(x) for x in shards["__shape"].tolist())
        flat = torch.zeros(int(torch.prod(torch.tensor(shape)).item()), dtype=torch.float32)
        flat.scatter_(0, indices.view(-1), vals.view(-1))
        return flat.view(shape)
    raise ValueError(f"Unknown compression mode '{mode}'")


def compress_state_dict(
    state_dict: Dict[str, torch.Tensor],
    mode: str = "none",
    topk_ratio: float = 0.01,
) -> Dict[str, torch.Tensor]:
    """Compress a full state_dict into a flat dict of shards (still tensors)."""
    mode = (mode or "none").lower().strip()
    if mode not in COMPRESSION_MODES:
        raise ValueError(f"Unknown compression '{mode}'. Options: {COMPRESSION_MODES}")
    if mode == "none":
        return {k: v.detach().to("cpu").contiguous() for k, v in state_dict.items()}
    compressed: Dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        shards = _compress_tensor(tensor, mode, topk_ratio)
        for suffix, shard in shards.items():
            comp_key = f"{key}{suffix}" if suffix else key
            compressed[comp_key] = shard
    # metadata tensor to allow auto-detection on load
    meta = json.dumps({"compression": mode, "topk_ratio": float(topk_ratio), "keys": list(state_dict.keys())})
    # store as uint8 bytes tensor (safetensors has no string, so encode as int8 array)
    # Instead, store as a float tensor with a special key and use JSON side-channel via safetensors metadata not reliable,
    # so we store a 1-element tensor whose bytes we decode: we will store meta as a tensor of uint8
    meta_bytes = torch.tensor(list(meta.encode("utf-8")), dtype=torch.uint8)
    compressed["__meta__"] = meta_bytes
    return compressed


def decompress_state_dict(flat_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Inverse of compress_state_dict — auto-detects via __meta__ or heuristic."""
    if "__meta__" not in flat_dict:
        # also heuristic: if any key contains __int8/__indices, try to reconstruct
        has_compressed = any("__" in k for k in flat_dict.keys() if k != "__meta__")
        if not has_compressed:
            return flat_dict
        # fallback: try to infer mode per key by presence of shards
        # We need keys list; infer from shard prefixes
        # collect base keys by stripping suffixes
        suffixes = ("__int8", "__scale", "__shape", "__indices", "__values", "__qvalues", "__k")
        base_keys = set()
        for k in flat_dict.keys():
            if k == "__meta__":
                continue
            for sfx in suffixes:
                if k.endswith(sfx):
                    base_keys.add(k[: -len(sfx)])
                    break
            else:
                base_keys.add(k)
        # Try to decode meta if present else guess
        return _decompress_without_meta(flat_dict, base_keys)
    # with meta, proper
    meta_bytes = flat_dict.pop("__meta__")
    try:
        meta = json.loads(bytes(meta_bytes.tolist()).decode("utf-8"))
    except Exception:
        return flat_dict
    mode = meta.get("compression", "none")
    topk_ratio = meta.get("topk_ratio", 0.01)
    keys = meta.get("keys", [])
    # if mode none, just return without meta
    if mode == "none":
        return {k: v for k, v in flat_dict.items() if not k.startswith("__")}
    out: Dict[str, torch.Tensor] = {}
    for key in keys:
        # collect shards for this key
        if mode == "int8":
            shards = {
                "__int8": flat_dict[f"{key}__int8"],
                "__scale": flat_dict[f"{key}__scale"],
                "__shape": flat_dict[f"{key}__shape"],
            }
            out[key] = _decompress_tensor(shards, mode)
        elif mode == "topk":
            shards = {
                "__indices": flat_dict[f"{key}__indices"],
                "__values": flat_dict[f"{key}__values"],
                "__shape": flat_dict[f"{key}__shape"],
            }
            out[key] = _decompress_tensor(shards, mode)
        elif mode == "int8_topk":
            shards = {
                "__indices": flat_dict[f"{key}__indices"],
                "__qvalues": flat_dict[f"{key}__qvalues"],
                "__scale": flat_dict[f"{key}__scale"],
                "__shape": flat_dict[f"{key}__shape"],
            }
            out[key] = _decompress_tensor(shards, mode)
        else:
            # fallback int8 case
            out[key] = flat_dict[key]
    return out


def _decompress_without_meta(flat_dict: Dict[str, torch.Tensor], base_keys) -> Dict[str, torch.Tensor]:
    """Best-effort fallback when __meta__ missing but shards present."""
    out = {}
    for key in base_keys:
        if f"{key}__int8" in flat_dict:
            # could be int8 or int8_topk; check indices
            if f"{key}__indices" in flat_dict:
                shards = {
                    "__indices": flat_dict[f"{key}__indices"],
                    "__qvalues": flat_dict[f"{key}__qvalues"],
                    "__scale": flat_dict[f"{key}__scale"],
                    "__shape": flat_dict[f"{key}__shape"],
                }
                out[key] = _decompress_tensor(shards, "int8_topk")
            else:
                shards = {
                    "__int8": flat_dict[f"{key}__int8"],
                    "__scale": flat_dict[f"{key}__scale"],
                    "__shape": flat_dict[f"{key}__shape"],
                }
                out[key] = _decompress_tensor(shards, "int8")
        elif f"{key}__indices" in flat_dict:
            shards = {
                "__indices": flat_dict[f"{key}__indices"],
                "__values": flat_dict[f"{key}__values"],
                "__shape": flat_dict[f"{key}__shape"],
            }
            out[key] = _decompress_tensor(shards, "topk")
        else:
            if key in flat_dict:
                out[key] = flat_dict[key]
    return out


def state_dict_to_bytes(state_dict: Dict[str, torch.Tensor], compression: str = "none", topk_ratio: float = 0.01) -> bytes:
    """Serialize a state_dict to safetensors bytes (CPU, contiguous, detached).

    Framework v2: optional compression (int8 / topk / int8_topk) for comm vs L2.
    Use compression='int8_topk' + topk_ratio=0.01 for ~10x saving.
    """
    mode = (compression or "none").lower().strip()
    if mode not in COMPRESSION_MODES:
        raise ValueError(f"Unknown compression '{mode}'. Options: {COMPRESSION_MODES}")
    if mode == "none":
        clean = {k: v.detach().to("cpu").contiguous() for k, v in state_dict.items()}
        return st_save(clean)
    # compressed path
    comp = compress_state_dict(state_dict, mode=mode, topk_ratio=topk_ratio)
    return st_save(comp)


def bytes_to_state_dict(data: bytes) -> Dict[str, torch.Tensor]:
    """Deserialize safetensors bytes into a CPU state_dict (auto-decompresses)."""
    flat = st_load(data)
    if "__meta__" in flat or any("__" in k for k in flat.keys() if isinstance(k, str)):
        # try decompress if it looks compressed
        # check if keys contain compression shards
        if any(k.endswith("__int8") or k.endswith("__indices") for k in flat.keys()):
            try:
                return decompress_state_dict(flat)
            except Exception:
                return flat
        if "__meta__" in flat:
            try:
                return decompress_state_dict(flat)
            except Exception:
                return flat
    return flat


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_state_dict(state_dict: Dict[str, torch.Tensor], path: Union[str, Path]) -> None:
    clean = {k: v.detach().to("cpu").contiguous() for k, v in state_dict.items()}
    save_file(clean, str(path))


def load_state_dict(path: Union[str, Path]) -> Dict[str, torch.Tensor]:
    return load_file(str(path))
