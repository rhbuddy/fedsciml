"""Noisy / adversarial client simulation (SRS FR-NOISE 1-4)."""

from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

import torch

StateDict = Dict[str, torch.Tensor]


def select_compromised(k: int, fraction: float, seed: int = 0) -> List[int]:
    """Deterministically select compromised client indices given fraction."""
    if fraction <= 0 or k == 0:
        return []
    n_bad = int(round(float(fraction) * k))
    n_bad = max(0, min(n_bad, k))
    if n_bad == 0:
        return []
    rng = random.Random(seed)
    return sorted(rng.sample(range(k), n_bad))


def corrupt_state(
    state: StateDict, mode: str, scale: float = 1.0, seed: int = 0
) -> StateDict:
    """Return a corrupted copy of ``state`` according to mode."""
    mode = (mode or "none").lower().strip()
    if mode == "none":
        return state
    out: StateDict = {}
    g = torch.Generator()
    g.manual_seed(int(seed) % (2**32))
    for key, tensor in state.items():
        t = tensor.detach().to("cpu", torch.float32)
        if mode == "noisy":
            # Add large Gaussian noise (10x std)
            noise = torch.randn(t.shape, generator=g, dtype=torch.float32) * (t.std().item() * 10 + 1.0)
            out[key] = t + noise * float(scale)
        elif mode == "adversarial":
            # Invert and scale: worst-case Byzantine - flip sign and amplify
            out[key] = -t * 5.0 * float(scale)
        else:
            out[key] = t
    return out


def apply_noise_to_updates(
    states: Sequence[StateDict],
    mode: str,
    fraction: float,
    seed: int = 0,
) -> Tuple[List[StateDict], List[int]]:
    """Apply corruption to a fraction of updates; return new list + compromised indices."""
    mode = (mode or "none").lower().strip()
    if mode not in ("noisy", "adversarial") or fraction <= 0:
        return list(states), []
    idx = select_compromised(len(states), fraction, seed)
    idx_set = set(idx)
    new_states: List[StateDict] = []
    for i, state in enumerate(states):
        if i in idx_set:
            new_states.append(corrupt_state(state, mode, scale=1.0, seed=seed + i * 1009))
        else:
            new_states.append(state)
    return new_states, idx
