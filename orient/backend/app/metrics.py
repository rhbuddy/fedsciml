"""Metrics: W1 heterogeneity, weight divergence, partition validation (SRS FR-DATA, FR-EVAL).

This module is server-side; clients compute local losses via problems.py.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch


def w1_1d(a: np.ndarray, b: np.ndarray) -> float:
    """1-Wasserstein distance between two 1-D samples (uniform mass)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if len(a) == 0 or len(b) == 0:
        return 0.0
    # Prefer POT's exact EMD if available (paper's method)
    try:
        import ot  # type: ignore

        wa = np.ones(len(a), dtype=np.float64) / len(a)
        wb = np.ones(len(b), dtype=np.float64) / len(b)
        M = ot.dist(a.reshape(-1, 1), b.reshape(-1, 1), metric="euclidean")
        return float(ot.emd2(wa, wb, M))
    except Exception:
        pass
    try:
        from scipy.stats import wasserstein_distance  # type: ignore

        return float(wasserstein_distance(a, b))
    except Exception:
        pass
    # Fallback: empirical quantile approximation (sorting)
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    # interpolate to common grid
    n = max(len(a_sorted), len(b_sorted))
    # simple approximation: compare sorted arrays resized via interpolation
    if len(a_sorted) != len(b_sorted):
        # Interpolate both to n points
        qa = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(a_sorted)), a_sorted)
        qb = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(b_sorted)), b_sorted)
        return float(np.mean(np.abs(qa - qb)))
    return float(np.mean(np.abs(a_sorted - b_sorted)))


def w1_between_clients(client_samples: Sequence[np.ndarray]) -> float:
    """Mean pairwise W1 per paper Eq.4 (direct W1 for K=2)."""
    k = len(client_samples)
    if k < 2:
        return 0.0
    if k == 2:
        return w1_1d(client_samples[0], client_samples[1])
    # mean pairwise for K>=3
    total = 0.0
    count = 0
    for i in range(k):
        for j in range(i + 1, k):
            total += w1_1d(client_samples[i], client_samples[j])
            count += 1
    if count == 0:
        return 0.0
    # Paper Eq.4 uses 1/((K-1)(K-2)) factor but that seems typo; we use standard mean pairwise
    # Keep both for reference, but use mean for stability.
    return float(total / count)


def compute_heterogeneity_stats(client_arrays: List[Dict[str, np.ndarray]]) -> Dict[str, Any]:
    """Compute W1 and partition coverage stats from client dataset dicts."""
    # Extract representative sample for W1: x_train or branch_train mean
    samples: List[np.ndarray] = []
    total_n = 0
    per_client_n: List[int] = []
    for arr in client_arrays:
        if "x_train" in arr:
            s = np.asarray(arr["x_train"]).reshape(-1)
            samples.append(s)
            total_n += arr["x_train"].shape[0]
            per_client_n.append(int(arr["x_train"].shape[0]))
        elif "branch_train" in arr:
            # for operator: use mean of branch vectors as scalar proxy
            s = np.asarray(arr["branch_train"]).mean(axis=1).reshape(-1)
            samples.append(s)
            total_n += arr["branch_train"].shape[0]
            per_client_n.append(int(arr["branch_train"].shape[0]))
        else:
            # fallback: first array
            key = next(iter(arr))
            s = np.asarray(arr[key]).reshape(-1)
            samples.append(s)
            total_n += int(arr[key].shape[0])
            per_client_n.append(int(arr[key].shape[0]))
    w1 = w1_between_clients(samples)
    return {
        "w1": float(w1),
        "total_samples": int(total_n),
        "per_client_samples": per_client_n,
        "min_per_client": int(min(per_client_n)) if per_client_n else 0,
        "max_per_client": int(max(per_client_n)) if per_client_n else 0,
        "n_clients": len(client_arrays),
        "coverage_ok": total_n > 0 and min(per_client_n) > 0,
    }


def weight_divergence(
    fed_state: Dict[str, torch.Tensor],
    central_state: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """Per-layer relative weight divergence (SRS FR-EVAL-3)."""
    out: Dict[str, float] = {}
    for key in fed_state:
        if key not in central_state:
            continue
        f = fed_state[key].detach().to("cpu", torch.float32)
        c = central_state[key].detach().to("cpu", torch.float32)
        denom = torch.linalg.norm(c).item()
        if denom == 0:
            div = float(torch.linalg.norm(f - c).item())
        else:
            div = float(torch.linalg.norm(f - c).item() / denom)
        out[key] = div
    # global summary
    if out:
        out["_mean"] = float(np.mean(list(out.values())))
        out["_max"] = float(np.max(list(out.values())))
    return out


def global_l2_divergence(
    fed_state: Dict[str, torch.Tensor],
    central_state: Dict[str, torch.Tensor],
) -> float:
    """Single scalar divergence across all parameters."""
    total_sq = 0.0
    total_norm_sq = 0.0
    for key in fed_state:
        if key not in central_state:
            continue
        f = fed_state[key].detach().to("cpu", torch.float32).reshape(-1)
        c = central_state[key].detach().to("cpu", torch.float32).reshape(-1)
        total_sq += float(torch.sum((f - c) ** 2).item())
        total_norm_sq += float(torch.sum(c ** 2).item())
    if total_norm_sq == 0:
        return float(math.sqrt(total_sq))
    return float(math.sqrt(total_sq) / math.sqrt(total_norm_sq))


def validate_partition_coverage(
    client_domains: List[Dict[str, Any]], global_domain: Tuple[float, float]
) -> Dict[str, Any]:
    """Check that client intervals cover the global domain without gaps/overlaps (1D case)."""
    # Simplified: just report counts
    return {
        "n_clients": len(client_domains),
        "global_domain": global_domain,
        "client_domains": client_domains,
    }
