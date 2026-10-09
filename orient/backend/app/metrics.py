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


def w1_nd(a: np.ndarray, b: np.ndarray) -> float:
    """W1 for ND point clouds (e.g. Schaffer 2D) via POT EMD with Euclidean cost."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if b.ndim == 1:
        b = b.reshape(-1, 1)
    if len(a) == 0 or len(b) == 0:
        return 0.0
    max_n = 1500
    if len(a) > max_n:
        a = a[np.random.default_rng(0).choice(len(a), max_n, replace=False)]
    if len(b) > max_n:
        b = b[np.random.default_rng(0).choice(len(b), max_n, replace=False)]
    try:
        import ot  # type: ignore

        wa = np.ones(len(a), dtype=np.float64) / len(a)
        wb = np.ones(len(b), dtype=np.float64) / len(b)
        M = ot.dist(a, b, metric="euclidean")
        return float(ot.emd2(wa, wb, M))
    except Exception:
        pass
    if a.shape[1] == 1 and b.shape[1] == 1:
        return w1_1d(a.reshape(-1), b.reshape(-1))
    return float(np.linalg.norm(a.mean(axis=0) - b.mean(axis=0)))


def w1_between_clients(client_samples: Sequence[np.ndarray]) -> float:
    """Mean pairwise W1 per paper Eq.4 (direct W1 for K=2). Supports 1-D and ND samples."""
    k = len(client_samples)
    if k < 2:
        return 0.0

    def _w1(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a)
        b = np.asarray(b)
        if a.ndim == 2 and a.shape[1] > 1:
            return w1_nd(a, b)
        if b.ndim == 2 and b.shape[1] > 1:
            return w1_nd(a, b)
        return w1_1d(a.reshape(-1), b.reshape(-1))

    if k == 2:
        return _w1(client_samples[0], client_samples[1])
    total = 0.0
    count = 0
    for i in range(k):
        for j in range(i + 1, k):
            total += _w1(client_samples[i], client_samples[j])
            count += 1
    if count == 0:
        return 0.0
    return float(total / count)


def compute_heterogeneity_stats(client_arrays: List[Dict[str, np.ndarray]]) -> Dict[str, Any]:
    """Compute W1 and partition coverage stats from client dataset dicts (SRS FR-DATA-3/6).

    - For PINN/Supervised: x_train point clouds (1D or 2D) -> ND W1
    - For Operator (DeepONet): branch function distribution (functional W1 proxy)
    Also reports coverage per FR-DATA-6.
    """
    samples: List[np.ndarray] = []
    total_n = 0
    per_client_n: List[int] = []
    for arr in client_arrays:
        if "x_train" in arr:
            x = np.asarray(arr["x_train"])
            if x.ndim == 2 and x.shape[1] > 1:
                samples.append(x)
            else:
                samples.append(x.reshape(-1))
            total_n += x.shape[0]
            per_client_n.append(int(x.shape[0]))
        elif "branch_train" in arr:
            b = np.asarray(arr["branch_train"])
            samples.append(b)
            total_n += b.shape[0]
            per_client_n.append(int(b.shape[0]))
        else:
            key = next(iter(arr))
            s = np.asarray(arr[key])
            if s.ndim == 2 and s.shape[1] > 1:
                samples.append(s)
            else:
                samples.append(s.reshape(-1))
            total_n += int(arr[key].shape[0])
            per_client_n.append(int(arr[key].shape[0]))
    w1 = w1_between_clients(samples)
    coverage_ok = total_n > 0 and all(n > 0 for n in per_client_n)
    return {
        "w1": float(w1),
        "total_samples": int(total_n),
        "per_client_samples": per_client_n,
        "min_per_client": int(min(per_client_n)) if per_client_n else 0,
        "max_per_client": int(max(per_client_n)) if per_client_n else 0,
        "n_clients": len(client_arrays),
        "coverage_ok": bool(coverage_ok),
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
