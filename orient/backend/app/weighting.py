"""Client weighting modes for aggregation: ``uniform`` | ``data_size`` | ``quality``.

Reference: Flower's ``num-examples`` weighting (FedAvg default) and the
quality-based weighting described in SRS v1.4 §4.5.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

MODES = ["uniform", "data_size", "quality"]


def compute_weights(
    mode: str,
    n_samples: Sequence[int],
    losses: Optional[Sequence[float]] = None,
) -> List[float]:
    """Return one non-negative weight per client (normalized inside `aggregate`)."""
    mode = (mode or "uniform").lower().strip()
    k = len(n_samples)
    if k == 0:
        return []

    if mode == "uniform":
        return [1.0] * k

    if mode == "data_size":
        total = float(sum(max(0, int(n)) for n in n_samples))
        if total <= 0:
            return [1.0] * k
        return [max(0.0, float(n)) / total for n in n_samples]

    if mode == "quality":
        if not losses:
            return [1.0] * k
        # Clamp the loss so a near-zero local loss cannot produce a weight of ~1e12
        # that would completely dominate (and destabilize) the aggregation.
        inv = [1.0 / max(float(l), 1e-6) for l in losses]
        total = sum(inv)
        if total <= 0:
            return [1.0] * k
        return [i / total for i in inv]

    raise ValueError(f"Unknown weighting mode '{mode}'. Options: {MODES}")
