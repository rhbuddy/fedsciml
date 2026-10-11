"""Client weighting modes for aggregation: ``uniform`` | ``data_size`` | ``quality``.

Reference: Flower's ``num-examples`` weighting (FedAvg default) and the
quality-based weighting described in SRS v1.4 §4.5.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Sequence

logger = logging.getLogger("orient.weighting")

# Data-size weighting is the FedAvg default in the FedSciML paper/SRS.
MODES = ["data_size", "uniform", "quality"]


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
        # SRS FR-AGG-12/FR-CLIENT-5: w_k = (1/L_k) / sum(1/L_i). This is
        # unbounded as L_k -> 0 by design, so the only thing to guard is the
        # division itself. The epsilon is scaled to the observed losses rather
        # than an absolute 1e-6, which was small enough that a zero-loss client
        # still took ~100% of the aggregate (weight 0.999999) while looking
        # "clamped".
        clean = [max(float(l), 0.0) for l in losses]
        scale = max(clean)
        eps = 1e-6 * scale if scale > 0 else 1e-12
        inv = [1.0 / max(l, eps) for l in clean]
        total = sum(inv)
        if not math.isfinite(total) or total <= 0:
            return [1.0] * k
        out = [i / total for i in inv]
        top = max(out)
        if top > 0.9:
            logger.warning(
                "quality weighting: one client holds %.4f of the aggregate "
                "(losses=%s). The 1/L formula is unbounded near L=0; consider "
                "'data_size' weighting if this is not intended.",
                top, [f"{l:.4g}" for l in clean],
            )
        return out

    raise ValueError(f"Unknown weighting mode '{mode}'. Options: {MODES}")
