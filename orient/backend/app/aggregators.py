"""Aggregation algorithms: standard (FedAvg/FedProx/FedAdam/FedAdagrad/FedYogi)
and Byzantine-robust (Median/Trimmed-Mean/Krum).

Reference: Flower's strategy set
(https://flower.ai/docs/framework/ref-api/flwr.server.strategy.html) and
Reddi et al., "Adaptive Federated Optimization" (arXiv:2003.00295).

All algorithms operate purely on ``state_dict`` tensors, so they are
architecture-agnostic (NFR-MNT-3).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger("orient.aggregators")

StateDict = Dict[str, torch.Tensor]


@dataclass
class ServerOptimizerState:
    """Persistent moment buffers for server-side adaptive optimizers."""

    m: Dict[str, torch.Tensor] = field(default_factory=dict)
    v: Dict[str, torch.Tensor] = field(default_factory=dict)
    t: int = 0


@dataclass
class AggregationResult:
    state_dict: StateDict
    info: Dict[str, Any] = field(default_factory=dict)


def _f32(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to("cpu", torch.float32)


def _weighted_average(states: Sequence[StateDict], weights: Sequence[float]) -> StateDict:
    out: StateDict = {}
    for key in states[0]:
        acc = torch.zeros_like(_f32(states[0][key]))
        for state, w in zip(states, weights):
            acc += float(w) * _f32(state[key])
        out[key] = acc
    return out


def _stack(states: Sequence[StateDict], key: str) -> torch.Tensor:
    return torch.stack([_f32(s[key]) for s in states], dim=0)


def _elementwise_median(states: Sequence[StateDict]) -> StateDict:
    out: StateDict = {}
    for key in states[0]:
        out[key] = _stack(states, key).median(dim=0).values
    return out


def _elementwise_trimmed_mean(states: Sequence[StateDict], trim_ratio: float = 0.1) -> StateDict:
    k = len(states)
    n_trim = int(math.floor(k * float(trim_ratio)))
    out: StateDict = {}
    for key in states[0]:
        stack = _stack(states, key)
        if n_trim > 0 and 2 * n_trim < k:
            sorted_stack, _ = torch.sort(stack, dim=0)
            kept = sorted_stack[n_trim : k - n_trim]
        else:
            kept = stack
        out[key] = kept.mean(dim=0)
    return out


def _flatten(state: StateDict) -> torch.Tensor:
    return torch.cat([_f32(state[key]).reshape(-1) for key in state])


def _krum(states: Sequence[StateDict], n_byzantine: int = 1, multi_k: int = 1) -> Tuple[StateDict, Dict[str, Any]]:
    """Krum / Multi-Krum: pick the update(s) closest to their peers."""
    k = len(states)
    flats = [_flatten(s) for s in states]
    dist = torch.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            d = torch.norm(flats[i] - flats[j])
            dist[i, j] = d
            dist[j, i] = d

    n_closest = max(1, k - int(n_byzantine) - 2)
    scores: List[float] = []
    for i in range(k):
        d = dist[i].clone()
        d[i] = float("inf")
        d_sorted, _ = torch.sort(d)
        scores.append(float(d_sorted[:n_closest].sum()))

    order = sorted(range(k), key=lambda i: scores[i])
    chosen = order[: max(1, min(int(multi_k), k))]
    out: StateDict = {}
    for key in states[0]:
        out[key] = torch.stack([_f32(states[i][key]) for i in chosen], dim=0).mean(dim=0)
    return out, {"selected_clients": chosen, "krum_scores": scores}


def _server_adaptive(
    states: Sequence[StateDict],
    weights: Sequence[float],
    global_state: StateDict,
    server: ServerOptimizerState,
    params: Dict[str, Any],
    kind: str,
) -> StateDict:
    """FedAdam / FedAdagrad / FedYogi server-side adaptive update."""
    avg = _weighted_average(states, weights)
    lr = float(params.get("server_lr", 0.1))
    beta1 = float(params.get("beta1", 0.9))
    beta2 = float(params.get("beta2", 0.99))
    tau = float(params.get("tau", 1e-3))
    server.t += 1

    out: StateDict = {}
    for key in avg:
        base = _f32(global_state[key])
        g = base - avg[key]                      # pseudo-gradient (Reddi et al.)
        if key not in server.m:
            server.m[key] = torch.zeros_like(g)
            server.v[key] = torch.zeros_like(g)
        m, v = server.m[key], server.v[key]

        if kind == "adam":
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g * g
        elif kind == "adagrad":
            m = g
            v = v + g * g
        elif kind == "yogi":
            m = beta1 * m + (1 - beta1) * g
            v = v - (1 - beta2) * torch.sign(v - g * g) * g * g
        else:  # pragma: no cover - guarded by the dispatcher
            raise ValueError(f"Unknown server optimizer '{kind}'")

        server.m[key], server.v[key] = m, v
        update = lr * m / (torch.sqrt(torch.abs(v)) + tau)
        out[key] = base - update
    return out


STANDARD_AGGREGATORS = ["fedavg", "fedprox", "fedadam", "fedagrad", "fedyogi"]
ROBUST_AGGREGATORS = ["median", "trimmed_mean", "krum"]
ALL_AGGREGATORS = STANDARD_AGGREGATORS + ROBUST_AGGREGATORS


def aggregate(
    name: str,
    client_states: Sequence[StateDict],
    weights: Sequence[float],
    global_state: StateDict,
    params: Optional[Dict[str, Any]] = None,
    server: Optional[ServerOptimizerState] = None,
) -> AggregationResult:
    """Aggregate client ``state_dict`` updates.

    Parameters
    ----------
    name:
        One of :data:`ALL_AGGREGATORS` (case-insensitive).
    client_states:
        One ``state_dict`` per participating client.
    weights:
        Non-negative weights (normalized to sum 1 inside) controlling each
        client's influence. Use ``weighting.mode`` helpers to build these.
    global_state:
        The current global ``state_dict`` (needed by server-adaptive methods).
    params:
        Algorithm hyper-parameters (``mu``, ``server_lr``, ``beta1``,
        ``beta2``, ``tau``, ``trim_ratio``, ``n_byzantine``, ``multi_k``).
    server:
        Persistent :class:`ServerOptimizerState` (required by adaptive methods).
    """
    name = name.lower().strip()
    params = dict(params or {})
    if not client_states:
        raise ValueError("No client updates to aggregate")

    total = float(sum(weights)) or float(len(weights))
    norm_w = [float(w) / total for w in weights]

    if name == "fedavg":
        return AggregationResult(_weighted_average(client_states, norm_w))

    if name == "fedprox":
        # Proximal term acts client-side (prox_mu); the server aggregates as FedAvg.
        return AggregationResult(_weighted_average(client_states, norm_w))

    if name in {"fedadam", "fedagrad", "fedyogi"}:
        if server is None:
            raise ValueError(f"Aggregator '{name}' requires a ServerOptimizerState")
        kind = {"fedadam": "adam", "fedagrad": "adagrad", "fedyogi": "yogi"}[name]
        return AggregationResult(
            _server_adaptive(client_states, norm_w, global_state, server, params, kind),
            {"server_round": server.t},
        )

    if name == "median":
        return AggregationResult(_elementwise_median(client_states))

    if name == "trimmed_mean":
        return AggregationResult(
            _elementwise_trimmed_mean(client_states, params.get("trim_ratio", 0.1)),
            {"trim_ratio": params.get("trim_ratio", 0.1)},
        )

    if name == "krum":
        n_f = int(params.get("n_byzantine", 1))
        required = 2 * n_f + 3
        if len(client_states) < required:
            # Krum's robustness guarantee needs K >= 2f+3. Below that the two
            # candidates are always mutually nearest, so it degenerates into
            # "pick whichever client happens to sort first" - not an aggregation.
            logger.warning(
                "krum: only %d client update(s) for f=%d; the method needs K >= 2f+3 = %d "
                "to be meaningful. Results will be unreliable.",
                len(client_states), n_f, required,
            )
        state, info = _krum(
            client_states,
            n_byzantine=n_f,
            multi_k=int(params.get("multi_k", 1)),
        )
        info["krum_clients_required"] = required
        return AggregationResult(state, info)

    raise ValueError(f"Unknown aggregator '{name}'. Options: {ALL_AGGREGATORS}")


def prox_mu_for(name: str, params: Optional[Dict[str, Any]]) -> float:
    """Return the client-side proximal coefficient (mu) for the given aggregator."""
    if name.lower().strip() == "fedprox":
        return float((params or {}).get("mu", 0.01))
    return 0.0

