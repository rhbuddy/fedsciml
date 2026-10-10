"""Aggregation algorithms: standard (FedAvg/FedProx/FedAdam/FedAdagrad/FedYogi)
and Byzantine-robust (Median/Trimmed-Mean/Krum) — FAST vectorized version.

Reference: Flower's strategy set
(https://flower.ai/docs/framework/ref-api/flwr.server.strategy.html) and
Reddi et al., "Adaptive Federated Optimization" (arXiv:2003.00295).

All algorithms operate purely on ``state_dict`` tensors, so they are
architecture-agnostic (NFR-MNT-3). Speed optimization (Num 1): flatten all
params to a single vector -> one kernel instead of ~12 Python loops. Krum uses
torch.cdist (one batched kernel vs K² loops).

Architecture note: public API unchanged, so tests/client/server need no change.
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
    """Persistent moment buffers for server-side adaptive optimizers, SCAFFOLD and FedDyn."""

    m: Dict[str, torch.Tensor] = field(default_factory=dict)
    v: Dict[str, torch.Tensor] = field(default_factory=dict)
    t: int = 0
    # SCAFFOLD control variates (Karimireddy et al., ICML 2020)
    scaffold_c: Dict[str, torch.Tensor] = field(default_factory=dict)
    scaffold_cis: List[Dict[str, torch.Tensor]] = field(default_factory=list)
    # FedDyn dynamic regularizer state (Acar et al., AISTATS 2021)
    feddyn_h: Dict[str, torch.Tensor] = field(default_factory=dict)
    # FAST vector cache for adaptive (optional, auto-managed)
    _m_vec: Optional[torch.Tensor] = field(default=None, repr=False)
    _v_vec: Optional[torch.Tensor] = field(default=None, repr=False)
    _c_vec: Optional[torch.Tensor] = field(default=None, repr=False)
    _cis_vec: Optional[List[torch.Tensor]] = field(default=None, repr=False)
    _feddyn_h_vec: Optional[torch.Tensor] = field(default=None, repr=False)


@dataclass
class AggregationResult:
    state_dict: StateDict
    info: Dict[str, Any] = field(default_factory=dict)


def _f32(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to("cpu", torch.float32)


def _require_finite_tensor(tensor: torch.Tensor, label: str) -> None:
    if not torch.isfinite(_f32(tensor)).all().item():
        raise ValueError(f"Non-finite value detected in {label}")


def _state_keys(states: Sequence[StateDict], global_state: StateDict) -> List[str]:
    return list(global_state.keys()) if global_state else list(states[0].keys())


def _validate_states(states: Sequence[StateDict], global_state: StateDict) -> None:
    """Fail early on incompatible or poisoned client updates."""
    if not states:
        raise ValueError("No client updates to aggregate")

    expected_keys = _state_keys(states, global_state)
    expected_key_set = set(expected_keys)
    expected_shapes = {key: tuple((global_state or states[0])[key].shape) for key in expected_keys}

    for index, state in enumerate(states):
        if set(state.keys()) != expected_key_set:
            raise ValueError(
                f"Client update {index} has state_dict keys {list(state.keys())}; "
                f"expected {expected_keys}"
            )
        for key in expected_keys:
            if tuple(state[key].shape) != expected_shapes[key]:
                raise ValueError(
                    f"Client update {index} tensor '{key}' has shape {tuple(state[key].shape)}; "
                    f"expected {expected_shapes[key]}"
                )
            _require_finite_tensor(state[key], f"client update {index} tensor '{key}'")


def _normalize_weights(weights: Sequence[float], k: int) -> List[float]:
    if len(weights) != k:
        raise ValueError(f"Received {len(weights)} weights for {k} client update(s)")

    clean: List[float] = []
    for index, weight in enumerate(weights):
        value = float(weight)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"Aggregation weight {index} must be finite and non-negative, got {weight!r}")
        clean.append(value)

    total = float(sum(clean))
    if total <= 0.0:
        return [1.0 / k] * k
    return [value / total for value in clean]


# ------------------------------------------------------------------ FAST vector helpers

def _keys_and_sizes(global_state: StateDict, keys: Sequence[str]):
    sizes = {k: global_state[k].numel() if global_state else 0 for k in keys}
    # if global empty, infer from first state shapes
    return sizes

def _flatten_state(state: StateDict, keys: Sequence[str]) -> torch.Tensor:
    # one contiguous vector per client
    return torch.cat([_f32(state[k]).reshape(-1) for k in keys])

def _unflatten_vec(vec: torch.Tensor, template: StateDict, keys: Sequence[str]) -> StateDict:
    out: StateDict = {}
    offset = 0
    for k in keys:
        n = template[k].numel() if template else vec.numel() // len(keys)  # fallback
        shape = template[k].shape if template and k in template else (n,)
        out[k] = vec[offset: offset + n].view(shape).clone()
        offset += n
    return out

def _stack_vecs(states: Sequence[StateDict], keys: Sequence[str]) -> torch.Tensor:
    # (K, P) matrix
    return torch.stack([_flatten_state(s, keys) for s in states], dim=0)

def _weighted_average_vec(states: Sequence[StateDict], weights: Sequence[float], keys: Sequence[str], template: StateDict) -> StateDict:
    # FAST: one matmul instead of ~12 per-tensor loops
    mat = _stack_vecs(states, keys)  # K x P
    w = torch.as_tensor(weights, dtype=torch.float32)  # K
    # w is already normalized to 1
    avg_vec = (w.unsqueeze(1) * mat).sum(dim=0)  # P
    return _unflatten_vec(avg_vec, template, keys)

# keep legacy per-key for fallback / grad checks but route through vec for speed
def _weighted_average(states: Sequence[StateDict], weights: Sequence[float]) -> StateDict:
    # legacy entry, now vectorized via template inference
    keys = list(states[0].keys())
    # infer template from first state
    tmpl = states[0]
    return _weighted_average_vec(states, weights, keys, tmpl)


def _stack(states: Sequence[StateDict], key: str) -> torch.Tensor:
    return torch.stack([_f32(s[key]) for s in states], dim=0)


def _elementwise_median(states: Sequence[StateDict]) -> StateDict:
    # FAST vectorized median
    keys = list(states[0].keys())
    tmpl = states[0]
    mat = _stack_vecs(states, keys)  # K x P
    med_vec = mat.median(dim=0).values
    return _unflatten_vec(med_vec, tmpl, keys)


def _elementwise_trimmed_mean(states: Sequence[StateDict], trim_ratio: float = 0.1) -> StateDict:
    keys = list(states[0].keys())
    tmpl = states[0]
    k = len(states)
    trim_ratio = max(0.0, min(float(trim_ratio), 0.5))
    n_trim = int(math.floor(k * trim_ratio))
    mat = _stack_vecs(states, keys)  # K x P
    if n_trim > 0 and 2 * n_trim < k:
        sorted_mat, _ = torch.sort(mat, dim=0)
        kept = sorted_mat[n_trim: k - n_trim]
        avg_vec = kept.mean(dim=0)
    else:
        avg_vec = mat.mean(dim=0)
    return _unflatten_vec(avg_vec, tmpl, keys)


def _flatten(state: StateDict, keys: Sequence[str]) -> torch.Tensor:
    return torch.cat([_f32(state[key]).reshape(-1) for key in keys])


def _krum(
    states: Sequence[StateDict],
    keys: Sequence[str],
    n_byzantine: int = 1,
    multi_k: int = 1,
) -> Tuple[StateDict, Dict[str, Any]]:
    """Krum / Multi-Krum — FAST via torch.cdist.

    Original: O(K²·P) double loop. Now: one batched cdist kernel.
    """
    k = len(states)
    if k == 0:
        raise ValueError("No states for Krum")
    # stack flats as (K, P)
    flats = _stack_vecs(states, keys)  # K x P
    # pairwise squared Euclidean
    # cdist gives (K,K) Euclidean; square for Krum score (as in paper)
    try:
        dist = torch.cdist(flats, flats, p=2) ** 2  # K x K
    except Exception:
        # fallback double loop if cdist fails (e.g. huge P)
        dist = torch.zeros((k, k), dtype=torch.float32)
        for i in range(k):
            for j in range(i + 1, k):
                d = torch.sum((flats[i] - flats[j]) ** 2)
                dist[i, j] = d
                dist[j, i] = d

    n_closest = max(1, k - int(n_byzantine) - 2)
    scores: List[float] = []
    for i in range(k):
        d = dist[i].clone()
        d[i] = float("inf")
        d_sorted, _ = torch.sort(d)
        scores.append(float(d_sorted[:n_closest].sum().item()))

    order = sorted(range(k), key=lambda i: scores[i])
    chosen = order[: max(1, min(int(multi_k), k))]
    # average chosen vectors
    avg_vec = flats[chosen].mean(dim=0)
    # need template for unflatten
    template = states[0]
    out = _unflatten_vec(avg_vec, template, keys)
    return out, {"selected_clients": chosen, "krum_scores": scores}


def _server_adaptive(
    states: Sequence[StateDict],
    weights: Sequence[float],
    global_state: StateDict,
    server: ServerOptimizerState,
    params: Dict[str, Any],
    kind: str,
) -> StateDict:
    """FedAdam / FedAdagrad / FedYogi server-side adaptive update.

    FAST path: vectorize m/v to single vector (one kernel) while keeping dict
    for backward compat. Falls back to per-key if shapes mismatch.
    """
    keys = list(global_state.keys())
    # vectorized fast path
    try:
        # avg via vec
        avg_state = _weighted_average_vec(states, weights, keys, global_state)
        # global vec
        global_vec = _flatten_state(global_state, keys)
        avg_vec = _flatten_state(avg_state, keys)
        g_vec = global_vec - avg_vec  # pseudo-gradient

        lr = float(params.get("server_lr", 0.1))
        beta1 = float(params.get("beta1", 0.9))
        beta2 = float(params.get("beta2", 0.99))
        tau = float(params.get("tau", 1e-3))
        server.t += 1

        # init vec buffers if needed
        if server._m_vec is None or server._m_vec.numel() != g_vec.numel():
            server._m_vec = torch.zeros_like(g_vec)
            server._v_vec = torch.zeros_like(g_vec)
        m_vec = server._m_vec
        v_vec = server._v_vec

        if kind == "adam":
            m_vec.mul_(beta1).add_(g_vec, alpha=1 - beta1)
            v_vec.mul_(beta2).add_(g_vec * g_vec, alpha=1 - beta2)
        elif kind == "adagrad":
            # m = g
            m_vec.copy_(g_vec)
            v_vec.add_(g_vec * g_vec)
        elif kind == "yogi":
            m_vec.mul_(beta1).add_(g_vec, alpha=1 - beta1)
            # v = v - (1-beta2)*sign(v - g^2)*g^2
            v_vec.sub_((1 - beta2) * torch.sign(v_vec - g_vec * g_vec) * g_vec * g_vec)
        else:
            raise ValueError(f"Unknown server optimizer '{kind}'")
        server._m_vec, server._v_vec = m_vec, v_vec

        # keep dict mirrors for inspectability
        # update dict from vec for backward compat (small cost)
        # we lazily sync only when needed - here we sync
        offset = 0
        for k in keys:
            n = global_state[k].numel()
            server.m[k] = m_vec[offset: offset + n].view(global_state[k].shape).clone()
            server.v[k] = v_vec[offset: offset + n].view(global_state[k].shape).clone()
            offset += n

        update = lr * m_vec / (torch.sqrt(torch.abs(v_vec)) + tau)
        new_vec = global_vec - update
        return _unflatten_vec(new_vec, global_state, keys)
    except Exception:
        # fallback to legacy per-key
        avg = _weighted_average_vec(states, weights, keys, global_state)
        lr = float(params.get("server_lr", 0.1))
        beta1 = float(params.get("beta1", 0.9))
        beta2 = float(params.get("beta2", 0.99))
        tau = float(params.get("tau", 1e-3))
        server.t += 1
        out: StateDict = {}
        for key in avg:
            base = _f32(global_state[key])
            g = base - avg[key]
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
            else:
                raise ValueError(f"Unknown server optimizer '{kind}'")
            server.m[key], server.v[key] = m, v
            update = lr * m / (torch.sqrt(torch.abs(v)) + tau)
            out[key] = base - update
        return out


def _scaffold(
    states: Sequence[StateDict],
    weights: Sequence[float],
    global_state: StateDict,
    server: ServerOptimizerState,
    params: Dict[str, Any],
) -> Tuple[StateDict, Dict[str, Any]]:
    """SCAFFOLD FAST vectorized."""
    k = len(states)
    keys = list(global_state.keys()) if global_state else list(states[0].keys())
    # vectorize global and client states
    global_vec = _flatten_state(global_state, keys)
    client_mats = _stack_vecs(states, keys)  # K x P

    # lazy init global control vec
    if server._c_vec is None or server._c_vec.numel() != global_vec.numel():
        server._c_vec = torch.zeros_like(global_vec)
    if server._cis_vec is None or len(server._cis_vec) != k or server._cis_vec[0].numel() != global_vec.numel():
        # rebuild from dict if exists, else zeros
        if server.scaffold_c and server.scaffold_cis and len(server.scaffold_cis) == k:
            # convert dict to vec
            server._c_vec = _flatten_state(server.scaffold_c, keys)
            server._cis_vec = [_flatten_state(ci, keys) for ci in server.scaffold_cis]
        else:
            server._cis_vec = [torch.zeros_like(global_vec) for _ in range(k)]
    c_vec = server._c_vec
    cis_vec = server._cis_vec
    # ensure length
    while len(cis_vec) < k:
        cis_vec.append(torch.zeros_like(global_vec))
    if len(cis_vec) > k:
        cis_vec = cis_vec[:k]

    server.t += 1
    global_lr = float(params.get("server_lr", 1.0))
    damping = float(params.get("scaffold_damping", 0.1))

    # corrected = client + c - ci
    # client_mats: K x P, c_vec: P, cis: K x P
    cis_mat = torch.stack(cis_vec, dim=0)  # K x P
    corrected = client_mats + c_vec.unsqueeze(0) - cis_mat  # K x P
    w = torch.as_tensor(_normalize_weights(weights, k), dtype=torch.float32)  # K, already normalized
    avg_corrected = (w.unsqueeze(1) * corrected).sum(dim=0)  # P
    new_global_vec = global_vec + global_lr * (avg_corrected - global_vec)

    # update controls: vectorized
    # mean_drift for c
    mean_drift = (client_mats - new_global_vec.unsqueeze(0)).mean(dim=0)  # P
    new_c_vec = c_vec + damping * mean_drift
    new_cis = []
    for idx in range(k):
        drift = client_mats[idx] - new_global_vec
        new_cis.append(cis_vec[idx] + damping * drift)

    server._c_vec = new_c_vec
    server._cis_vec = new_cis
    # sync dict mirrors for inspection / backward compat
    server.scaffold_c = _unflatten_vec(new_c_vec, global_state, keys)
    server.scaffold_cis = [_unflatten_vec(v, global_state, keys) for v in new_cis]

    new_global = _unflatten_vec(new_global_vec, global_state, keys)
    info = {"server_round": server.t, "scaffold_c_norm": float(new_c_vec.norm().item())}
    return new_global, info


def _feddyn(
    states: Sequence[StateDict],
    weights: Sequence[float],
    global_state: StateDict,
    server: ServerOptimizerState,
    params: Dict[str, Any],
) -> Tuple[StateDict, Dict[str, Any]]:
    """FedDyn (Acar et al., AISTATS 2021) — dynamic regularization.

    Maintains a server state ``h`` updated as
        g = global - avg(client_models)   (pseudo-gradient)
        h <- beta * h + alpha * g
        new_global = avg - server_lr * h
    ``h`` is kept as a single vector for speed but also mirrored as a dict
    for inspectability. Converges faster than FedAvg under heterogeneity
    (small n_pieces / large W1). Defaults ``alpha=0.01, beta=0.9,
    server_lr=1.0`` follow the paper (``mu`` is aliased to ``alpha``).
    """
    keys = list(global_state.keys()) if global_state else list(states[0].keys())
    # allow mu as alias for alpha
    if "mu" in params and "alpha" not in params:
        params = dict(params)
        params["alpha"] = params["mu"]
    alpha = float(params.get("alpha", 0.01))
    beta = float(params.get("beta", 0.9))
    server_lr = float(params.get("server_lr", 1.0))
    if global_state:
        global_vec = _flatten_state(global_state, keys)
        tmpl = global_state
    else:
        tmpl = states[0]
        global_vec = _flatten_state(tmpl, keys) * 0
    avg_state = _weighted_average_vec(states, weights, keys, tmpl if global_state else states[0])
    avg_vec = _flatten_state(avg_state, keys)
    g_vec = global_vec - avg_vec
    if server._feddyn_h_vec is None or server._feddyn_h_vec.numel() != g_vec.numel():
        if server.feddyn_h and len(server.feddyn_h) == len(keys):
            try:
                server._feddyn_h_vec = _flatten_state(server.feddyn_h, keys)
            except Exception:
                server._feddyn_h_vec = torch.zeros_like(g_vec)
        else:
            server._feddyn_h_vec = torch.zeros_like(g_vec)
    h_vec = server._feddyn_h_vec
    server.t += 1
    h_vec = beta * h_vec + alpha * g_vec
    server._feddyn_h_vec = h_vec
    server.feddyn_h = _unflatten_vec(h_vec, tmpl, keys)
    new_vec = avg_vec - server_lr * h_vec
    new_global = _unflatten_vec(new_vec, tmpl, keys)
    info = {"server_round": server.t, "feddyn_h_norm": float(h_vec.norm().item()), "alpha": alpha, "beta": beta}
    return new_global, info


STANDARD_AGGREGATORS = ["fedavg", "fedprox", "fedadam", "fedadagrad", "fedyogi", "scaffold", "feddyn"]
ROBUST_AGGREGATORS = ["median", "trimmed_mean", "krum"]
ALL_AGGREGATORS = STANDARD_AGGREGATORS + ROBUST_AGGREGATORS
AGGREGATOR_ALIASES = {"fedagrad": "fedadagrad"}
SUPPORTED_AGGREGATORS = ALL_AGGREGATORS + sorted(AGGREGATOR_ALIASES)


def canonical_name(name: str) -> str:
    """Return the canonical SRS aggregator name, accepting legacy aliases."""
    normalized = name.lower().strip()
    normalized = AGGREGATOR_ALIASES.get(normalized, normalized)
    if normalized not in ALL_AGGREGATORS:
        raise ValueError(f"Unknown aggregator '{name}'. Options: {ALL_AGGREGATORS}")
    return normalized


def aggregate(
    name: str,
    client_states: Sequence[StateDict],
    weights: Sequence[float],
    global_state: StateDict,
    params: Optional[Dict[str, Any]] = None,
    server: Optional[ServerOptimizerState] = None,
) -> AggregationResult:
    """Aggregate client ``state_dict`` updates."""
    name = canonical_name(name)
    params = dict(params or {})
    _validate_states(client_states, global_state)
    keys = _state_keys(client_states, global_state)
    norm_w = _normalize_weights(weights, len(client_states))

    if name == "fedavg":
        # FAST vec path
        return AggregationResult(_weighted_average_vec(client_states, norm_w, keys, global_state or client_states[0]))

    if name == "fedprox":
        return AggregationResult(_weighted_average_vec(client_states, norm_w, keys, global_state or client_states[0]))

    if name in {"fedadam", "fedadagrad", "fedyogi"}:
        if server is None:
            raise ValueError(f"Aggregator '{name}' requires a ServerOptimizerState")
        kind = {"fedadam": "adam", "fedadagrad": "adagrad", "fedyogi": "yogi"}[name]
        return AggregationResult(
            _server_adaptive(client_states, norm_w, global_state, server, params, kind),
            {"server_round": server.t},
        )

    if name == "median":
        return AggregationResult(_elementwise_median(client_states))

    if name == "trimmed_mean":
        trim_ratio = max(0.0, min(float(params.get("trim_ratio", 0.1)), 0.5))
        return AggregationResult(
            _elementwise_trimmed_mean(client_states, trim_ratio),
            {"trim_ratio": trim_ratio},
        )

    if name == "krum":
        n_f = int(params.get("n_byzantine", 1))
        required = 2 * n_f + 3
        if len(client_states) < required:
            logger.warning(
                "krum: only %d client update(s) for f=%d; the method needs K >= 2f+3 = %d "
                "to be meaningful. Results will be unreliable.",
                len(client_states), n_f, required,
            )
        state, info = _krum(
            client_states,
            keys,
            n_byzantine=n_f,
            multi_k=int(params.get("multi_k", 1)),
        )
        info["krum_clients_required"] = required
        return AggregationResult(state, info)

    if name == "scaffold":
        if server is None:
            raise ValueError("Aggregator 'scaffold' requires a ServerOptimizerState")
        state, info = _scaffold(client_states, norm_w, global_state, server, params)
        return AggregationResult(state, info)

    if name == "feddyn":
        if server is None:
            raise ValueError("Aggregator 'feddyn' requires a ServerOptimizerState")
        state, info = _feddyn(client_states, norm_w, global_state, server, params)
        return AggregationResult(state, info)

    raise ValueError(f"Unknown aggregator '{name}'. Options: {ALL_AGGREGATORS}")


def prox_mu_for(name: str, params: Optional[Dict[str, Any]]) -> float:
    """Return the client-side proximal coefficient (mu) for the given aggregator."""
    if canonical_name(name) == "fedprox":
        return float((params or {}).get("mu", 0.01))
    return 0.0
