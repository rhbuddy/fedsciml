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
    # Per-client control variates keyed by client_id. The positional list above
    # cannot survive partial participation: with C<1 sampling a different set of
    # clients shows up each round, so "position i" is not the same client.
    scaffold_ci_by_id: Dict[str, torch.Tensor] = field(default_factory=dict)
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
    """Element-wise median across clients (FR-AGG-8).

    ``torch.median`` returns the LOWER of the two central values for an even
    number of inputs, which makes the aggregator an element-wise *minimum* at
    K=2 and biases it low at every even K. The reference implementations
    (Flower's ``Median`` / ``numpy.median``) average the two central values, so
    do the same here.
    """
    keys = list(states[0].keys())
    tmpl = states[0]
    mat = _stack_vecs(states, keys)  # K x P
    k = mat.shape[0]
    if k % 2 == 1:
        med_vec = mat.median(dim=0).values
    else:
        sorted_mat, _ = torch.sort(mat, dim=0)
        med_vec = (sorted_mat[k // 2 - 1] + sorted_mat[k // 2]) / 2.0
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
    """FedAdam / FedAdagrad / FedYogi server-side adaptive update (Reddi et al. 2020).

    Vectorized: ``m``/``v`` live in a single flat vector (one kernel) and are
    mirrored back into the per-key dicts for inspection.

    There is deliberately NO blanket ``except`` retry here. The previous
    implementation caught every exception and re-ran a per-key path, but ``m``
    and ``v`` had already been mutated *in place* by then, so the retry applied
    the momentum update a second time and double-incremented ``server.t`` while
    hiding the original error. ``_validate_states`` already guarantees identical
    keys/shapes, so a failure here is a real bug and must propagate.
    """
    keys = list(global_state.keys())
    if kind not in ("adam", "adagrad", "yogi"):
        raise ValueError(f"Unknown server optimizer '{kind}'")

    avg_state = _weighted_average_vec(states, weights, keys, global_state)
    global_vec = _flatten_state(global_state, keys)
    avg_vec = _flatten_state(avg_state, keys)
    g_vec = global_vec - avg_vec  # pseudo-gradient

    lr = float(params.get("server_lr", 0.1))
    beta1 = float(params.get("beta1", 0.9))
    beta2 = float(params.get("beta2", 0.99))
    tau = float(params.get("tau", 1e-3))
    server.t += 1

    if server._m_vec is None or server._m_vec.numel() != g_vec.numel():
        server._m_vec = torch.zeros_like(g_vec)
        server._v_vec = torch.zeros_like(g_vec)
    m_vec = server._m_vec
    v_vec = server._v_vec

    if kind == "adam":
        m_vec.mul_(beta1).add_(g_vec, alpha=1 - beta1)
        v_vec.mul_(beta2).add_(g_vec * g_vec, alpha=1 - beta2)
    elif kind == "adagrad":
        m_vec.copy_(g_vec)
        v_vec.add_(g_vec * g_vec)
    else:  # yogi
        m_vec.mul_(beta1).add_(g_vec, alpha=1 - beta1)
        # v = v - (1-beta2)*sign(v - g^2)*g^2
        v_vec.sub_((1 - beta2) * torch.sign(v_vec - g_vec * g_vec) * g_vec * g_vec)

    server._m_vec, server._v_vec = m_vec, v_vec

    # mirror the flat buffers back into per-key dicts for inspectability
    offset = 0
    for k in keys:
        n = global_state[k].numel()
        server.m[k] = m_vec[offset: offset + n].view(global_state[k].shape).clone()
        server.v[k] = v_vec[offset: offset + n].view(global_state[k].shape).clone()
        offset += n

    update = lr * m_vec / (torch.sqrt(torch.abs(v_vec)) + tau)
    return _unflatten_vec(global_vec - update, global_state, keys)


def _scaffold(
    states: Sequence[StateDict],
    weights: Sequence[float],
    global_state: StateDict,
    server: ServerOptimizerState,
    params: Dict[str, Any],
    client_ids: Optional[Sequence[str]] = None,
) -> Tuple[StateDict, Dict[str, Any]]:
    """SCAFFOLD (Karimireddy et al., ICML 2020), vectorized.

    Control variates are keyed by ``client_id`` when ids are supplied. That
    matters under partial participation: with ``client_fraction < 1`` a
    different subset of clients reports each round, so indexing ``c_i`` by the
    position a client happens to occupy would hand one client's control
    variate to a completely different client (and silently reset them all
    whenever K changed).
    """
    k = len(states)
    keys = list(global_state.keys()) if global_state else list(states[0].keys())
    global_vec = _flatten_state(global_state, keys)
    client_mats = _stack_vecs(states, keys)  # K x P

    if server._c_vec is None or server._c_vec.numel() != global_vec.numel():
        server._c_vec = torch.zeros_like(global_vec)
    c_vec = server._c_vec

    ids = list(client_ids) if client_ids is not None else [None] * k
    if len(ids) != k:
        raise ValueError(f"Got {len(ids)} client ids for {k} client update(s)")

    # Per-client control variates: look up by id, default to zeros for a client
    # seen for the first time.
    cis_vec: List[torch.Tensor] = []
    for cid in ids:
        prev = server.scaffold_ci_by_id.get(cid) if cid is not None else None
        if prev is None or prev.numel() != global_vec.numel():
            cis_vec.append(torch.zeros_like(global_vec))
        else:
            cis_vec.append(prev)

    server.t += 1
    global_lr = float(params.get("server_lr", 1.0))
    damping = float(params.get("scaffold_damping", 0.1))

    # corrected = client + c - ci
    cis_mat = torch.stack(cis_vec, dim=0)  # K x P
    corrected = client_mats + c_vec.unsqueeze(0) - cis_mat  # K x P
    w = torch.as_tensor(_normalize_weights(weights, k), dtype=torch.float32)
    avg_corrected = (w.unsqueeze(1) * corrected).sum(dim=0)  # P
    new_global_vec = global_vec + global_lr * (avg_corrected - global_vec)

    # control-variate updates
    mean_drift = (client_mats - new_global_vec.unsqueeze(0)).mean(dim=0)  # P
    new_c_vec = c_vec + damping * mean_drift
    new_cis = [cis_vec[idx] + damping * (client_mats[idx] - new_global_vec) for idx in range(k)]

    server._c_vec = new_c_vec
    server._cis_vec = new_cis
    # dict mirrors for inspection / backward compat
    server.scaffold_c = _unflatten_vec(new_c_vec, global_state, keys)
    server.scaffold_cis = [_unflatten_vec(v, global_state, keys) for v in new_cis]
    # the authoritative store, keyed so it survives a changing participant set
    for cid, vec in zip(ids, new_cis):
        if cid is not None:
            server.scaffold_ci_by_id[cid] = vec.clone()

    new_global = _unflatten_vec(new_global_vec, global_state, keys)
    info = {
        "server_round": server.t,
        "scaffold_c_norm": float(new_c_vec.norm().item()),
        "clients": [c for c in ids if c is not None],
    }
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
    client_ids: Optional[Sequence[str]] = None,
) -> AggregationResult:
    """Aggregate client ``state_dict`` updates.

    ``client_ids`` (optional) must be aligned with ``client_states``; SCAFFOLD
    uses it to keep control variates attached to the right client across rounds
    with partial participation.
    """
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
        # With the default trim_ratio=0.1, floor(K*0.1) == 0 for every K < 10, so
        # "trimmed mean" silently degenerates into plain FedAvg and provides no
        # robustness at all. Say so loudly in the logs and in the run metrics.
        k = len(client_states)
        n_trim = int(math.floor(k * trim_ratio))
        if trim_ratio > 0.0 and n_trim == 0:
            logger.warning(
                "trimmed_mean: trim_ratio=%.3g with K=%d trims floor(K*trim_ratio)=%d client(s) "
                "per side, i.e. nothing - this round is a plain weighted mean. "
                "Raise trim_ratio or use more clients for real trimming.",
                trim_ratio, k, n_trim,
            )
        return AggregationResult(
            _elementwise_trimmed_mean(client_states, trim_ratio),
            {
                "trim_ratio": trim_ratio,
                "trimmed_per_side": n_trim,
                "effectively_untrimmed": trim_ratio > 0.0 and n_trim == 0,
            },
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
        state, info = _scaffold(client_states, norm_w, global_state, server, params, client_ids)
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
