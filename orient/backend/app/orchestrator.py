"""Federation state and the synchronous round engine.

Round lifecycle (SRS v1.4 §4.11):
    broadcast global model -> collect client updates -> aggregate -> advance

The engine is thread-safe (a single ``RLock``) and designed for a single
process; aggregation cost is small relative to client training.
"""

from __future__ import annotations

import math
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np
import torch

from . import aggregators, weighting
from .config import settings
from .models import build_model, count_parameters
from .problems import get_problem
from .protocol import (
    AssignmentResponse,
    ClientInfo,
    ClientRegistration,
    OptimizerSpec,
    RunStartRequest,
    RunStatus,
)
from .storage import RunStorage
from .weights import (
    bytes_to_state_dict,
    sha256_hex,
    state_dict_to_bytes,
)

MIN_CLIENTS_PER_ROUND = 2


@dataclass
class ClientRecord:
    registration: ClientRegistration
    registered_at: float = 0.0
    last_seen_round: int = 0
    last_seen_at: float = 0.0


@dataclass
class RunState:
    phase: str = "idle"  # idle | collecting | complete
    round: int = 0
    total_rounds: int = 0
    problem: str = ""
    aggregator: str = "fedavg"
    weighting: str = "data_size"
    local_epochs: int = 0
    learning_rate: float = 0.0
    aggregator_params: Dict[str, Any] = field(default_factory=dict)
    expected: List[str] = field(default_factory=list)
    updates: Dict[str, bytes] = field(default_factory=dict)
    update_meta: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metrics: List[Dict[str, Any]] = field(default_factory=list)
    round_started_at: float = 0.0


class Federation:
    """Thread-safe holder of the client registry, global model, and run state."""

    def __init__(self, results_dir: Optional[str] = None, seed: Optional[int] = None) -> None:
        self._lock = threading.RLock()
        self.clients: Dict[str, ClientRecord] = {}
        # Clients that completed the previous run are kept visible for status, but
        # are not auto-selected for a new run unless they explicitly register again.
        self._completed_clients: Set[str] = set()
        self.run = RunState()
        self.server_opt = aggregators.ServerOptimizerState()
        self.results_dir = results_dir or settings.results_dir
        self.seed = settings.seed if seed is None else seed
        self.storage: Optional[RunStorage] = None
        self._problem = None
        self._global_model = None
        self._global_bytes: bytes = b""
        self._global_sha: str = ""

    # ------------------------------------------------------------------ registry
    def register(self, reg: ClientRegistration) -> ClientInfo:
        with self._lock:
            # Reject a client whose problem differs from the active run: it would be
            # added to `expected`, never produce a compatible update, and stall the
            # round for everyone else. Do this only while collecting; after a run is
            # complete, clients for a different next problem must be able to register.
            if self.run.phase == "collecting" and reg.problem != self.run.problem:
                raise ValueError(
                    f"Run is on problem '{self.run.problem}'; client '{reg.client_id}' "
                    f"registered for '{reg.problem}'. Refusing to add it to the round."
                )

            self.clients[reg.client_id] = ClientRecord(
                registration=reg, registered_at=time.time(), last_seen_at=time.time()
            )
            self._completed_clients.discard(reg.client_id)
            # A client that arrives during an unfinished round joins that round,
            # but only before any update has been accepted. Once collection has
            # started, changing the expected set would make completion ambiguous.
            if (
                self.run.phase == "collecting"
                and reg.problem == self.run.problem
                and reg.client_id not in self.run.expected
                and not self.run.updates
            ):
                self.run.expected.append(reg.client_id)
                self.run.expected.sort()
            return ClientInfo(
                client_id=reg.client_id,
                problem=reg.problem,
                n_samples=reg.n_samples,
                domain=reg.domain,
                last_seen_round=self.clients[reg.client_id].last_seen_round,
            )

    def reap_stale(self) -> None:
        """Drop clients that stopped polling (crashed / closed process).

        Without this a dead client stays in ``expected`` and blocks every
        subsequent round forever. Set ``ORIENT_CLIENT_TTL=0`` to disable.
        """
        ttl = settings.client_ttl_seconds
        if ttl <= 0:
            return
        now = time.time()
        stale = [
            cid for cid, record in self.clients.items()
            if record.last_seen_at and now - record.last_seen_at > ttl
        ]
        for client_id in stale:
            del self.clients[client_id]
            self._completed_clients.discard(client_id)
            if self.run.phase == "collecting" and client_id not in self.run.updates:
                self.run.expected = [cid for cid in self.run.expected if cid != client_id]

    def live_client_ids(self, problem: Optional[str] = None) -> List[str]:
        with self._lock:
            self.reap_stale()
            return sorted(
                cid for cid, record in self.clients.items()
                if cid not in self._completed_clients
                and (problem is None or record.registration.problem == problem)
            )

    def is_registered(self, client_id: str) -> bool:
        """True if this client has completed /clients/register at least once."""
        with self._lock:
            return client_id in self.clients

    def list_clients(self) -> List[ClientInfo]:
        with self._lock:
            return [
                ClientInfo(
                    client_id=c.registration.client_id,
                    problem=c.registration.problem,
                    n_samples=c.registration.n_samples,
                    domain=c.registration.domain,
                    last_seen_round=c.last_seen_round,
                )
                for c in self.clients.values()
            ]

    # ---------------------------------------------------------------------- run
    def start_run(self, req: RunStartRequest) -> RunStatus:
        with self._lock:
            problem = get_problem(req.problem)

            # Validate up-front: a bad aggregator/weighting used to raise deep inside
            # aggregation(), which left the run stuck in "collecting" forever.
            aggregator_name = aggregators.canonical_name(req.aggregator)
            weighting_name = (req.weighting or "data_size").lower().strip()
            if weighting_name not in weighting.MODES:
                raise ValueError(
                    f"Unknown weighting '{req.weighting}'. Options: {weighting.MODES}"
                )
            if int(req.total_rounds) < 1:
                raise ValueError("total_rounds must be >= 1")
            if int(req.local_epochs) < 1:
                raise ValueError("local_epochs must be >= 1")

            if req.expected_clients:
                requested_clients = sorted(set(req.expected_clients))
                unknown = [cid for cid in requested_clients if cid not in self.clients]
                if unknown:
                    raise ValueError(f"expected_clients contains unregistered client(s): {unknown}")
                wrong_problem = [
                    cid for cid in requested_clients
                    if self.clients[cid].registration.problem != req.problem
                ]
                if wrong_problem:
                    raise ValueError(
                        f"expected_clients registered for a different problem: {wrong_problem}"
                    )
                completed = [cid for cid in requested_clients if cid in self._completed_clients]
                if completed:
                    raise ValueError(
                        f"expected_clients must re-register before a new run: {completed}"
                    )
                expected = requested_clients
            else:
                # Do not carry idle clients from a different problem into the new run.
                expected = self.live_client_ids(req.problem)

            # Reproducibility (NFR-REP-1): seed before building the global model so
            # its random initialization is identical for a given seed.
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            np.random.seed(self.seed % (2**32))

            model = build_model(problem.model_spec())
            self._problem = problem
            self._global_model = model
            self.server_opt = aggregators.ServerOptimizerState()
            self._global_bytes = state_dict_to_bytes(model.state_dict())
            self._global_sha = sha256_hex(self._global_bytes)

            self.run = RunState(
                phase="collecting",
                round=1,
                total_rounds=int(req.total_rounds),
                problem=req.problem,
                aggregator=aggregator_name,
                weighting=weighting_name,
                local_epochs=int(req.local_epochs),
                learning_rate=float(req.learning_rate),
                aggregator_params=dict(req.aggregator_params),
                expected=expected,
                round_started_at=time.time(),
            )
            self.storage = RunStorage(self.results_dir, req.problem, aggregator_name)
            self.storage.write_config(
                {
                    **req.model_dump(),
                    "aggregator": aggregator_name,
                    "weighting": weighting_name,
                    "n_parameters": count_parameters(model),
                    "seed": self.seed,
                    "min_clients_per_round": MIN_CLIENTS_PER_ROUND,
                }
            )
            return self.status()

    def stop_run(self) -> RunStatus:
        with self._lock:
            self.run.phase = "complete"
            self._completed_clients.update(self.run.expected)
            if self.storage and self._global_model is not None:
                self.storage.log_round({"event": "stopped", "round": self.run.round})
            return self.status()

    # ------------------------------------------------------------- client cycle
    def assignment(self, client_id: str) -> AssignmentResponse:
        with self._lock:
            if client_id not in self.clients:
                return AssignmentResponse(status="error", message="Client not registered")

            self.clients[client_id].last_seen_round = self.run.round
            self.clients[client_id].last_seen_at = time.time()
            self.reap_stale()  # after refreshing ourselves
            self._maybe_timeout()

            if self.run.phase == "idle":
                return AssignmentResponse(
                    status="wait",
                    round=self.run.round,
                    total_rounds=self.run.total_rounds,
                    message="No run in progress. Waiting for the server operator to start one.",
                )

            if self.run.phase == "complete":
                return AssignmentResponse(
                    status="done",
                    round=self.run.round,
                    total_rounds=self.run.total_rounds,
                    message="Run complete.",
                )

            if client_id not in self.run.expected:
                return AssignmentResponse(
                    status="wait",
                    round=self.run.round,
                    total_rounds=self.run.total_rounds,
                    message="Not selected for this round (joined late or different problem). Waiting for the next round.",
                )

            if len(self.run.expected) < MIN_CLIENTS_PER_ROUND:
                return AssignmentResponse(
                    status="wait",
                    round=self.run.round,
                    total_rounds=self.run.total_rounds,
                    message=(
                        f"Waiting for at least {MIN_CLIENTS_PER_ROUND} clients before training; "
                        f"currently have {len(self.run.expected)}."
                    ),
                )

            if client_id in self.run.updates:
                return AssignmentResponse(
                    status="wait",
                    round=self.run.round,
                    total_rounds=self.run.total_rounds,
                    message="Update already received. Waiting for peers.",
                )

            problem = self._problem
            return AssignmentResponse(
                status="train",
                round=self.run.round,
                total_rounds=self.run.total_rounds,
                message=f"Train locally for {self.run.local_epochs} epoch(s), then upload weights.",
                model_spec=problem.model_spec(),
                problem_spec=problem.problem_spec(),
                local_epochs=self.run.local_epochs,
                optimizer=OptimizerSpec(name="adam", lr=self.run.learning_rate),
                prox_mu=aggregators.prox_mu_for(self.run.aggregator, self.run.aggregator_params),
                weights_sha256=self._global_sha,
            )

    def get_global_bytes(self, round_no: int) -> bytes:
        with self._lock:
            if self.run.phase == "idle" or self._global_model is None:
                raise ValueError("No run in progress")
            if int(round_no) != self.run.round:
                raise ValueError(f"Round mismatch: requested {round_no}, current {self.run.round}")
            return self._global_bytes

    def submit_update(
        self,
        client_id: str,
        round_no: int,
        n_samples: int,
        local_loss: float,
        data: bytes,
    ) -> AssignmentResponse:
        with self._lock:
            if client_id not in self.clients:
                return AssignmentResponse(status="error", message="Client not registered")

            if self.run.phase == "complete":
                return AssignmentResponse(status="done", round=self.run.round, message="Run complete.")

            if self.run.phase != "collecting" or int(round_no) != self.run.round:
                return AssignmentResponse(
                    status="wait",
                    round=self.run.round,
                    message=f"Stale update ignored (server is at round {self.run.round}).",
                )

            if client_id not in self.run.expected or client_id in self.run.updates:
                return AssignmentResponse(
                    status="wait", round=self.run.round, message="Update not applicable."
                )

            if len(self.run.expected) < MIN_CLIENTS_PER_ROUND:
                return AssignmentResponse(
                    status="wait",
                    round=self.run.round,
                    message=f"Need at least {MIN_CLIENTS_PER_ROUND} clients before accepting updates.",
                )

            if int(n_samples) <= 0:
                return AssignmentResponse(status="error", message="Client update has no samples")
            if not math.isfinite(float(local_loss)):
                return AssignmentResponse(status="error", message="Client update has non-finite local_loss")

            # Validate the payload deserializes, is finite, and matches the global shapes.
            try:
                state = bytes_to_state_dict(data)
                expected_shapes = {k: tuple(v.shape) for k, v in self._global_model.state_dict().items()}
                got_shapes = {k: tuple(v.shape) for k, v in state.items()}
                if got_shapes != expected_shapes:
                    raise ValueError("state_dict keys/shapes do not match the global model")
                for key, tensor in state.items():
                    if not torch.isfinite(tensor.detach().to("cpu", torch.float32)).all().item():
                        raise ValueError(f"state_dict tensor '{key}' contains NaN/Inf")
            except Exception as exc:  # noqa: BLE001
                return AssignmentResponse(status="error", message=f"Invalid weight payload: {exc}")

            self.run.updates[client_id] = data
            self.run.update_meta[client_id] = {
                "n_samples": int(n_samples),
                "local_loss": float(local_loss),
            }
            self._maybe_complete_round()
            return AssignmentResponse(
                status="wait",
                round=self.run.round,
                total_rounds=self.run.total_rounds,
                message="Update accepted. Waiting for peers.",
            )

    # -------------------------------------------------------------- aggregation
    def _maybe_timeout(self) -> None:
        """Advance with the updates received so far if the round has timed out."""
        if self.run.phase != "collecting" or len(self.run.updates) < MIN_CLIENTS_PER_ROUND:
            return
        timeout = settings.round_timeout_seconds
        if timeout <= 0 or self.run.round_started_at <= 0:
            return
        if time.time() - self.run.round_started_at > timeout:
            self._aggregate()

    def _maybe_complete_round(self) -> None:
        if self.run.phase != "collecting":
            return
        missing = [c for c in self.run.expected if c not in self.run.updates]
        if self.run.expected and not missing and len(self.run.updates) >= MIN_CLIENTS_PER_ROUND:
            self._aggregate()

    def force_aggregate(self) -> RunStatus:
        """Operator override: aggregate using whatever updates have arrived."""
        with self._lock:
            if self.run.phase == "collecting" and len(self.run.updates) >= MIN_CLIENTS_PER_ROUND:
                self._aggregate()
            elif self.run.phase == "collecting" and self.storage:
                self.storage.log_round(
                    {
                        "event": "force_aggregate_skipped",
                        "round": self.run.round,
                        "reason": f"need at least {MIN_CLIENTS_PER_ROUND} updates",
                        "n_updates": len(self.run.updates),
                    }
                )
            return self.status()

    def _aggregate(self) -> None:
        order = [c for c in self.run.expected if c in self.run.updates]
        if len(order) < MIN_CLIENTS_PER_ROUND:
            raise ValueError(f"Need at least {MIN_CLIENTS_PER_ROUND} client updates to aggregate")

        states = [bytes_to_state_dict(self.run.updates[c]) for c in order]
        n_samples = [self.run.update_meta[c]["n_samples"] for c in order]
        losses = [self.run.update_meta[c]["local_loss"] for c in order]

        weights = weighting.compute_weights(self.run.weighting, n_samples, losses)
        result = aggregators.aggregate(
            self.run.aggregator,
            states,
            weights,
            self._global_model.state_dict(),
            self.run.aggregator_params,
            self.server_opt,
        )
        self._global_model.load_state_dict(result.state_dict)
        self._global_bytes = state_dict_to_bytes(self._global_model.state_dict())
        self._global_sha = sha256_hex(self._global_bytes)

        metrics = self._problem.evaluate(self._global_model)
        record: Dict[str, Any] = {
            "round": self.run.round,
            "aggregator": self.run.aggregator,
            "weighting": self.run.weighting,
            "n_clients": len(order),
            "clients": order,
            "mean_local_loss": float(np.mean(losses)) if losses else None,
            "n_samples": n_samples,
            **metrics,
            "aggregation_info": result.info,
        }
        self.run.metrics.append(record)
        if self.storage:
            self.storage.log_round(record)

        if self.run.round >= self.run.total_rounds:
            self.run.phase = "complete"
            self._completed_clients.update(order)
            if self.storage:
                self.storage.save_model(self._global_model.state_dict())
        else:
            self.run.round += 1
            self.run.updates = {}
            self.run.update_meta = {}
            # Only clients on this problem that are actually still polling are
            # expected next round; otherwise a dead or mismatched participant would
            # stall the run.
            self.run.expected = self.live_client_ids(self.run.problem)
            self.run.round_started_at = time.time()

    # ------------------------------------------------------------------- status
    def status(self) -> RunStatus:
        with self._lock:
            self.reap_stale()
            return RunStatus(
                phase=self.run.phase,  # type: ignore[arg-type]
                round=self.run.round,
                total_rounds=self.run.total_rounds,
                problem=self.run.problem,
                aggregator=self.run.aggregator,
                weighting=self.run.weighting,
                registered_clients=sorted(self.clients),
                expected_clients=list(self.run.expected),
                submitted_clients=sorted(self.run.updates),
                metrics=list(self.run.metrics),
            )

    def metrics(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self.run.metrics)
