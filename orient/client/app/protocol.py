"""Wire protocol schemas shared by the Server app and the Client app.

This module is duplicated verbatim in ``client/app/protocol.py`` so that each
app is self-contained. Keep both copies identical.

Reference: SRS v1.4 §4.12 (FR-PROTO-1..5).
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

PROTOCOL_VERSION = "1.0"


class ModelSpec(BaseModel):
    """Fully determines the architecture so both apps build identical models."""

    family: str = Field(description="'mlp' or 'deeponet'")
    params: Dict[str, Any] = Field(default_factory=dict)
    input_dim: int = 1
    output_dim: int = 1


class ProblemSpec(BaseModel):
    """References a problem by name from the fixed registry (no code on the wire)."""

    name: str
    family: str = Field(description="'supervised' | 'pinn' | 'operator'")
    description: str = ""
    required_arrays: List[str] = Field(default_factory=list)


class ClientRegistration(BaseModel):
    client_id: str
    problem: str
    n_samples: int = 0
    domain: Optional[Dict[str, Any]] = None
    protocol_version: str = PROTOCOL_VERSION


class OptimizerSpec(BaseModel):
    name: str = "adam"
    lr: float = 1e-3


class AssignmentResponse(BaseModel):
    """Server -> client: what the client should do next."""

    status: Literal["wait", "train", "done", "error"]
    round: int = 0
    total_rounds: int = 0
    message: str = ""
    model_spec: Optional[ModelSpec] = None
    problem_spec: Optional[ProblemSpec] = None
    local_epochs: int = 0
    optimizer: Optional[OptimizerSpec] = None
    prox_mu: float = 0.0
    weights_sha256: str = ""


class ClientUpdateMeta(BaseModel):
    """Metadata accompanying a binary weight upload."""

    client_id: str
    round: int
    n_samples: int
    local_loss: float
    protocol_version: str = PROTOCOL_VERSION


class RunStartRequest(BaseModel):
    problem: str
    aggregator: str = "fedavg"
    total_rounds: int = 10
    local_epochs: int = 5
    learning_rate: float = 1e-3
    weighting: str = Field(default="uniform", description="uniform | data_size | quality")
    aggregator_params: Dict[str, Any] = Field(default_factory=dict)
    expected_clients: Optional[List[str]] = None


class RunStatus(BaseModel):
    phase: Literal["idle", "collecting", "complete"] = "idle"
    round: int = 0
    total_rounds: int = 0
    problem: str = ""
    aggregator: str = ""
    weighting: str = "uniform"
    registered_clients: List[str] = Field(default_factory=list)
    expected_clients: List[str] = Field(default_factory=list)
    submitted_clients: List[str] = Field(default_factory=list)
    metrics: List[Dict[str, Any]] = Field(default_factory=list)


class ClientInfo(BaseModel):
    client_id: str
    problem: str
    n_samples: int
    domain: Optional[Dict[str, Any]] = None
    last_seen_round: int = 0


class SpawnClientsRequest(BaseModel):
    """Ask the server to start N *local* demo clients (development convenience)."""

    count: int = 2
    problem: str = "gramacy_lee"
