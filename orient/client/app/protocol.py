"""Wire protocol schemas shared by the Server app and the Client app.

This module is duplicated verbatim in ``client/app/protocol.py`` so that each
app is self-contained. Keep both copies identical.

Reference: SRS v1.4 §4.12 (FR-PROTO-1..5) and §4.1 FR-CFG-2..4.
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
    # FR-CLIENT-9: gradient clipping config per round
    gradient_clip: str = Field(default="none", description="none | value | norm")
    max_norm: float = 1.0
    clip_value: float = 0.5
    # heterogeneity / noise passthrough for logging
    noise_mode: str = "none"
    heterogeneity: Optional[Dict[str, Any]] = None
    # Framework v2: compression & DP (told to client per round)
    compression: str = Field(default="none", description="none | int8 | topk | int8_topk")
    topk_ratio: float = Field(default=0.01, ge=0.0, le=1.0)
    dp_noise_multiplier: float = Field(default=0.0, ge=0.0)
    dp_delta: float = Field(default=1e-5, gt=0, le=1.0)


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
    # FedAvg in the paper/SRS is data-size proportional by default.
    weighting: str = Field(default="data_size", description="uniform | data_size | quality")
    aggregator_params: Dict[str, Any] = Field(default_factory=dict)
    expected_clients: Optional[List[str]] = None
    # SRS FR-CFG-2 extended fields
    heterogeneity: Optional[Dict[str, Any]] = Field(default=None, description="e.g. {mode: 1d_partition, n_pieces: 10}")
    seed: Optional[int] = Field(default=None, description="reproducibility seed")
    noise_mode: str = Field(default="none", description="none | noisy | adversarial")
    noise_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    gradient_clip: str = Field(default="none", description="none | value | norm")
    max_norm: float = Field(default=1.0, gt=0)
    clip_value: float = Field(default=0.5, gt=0)
    optimizer: str = Field(default="adam", description="adam | sgd | lion")
    n_clients: Optional[int] = Field(default=None, ge=2, description="expected K (>=2) for validation")
    # Framework v2: async sampling, compression, DP (user requested 3 — exclude postgres+s3+redis)
    client_fraction: float = Field(default=1.0, ge=0.0, le=1.0, description="Fraction of clients sampled per round (C in FedAvg, 1.0=sync)")
    round_timeout: int = Field(default=0, ge=0, description="Seconds to wait before aggregating whatever arrived (0=wait all)")
    compression: str = Field(default="none", description="none | int8 | topk | int8_topk")
    topk_ratio: float = Field(default=0.01, ge=0.0, le=1.0, description="Top-k sparsity for compression (0.01=1%)")
    dp_noise_multiplier: float = Field(default=0.0, ge=0.0, description="DP Gaussian noise multiplier sigma (0=off, 0.1=on, pairs with gradient_clip norm)")
    dp_delta: float = Field(default=1e-5, gt=0, le=1.0, description="DP delta for RDP accounting")


class RunStatus(BaseModel):
    phase: Literal["idle", "collecting", "complete"] = "idle"
    round: int = 0
    total_rounds: int = 0
    problem: str = ""
    aggregator: str = ""
    weighting: str = "data_size"
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
