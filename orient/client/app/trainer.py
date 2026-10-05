"""Local training for the client app.

The model and the Adam optimizer persist across rounds so that momentum is not
reset each round. The client never shares raw data — only ``state_dict`` tensors
leave the machine (FR-CLIENTAPP-6).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .models import build_model
from .problems import get_problem
from .protocol import ModelSpec, OptimizerSpec


def _steps_per_epoch(arrays: Dict[str, np.ndarray], batch_size: int) -> int:
    if "branch_train" in arrays:
        n = int(arrays["branch_train"].shape[0])
    elif "x_train" in arrays:
        n = int(arrays["x_train"].shape[0])
    else:
        n = batch_size
    return max(1, n // max(1, batch_size))


class LocalTrainer:
    """Owns the local model + optimizer for the lifetime of the client process."""

    def __init__(self, problem_name: str, device: str = "cpu") -> None:
        self.problem_name = problem_name
        self.problem = get_problem(problem_name)
        self.device = torch.device(device)
        self.model = build_model(self.problem.model_spec()).to(self.device)
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self._global_ref: Optional[List[torch.Tensor]] = None

    # ------------------------------------------------------------------ spec/io
    def expected_spec(self) -> ModelSpec:
        return self.problem.model_spec()

    def assert_spec_compatible(self, spec: ModelSpec) -> None:
        """Fail loudly if the server's architecture differs from the local one."""
        mine = self.expected_spec()
        if spec.family != mine.family or spec.params != mine.params:
            raise ValueError(
                f"Server ModelSpec {spec.family}/{spec.params} does not match the local "
                f"model for problem '{self.problem_name}': {mine.family}/{mine.params}"
            )

    def load_weights(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict({k: v.to(self.device) for k, v in state_dict.items()})
        # Snapshot the global weights for the FedProx proximal term.
        self._global_ref = [p.detach().clone() for p in self.model.parameters()]

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.model.state_dict()

    # ------------------------------------------------------------------ training
    def train(
        self,
        arrays: Dict[str, np.ndarray],
        epochs: int,
        optimizer_spec: OptimizerSpec,
        batch_size: int = 256,
        prox_mu: float = 0.0,
        grad_clip: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run ``epochs`` local epochs and return ``{"loss": ..., "steps": ...}``."""
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=float(optimizer_spec.lr)
            )
        else:
            for group in self.optimizer.param_groups:
                group["lr"] = float(optimizer_spec.lr)

        self.model.train()
        steps = _steps_per_epoch(arrays, batch_size)
        losses: List[float] = []

        for _ in range(max(1, int(epochs))):
            for _ in range(steps):
                self.optimizer.zero_grad(set_to_none=True)
                loss = self.problem.local_loss(self.model, arrays, batch_size)

                if prox_mu > 0.0 and self._global_ref is not None:
                    # NOTE: must stay in the autograd graph, otherwise the proximal
                    # term is a constant offset and FedProx degenerates to FedAvg.
                    prox = sum(
                        ((p - ref) ** 2).sum()
                        for p, ref in zip(self.model.parameters(), self._global_ref)
                    )
                    loss = loss + 0.5 * float(prox_mu) * prox

                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(grad_clip))
                self.optimizer.step()
                losses.append(float(loss.detach().cpu()))

        return {
            "loss": float(np.mean(losses)) if losses else 0.0,
            "steps": len(losses),
        }
