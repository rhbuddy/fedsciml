"""Local training for the client app.

The model and optimizer persist across rounds so that optimizer state (for
example Adam momentum) is not reset each round. The client never shares raw data
— only ``state_dict`` tensors leave the machine (FR-CLIENTAPP-6).
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


def _optimizer_factory(name: str, parameters, lr: float) -> torch.optim.Optimizer:
    """Build a supported local optimizer from the wire ``OptimizerSpec``."""
    normalized = (name or "adam").lower().strip()
    if normalized == "adam":
        return torch.optim.Adam(parameters, lr=lr)
    if normalized == "sgd":
        return torch.optim.SGD(parameters, lr=lr)
    # Lion is listed in the SRS as configurable, but it is not part of standard
    # torch.optim in common PyTorch releases. Fail clearly instead of silently
    # falling back to Adam.
    if normalized == "lion":
        raise ValueError(
            "Optimizer 'lion' is not available in standard torch.optim; install/add "
            "a Lion optimizer implementation before selecting it."
        )
    raise ValueError("Unknown optimizer '%s'. Options: adam, sgd%s" % (name, ", lion (if implemented)"))


class LocalTrainer:
    """Owns the local model + optimizer for the lifetime of the client process."""

    def __init__(self, problem_name: str, device: str = "cpu") -> None:
        self.problem_name = problem_name
        self.problem = get_problem(problem_name)
        self.device = torch.device(device)
        self.model = build_model(self.problem.model_spec()).to(self.device)
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self._optimizer_name: Optional[str] = None
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

    def _ensure_optimizer(self, optimizer_spec: OptimizerSpec) -> None:
        name = (optimizer_spec.name or "adam").lower().strip()
        lr = float(optimizer_spec.lr)
        if self.optimizer is None or self._optimizer_name != name:
            self.optimizer = _optimizer_factory(name, self.model.parameters(), lr)
            self._optimizer_name = name
            return
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def _check_gradients_finite(self) -> None:
        bad: List[str] = []
        for name, param in self.model.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all().item():
                bad.append(name)
        if bad:
            raise ValueError(f"Non-finite gradient(s) detected in {bad}")

    def _check_parameters_finite(self) -> None:
        bad = [name for name, param in self.model.named_parameters() if not torch.isfinite(param).all().item()]
        if bad:
            raise ValueError(f"Non-finite model parameter(s) after optimizer step: {bad}")

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
        """Run ``epochs`` local epochs and return ``{"loss": ..., "steps": ...}``.

        Raises ``ValueError`` on non-finite losses/gradients/parameters so a bad
        client exits instead of uploading poisoned weights to the server.
        """
        self._ensure_optimizer(optimizer_spec)
        assert self.optimizer is not None  # for type checkers

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

                if not torch.isfinite(loss).all().item():
                    raise ValueError(f"Non-finite local loss detected: {float(loss.detach().cpu())}")

                loss.backward()
                self._check_gradients_finite()
                if grad_clip:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), float(grad_clip)
                    )
                    if not torch.isfinite(torch.as_tensor(grad_norm)).all().item():
                        raise ValueError(f"Non-finite gradient norm during clipping: {grad_norm}")
                self.optimizer.step()
                self._check_parameters_finite()
                losses.append(float(loss.detach().cpu()))

        return {
            "loss": float(np.mean(losses)) if losses else 0.0,
            "steps": len(losses),
        }
