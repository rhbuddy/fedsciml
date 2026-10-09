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


class Lion(torch.optim.Optimizer):
    """Lion optimizer (Chen et al. 2023) - lightweight pytorch implementation.

    Drop-in so SRS FR-CLIENT-2 optimizer: Adam, SGD, Lion all work.
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]
                beta1, beta2 = group["betas"]
                # update direction via sign interpolation
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(torch.sign(update), alpha=-group["lr"])
                # momentum update
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
                if group["weight_decay"] != 0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])
        return loss


def _optimizer_factory(name: str, parameters, lr: float) -> torch.optim.Optimizer:
    """Build a supported local optimizer from the wire ``OptimizerSpec``."""
    normalized = (name or "adam").lower().strip()
    if normalized == "adam":
        return torch.optim.Adam(parameters, lr=lr)
    if normalized == "sgd":
        return torch.optim.SGD(parameters, lr=lr)
    if normalized == "lion":
        # Try external package first, fallback to bundled Lion
        try:
            from lion_pytorch import Lion as ExtLion  # type: ignore

            return ExtLion(parameters, lr=lr)
        except Exception:
            pass
        return Lion(parameters, lr=lr)
    raise ValueError("Unknown optimizer '%s'. Options: adam, sgd, lion" % name)


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

    def _split_train_val(self, arrays: Dict[str, np.ndarray], val_frac: float = 0.2) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray] | None]:
        """SRS FR-AGG-12: 80/20 train/val split for quality weighting (deterministic)."""
        try:
            n = int(next(iter(arrays.values())).shape[0])
            if n < 10 or val_frac <= 0:
                return arrays, None
            n_train = int(n * (1 - val_frac))
            if n_train <= 0 or n_train >= n:
                return arrays, None
            train: Dict[str, np.ndarray] = {}
            val: Dict[str, np.ndarray] = {}
            for k, v in arrays.items():
                train[k] = v[:n_train]
                val[k] = v[n_train:]
            return train, val
        except Exception:
            return arrays, None

    # ------------------------------------------------------------------ training
    def train(
        self,
        arrays: Dict[str, np.ndarray],
        epochs: int,
        optimizer_spec: OptimizerSpec,
        batch_size: int = 256,
        prox_mu: float = 0.0,
        grad_clip: Optional[float] = None,
        gradient_clip: str = "none",
        max_norm: float = 1.0,
        clip_value: float = 0.5,
    ) -> Dict[str, Any]:
        """Run ``epochs`` local epochs and return ``{\"loss\": ..., \"val_loss\": ..., \"steps\": ...}``.

        Supports SRS FR-CLIENT-9 gradient clipping modes: ``none`` | ``value`` | ``norm``.
        ``grad_clip`` is backward-compat alias for ``max_norm`` when ``gradient_clip=='norm'``.
        Also computes ``val_loss`` on a 20% held-out shard for quality weighting (FR-AGG-12).

        Raises ``ValueError`` on non-finite losses/gradients/parameters so a bad
        client exits instead of uploading poisoned weights to the server.
        """
        self._ensure_optimizer(optimizer_spec)
        assert self.optimizer is not None  # for type checkers

        # Resolve clipping mode (new protocol fields take precedence, fallback to legacy grad_clip)
        clip_mode = (gradient_clip or "none").lower().strip()
        if grad_clip is not None and clip_mode == "none":
            # legacy CLI --grad-clip provided as norm value
            clip_mode = "norm"
            max_norm = float(grad_clip)

        # SRS FR-AGG-12: split 80/20 for quality weighting validation loss
        train_arrays, val_arrays = self._split_train_val(arrays, 0.2)

        self.model.train()
        steps = _steps_per_epoch(train_arrays, batch_size)
        losses: List[float] = []

        for _ in range(max(1, int(epochs))):
            for _ in range(steps):
                self.optimizer.zero_grad(set_to_none=True)
                loss = self.problem.local_loss(self.model, train_arrays, batch_size)

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
                # SRS gradient clipping: none | value | norm
                if clip_mode == "norm":
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), float(max_norm)
                    )
                    if not torch.isfinite(torch.as_tensor(grad_norm)).all().item():
                        raise ValueError(f"Non-finite gradient norm during clipping: {grad_norm}")
                elif clip_mode == "value":
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), float(clip_value))
                elif clip_mode != "none":
                    raise ValueError(f"Unknown gradient_clip mode '{clip_mode}'. Options: none, value, norm")

                self.optimizer.step()
                self._check_parameters_finite()
                losses.append(float(loss.detach().cpu()))

        # Compute held-out validation loss for quality weighting (FR-AGG-12) — no grad, same loss fn on val shard
        val_loss: float | None = None
        if val_arrays is not None:
            self.model.eval()
            try:
                with torch.no_grad():
                    v_losses = []
                    for _ in range(5):
                        v_losses.append(float(self.problem.local_loss(self.model, val_arrays, batch_size).detach().cpu()))
                    val_loss = float(np.mean(v_losses))
                    if not np.isfinite(val_loss):
                        val_loss = float(np.mean(losses)) if losses else 0.0
            except Exception:
                val_loss = float(np.mean(losses)) if losses else 0.0
            self.model.train()
        else:
            val_loss = float(np.mean(losses)) if losses else 0.0

        return {
            "loss": float(np.mean(losses)) if losses else 0.0,
            "val_loss": val_loss,
            "steps": len(losses),
        }
