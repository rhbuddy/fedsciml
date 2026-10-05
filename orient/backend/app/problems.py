"""Problem registry — the scientific core both apps reference *by name*.

Server and client share this module so the training loss, the PDE residual, and
the analytic ground truth used for evaluation are identical on both sides
(SRS v1.4 §4.12 FR-PROTO-3: no arbitrary code on the wire).

This module is duplicated verbatim in ``client/app/problems.py``.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch

from .models import build_model
from .protocol import ModelSpec, ProblemSpec


def relative_l2(pred: np.ndarray, true: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    true = np.asarray(true, dtype=np.float64).reshape(-1)
    denom = np.linalg.norm(true)
    if denom == 0.0:
        return float(np.linalg.norm(pred))
    return float(np.linalg.norm(pred - true) / denom)


class Problem:
    """Base class for a federated SciML benchmark problem."""

    name: str = "base"
    family: str = "supervised"          # supervised | pinn | operator
    description: str = ""
    required_arrays: List[str] = []

    # -- specs -----------------------------------------------------------------
    def model_spec(self) -> ModelSpec:  # pragma: no cover - abstract
        raise NotImplementedError

    def problem_spec(self) -> ProblemSpec:
        return ProblemSpec(
            name=self.name,
            family=self.family,
            description=self.description,
            required_arrays=list(self.required_arrays),
        )

    # -- data ------------------------------------------------------------------
    def sample_dataset(self, n: int, seed: int = 0) -> Dict[str, np.ndarray]:
        """Generate an example dataset bundle (used to bootstrap demo clients)."""
        raise NotImplementedError

    # -- training / evaluation -------------------------------------------------
    def local_loss(self, model, arrays: Dict[str, np.ndarray], batch_size: int = 256) -> torch.Tensor:
        raise NotImplementedError

    def evaluate(self, model, num_test: int = 400, seed: int = 1234) -> Dict[str, float]:
        raise NotImplementedError


class GramacyLee1D(Problem):
    """1D Gramacy & Lee function approximation (FedFuncApprox benchmark)."""

    name = "gramacy_lee"
    family = "supervised"
    description = "1D Gramacy & Lee function approximation f(x)=sin(10*pi*x)/(2x)+(x-1)^4"
    required_arrays = ["x_train", "y_train"]
    domain = (0.5, 2.5)

    @staticmethod
    def f(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return np.sin(10.0 * np.pi * x) / (2.0 * x) + (x - 1.0) ** 4

    def model_spec(self) -> ModelSpec:
        return ModelSpec(
            family="mlp",
            input_dim=1,
            output_dim=1,
            # Paper/SRS Table 4 default: width 64, depth 3, tanh.
            params={"layer_sizes": [1, 64, 64, 64, 1], "activation": "tanh"},
        )

    def sample_dataset(self, n: int, seed: int = 0) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        x = rng.uniform(self.domain[0], self.domain[1], size=(n, 1))
        return {
            "x_train": x.astype(np.float32),
            "y_train": self.f(x).astype(np.float32),
        }

    def local_loss(self, model, arrays, batch_size: int = 256) -> torch.Tensor:
        x = torch.as_tensor(arrays["x_train"], dtype=torch.float32)
        y = torch.as_tensor(arrays["y_train"], dtype=torch.float32)
        idx = torch.randint(0, x.shape[0], (min(batch_size, x.shape[0]),))
        return torch.mean((model(x[idx]) - y[idx]) ** 2)

    def evaluate(self, model, num_test: int = 400, seed: int = 1234) -> Dict[str, float]:
        rng = np.random.default_rng(seed)
        x = rng.uniform(self.domain[0], self.domain[1], size=(num_test, 1)).astype(np.float32)
        with torch.no_grad():
            pred = model(torch.as_tensor(x)).cpu().numpy()
        return {"l2_relative_error": relative_l2(pred, self.f(x))}


class Schaffer2D(Problem):
    """2D Schaffer function approximation (FedFuncApprox benchmark)."""

    name = "schaffer"
    family = "supervised"
    description = "2D Schaffer N.2: f=0.5+(sin^2(x1^2-x2^2)-0.5)/(1+0.001(x1^2+x2^2))^2"
    required_arrays = ["x_train", "y_train"]
    domain = (-2.0, 2.0)

    @staticmethod
    def f(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        x1, x2 = x[..., 0:1], x[..., 1:2]
        num = np.sin(x1**2 - x2**2) ** 2 - 0.5
        den = (1.0 + 0.001 * (x1**2 + x2**2)) ** 2
        return 0.5 + num / den

    def model_spec(self) -> ModelSpec:
        return ModelSpec(
            family="mlp",
            input_dim=2,
            output_dim=1,
            # Paper/SRS Table 4 default: width 64, depth 3, tanh.
            params={"layer_sizes": [2, 64, 64, 64, 1], "activation": "tanh"},
        )

    def sample_dataset(self, n: int, seed: int = 0) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        x = rng.uniform(self.domain[0], self.domain[1], size=(n, 2))
        return {
            "x_train": x.astype(np.float32),
            "y_train": self.f(x).astype(np.float32),
        }

    def local_loss(self, model, arrays, batch_size: int = 256) -> torch.Tensor:
        x = torch.as_tensor(arrays["x_train"], dtype=torch.float32)
        y = torch.as_tensor(arrays["y_train"], dtype=torch.float32)
        idx = torch.randint(0, x.shape[0], (min(batch_size, x.shape[0]),))
        return torch.mean((model(x[idx]) - y[idx]) ** 2)

    def evaluate(self, model, num_test: int = 400, seed: int = 1234) -> Dict[str, float]:
        rng = np.random.default_rng(seed)
        x = rng.uniform(self.domain[0], self.domain[1], size=(num_test, 2)).astype(np.float32)
        with torch.no_grad():
            pred = model(torch.as_tensor(x)).cpu().numpy()
        return {"l2_relative_error": relative_l2(pred, self.f(x))}


class Poisson1D(Problem):
    """1D Poisson equation ``-u'' = f`` with hard-constrained PINN (paper's setup)."""

    name = "poisson"
    family = "pinn"
    description = "1D Poisson -u''=f with u(0)=0,u(pi)=pi enforced exactly by output transform"
    required_arrays = ["x_train"]
    domain = (0.0, float(np.pi))

    @staticmethod
    def u_true(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        sol = x + (1.0 / 8.0) * np.sin(8.0 * x)
        for i in range(1, 5):
            sol = sol + (1.0 / i) * np.sin(i * x)
        return sol

    @staticmethod
    def forcing(x: torch.Tensor) -> torch.Tensor:
        f = 8.0 * torch.sin(8.0 * x)
        for i in range(1, 5):
            f = f + i * torch.sin(i * x)
        return f

    def model_spec(self) -> ModelSpec:
        return ModelSpec(
            family="mlp",
            input_dim=1,
            output_dim=1,
            params={
                "layer_sizes": [1, 20, 20, 20, 1],
                "activation": "tanh",
                "output_transform": "poisson1d_hard",
            },
        )

    def sample_dataset(self, n: int, seed: int = 0) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        x = rng.uniform(self.domain[0], self.domain[1], size=(n, 1))
        return {"x_train": x.astype(np.float32)}

    def local_loss(self, model, arrays, batch_size: int = 128) -> torch.Tensor:
        x = torch.as_tensor(arrays["x_train"], dtype=torch.float32)
        idx = torch.randint(0, x.shape[0], (min(batch_size, x.shape[0]),))
        xb = x[idx].clone().detach().requires_grad_(True)
        y = model(xb)
        dy = torch.autograd.grad(y, xb, torch.ones_like(y), create_graph=True)[0]
        d2y = torch.autograd.grad(dy, xb, torch.ones_like(dy), create_graph=True)[0]
        residual = -d2y - self.forcing(xb)
        return torch.mean(residual ** 2)

    def evaluate(self, model, num_test: int = 100, seed: int = 1234) -> Dict[str, float]:
        x = np.linspace(self.domain[0], self.domain[1], num_test).reshape(-1, 1).astype(np.float32)
        with torch.no_grad():
            pred = model(torch.as_tensor(x)).cpu().numpy()
        return {"l2_relative_error": relative_l2(pred, self.u_true(x))}


class Antiderivative(Problem):
    """Antiderivative operator with DeepONet (FedDeepONet benchmark).

    Sample functions are truncated Fourier series ``u(x) = sum a_k sin(k*pi*x)``
    with ``u(0)=0``; the branch input is ``v = u'`` evaluated at fixed sensors.
    """

    name = "antiderivative"
    family = "operator"
    description = "DeepONet learns u(x)=int_0^x v(t)dt from 50 sensor values of v"
    required_arrays = ["branch_train", "trunk_train", "y_train"]
    K = 5
    N_SENSORS = 50
    N_TRUNK = 40

    def __init__(self) -> None:
        self._k = np.arange(1, self.K + 1, dtype=np.float64)
        self.sensors = np.linspace(0.0, 1.0, self.N_SENSORS)
        self.trunk = np.linspace(0.0, 1.0, self.N_TRUNK).reshape(-1, 1)

    def _sample_coeffs(self, n: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.normal(size=(n, self.K)) / self._k[None, :]

    def _u(self, coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
        s = np.sin(np.pi * np.outer(np.asarray(x).reshape(-1), self._k))
        return coeffs @ s.T

    def _v(self, coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
        c = np.cos(np.pi * np.outer(np.asarray(x).reshape(-1), self._k))
        return coeffs @ (self._k[None, :] * np.pi * c).T

    def model_spec(self) -> ModelSpec:
        return ModelSpec(
            family="deeponet",
            input_dim=self.N_SENSORS,
            output_dim=1,
            params={
                # Paper/SRS Table 4 default: width 40, depth 2, ReLU.
                "branch_sizes": [self.N_SENSORS, 40, 40],
                "trunk_sizes": [1, 40, 40],
                "activation": "relu",
            },
        )

    def sample_dataset(self, n: int, seed: int = 0) -> Dict[str, np.ndarray]:
        coeffs = self._sample_coeffs(n, seed)
        return {
            "branch_train": self._v(coeffs, self.sensors).astype(np.float32),
            "trunk_train": self.trunk.astype(np.float32),
            "y_train": self._u(coeffs, self.trunk).astype(np.float32),
        }

    def local_loss(self, model, arrays, batch_size: int = 32) -> torch.Tensor:
        branch = torch.as_tensor(arrays["branch_train"], dtype=torch.float32)
        trunk = torch.as_tensor(arrays["trunk_train"], dtype=torch.float32)
        y = torch.as_tensor(arrays["y_train"], dtype=torch.float32)
        idx = torch.randint(0, branch.shape[0], (min(batch_size, branch.shape[0]),))
        pred = model(branch[idx], trunk)
        return torch.mean((pred - y[idx]) ** 2)

    def evaluate(self, model, num_test: int = 64, seed: int = 4321) -> Dict[str, float]:
        coeffs = self._sample_coeffs(num_test, seed)
        branch = torch.as_tensor(self._v(coeffs, self.sensors), dtype=torch.float32)
        trunk = torch.as_tensor(self.trunk, dtype=torch.float32)
        with torch.no_grad():
            pred = model(branch, trunk).cpu().numpy()
        return {"l2_relative_error": relative_l2(pred, self._u(coeffs, self.trunk))}


_PROBLEM_CLASSES = [GramacyLee1D, Schaffer2D, Poisson1D, Antiderivative]
_REGISTRY: Dict[str, Problem] = {cls().name: cls() for cls in _PROBLEM_CLASSES}


def get_problem(name: str) -> Problem:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown problem '{name}'. Options: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_problems() -> List[str]:
    return sorted(_REGISTRY)


def build_problem_model(problem_name: str):
    """Return ``(problem, model)`` for the given problem name."""
    problem = get_problem(problem_name)
    return problem, build_model(problem.model_spec())
