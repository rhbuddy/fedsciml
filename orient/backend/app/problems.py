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


class Helmholtz2D(Problem):
    """2D Helmholtz equation -Δu - k0² u = f on [0,1]² with u=0 on boundary (hard constraint)."""

    name = "helmholtz"
    family = "pinn"
    description = "2D Helmholtz -Δu - k0² u = k0² sin(k0 x) sin(k0 y), k0=4π, u=0 on boundary"
    required_arrays = ["x_train"]
    k0 = 4 * np.pi  # n=2 in paper
    domain = (0.0, 1.0)

    @staticmethod
    def u_true(xy: np.ndarray) -> np.ndarray:
        k0 = 4 * np.pi
        return np.sin(k0 * xy[:, 0:1]) * np.sin(k0 * xy[:, 1:2])

    def model_spec(self) -> ModelSpec:
        return ModelSpec(
            family="mlp",
            input_dim=2,
            output_dim=1,
            params={
                "layer_sizes": [2, 64, 64, 64, 1],
                "activation": "sine",
                "output_transform": "helmholtz_hard",
            },
        )

    def sample_dataset(self, n: int, seed: int = 0) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        x = rng.uniform(0, 1, size=(n, 2))
        return {"x_train": x.astype(np.float32)}

    def local_loss(self, model, arrays, batch_size: int = 256) -> torch.Tensor:
        x = torch.as_tensor(arrays["x_train"], dtype=torch.float32)
        idx = torch.randint(0, x.shape[0], (min(batch_size, x.shape[0]),))
        xb = x[idx].clone().detach().requires_grad_(True)
        y = model(xb)
        grads = torch.autograd.grad(y, xb, torch.ones_like(y), create_graph=True)[0]
        dy_dx = grads[:, 0:1]
        dy_dy = grads[:, 1:2]
        d2y_dx2 = torch.autograd.grad(dy_dx, xb, torch.ones_like(dy_dx), create_graph=True, retain_graph=True)[0][:, 0:1]
        d2y_dy2 = torch.autograd.grad(dy_dy, xb, torch.ones_like(dy_dy), create_graph=True)[0][:, 1:2]
        lap = d2y_dx2 + d2y_dy2
        k0 = self.k0
        f = (k0 ** 2) * torch.sin(k0 * xb[:, 0:1]) * torch.sin(k0 * xb[:, 1:2])
        residual = -lap - (k0 ** 2) * y - f
        return torch.mean(residual ** 2)

    def evaluate(self, model, num_test: int = 400, seed: int = 1234) -> Dict[str, float]:
        rng = np.random.default_rng(seed)
        xy = rng.uniform(0, 1, size=(num_test, 2)).astype(np.float32)
        with torch.no_grad():
            pred = model(torch.as_tensor(xy)).cpu().numpy()
        return {"l2_relative_error": relative_l2(pred, self.u_true(xy))}


class AllenCahn(Problem):
    """Allen-Cahn equation u_t - 0.001 u_xx + 5(u³ - u)=0 on x∈[-1,1], t∈[0,1]."""

    name = "allen_cahn"
    family = "pinn"
    description = "Allen-Cahn u_t -0.001 u_xx +5(u³-u)=0, u(x,0)=x² cos(π x)"
    required_arrays = ["x_train"]
    domain = (-1.0, 1.0)  # spatial; temporal is [0,1] in second column

    @staticmethod
    def u_true(xt: np.ndarray) -> np.ndarray:
        # Manufactured smooth solution for evaluation: decay of initial hump
        x = xt[:, 0:1]
        t = xt[:, 1:2]
        return (x ** 2) * np.cos(np.pi * x) * np.exp(-5 * t)

    def model_spec(self) -> ModelSpec:
        return ModelSpec(
            family="mlp",
            input_dim=2,
            output_dim=1,
            params={
                "layer_sizes": [2, 64, 64, 64, 1],
                "activation": "tanh",
                "output_transform": "allen_cahn",
            },
        )

    def sample_dataset(self, n: int, seed: int = 0) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        x = rng.uniform(-1, 1, size=(n, 1))
        t = rng.uniform(0, 1, size=(n, 1))
        xt = np.hstack([x, t])
        return {"x_train": xt.astype(np.float32)}

    def local_loss(self, model, arrays, batch_size: int = 256) -> torch.Tensor:
        x = torch.as_tensor(arrays["x_train"], dtype=torch.float32)
        idx = torch.randint(0, x.shape[0], (min(batch_size, x.shape[0]),))
        xb = x[idx].clone().detach().requires_grad_(True)
        y = model(xb)
        grads = torch.autograd.grad(y, xb, torch.ones_like(y), create_graph=True)[0]
        dy_dx = grads[:, 0:1]
        dy_dt = grads[:, 1:2]
        d2y_dx2 = torch.autograd.grad(dy_dx, xb, torch.ones_like(dy_dx), create_graph=True)[0][:, 0:1]
        residual = dy_dt - 0.001 * d2y_dx2 + 5 * (y ** 3 - y)
        return torch.mean(residual ** 2)

    def evaluate(self, model, num_test: int = 400, seed: int = 1234) -> Dict[str, float]:
        rng = np.random.default_rng(seed)
        x = rng.uniform(-1, 1, size=(num_test, 1))
        t = rng.uniform(0, 1, size=(num_test, 1))
        xt = np.hstack([x, t]).astype(np.float32)
        with torch.no_grad():
            pred = model(torch.as_tensor(xt)).cpu().numpy()
        return {"l2_relative_error": relative_l2(pred, self.u_true(xt))}


class InverseNS(Problem):
    """Inverse Navier-Stokes: infer velocity field from sparse observations.

    Simplified to a supervised operator: (x,y,t) -> (u,v,p) with synthetic vortex.
    Retains the paper's architecture [3,50×6,3] tanh and physics flavor.
    """

    name = "inverse_ns"
    family = "pinn"
    description = "Inverse NS (simplified): (x,y,t)->(u,v,p) vortex, infer flow"
    required_arrays = ["x_train", "y_train"]

    def model_spec(self) -> ModelSpec:
        return ModelSpec(
            family="mlp",
            input_dim=3,
            output_dim=3,
            params={"layer_sizes": [3, 50, 50, 50, 50, 50, 50, 3], "activation": "tanh"},
        )

    @staticmethod
    def _field(xyt: np.ndarray) -> np.ndarray:
        x = xyt[:, 0:1]
        y = xyt[:, 1:2]
        t = xyt[:, 2:3]
        # synthetic divergence-free vortex decaying in time
        u = -np.sin(np.pi * x) * np.cos(np.pi * y) * np.exp(-t)
        v = np.cos(np.pi * x) * np.sin(np.pi * y) * np.exp(-t)
        p = np.sin(np.pi * x) * np.sin(np.pi * y) * np.exp(-t) * 0.5
        return np.hstack([u, v, p])

    def sample_dataset(self, n: int, seed: int = 0) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        # x∈[1,8], y∈[-2,2], t∈[0,1] per paper
        x = rng.uniform(1, 8, size=(n, 1))
        y = rng.uniform(-2, 2, size=(n, 1))
        t = rng.uniform(0, 1, size=(n, 1))
        xyt = np.hstack([x, y, t])
        return {
            "x_train": xyt.astype(np.float32),
            "y_train": self._field(xyt).astype(np.float32),
        }

    def local_loss(self, model, arrays, batch_size: int = 256) -> torch.Tensor:
        x = torch.as_tensor(arrays["x_train"], dtype=torch.float32)
        y = torch.as_tensor(arrays["y_train"], dtype=torch.float32)
        idx = torch.randint(0, x.shape[0], (min(batch_size, x.shape[0]),))
        return torch.mean((model(x[idx]) - y[idx]) ** 2)

    def evaluate(self, model, num_test: int = 400, seed: int = 1234) -> Dict[str, float]:
        rng = np.random.default_rng(seed)
        x = rng.uniform(1, 8, size=(num_test, 1))
        y = rng.uniform(-2, 2, size=(num_test, 1))
        t = rng.uniform(0, 1, size=(num_test, 1))
        xyt = np.hstack([x, y, t]).astype(np.float32)
        with torch.no_grad():
            pred = model(torch.as_tensor(xyt)).cpu().numpy()
        return {"l2_relative_error": relative_l2(pred, self._field(xyt))}


class InverseDR(Problem):
    """Inverse diffusion-reaction: recover reaction rate k(x) and state u(x).

    Simplified to supervised (x -> [u,k]) with k(x)=1+exp(-0.5*(x-0.5)²/1.05²),
    u(x) = BVP solution approximated by sin envelope.
    """

    name = "inverse_dr"
    family = "pinn"
    description = "Inverse diffusion-reaction x->(u,k), l=0.01, 1D BVP"
    required_arrays = ["x_train", "y_train"]

    @staticmethod
    def k_true(x: np.ndarray) -> np.ndarray:
        return 1.0 + np.exp(-0.5 * (x - 0.5) ** 2 / 1.05 ** 2)

    @staticmethod
    def u_true(x: np.ndarray) -> np.ndarray:
        return np.sin(np.pi * x) * np.exp(-10 * (x - 0.5) ** 2)

    def model_spec(self) -> ModelSpec:
        return ModelSpec(
            family="mlp",
            input_dim=1,
            output_dim=2,
            params={"layer_sizes": [1, 20, 20, 20, 2], "activation": "tanh"},
        )

    def sample_dataset(self, n: int, seed: int = 0) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        x = rng.uniform(0, 1, size=(n, 1))
        k = self.k_true(x)
        u = self.u_true(x)
        y = np.hstack([u, k])
        return {"x_train": x.astype(np.float32), "y_train": y.astype(np.float32)}

    def local_loss(self, model, arrays, batch_size: int = 256) -> torch.Tensor:
        x = torch.as_tensor(arrays["x_train"], dtype=torch.float32)
        y = torch.as_tensor(arrays["y_train"], dtype=torch.float32)
        idx = torch.randint(0, x.shape[0], (min(batch_size, x.shape[0]),))
        return torch.mean((model(x[idx]) - y[idx]) ** 2)

    def evaluate(self, model, num_test: int = 400, seed: int = 1234) -> Dict[str, float]:
        rng = np.random.default_rng(seed)
        x = rng.uniform(0, 1, size=(num_test, 1)).astype(np.float32)
        with torch.no_grad():
            pred = model(torch.as_tensor(x)).cpu().numpy()
        true = np.hstack([self.u_true(x), self.k_true(x)]).astype(np.float64)
        return {"l2_relative_error": relative_l2(pred, true)}


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


class BurgersDeepONet(Problem):
    """Burgers operator: initial condition -> solution field (DeepONet).

    Simplified Fourier IC + viscous Burgers solver approximation via pseudo-spectral
    proxy; branch sees IC at 101 sensors, trunk is (x,t) in [0,1]².
    """

    name = "burgers"
    family = "operator"
    description = "DeepONet Burgers: IC (101 sensors) -> u(x,t) field, nu=0.1"
    required_arrays = ["branch_train", "trunk_train", "y_train"]
    N_SENSORS = 101
    N_TRUNK = 101

    def __init__(self) -> None:
        self.sensors = np.linspace(0, 1, self.N_SENSORS)
        xs = np.linspace(0, 1, 11)
        ts = np.linspace(0, 1, 11)
        xx, tt = np.meshgrid(xs, ts)
        self.trunk = np.stack([xx.ravel(), tt.ravel()], axis=1)  # (121,2)

    def _sample_ic(self, n: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        coeffs = rng.normal(size=(n, 5)) * 0.5
        x = self.sensors[None, :]
        waves = np.array([np.sin(k * np.pi * x) for k in range(1, 6)])  # (5,1,101)
        # combine (n,5) @ (5,101) -> (n,101)
        ics = np.tensordot(coeffs, waves[:, 0, :], axes=(1, 0))
        return ics

    def _solution(self, ic: np.ndarray, trunk: np.ndarray) -> np.ndarray:
        # proxy: advect+diffuse: u(x,t) ≈ IC(x - c*t) * exp(-nu*k² t)
        # for demo we use linear superposition decay
        n = ic.shape[0]
        m = trunk.shape[0]
        out = np.zeros((n, m), dtype=np.float64)
        for i in range(n):
            # reconstruct IC as Fourier, evaluate at shifted x
            # simple: average IC value as proxy for field
            base = float(np.mean(ic[i]))
            for j, (x, t) in enumerate(trunk):
                out[i, j] = base * np.exp(-0.1 * t) * np.cos(np.pi * (x - 0.2 * t))
        return out

    def model_spec(self) -> ModelSpec:
        return ModelSpec(
            family="deeponet",
            input_dim=self.N_SENSORS,
            output_dim=1,
            params={
                "branch_sizes": [self.N_SENSORS, 64, 64],
                "trunk_sizes": [2, 64, 64],
                "activation": "relu",
            },
        )

    def sample_dataset(self, n: int, seed: int = 0) -> Dict[str, np.ndarray]:
        ics = self._sample_ic(n, seed)
        y = self._solution(ics, self.trunk)
        return {
            "branch_train": ics.astype(np.float32),
            "trunk_train": self.trunk.astype(np.float32),
            "y_train": y.astype(np.float32),
        }

    def local_loss(self, model, arrays, batch_size: int = 32) -> torch.Tensor:
        branch = torch.as_tensor(arrays["branch_train"], dtype=torch.float32)
        trunk = torch.as_tensor(arrays["trunk_train"], dtype=torch.float32)
        y = torch.as_tensor(arrays["y_train"], dtype=torch.float32)
        idx = torch.randint(0, branch.shape[0], (min(batch_size, branch.shape[0]),))
        pred = model(branch[idx], trunk)
        return torch.mean((pred - y[idx]) ** 2)

    def evaluate(self, model, num_test: int = 64, seed: int = 4321) -> Dict[str, float]:
        ics = self._sample_ic(num_test, seed)
        branch = torch.as_tensor(ics, dtype=torch.float32)
        trunk = torch.as_tensor(self.trunk, dtype=torch.float32)
        with torch.no_grad():
            pred = model(branch, trunk).cpu().numpy()
        true = self._solution(ics, self.trunk)
        return {"l2_relative_error": relative_l2(pred, true)}


class DiffusionReactionDeepONet(Problem):
    """Diffusion-reaction operator (DeepONet): source -> solution field."""

    name = "diffusion_reaction"
    family = "operator"
    description = "DeepONet diffusion-reaction: source (101 sensors) -> u field, 3×100 ReLU"
    required_arrays = ["branch_train", "trunk_train", "y_train"]
    N_SENSORS = 101
    N_TRUNK = 100

    def __init__(self) -> None:
        self.sensors = np.linspace(0, 1, self.N_SENSORS)
        self.trunk = np.linspace(0, 1, self.N_TRUNK).reshape(-1, 1)

    def _sample_source(self, n: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        # Chebyshev-like source: sum a_i T_i(x)
        coeffs = rng.uniform(-1, 1, size=(n, 6))
        x = self.sensors[None, :]
        # T_i approx cos(i arccos(2x-1))
        ar = np.arccos(2 * x - 1)
        basis = np.array([np.cos(i * ar[0]) for i in range(6)])  # (6,101)
        src = coeffs @ basis
        return src

    def _solution(self, src: np.ndarray) -> np.ndarray:
        # proxy elliptic solve: u ≈ -0.1 * src + smoothing
        n = src.shape[0]
        m = self.trunk.shape[0]
        out = np.zeros((n, m), dtype=np.float64)
        for i in range(n):
            # simple convolution smoothing
            smoothed = np.convolve(src[i], np.ones(5) / 5, mode="same")
            # interpolate to trunk points (same grid)
            xp = self.sensors
            fp = smoothed
            xt = self.trunk[:, 0]
            out[i] = np.interp(xt, xp, fp) * 0.3
        return out

    def model_spec(self) -> ModelSpec:
        return ModelSpec(
            family="deeponet",
            input_dim=self.N_SENSORS,
            output_dim=1,
            params={
                "branch_sizes": [self.N_SENSORS, 100, 100, 100],
                "trunk_sizes": [1, 100, 100, 100],
                "activation": "relu",
            },
        )

    def sample_dataset(self, n: int, seed: int = 0) -> Dict[str, np.ndarray]:
        src = self._sample_source(n, seed)
        y = self._solution(src)
        return {
            "branch_train": src.astype(np.float32),
            "trunk_train": self.trunk.astype(np.float32),
            "y_train": y.astype(np.float32),
        }

    def local_loss(self, model, arrays, batch_size: int = 32) -> torch.Tensor:
        branch = torch.as_tensor(arrays["branch_train"], dtype=torch.float32)
        trunk = torch.as_tensor(arrays["trunk_train"], dtype=torch.float32)
        y = torch.as_tensor(arrays["y_train"], dtype=torch.float32)
        idx = torch.randint(0, branch.shape[0], (min(batch_size, branch.shape[0]),))
        pred = model(branch[idx], trunk)
        return torch.mean((pred - y[idx]) ** 2)

    def evaluate(self, model, num_test: int = 64, seed: int = 4321) -> Dict[str, float]:
        src = self._sample_source(num_test, seed)
        branch = torch.as_tensor(src, dtype=torch.float32)
        trunk = torch.as_tensor(self.trunk, dtype=torch.float32)
        with torch.no_grad():
            pred = model(branch, trunk).cpu().numpy()
        true = self._solution(src)
        return {"l2_relative_error": relative_l2(pred, true)}


_PROBLEM_CLASSES = [
    GramacyLee1D,
    Schaffer2D,
    Poisson1D,
    Helmholtz2D,
    AllenCahn,
    InverseNS,
    InverseDR,
    Antiderivative,
    BurgersDeepONet,
    DiffusionReactionDeepONet,
]
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
