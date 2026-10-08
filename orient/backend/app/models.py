"""Pure-PyTorch model factory shared by the Server app and the Client app.

Only depends on ``torch`` so that both apps can rebuild the *exact* same
architecture from a :class:`ModelSpec`, guaranteeing identical ``state_dict``
keys and shapes (FR-PROTO-2).

This module is duplicated verbatim in ``client/app/models.py``.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from .protocol import ModelSpec


class Sine(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


_ACTIVATIONS: Dict[str, type] = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "sigmoid": nn.Sigmoid,
    "sin": Sine,
    "sine": Sine,
}


def _activation(name: str) -> nn.Module:
    if name not in _ACTIVATIONS:
        raise ValueError(f"Unknown activation '{name}'. Options: {sorted(_ACTIVATIONS)}")
    return _ACTIVATIONS[name]()


def poisson1d_hard_transform(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Hard-constraint output transform for u(0)=0, u(pi)=pi (paper's PINN)."""
    return x + torch.tanh(x) * torch.tanh(np.pi - x) * y


def helmholtz_hard_transform(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Hard constraint for 2D Helmholtz: u=0 on boundary of [0,1]^2."""
    # x shape (...,2), y shape (...,1)
    if x.shape[-1] == 2:
        x0 = x[..., 0:1]
        x1 = x[..., 1:2]
        return x0 * (1 - x0) * x1 * (1 - x1) * y
    return y


def allen_cahn_transform(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Allen-Cahn transform: initial condition u(x,0)=x^2 cos(pi x), periodic?"""
    # x shape (...,2) where [:,0]=x_spatial, [:,1]=t
    if x.shape[-1] == 2:
        x_sp = x[..., 0:1]
        t = x[..., 1:2]
        # u = x^2 cos(pi x) at t=0, smooth transition
        ic = torch.square(x_sp) * torch.cos(np.pi * x_sp)
        return t * (1 - torch.square(x_sp)) * y + ic
    return y


_TRANSFORMS: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "poisson1d_hard": poisson1d_hard_transform,
    "helmholtz_hard": helmholtz_hard_transform,
    "allen_cahn": allen_cahn_transform,
}


def resolve_transform(name: Optional[str]):
    if not name:
        return None
    if name not in _TRANSFORMS:
        raise ValueError(f"Unknown output transform '{name}'. Options: {sorted(_TRANSFORMS)}")
    return _TRANSFORMS[name]


class MLP(nn.Module):
    """Fully-connected network with an optional output transform."""

    def __init__(
        self,
        layer_sizes: List[int],
        activation: str = "tanh",
        output_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(_activation(activation))
        self.net = nn.Sequential(*layers)
        self.output_transform = output_transform

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        if self.output_transform is not None:
            y = self.output_transform(x, y)
        return y


class DeepONet(nn.Module):
    """DeepONet with a Cartesian-product output: ``B(branch) @ T(trunk)^T + bias``."""

    def __init__(
        self,
        branch_sizes: List[int],
        trunk_sizes: List[int],
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        self.branch = MLP(branch_sizes, activation)
        self.trunk = MLP(trunk_sizes, activation)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, branch_x: torch.Tensor, trunk_x: torch.Tensor) -> torch.Tensor:
        b = self.branch(branch_x)          # (n, p)
        t = self.trunk(trunk_x)            # (m, p)
        return b @ t.T + self.bias         # (n, m)


def build_model(spec: ModelSpec) -> nn.Module:
    """Build the architecture described by ``spec``. Raises ValueError if unknown."""
    family = spec.family.lower()
    params = dict(spec.params)

    if family == "mlp":
        transform = resolve_transform(params.pop("output_transform", None))
        return MLP(params["layer_sizes"], params.get("activation", "tanh"), transform)

    if family == "deeponet":
        return DeepONet(
            params["branch_sizes"],
            params["trunk_sizes"],
            params.get("activation", "tanh"),
        )

    raise ValueError(f"Unknown model family '{spec.family}'. Options: ['mlp', 'deeponet']")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
