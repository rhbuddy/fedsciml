"""Load and validate a client dataset bundle (SRS v1.4 §4.13).

A bundle is a directory::

    mydata/
    ├── dataset.json      {"problem": "poisson", "n_samples": 3200, "domain": {...}}
    └── data.npz          arrays keyed by problem family:
                          #   supervised : x_train, y_train
                          #   pinn       : x_train
                          #   operator   : branch_train, trunk_train, y_train

Raw arrays never leave this machine (FR-CLIENTAPP-9).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

DEFAULT_MANIFEST = "dataset.json"
DEFAULT_ARRAYS = "data.npz"


@dataclass
class DatasetBundle:
    path: Path
    problem: str
    n_samples: int
    domain: Dict[str, Any] = field(default_factory=dict)
    arrays: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def size(self) -> int:
        """Number of local samples (used for data-size weighting upstream)."""
        for key in ("y_train", "x_train", "branch_train"):
            if key in self.arrays:
                return int(self.arrays[key].shape[0])
        return int(self.n_samples)


def _resolve_paths(path: str | Path) -> tuple[Path, Path]:
    root = Path(path)
    if root.is_dir():
        return root / DEFAULT_MANIFEST, root / DEFAULT_ARRAYS
    return root, root.parent / DEFAULT_ARRAYS


def load_bundle(path: str | Path, problem: Optional[str] = None) -> DatasetBundle:
    """Load a bundle, validating it against the problem registry."""
    manifest_path, arrays_path = _resolve_paths(path)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")
    if not arrays_path.exists():
        raise FileNotFoundError(f"Dataset arrays not found: {arrays_path}")

    manifest: Dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    bundle_problem = manifest.get("problem")
    if problem and bundle_problem and problem != bundle_problem:
        raise ValueError(
            f"Dataset is for problem '{bundle_problem}', "
            f"but the client is configured for '{problem}'"
        )
    resolved_problem = problem or bundle_problem
    if not resolved_problem:
        raise ValueError("No problem specified in the manifest or the CLI arguments")

    # Imported lazily so that a bad problem name fails with a friendly message.
    from .problems import get_problem

    problem_obj = get_problem(resolved_problem)

    arrays = {k: np.asarray(v) for k, v in np.load(str(arrays_path)).items()}
    missing = [k for k in problem_obj.required_arrays if k not in arrays]
    if missing:
        raise ValueError(
            f"Dataset bundle is missing arrays {missing} required by "
            f"problem '{resolved_problem}' (expected {problem_obj.required_arrays})"
        )

    # FR-CLIENTAPP-8: reject arrays that do not match the ProblemSpec's shapes.
    for name in problem_obj.required_arrays:
        array = arrays[name]
        if array.ndim != 2:
            raise ValueError(
                f"Array '{name}' must be 2-D (samples x features), got shape {array.shape}. "
                f"Reshape it to (n, d) and re-upload."
            )
        if array.shape[0] == 0:
            raise ValueError(f"Array '{name}' is empty (shape {array.shape}).")
        if not np.issubdtype(array.dtype, np.number):
            raise ValueError(f"Array '{name}' must be numeric, got dtype {array.dtype}.")
        if not np.isfinite(array).all():
            raise ValueError(f"Array '{name}' contains NaN/Inf values.")

    if "branch_train" in arrays and "y_train" in arrays:
        if arrays["branch_train"].shape[0] != arrays["y_train"].shape[0]:
            raise ValueError(
                f"branch_train ({arrays['branch_train'].shape}) and y_train "
                f"({arrays['y_train'].shape}) must have the same number of samples."
            )
    if "x_train" in arrays and "y_train" in arrays:
        if arrays["x_train"].shape[0] != arrays["y_train"].shape[0]:
            raise ValueError(
                f"x_train ({arrays['x_train'].shape}) and y_train "
                f"({arrays['y_train'].shape}) must have the same number of samples."
            )

    bundle = DatasetBundle(
        path=manifest_path,
        problem=resolved_problem,
        n_samples=int(manifest.get("n_samples", 0)),
        domain=dict(manifest.get("domain") or {}),
        arrays=arrays,
    )
    if bundle.n_samples <= 0:
        bundle.n_samples = bundle.size
    return bundle


def save_bundle(
    directory: str | Path,
    problem: str,
    arrays: Dict[str, np.ndarray],
    domain: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write a bundle to disk (used to bootstrap example datasets)."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    n_samples = 0
    for key in ("y_train", "x_train", "branch_train"):
        if key in arrays:
            n_samples = int(np.asarray(arrays[key]).shape[0])
            break
    manifest = {
        "problem": problem,
        "n_samples": n_samples,
        "domain": dict(domain or {}),
    }
    (root / DEFAULT_MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    np.savez_compressed(root / DEFAULT_ARRAYS, **{k: np.asarray(v) for k, v in arrays.items()})
    return root
