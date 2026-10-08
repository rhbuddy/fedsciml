"""Generate example dataset bundles for demo clients.

Writes *heterogeneous* local datasets so the federated effect is visible:
each client only sees a sub-domain (or a subset of Fourier modes).

Usage (from the ``client/`` directory)::

    python examples/make_datasets.py
    python examples/make_datasets.py --clients 5      # K>=2f+3, needed for Krum
    python examples/make_datasets.py --out examples/bundles --samples 1500

Why more than 2 clients? Byzantine-robust rules such as Krum need ``K >= 2f + 3``
participants to be meaningful; with only 2 clients Krum just picks one of the two
updates, which is not a real aggregation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.dataset import save_bundle  # noqa: E402
from app.problems import Antiderivative, GramacyLee1D, Schaffer2D, get_problem  # noqa: E402


def make_gramacy_lee(out: Path, name: str, n: int, seed: int, lo: float, hi: float) -> None:
    rng = np.random.default_rng(seed)
    x = rng.uniform(lo, hi, size=(n, 1))
    save_bundle(
        out / name,
        "gramacy_lee",
        {"x_train": x.astype(np.float32), "y_train": GramacyLee1D.f(x).astype(np.float32)},
        domain={"interval": [lo, hi]},
    )


def make_schaffer(out: Path, name: str, n: int, seed: int, lo: float, hi: float) -> None:
    rng = np.random.default_rng(seed)
    # Split along x1 and keep the full x2 range so the clients form a clean partition.
    x = np.column_stack([rng.uniform(lo, hi, size=n), rng.uniform(-2.0, 2.0, size=n)])
    save_bundle(
        out / name,
        "schaffer",
        {"x_train": x.astype(np.float32), "y_train": Schaffer2D.f(x).astype(np.float32)},
        domain={"x1_interval": [lo, hi], "x2_interval": [-2.0, 2.0]},
    )


def make_poisson(out: Path, name: str, n: int, seed: int, lo: float, hi: float) -> None:
    rng = np.random.default_rng(seed)
    x = rng.uniform(lo, hi, size=(n, 1))
    save_bundle(out / name, "poisson", {"x_train": x.astype(np.float32)}, domain={"interval": [lo, hi]})


def make_antiderivative(out: Path, name: str, n: int, seed: int, modes: list[int]) -> None:
    problem: Antiderivative = get_problem("antiderivative")  # type: ignore[assignment]
    rng = np.random.default_rng(seed)
    coeffs = rng.normal(size=(n, problem.K)) / problem._k[None, :]
    mask = np.ones(problem.K, dtype=bool)
    mask[[m - 1 for m in modes]] = False
    coeffs[:, mask] = 0.0
    save_bundle(
        out / name,
        "antiderivative",
        {
            "branch_train": problem._v(coeffs, problem.sensors).astype(np.float32),
            "trunk_train": problem.trunk.astype(np.float32),
            "y_train": problem._u(coeffs, problem.trunk).astype(np.float32),
        },
        domain={"fourier_modes": modes},
    )


def build(out: Path, n: int, n_clients: int) -> None:
    """Partition each problem's domain into ``n_clients`` disjoint slices."""
    print(f"Writing bundles to {out} ({n} samples, {n_clients} clients per problem)")

    gl_edges = np.linspace(0.5, 2.5, n_clients + 1)
    for i in range(n_clients):
        make_gramacy_lee(out, f"gramacy_lee_client{i + 1}", n, 100 + i, gl_edges[i], gl_edges[i + 1])

    sc_edges = np.linspace(-2.0, 2.0, n_clients + 1)
    for i in range(n_clients):
        make_schaffer(out, f"schaffer_client{i + 1}", n, 200 + i, sc_edges[i], sc_edges[i + 1])

    po_edges = np.linspace(0.0, float(np.pi), n_clients + 1)
    for i in range(n_clients):
        make_poisson(out, f"poisson_client{i + 1}", n, 300 + i, po_edges[i], po_edges[i + 1])

    # Each client only sees a subset of the Fourier modes -> operator heterogeneity.
    rng = np.random.default_rng(999)
    problem = get_problem("antiderivative")
    for i in range(n_clients):
        size = int(rng.integers(2, problem.K + 1))
        modes = sorted(int(m) for m in rng.choice(np.arange(1, problem.K + 1), size=size, replace=False))
        make_antiderivative(out, f"antiderivative_client{i + 1}", n, 400 + i, modes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create example dataset bundles.")
    parser.add_argument("--out", default=str(Path(__file__).parent / "bundles"))
    parser.add_argument("--samples", type=int, default=1500)
    parser.add_argument(
        "--clients",
        type=int,
        default=2,
        help="clients per problem (use >=5 to exercise Krum with f=1)",
    )
    args = parser.parse_args()

    out = Path(args.out)
    build(out, args.samples, max(1, args.clients))

    names = sorted(p.name for p in out.iterdir() if p.is_dir())
    print(f"  - {len(names)} bundles:")
    for name in names:
        print(f"      {name}")
    print("Done.")


if __name__ == "__main__":
    main()
