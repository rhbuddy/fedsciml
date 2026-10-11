"""Generate example dataset bundles for demo clients.

Writes *heterogeneous* local datasets so the federated effect is visible:
each client only sees a sub-domain (or a subset of Fourier modes).

Usage (from the ``client/`` directory)::

    python examples/make_datasets.py
    python examples/make_datasets.py --clients 5      # K>=2f+3, needed for Krum
    python examples/make_datasets.py --out examples/bundles --samples 1500
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


def make_helmholtz(out: Path, name: str, n: int, seed: int, lo_x: float, hi_x: float) -> None:
    """Heterogeneity via x-slice; y stays full [0,1]."""
    rng = np.random.default_rng(seed)
    x = np.column_stack([rng.uniform(lo_x, hi_x, size=n), rng.uniform(0, 1, size=n)])
    save_bundle(out / name, "helmholtz", {"x_train": x.astype(np.float32)}, domain={"x_interval": [lo_x, hi_x], "y_interval": [0, 1]})


def make_allen_cahn(out: Path, name: str, n: int, seed: int, lo_x: float, hi_x: float) -> None:
    rng = np.random.default_rng(seed)
    x = rng.uniform(lo_x, hi_x, size=n)
    t = rng.uniform(0, 1, size=n)
    xt = np.column_stack([x, t])
    save_bundle(out / name, "allen_cahn", {"x_train": xt.astype(np.float32)}, domain={"x_interval": [lo_x, hi_x]})


def make_inverse_ns(out: Path, name: str, n: int, seed: int, lo_x: float, hi_x: float) -> None:
    problem = get_problem("inverse_ns")
    ds = problem.sample_dataset(n, seed=seed)
    # slice by x coordinate is already heterogeneous via sampling; just tag domain
    save_bundle(out / name, "inverse_ns", ds, domain={"x_interval": [lo_x, hi_x]})


def make_inverse_dr(out: Path, name: str, n: int, seed: int, lo: float, hi: float) -> None:
    rng = np.random.default_rng(seed)
    x = rng.uniform(lo, hi, size=(n, 1))
    # generate from problem's own distribution but restrict interval
    from app.problems import InverseDR
    p = InverseDR()
    k = p.k_true(x)
    u = p.u_true(x)
    y = np.hstack([u, k])
    save_bundle(out / name, "inverse_dr", {"x_train": x.astype(np.float32), "y_train": y.astype(np.float32)}, domain={"interval": [lo, hi]})


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


def make_burgers(out: Path, name: str, n: int, seed: int, modes: list[int]) -> None:
    problem = get_problem("burgers")
    # heterogeneity via Fourier mode subset similar to antiderivative
    # Use antiderivative's scheme to generate IC modes
    sensors = problem.sensors  # type: ignore[attr-defined]
    rng = np.random.default_rng(seed)
    coeffs = rng.normal(size=(n, 5)) * 0.5
    # zero out modes not in subset to simulate functional heterogeneity
    mask = np.ones(5, dtype=bool)
    valid = [m for m in modes if 1 <= m <= 5]
    mask[:] = False
    for m in valid:
        mask[m - 1] = True
    coeffs[:, ~mask] = 0.0
    # generate branch via sensors
    x = sensors[None, :]
    waves = np.array([np.sin(k * np.pi * x) for k in range(1, 6)])[:, 0, :]  # (5,101)
    ics = coeffs @ waves
    y = problem._solution(ics, problem.trunk)  # type: ignore[attr-defined]
    save_bundle(out / name, "burgers", {"branch_train": ics.astype(np.float32), "trunk_train": problem.trunk.astype(np.float32), "y_train": y.astype(np.float32)}, domain={"fourier_modes": modes})


def make_diffusion_reaction(out: Path, name: str, n: int, seed: int, modes: list[int]) -> None:
    problem = get_problem("diffusion_reaction")
    rng = np.random.default_rng(seed)
    # functional heterogeneity: Chebyshev-like sources with masked basis
    coeffs = rng.uniform(-1, 1, size=(n, 6))
    mask = np.ones(6, dtype=bool)
    mask[:] = False
    for m in modes:
        if 0 <= m < 6:
            mask[m] = True
    # if no valid, keep at least 2
    if not mask.any():
        mask[0:2] = True
    coeffs[:, ~mask] = 0.0
    sensors = problem.sensors  # type: ignore[attr-defined]
    # reconstruct source as sum a_i T_i
    x = sensors[None, :]
    ar = np.arccos(2 * x - 1)
    basis = np.array([np.cos(i * ar[0]) for i in range(6)])  # (6,101)
    src = coeffs @ basis
    y = problem._solution(src)  # type: ignore[attr-defined]
    save_bundle(out / name, "diffusion_reaction", {"branch_train": src.astype(np.float32), "trunk_train": problem.trunk.astype(np.float32), "y_train": y.astype(np.float32)}, domain={"fourier_modes": modes})


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

    hel_edges = np.linspace(0, 1, n_clients + 1)
    for i in range(n_clients):
        make_helmholtz(out, f"helmholtz_client{i + 1}", n, 350 + i, hel_edges[i], hel_edges[i + 1])

    ac_edges = np.linspace(-1, 1, n_clients + 1)
    for i in range(n_clients):
        make_allen_cahn(out, f"allen_cahn_client{i + 1}", n, 360 + i, ac_edges[i], ac_edges[i + 1])

    # Inverse problems: slices on x domain
    ns_edges = np.linspace(1, 8, n_clients + 1)
    for i in range(n_clients):
        make_inverse_ns(out, f"inverse_ns_client{i + 1}", n, 370 + i, ns_edges[i], ns_edges[i + 1])

    dr_edges = np.linspace(0, 1, n_clients + 1)
    for i in range(n_clients):
        make_inverse_dr(out, f"inverse_dr_client{i + 1}", n, 380 + i, dr_edges[i], dr_edges[i + 1])

    # Operator heterogeneity via Fourier mode subsets
    rng = np.random.default_rng(999)
    for prob_name, maker in [("antiderivative", make_antiderivative), ("burgers", make_burgers), ("diffusion_reaction", make_diffusion_reaction)]:
        problem = get_problem(prob_name)
        K = getattr(problem, "K", 5) if prob_name == "antiderivative" else 6
        for i in range(n_clients):
            size = int(rng.integers(2, K + 1))
            modes = sorted(int(m) for m in rng.choice(np.arange(1, K + 1) if prob_name != "diffusion_reaction" else np.arange(0, 6), size=size, replace=False))
            maker(out, f"{prob_name}_client{i + 1}", n, 400 + i * 10 + hash(prob_name) % 100, modes)


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
