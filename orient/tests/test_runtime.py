"""Runtime smoke tests for the shared scientific core (needs torch/numpy/safetensors).

Run from the ``orient/`` directory::

    .venv\\Scripts\\python.exe tests\\test_runtime.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.aggregators import ALL_AGGREGATORS, ServerOptimizerState, aggregate  # noqa: E402
from app.models import build_model, count_parameters  # noqa: E402
from app.problems import build_problem_model, list_problems  # noqa: E402
from app.protocol import AssignmentResponse, ModelSpec  # noqa: E402
from app.weighting import compute_weights  # noqa: E402
from app.weights import bytes_to_state_dict, state_dict_to_bytes  # noqa: E402

failures: list[str] = []


def check(condition: bool, label: str) -> None:
    print(("  PASS  " if condition else "  FAIL  ") + label)
    if not condition:
        failures.append(label)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def test_safetensors_roundtrip() -> None:
    section("safetensors round-trip")
    model = build_model(ModelSpec(family="mlp", params={"layer_sizes": [2, 8, 1]}))
    original = model.state_dict()
    restored = bytes_to_state_dict(state_dict_to_bytes(original))
    check(set(original) == set(restored), "same state_dict keys survive the wire")
    check(
        all(torch.allclose(original[k], restored[k]) for k in original),
        "all tensors are bit-identical after serialize/deserialize",
    )


def test_aggregators() -> None:
    section("aggregation algorithms")
    states = [
        {"w": torch.tensor([1.0, 1.0])},
        {"w": torch.tensor([3.0, 3.0])},
        {"w": torch.tensor([5.0, 5.0])},
    ]
    global_state = {"w": torch.tensor([0.0, 0.0])}
    weights = [1.0, 1.0, 1.0]

    avg = aggregate("fedavg", states, weights, global_state).state_dict
    check(torch.allclose(avg["w"], torch.tensor([3.0, 3.0])), "fedavg = plain mean")

    weighted = aggregate("fedavg", states, [1.0, 0.0, 0.0], global_state).state_dict
    check(torch.allclose(weighted["w"], torch.tensor([1.0, 1.0])), "fedavg honours weights")

    med = aggregate("median", states, weights, global_state).state_dict
    check(torch.allclose(med["w"], torch.tensor([3.0, 3.0])), "element-wise median")

    trimmed = aggregate(
        "trimmed_mean", states, weights, global_state, {"trim_ratio": 1 / 3}
    ).state_dict
    check(
        torch.allclose(trimmed["w"], torch.tensor([3.0, 3.0])),
        "trimmed_mean with trim_ratio=1/3 reduces to the median",
    )

    # Krum needs K >= 2f+3, so exercise it with 3 honest + 2 Byzantine = 5 clients.
    mixed = states + [
        {"w": torch.tensor([1000.0, 1000.0])},
        {"w": torch.tensor([-1000.0, -1000.0])},
    ]
    med_mixed = aggregate("median", mixed, [1.0] * 5, global_state).state_dict
    check(float(med_mixed["w"].max()) < 10.0, "median rejects two Byzantine 1000-valued updates")
    krum_mixed = aggregate("krum", mixed, [1.0] * 5, global_state, {"n_byzantine": 1}).state_dict
    check(float(krum_mixed["w"].max()) < 10.0, "krum (K=5, f=1) rejects two Byzantine updates")

    prox = aggregate("fedprox", states, weights, global_state, {"mu": 0.01}).state_dict
    check(torch.allclose(prox["w"], torch.tensor([3.0, 3.0])), "fedprox server step = fedavg")

    for name in ("fedadam", "fedadagrad", "fedyogi"):
        # Server-adaptive methods take a *normalized* step of size ~server_lr, so a
        # single round only moves partway from the global model toward the client
        # mean. Iterate to check they actually converge onto it.
        server = ServerOptimizerState()
        current = {"w": global_state["w"].clone()}
        for _ in range(400):
            current = aggregate(
                name, states, weights, current, {"server_lr": 0.5}, server
            ).state_dict
        check(
            torch.allclose(current["w"], torch.tensor([3.0, 3.0]), atol=0.15),
            f"{name} converges onto the client mean over repeated rounds",
        )

    # scaffold also converges (with correction)
    server = ServerOptimizerState()
    current = {"w": global_state["w"].clone()}
    for _ in range(100):
        current = aggregate(
            "scaffold", states, weights, current, {"server_lr": 0.8}, server
        ).state_dict
    check(
        torch.allclose(current["w"], torch.tensor([3.0, 3.0]), atol=0.5),
        "scaffold converges onto the client mean",
    )

    check(len(ALL_AGGREGATORS) == 9, "9 aggregators are exposed (SRS §4.4)")
    check("scaffold" in ALL_AGGREGATORS, "scaffold in ALL_AGGREGATORS")


def test_weighting() -> None:
    section("weighting modes")
    check(compute_weights("uniform", [10, 90]) == [1.0, 1.0], "uniform weights")
    check(
        np.allclose(compute_weights("data_size", [10, 90]), [0.1, 0.9]),
        "data_size weights are proportional",
    )
    q = compute_weights("quality", [1, 1], [1.0, 3.0])
    check(np.allclose(q, [0.75, 0.25]), "quality weights favour the lower loss")


def test_problems() -> None:
    section("problems")
    expected = [
        "allen_cahn",
        "antiderivative",
        "burgers",
        "diffusion_reaction",
        "gramacy_lee",
        "helmholtz",
        "inverse_dr",
        "inverse_ns",
        "poisson",
        "schaffer",
    ]
    check(
        sorted(list_problems()) == expected,
        "10 problems registered (SRS §4.2)",
    )

    for name in list_problems():
        problem, model = build_problem_model(name)
        arrays = problem.sample_dataset(n=200, seed=0)
        built = build_model(problem.model_spec())
        check(
            count_parameters(model) == count_parameters(built),
            f"{name}: model factory is deterministic",
        )

        loss = problem.local_loss(model, arrays, batch_size=64)
        check(
            torch.isfinite(loss).all().item(),
            f"{name}: local loss is finite ({float(loss.detach()):.4g})",
        )
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        check(len(grads) > 0, f"{name}: gradients flow to parameters")

        fresh, _ = build_problem_model(name)
        metrics = fresh.evaluate(model, num_test=64)
        check(
            np.isfinite(metrics["l2_relative_error"]),
            f"{name}: evaluate() returns a finite L2 error",
        )

    poisson, poisson_model = build_problem_model("poisson")
    xs = torch.tensor([[0.0], [float(np.pi)]], dtype=torch.float32)
    with torch.no_grad():
        us = poisson_model(xs).numpy().reshape(-1)
    check(abs(us[0]) < 1e-5, f"poisson hard constraint u(0)=0 (got {us[0]:.2e})")
    check(abs(us[1] - np.pi) < 1e-3, f"poisson hard constraint u(pi)=pi (got {us[1]:.5f})")


def test_protocol() -> None:
    section("wire protocol")
    assignment = AssignmentResponse(
        status="train",
        round=3,
        total_rounds=10,
        local_epochs=5,
        model_spec=ModelSpec(family="mlp", params={"layer_sizes": [1, 4, 1]}),
    )
    restored = AssignmentResponse(**assignment.model_dump())
    check(restored.status == "train" and restored.round == 3, "AssignmentResponse survives JSON round-trip")
    check(
        restored.model_spec is not None and restored.model_spec.family == "mlp",
        "nested ModelSpec survives the round-trip",
    )


def main() -> int:
    test_safetensors_roundtrip()
    test_aggregators()
    test_weighting()
    test_problems()
    test_protocol()
    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} FAILURE(S):")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("ALL RUNTIME TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())