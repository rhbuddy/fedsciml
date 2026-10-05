"""Regression tests for the server app (run validation, seeding, round engine).

Runs in its own process because it imports the *backend* ``app`` package.
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.orchestrator import Federation  # noqa: E402
from app.protocol import ClientRegistration, RunStartRequest  # noqa: E402
from app.weighting import compute_weights  # noqa: E402
from app.weights import bytes_to_state_dict, state_dict_to_bytes  # noqa: E402

failures: list[str] = []


def check(condition: bool, label: str) -> None:
    print(("  PASS  " if condition else "  FAIL  ") + label)
    if not condition:
        failures.append(label)


def federation(tmp: str, seed: int = 0) -> Federation:
    return Federation(results_dir=tmp, seed=seed)


def test_run_validation() -> None:
    """Regression: a bad aggregator used to raise mid-run and stall the round."""
    print("\n=== run configuration validation ===")
    with tempfile.TemporaryDirectory() as tmp:
        for label, kwargs in [
            ("unknown aggregator", {"aggregator": "bogus"}),
            ("unknown weighting", {"weighting": "bogus"}),
            ("total_rounds < 1", {"total_rounds": 0}),
            ("local_epochs < 1", {"local_epochs": 0}),
        ]:
            fed = federation(tmp)
            try:
                fed.start_run(RunStartRequest(problem="poisson", **kwargs))
                check(False, f"{label} is rejected at start_run")
            except ValueError:
                check(True, f"{label} is rejected at start_run")


def test_registration_guard() -> None:
    """Regression: a mismatched client was added to `expected` and stalled the round."""
    print("\n=== registration guards ===")
    with tempfile.TemporaryDirectory() as tmp:
        fed = federation(tmp)
        fed.start_run(
            RunStartRequest(problem="poisson", aggregator="fedavg", total_rounds=2, local_epochs=1)
        )
        try:
            fed.register(ClientRegistration(client_id="bad", problem="schaffer", n_samples=10))
            check(False, "client with a different problem is rejected")
        except ValueError as exc:
            check("Refusing to add it" in str(exc), "client with a different problem is rejected")
        check("bad" not in fed.run.expected, "the rejected client was not added to the round")

        fed.register(ClientRegistration(client_id="good", problem="poisson", n_samples=10))
        check("good" in fed.run.expected, "a matching client joins the round")
        check(fed.is_registered("good") and not fed.is_registered("nope"), "is_registered works")


def test_reproducibility() -> None:
    """Regression: the seed was recorded in config.json but never applied."""
    print("\n=== reproducibility (NFR-REP-1) ===")
    with tempfile.TemporaryDirectory() as tmp:
        a, b, c = federation(tmp, 42), federation(tmp, 42), federation(tmp, 7)
        for fed in (a, b, c):
            fed.start_run(
                RunStartRequest(problem="poisson", aggregator="fedavg", total_rounds=1, local_epochs=1)
            )
        wa, wb, wc = a.get_global_bytes(1), b.get_global_bytes(1), c.get_global_bytes(1)
        check(wa == wb, "same seed -> byte-identical initial global weights")
        check(wa != wc, "different seed -> different initial global weights")


def test_round_engine() -> None:
    print("\n=== round engine ===")
    with tempfile.TemporaryDirectory() as tmp:
        fed = federation(tmp)
        fed.start_run(
            RunStartRequest(problem="poisson", aggregator="fedavg", total_rounds=2, local_epochs=1)
        )
        fed.register(ClientRegistration(client_id="c1", problem="poisson", n_samples=100))
        fed.register(ClientRegistration(client_id="c2", problem="poisson", n_samples=100))
        check(set(fed.run.expected) == {"c1", "c2"}, "both clients are expected in round 1")

        junk = state_dict_to_bytes({"not_a_real_key": torch.zeros(2)})
        resp = fed.submit_update("c1", 1, 100, 0.5, junk)
        check(resp.status == "error", "a state_dict with wrong keys is rejected")

        good_payload = state_dict_to_bytes(bytes_to_state_dict(fed.get_global_bytes(1)))
        resp = fed.submit_update("c1", 99, 100, 0.5, good_payload)
        check(resp.status == "wait" and "Stale" in resp.message, "a stale round number is ignored")

        for cid in ("c1", "c2"):
            state = bytes_to_state_dict(fed.get_global_bytes(1))
            perturbed = {k: v + 0.01 for k, v in state.items()}
            resp = fed.submit_update(cid, 1, 100, 0.5, state_dict_to_bytes(perturbed))
            check(resp.status == "wait", f"{cid} update accepted")

        status = fed.status()
        check(len(status.metrics) == 1, "one metrics record after all clients reported")
        check(status.round == 2, "round advanced to 2")
        check(status.metrics[0]["n_clients"] == 2, "metrics record both clients")
        check(
            math.isfinite(status.metrics[0]["l2_relative_error"]),
            "metrics record contains a finite L2 error",
        )


def test_weighting_robustness() -> None:
    """Regression: a near-zero local loss produced a ~1e12 weight."""
    print("\n=== weighting robustness ===")
    weights = compute_weights("quality", [1, 1], [0.0, 1.0])
    check(all(math.isfinite(w) for w in weights), "quality weights stay finite at loss=0")
    check(abs(sum(weights) - 1.0) < 1e-9, "quality weights are normalized")
    check(max(weights) < 1.0, "no single client dominates when loss=0")
    check(compute_weights("uniform", [0, 0]) == [1.0, 1.0], "uniform handles zero sizes")
    fallback = compute_weights("data_size", [0, 0])
    check(
        fallback == [1.0, 1.0],
        "data_size falls back to uniform when every size is 0",
    )


def main() -> int:
    test_run_validation()
    test_registration_guard()
    test_reproducibility()
    test_round_engine()
    test_weighting_robustness()
    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} FAILURE(S):")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("ALL SERVER TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())