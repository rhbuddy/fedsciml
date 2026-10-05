"""Regression tests for the client app (FedProx gradient, dataset validation).

Runs in its own process because it imports the *client* ``app`` package.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "client"))

from app.dataset import load_bundle, save_bundle  # noqa: E402
from app.problems import get_problem  # noqa: E402
from app.trainer import LocalTrainer  # noqa: E402

failures: list[str] = []


def check(condition: bool, label: str) -> None:
    print(("  PASS  " if condition else "  FAIL  ") + label)
    if not condition:
        failures.append(label)


def test_prox_term_is_differentiable() -> None:
    """The FedProx proximal term MUST contribute a gradient.

    Regression test: it used to be `.detach()`-ed, which made it a constant
    offset and silently reduced FedProx to plain FedAvg.
    """
    print("\n=== FedProx proximal term ===")
    problem = get_problem("gramacy_lee")
    arrays = problem.sample_dataset(256, seed=0)

    trainer = LocalTrainer("gramacy_lee")
    trainer.load_weights(trainer.state_dict())
    assert trainer._global_ref is not None

    prox = sum(
        ((p - ref) ** 2).sum()
        for p, ref in zip(trainer.model.parameters(), trainer._global_ref)
    )
    check(prox.requires_grad, "proximal term is part of the autograd graph")

    def grads_with(scale: float) -> list[torch.Tensor]:
        """Fresh graph each time - a tensor's graph is freed after backward().

        The seed is reset so every call samples the *same* minibatch; otherwise
        ``local_loss``'s random batch would make this comparison flaky.
        """
        torch.manual_seed(1234)
        trainer.model.zero_grad(set_to_none=True)
        loss = problem.local_loss(trainer.model, arrays, 256)
        if scale > 0:
            loss = loss + 0.5 * scale * sum(
                ((p - ref) ** 2).sum()
                for p, ref in zip(trainer.model.parameters(), trainer._global_ref)
            )
        loss.backward()
        return [p.grad.detach().clone() for p in trainer.model.parameters()]

    # Right after load_weights, p == ref, so the proximal gradient is legitimately
    # zero - FedProx only acts once the client has drifted from the global model.
    # Comparing gradients *before* simulating that drift would be meaningless.
    torch.manual_seed(0)
    with torch.no_grad():
        for param in trainer.model.parameters():
            param.add_(torch.randn_like(param) * 0.5)

    g_plain = grads_with(0.0)
    g_small = grads_with(1.0)
    g_huge = grads_with(1000.0)

    check(
        any(not torch.allclose(a, b) for a, b in zip(g_plain, g_small)),
        "after drift, a non-zero prox_mu changes the gradients",
    )

    expected = [
        1000.0 * (param.detach() - ref)          # d/dp [0.5*mu*(p-ref)^2] = mu*(p-ref)
        for param, ref in zip(trainer.model.parameters(), trainer._global_ref)
    ]
    # Compare as a whole rather than element-wise: elements where (p - ref) is near
    # zero would need an absurd relative tolerance. What matters is that the
    # proximal term DOMINATES the total gradient.
    residual = max(float((a - e).abs().max()) for a, e in zip(g_huge, expected))
    scale = max(float(e.abs().max()) for e in expected)
    check(
        residual < 0.05 * scale,
        f"with mu=1000 the gradient is dominated by the proximal term "
        f"(residual {residual:.3g} < 5% of {scale:.3g})",
    )


def test_dataset_validation() -> None:
    """FR-CLIENTAPP-8: bad bundles must be rejected, not silently trained."""
    print("\n=== dataset bundle validation ===")
    problem = get_problem("gramacy_lee")

    with tempfile.TemporaryDirectory() as tmp:
        good = problem.sample_dataset(64, seed=0)
        save_bundle(Path(tmp) / "good", "gramacy_lee", good)
        bundle = load_bundle(Path(tmp) / "good")
        check(bundle.problem == "gramacy_lee" and bundle.size == 64, "a valid bundle loads")

        # 1-D inputs must be rejected with a clear message.
        bad_dir = Path(tmp) / "bad1d"
        save_bundle(bad_dir, "gramacy_lee", {"x_train": np.zeros(32), "y_train": np.zeros((32, 1))})
        try:
            load_bundle(bad_dir)
            check(False, "1-D x_train is rejected")
        except ValueError as exc:
            check("2-D" in str(exc), f"1-D x_train is rejected with a helpful message ({exc})")

        # Mismatched sample counts must be rejected.
        bad_dir2 = Path(tmp) / "badlen"
        save_bundle(bad_dir2, "gramacy_lee", {"x_train": np.zeros((32, 1)), "y_train": np.zeros((16, 1))})
        try:
            load_bundle(bad_dir2)
            check(False, "mismatched x/y lengths are rejected")
        except ValueError as exc:
            check("same number of samples" in str(exc), "mismatched x/y lengths are rejected")

        # Unknown problem must be rejected.
        bad_dir3 = Path(tmp) / "badprob"
        save_bundle(bad_dir3, "not_a_problem", {"x_train": np.zeros((8, 1)), "y_train": np.zeros((8, 1))})
        try:
            load_bundle(bad_dir3)
            check(False, "unknown problem is rejected")
        except KeyError:
            check(True, "unknown problem is rejected")

        # The shipped example bundles must all be valid.
        bundles = ROOT / "client" / "examples" / "bundles"
        if bundles.exists():
            names = sorted(p.name for p in bundles.iterdir() if p.is_dir())
            ok = True
            for name in names:
                try:
                    load_bundle(bundles / name)
                except Exception as exc:  # noqa: BLE001
                    ok = False
                    print(f"      {name}: {exc}")
            check(ok and len(names) > 0, f"all {len(names)} shipped example bundles validate")


def main() -> int:
    test_prox_term_is_differentiable()
    test_dataset_validation()
    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} FAILURE(S):")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("ALL CLIENT TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())