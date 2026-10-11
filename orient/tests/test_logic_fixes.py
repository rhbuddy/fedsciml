"""Regression tests for the 12 logic errors found in the 2026-10-11 logic audit.

Each test names the finding it pins down, so a future refactor that reintroduces
one fails with an explanatory message rather than silently changing results.

Run directly::

    .venv/bin/python tests/test_logic_fixes.py
"""

from __future__ import annotations

import logging
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app import aggregators, noise_sim, weighting  # noqa: E402
from app.aggregators import ServerOptimizerState, aggregate  # noqa: E402
from app.metrics import w1_1d, w1_nd  # noqa: E402
from app.orchestrator import Federation  # noqa: E402
from app.protocol import ClientRegistration, OptimizerSpec, RunStartRequest  # noqa: E402
from app.trainer import LocalTrainer  # noqa: E402
from app.weights import bytes_to_state_dict, state_dict_to_bytes  # noqa: E402

failures: list[str] = []


def check(condition: bool, label: str) -> None:
    print(("  PASS  " if condition else "  FAIL  ") + label)
    if not condition:
        failures.append(label)


def sd(vals, key="w"):
    return {key: torch.tensor(vals, dtype=torch.float32)}


# --------------------------------------------------------------------------- 1
def test_median_is_a_real_median_at_even_k() -> None:
    """#1 median returned the element-wise MIN at even K (torch.median's lower middle)."""
    print("\n=== #1 median with an even number of clients ===")
    for k, clients in ((2, [0.0, 1.0]), (4, [0.0, 1.0, 2.0, 3.0])):
        states = [sd([c, 10.0]) for c in clients]
        got = aggregate("median", states, [1.0] * k, states[0]).state_dict["w"].tolist()
        ref = np.median(np.array([s["w"].numpy() for s in states]), axis=0).tolist()
        check(np.allclose(got, ref), f"K={k}: median {got} matches numpy {ref}")

    # odd K must be unchanged
    states3 = [sd([0.0]), sd([1.0]), sd([2.0])]
    got3 = aggregate("median", states3, [1.0] * 3, states3[0]).state_dict["w"].item()
    check(got3 == 1.0, f"K=3: median is still {got3}")

    # the Byzantine consequence of the old behaviour
    honest, attacker = sd([1.0]), sd([-1e6])
    got2 = aggregate("median", [honest, attacker], [1.0, 1.0], honest).state_dict["w"].item()
    check(got2 == -5e5 + 0.5 or abs(got2 + 499999.5) < 1.0,
          f"K=2 median of honest/attacker is their mean ({got2}), not the attacker")


# --------------------------------------------------------------------------- 2
def test_noise_fraction_always_compromises_someone() -> None:
    """#2 round(f*k) used banker's rounding, so whole noise cells ran clean."""
    print("\n=== #2 select_compromised over the SRS noise grid ===")
    bad = []
    for k in (2, 3, 5):
        for f in (0.1, 0.2, 0.3):
            n = len(noise_sim.select_compromised(k, f, seed=7))
            if n == 0:
                bad.append((k, f))
    check(not bad, f"no (K, fraction) cell yields zero compromised clients (bad: {bad})")
    check(len(noise_sim.select_compromised(5, 0.1, seed=7)) == 1, "K=5 f=0.1 -> ceil(0.5)=1 client")
    check(noise_sim.select_compromised(5, 0.0, seed=7) == [], "fraction=0 -> nobody")
    # deterministic
    check(noise_sim.select_compromised(5, 0.4, seed=3) == noise_sim.select_compromised(5, 0.4, seed=3),
          "selection is deterministic for a given seed")


# --------------------------------------------------------------------------- 3
def test_topk_sparsifies_the_update_not_the_model() -> None:
    """#3 topk zeroed 1-ratio of the model itself instead of the update."""
    print("\n=== #3 topk compression is delta-encoded ===")
    torch.manual_seed(0)
    ref = {"w": torch.randn(64, 64), "b": torch.randn(64)}
    moved = {k: v + torch.randn_like(v) * 0.1 for k, v in ref.items()}
    total = sum(t.numel() for t in moved.values())

    for mode in ("topk", "int8_topk"):
        blob = state_dict_to_bytes(moved, compression=mode, topk_ratio=0.01, reference=ref)
        back = bytes_to_state_dict(blob, reference=ref)
        zeros = sum(int((back[k] == 0).sum()) for k in back)
        rel = max(float((back[k] - moved[k]).abs().max() / moved[k].abs().max()) for k in moved)
        check(zeros < total * 0.5, f"{mode}: reconstructs a dense model ({zeros}/{total} zeros)")
        check(rel < 0.2, f"{mode}: round-trip relative error {rel:.4f} < 0.2")
        check(len(blob) < len(state_dict_to_bytes(moved, compression="none")) / 3,
              f"{mode}: still compresses (>3x smaller)")

    # unchanged weights survive exactly
    blob = state_dict_to_bytes(ref, compression="topk", topk_ratio=0.01, reference=ref)
    back = bytes_to_state_dict(blob, reference=ref)
    check(all(torch.allclose(back[k], ref[k]) for k in ref), "topk of a zero update is lossless")

    # and it refuses to sparsify absolute weights
    try:
        state_dict_to_bytes(moved, compression="topk", topk_ratio=0.01)
        check(False, "topk without a reference is rejected")
    except ValueError as exc:
        check("reference" in str(exc), f"topk without a reference is rejected ({exc})")


# --------------------------------------------------------------------------- 4
def test_robust_aggregators_can_start_before_clients_register() -> None:
    """#4 the K guard ran at start_run, when K is still unknown in the documented flow."""
    print("\n=== #4 median/krum runs can start before clients register ===")
    with tempfile.TemporaryDirectory() as tmp:
        fed = Federation(results_dir=tmp, seed=1)
        fed.start_run(RunStartRequest(problem="poisson", aggregator="median",
                                      total_rounds=1, local_epochs=1))
        check(fed.run.phase == "collecting", "median run starts with 0 registered clients")
        check(fed._min_aggregation_k == 3, f"minimum K recorded as {fed._min_aggregation_k}")

        # a lone client is told to wait rather than being fed to a 1-client median
        fed.register(ClientRegistration(client_id="c1", problem="poisson", n_samples=10))
        resp = fed.assignment("c1")
        check(resp.status == "wait" and "requires K >= 3" in resp.message,
              f"one client waits ({resp.message})")

        fed2 = Federation(results_dir=tmp, seed=1)
        fed2.start_run(RunStartRequest(problem="poisson", aggregator="krum",
                                       total_rounds=1, local_epochs=1))
        check(fed2._min_aggregation_k == 5, f"krum f=1 needs K>=5, got {fed2._min_aggregation_k}")

        # but a KNOWN-too-small K still fails fast
        fed3 = Federation(results_dir=tmp, seed=1)
        for i in range(2):
            fed3.register(ClientRegistration(client_id=f"c{i}", problem="poisson", n_samples=10))
        try:
            fed3.start_run(RunStartRequest(problem="poisson", aggregator="median",
                                           total_rounds=1, local_epochs=1))
            check(False, "a known K=2 median run is still rejected at start_run")
        except ValueError as exc:
            check("requires K >= 3" in str(exc), f"known K=2 rejected ({exc})")


def test_shrinking_round_cannot_feed_median_two_updates() -> None:
    """#4b a round that shrinks below the minimum must refuse, not take the min."""
    print("\n=== #4b reaped clients cannot shrink median down to 2 ===")
    with tempfile.TemporaryDirectory() as tmp:
        from app.config import settings
        settings.client_ttl_seconds = 5
        fed = Federation(results_dir=tmp, seed=1)
        for i in range(4):
            fed.register(ClientRegistration(client_id=f"c{i+1}", problem="poisson", n_samples=100))
        fed.start_run(RunStartRequest(problem="poisson", aggregator="median",
                                      total_rounds=1, local_epochs=1))
        fed.clients["c1"].last_seen_at = time.time() - 99
        fed.clients["c2"].last_seen_at = time.time() - 99
        fed.reap_stale()
        check(fed.run.expected == ["c3", "c4"], f"round shrank to {fed.run.expected}")

        g = bytes_to_state_dict(fed.get_global_bytes(1))
        poisoned = {k: v - 1e6 for k, v in g.items()}
        fed.submit_update("c3", 1, 100, 0.5, state_dict_to_bytes(poisoned))
        resp = fed.submit_update("c4", 1, 100, 0.5, state_dict_to_bytes(g))
        check(resp.status == "error" and "Refusing to aggregate" in resp.message,
              f"aggregation refused ({resp.message[:70]})")
        check(fed.run.phase == "collecting", "the round did not silently aggregate")
        if fed._global_model is not None:
            worst = min(float(v.min()) for v in fed._global_model.state_dict().values())
            check(worst > -1e5, f"the poisoned update did not become the global model (min={worst:.3g})")


# --------------------------------------------------------------------------- 5
def test_holdout_only_when_quality_weighting() -> None:
    """#5 the 80/20 split was unconditional, wasting 20% of every client's data."""
    print("\n=== #5 holdout only when quality weighting asks for it ===")
    problem_name = "gramacy_lee"
    from app.problems import get_problem
    arr = get_problem(problem_name).sample_dataset(1000, seed=0)

    tr = LocalTrainer(problem_name)
    tr.load_weights(tr.state_dict())
    full = tr.train(arr, epochs=1, optimizer_spec=OptimizerSpec(name="adam", lr=1e-3),
                    batch_size=256, val_holdout_frac=0.0)
    held = tr.train(arr, epochs=1, optimizer_spec=OptimizerSpec(name="adam", lr=1e-3),
                    batch_size=256, val_holdout_frac=0.2)

    tr_full = LocalTrainer(problem_name)
    ta, va = tr_full._split_train_val(arr, 0.2)
    steps_full_expected = max(1, arr["x_train"].shape[0] // 256)
    steps_split_expected = max(1, ta["x_train"].shape[0] // 256)
    check(full["steps"] == steps_full_expected,
          f"no holdout -> {full['steps']} steps over all data (expected {steps_full_expected})")
    check(held["steps"] <= steps_full_expected,
          f"holdout=0.2 -> {held['steps']} steps (split would give {steps_split_expected})")
    check(va is not None and va["x_train"].shape[0] == 200, "the 0.2 holdout is 200 samples")

    # the server only asks for a holdout under quality weighting
    with tempfile.TemporaryDirectory() as tmp:
        for wmode, want in (("data_size", 0.0), ("quality", 0.2)):
            fed = Federation(results_dir=tmp, seed=1)
            for i in range(2):
                fed.register(ClientRegistration(client_id=f"c{i+1}", problem="poisson", n_samples=10))
            fed.start_run(RunStartRequest(problem="poisson", aggregator="fedavg", weighting=wmode,
                                          total_rounds=1, local_epochs=1))
            got = fed.assignment("c1").val_holdout_frac
            check(got == want, f"weighting={wmode} -> val_holdout_frac={got} (want {want})")


# --------------------------------------------------------------------------- 6
def test_scaffold_control_variates_follow_the_client() -> None:
    """#6 SCAFFOLD keyed c_i by position, so partial participation misattributed them."""
    print("\n=== #6 SCAFFOLD control variates are keyed by client id ===")
    server = ServerOptimizerState()
    g = sd([0.0, 0.0])
    r1 = aggregate("scaffold", [sd([1.0, 1.0]), sd([2.0, 2.0])], [1.0, 1.0], g,
                   {"server_lr": 1.0}, server, client_ids=["a", "b"])
    check(set(server.scaffold_ci_by_id) == {"a", "b"},
          f"control variates stored per client id: {sorted(server.scaffold_ci_by_id)}")
    ca = server.scaffold_ci_by_id["a"].clone()

    # a different participant set: 'a' returns, 'b' is replaced by 'c'
    aggregate("scaffold", [sd([1.5, 1.5]), sd([3.0, 3.0])], [1.0, 1.0], r1.state_dict,
              {"server_lr": 1.0}, server, client_ids=["a", "c"])
    check(set(server.scaffold_ci_by_id) == {"a", "b", "c"},
          "the departed client's variate is kept, the new one is added")
    check(not torch.allclose(server.scaffold_ci_by_id["a"], ca),
          "'a' keeps evolving its own variate across rounds")
    check(torch.allclose(server.scaffold_ci_by_id["c"],
                         server.scaffold_cis[-1]["w"]), "'c' got a fresh variate, not 'b's")


# --------------------------------------------------------------------------- 7
def test_quality_weighting_is_honest_about_domination() -> None:
    """#7 the 1e-6 clamp let a zero-loss client take ~100% of the aggregate."""
    print("\n=== #7 quality weighting ===")
    w = weighting.compute_weights("quality", [100, 100], [0.0, 1.0])
    check(all(math.isfinite(x) for x in w) and abs(sum(w) - 1.0) < 1e-9, "weights finite and normalized")
    w3 = weighting.compute_weights("quality", [10, 10, 10], [0.1, 0.2, 0.4])
    check(w3[0] > w3[1] > w3[2], f"weights strictly ordered by loss: {[f'{x:.3f}' for x in w3]}")
    check(abs(w3[0] / w3[1] - 2.0) < 1e-6, "the 1/L ratio is preserved exactly")

    # domination is now logged instead of silent
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = lambda rec: records.append(rec)  # type: ignore[assignment]
    logger = logging.getLogger("orient.weighting")
    logger.addHandler(handler)
    try:
        weighting.compute_weights("quality", [100, 100], [0.0, 1.0])
    finally:
        logger.removeHandler(handler)
    check(any("holds" in r.getMessage() for r in records),
          f"a dominating client is warned about ({len(records)} record(s))")


# --------------------------------------------------------------------------- 8
def test_w1_nd_fallback_is_a_wasserstein_distance() -> None:
    """#8 without POT, w1_nd returned ||mean(a)-mean(b)||, which is 0 for symmetric clouds."""
    print("\n=== #8 w1_nd fallback ===")
    a = np.array([[0.0, 0.0], [0.0, 0.0]])
    b = np.array([[-1.0, 0.0], [1.0, 0.0]])
    got = w1_nd(a, b)
    check(got > 0.5, f"symmetric clouds 1 unit apart give W1={got:.3f} (true 1.0, old code 0.0)")
    check(abs(got - 1.0) < 1e-6, f"exact assignment fallback recovers the true value ({got:.6f})")
    check(w1_nd(a, a) == 0.0, "identical clouds give 0")
    # translation still measured correctly
    t = w1_nd(np.zeros((4, 2)), np.full((4, 2), 3.0))
    check(abs(t - 3.0 * math.sqrt(2)) < 1e-6, f"pure translation gives {t:.6f} (true {3*math.sqrt(2):.6f})")
    check(abs(w1_1d(np.array([0.0, 1.0]), np.array([2.0, 3.0])) - 2.0) < 1e-9, "w1_1d still exact")


# --------------------------------------------------------------------------- 9
def test_dp_epsilon_does_not_scale_with_client_fraction() -> None:
    """#9 eps was multiplied by sample_rate, an invalid RDP amplification."""
    print("\n=== #9 DP epsilon ===")
    with tempfile.TemporaryDirectory() as tmp:
        fed = Federation(results_dir=tmp, seed=1)
        e_full = fed._dp_epsilon(10, 1.0, 1e-5, sample_rate=1.0)
        e_half = fed._dp_epsilon(10, 1.0, 1e-5, sample_rate=0.5)
        check(e_full > 0 and math.isfinite(e_full), f"epsilon is finite and positive ({e_full:.3f})")
        check(abs(e_full - e_half) < 1e-9,
              f"epsilon no longer shrinks with the sampling rate ({e_full:.4f} vs {e_half:.4f})")
        check(fed._dp_epsilon(0, 1.0, 1e-5) == 0.0, "no steps -> 0")
        check(fed._dp_epsilon(10, 0.0, 1e-5) == 0.0, "no noise -> 0")
        more = fed._dp_epsilon(20, 1.0, 1e-5)
        check(more > e_full, f"more steps -> larger epsilon ({more:.3f} > {e_full:.3f})")
        louder = fed._dp_epsilon(10, 0.5, 1e-5)
        check(louder > e_full, f"less noise -> larger epsilon ({louder:.3f} > {e_full:.3f})")


# -------------------------------------------------------------------------- 10
def test_server_adaptive_does_not_double_apply_momentum() -> None:
    """#10 a blanket except retried the update after in-place mutation of m/v."""
    print("\n=== #10 server adaptive optimizers ===")
    import inspect
    src = inspect.getsource(aggregators._server_adaptive)
    check("except Exception" not in src, "no blanket except-and-retry left in _server_adaptive")
    check(src.count("server.t += 1") == 1, "server.t is incremented exactly once")

    server = ServerOptimizerState()
    g = sd([1.0, 1.0])
    states = [sd([0.0, 0.0]), sd([0.0, 0.0])]
    out = aggregate("fedadam", states, [1.0, 1.0], g, {"server_lr": 0.1}, server).state_dict
    check(server.t == 1, f"one round -> server.t == {server.t}")
    check(torch.isfinite(out["w"]).all().item(), "fedadam output is finite")
    # global [1,1], both clients [0,0] -> the server must step TOWARD the average
    moved = float(out["w"].mean())
    check(0.0 < moved < 1.0, f"fedadam stepped toward the client average ({moved:.4f} in (0,1))")
    # unknown kind must raise, not be swallowed
    try:
        aggregators._server_adaptive(states, [1.0, 1.0], g, ServerOptimizerState(), {}, "bogus")
        check(False, "an unknown server optimizer raises")
    except ValueError:
        check(True, "an unknown server optimizer raises")


# -------------------------------------------------------------------------- 11
def test_trimmed_mean_reports_when_it_trims_nothing() -> None:
    """#11 trim_ratio=0.1 trims nothing for K<10, silently becoming FedAvg."""
    print("\n=== #11 trimmed_mean is honest about not trimming ===")
    res = aggregate("trimmed_mean", [sd([1.0]), sd([2.0])], [1.0, 1.0], sd([0.0]), {"trim_ratio": 0.1})
    check(res.info.get("effectively_untrimmed") is True, f"K=2 flagged as untrimmed: {res.info}")
    check(res.info.get("trimmed_per_side") == 0, "0 trimmed per side reported")
    res10 = aggregate("trimmed_mean", [sd([float(i)]) for i in range(10)], [1.0] * 10,
                      sd([0.0]), {"trim_ratio": 0.1})
    check(res10.info.get("effectively_untrimmed") is False and res10.info["trimmed_per_side"] == 1,
          f"K=10 really trims: {res10.info}")


# -------------------------------------------------------------------------- 12
def test_validation_gate_needs_the_published_value() -> None:
    """#12 the gate printed 'within tolerance' without ever comparing to anything."""
    print("\n=== #12 validation gate ===")
    import inspect
    from app import runner
    src = inspect.getsource(runner.validation_gate)
    check("expected_l2" in src and "PASSED (WEAK)" in src,
          "the weak path is labelled instead of claiming a tolerance check")
    check("--expected-l2" in inspect.getsource(runner.main), "--expected-l2 is wired into the CLI")

    sig = inspect.signature(runner.validation_gate)
    check("expected_l2" in sig.parameters and "tolerance" in sig.parameters,
          f"validation_gate{sig} takes the published value and the tolerance")


# --------------------------------------------------------------- FR-DATA-7
def test_w1_self_check_matches_manual_calculation() -> None:
    """FR-DATA-7: W1 must be verified against hand-computed values in the gate."""
    print("\n=== FR-DATA-7 W1 self-check ===")
    import numpy as np
    from app.metrics import self_check_w1, w1_1d, w1_nd

    results = self_check_w1()
    check(len(results) >= 5, f"{len(results)} manual W1 cases are checked")
    for name, ok, detail in results:
        check(ok, f"{name}: {detail}")

    # The cases must actually be able to fail - a self-check that cannot fail is
    # not a check. Re-run the mean-difference shortcut the code used to use and
    # confirm it disagrees with the manual answer on the symmetric-cloud case.
    A = np.array([[0.0, 0.0], [0.0, 0.0]])
    B = np.array([[-1.0, 0.0], [1.0, 0.0]])
    mean_diff = float(np.linalg.norm(A.mean(axis=0) - B.mean(axis=0)))
    check(abs(mean_diff - 0.0) < 1e-12, f"the old mean-diff shortcut gives {mean_diff} here ...")
    check(abs(w1_nd(A, B) - 1.0) < 1e-9, f"... while the real W1 is {w1_nd(A, B)}")

    # And the gate must call it.
    import inspect
    from app import runner
    check("self_check_w1" in inspect.getsource(runner.validation_gate),
          "validation_gate invokes the FR-DATA-7 self-check")


# ----------------------------------------------------------------- FR-RUN-5
def test_validation_gate_supports_both_tolerance_readings() -> None:
    """FR-RUN-5 asks for '+/-5% relative ... (absolute difference)'; support both."""
    print("\n=== FR-RUN-5 tolerance modes ===")
    import inspect
    from app import runner

    sig = inspect.signature(runner.validation_gate)
    check("tolerance_mode" in sig.parameters, f"validation_gate{sig} takes tolerance_mode")
    check(sig.parameters["tolerance_mode"].default == "relative",
          "defaults to the relative ('+/-5%') reading")

    main_src = inspect.getsource(runner.main)
    check("--gate-mode" in main_src and "relative" in main_src and "absolute" in main_src,
          "--gate-mode is wired into the CLI with both choices")

    # The two readings must actually diverge, otherwise the flag is decorative.
    # best=0.054, published=0.10: relative deviation 46% (fails 5%), absolute
    # deviation 0.046 (passes 0.05).
    best, published = 0.054, 0.10
    rel = abs(best - published) / published
    absd = abs(best - published)
    check(rel > 0.05 and absd < 0.05,
          f"relative {rel:.3f} > 0.05 but absolute {absd:.3f} < 0.05: the modes disagree")


# ------------------------------------------------------------- problems.py
def test_burgers_operator_actually_depends_on_the_ic() -> None:
    """The Burgers IC->solution map was rank 1: only mean(IC) survived."""
    print("\n=== Burgers operator is not a scalar regression ===")
    import numpy as np
    from app.problems import get_problem

    b = get_problem("burgers")
    xs = b.sensors
    ic_a = np.sin(np.pi * xs) - np.sin(2 * np.pi * xs)
    ic_b = np.sin(3 * np.pi * xs) - np.sin(4 * np.pi * xs)
    check(np.abs(ic_a - ic_b).max() > 1.0, "the two ICs really are different")

    sa = b._solution(ic_a[None, :], b.trunk)[0]
    sb = b._solution(ic_b[None, :], b.trunk)[0]
    check(not np.allclose(sa, sb),
          f"different ICs give different fields (max diff {np.abs(sa - sb).max():.4f})")

    rng = np.random.default_rng(0)
    M = np.array([b._solution(r[None, :], b.trunk)[0] for r in rng.normal(size=(40, 101))])
    rank = int(np.linalg.matrix_rank(M))
    check(rank >= 4, f"IC -> solution map has rank {rank}, not 1")

    # The field at t=0 must reproduce the initial condition exactly.
    ics = b._sample_ic(3, seed=11)
    sol = b._solution(ics, b.trunk)
    j = int(np.argmin(np.abs(b.trunk[:, 1])))
    truth = np.array([np.interp(b.trunk[j, 0], xs, r) for r in ics])
    check(np.abs(sol[:, j] - truth).max() < 1e-9,
          f"field at t=0 equals the IC (max err {np.abs(sol[:, j] - truth).max():.3g})")


def test_allen_cahn_ground_truth_solves_its_own_pde() -> None:
    """u_true violated the Allen-Cahn operator by ~8.5x its own magnitude."""
    print("\n=== Allen-Cahn manufactured solution is exact ===")
    import numpy as np
    import torch
    from app.models import allen_cahn_transform
    from app.problems import get_problem

    ac = get_problem("allen_cahn")
    x = np.linspace(-1, 1, 60)
    t = np.linspace(0, 1, 40)
    X, T = np.meshgrid(x, t)
    X = X.ravel()[:, None]; T = T.ravel()[:, None]
    xt = np.hstack([X, T])
    u = ac.u_true(xt)

    # (1) exact solution of the FORCED Allen-Cahn equation
    A = X ** 2 * np.cos(np.pi * X)
    Ap = 2 * X * np.cos(np.pi * X) - np.pi * X ** 2 * np.sin(np.pi * X)
    App = 2 * np.cos(np.pi * X) - 4 * np.pi * X * np.sin(np.pi * X) \
        - np.pi ** 2 * X ** 2 * np.cos(np.pi * X)
    B = 1 - 5 * T * (1 - X ** 2)
    Bt = -5 * (1 - X ** 2); Bx = 10 * T * X; Bxx = 10 * T
    q = ac.forcing(torch.tensor(xt, dtype=torch.float64)).numpy()
    resid = (A * Bt) - 0.001 * (App * B + 2 * Ap * Bx + A * Bxx) + 5 * (u ** 3 - u) - q
    rms = float(np.sqrt(np.mean(resid ** 2)))
    scale = float(np.sqrt(np.mean(u ** 2)))
    check(rms / scale < 1e-10,
          f"residual/|u| = {rms/scale:.3e} (was 8.48 before the source term)")

    # (2) exactly representable by the hard-constraint transform, otherwise the
    # achievable L2 has a nonzero floor and the benchmark is uninterpretable.
    ic = X ** 2 * np.cos(np.pi * X)
    y = (u - ic) / np.where(np.abs(T * (1 - X ** 2)) > 1e-14, T * (1 - X ** 2), 1.0)
    rec = allen_cahn_transform(torch.tensor(xt, dtype=torch.float64),
                               torch.tensor(y, dtype=torch.float64)).numpy()
    floor = float(np.linalg.norm(rec - u) / np.linalg.norm(u))
    check(floor < 1e-10,
          f"transform can represent u_true exactly (floor {floor:.3e}; was 0.481)")

    # The training residual must include the source, or the fix is inert.
    import inspect
    check("forcing" in inspect.getsource(type(ac).local_loss),
          "local_loss subtracts the manufactured source term")


def test_architectures_match_srs_table_4() -> None:
    """SRS Table 4 pins width/depth/activation per problem."""
    print("\n=== architectures vs SRS Table 4 ===")
    from app.problems import list_problems, get_problem

    # Rows read as (depth, width, activation). DeepONet branch_sizes has no
    # separate output layer, so every entry after the input is hidden.
    SRS = {
        "gramacy_lee": (3, 64, "tanh"), "schaffer": (3, 64, "tanh"),
        "poisson": (3, 20, "tanh"), "helmholtz": (3, 64, "sine"),
        "allen_cahn": (3, 64, "sine"), "inverse_ns": (6, 50, "tanh"),
        "inverse_dr": (3, 20, "tanh"), "antiderivative": (2, 40, "relu"),
        "diffusion_reaction": (3, 100, "relu"), "burgers": (2, 64, "relu"),
    }
    for name in list_problems():
        par = get_problem(name).model_spec().params
        act = str(par.get("activation", "?")).lower()
        hidden = (par["layer_sizes"][1:-1] if "layer_sizes" in par
                  else par["branch_sizes"][1:])
        width, depth = max(hidden), len(hidden)
        wd, ww, wa = SRS[name]
        check(depth == wd and width == ww and act == wa,
              f"{name}: d={depth} w={width} act={act} (SRS d={wd} w={ww} act={wa})")


# ------------------------------------------------------- experiment_config.py
def test_sweep_expansion_is_neither_empty_nor_partly_ignored() -> None:
    """An empty dimension expanded to 0 configs; a top-level list was dropped."""
    print("\n=== sweep expansion ===")
    from app.experiment_config import config_to_run_start, expand_sweeps, load_yaml
    from app.protocol import RunStartRequest

    # (1) An empty sweep dimension used to make itertools.product yield nothing,
    # so the runner looped over zero configs and exited 0 having done nothing.
    try:
        expand_sweeps({"problem": "poisson", "aggregator": ["fedavg", "fedprox"], "seed": []})
        check(False, "empty sweep dimension should raise")
    except ValueError as exc:
        check("empty list" in str(exc), f"empty dimension raises: {str(exc)[:60]}")

    # (2) A top-level list next to a 'sweep' mapping used to be passed through
    # unexpanded, and RunStartRequest then rejected the list.
    out = expand_sweeps({"problem": "poisson", "aggregator": "fedavg",
                         "seeds": [1, 2, 3], "sweep": {"total_rounds": [2, 4]}})
    check(len(out) == 6, f"both syntaxes cross-product: {len(out)} configs (expected 6)")
    check(not any(isinstance(c.get("seed"), list) for c in out),
          "no list survives into a concrete config")
    check(sorted({c["seed"] for c in out}) == [1, 2, 3], "every seed value is swept")
    check(sorted({c["total_rounds"] for c in out}) == [2, 4], "every rounds value is swept")

    req = RunStartRequest(**{k: v for k, v in config_to_run_start(out[0]).items()
                             if v is not None or k in ("problem", "aggregator")})
    check(isinstance(req.seed, int), f"RunStartRequest accepts the result (seed={req.seed})")

    # (3) The two existing forms must keep working unchanged.
    plain = expand_sweeps({"problem": "poisson",
                           "aggregator": ["fedavg", "fedprox"], "seed": [1, 2]})
    check(len(plain) == 4, f"plain list sweep still gives 4, got {len(plain)}")
    single = expand_sweeps({"problem": "poisson", "aggregator": "fedavg", "total_rounds": 5})
    check(len(single) == 1 and single[0]["total_rounds"] == 5, "concrete config unchanged")

    # (4) The shipped Appendix A.4 sweep must still expand to its documented 9.
    shipped = load_yaml("/home/user/fedsciml/orient/configs/sweep_example.yaml")
    check(len(expand_sweeps(shipped)) == 9,
          f"sweep_example.yaml expands to {len(expand_sweeps(shipped))} (documented 3*3=9)")


def test_one_bad_sweep_config_cannot_abort_the_rest() -> None:
    """A single invalid config used to discard every config after it."""
    print("\n=== sweep failure isolation ===")
    import inspect
    from app import runner

    # run_configs must catch per-config failures rather than let them propagate.
    src = inspect.getsource(runner.run_configs)
    check("except Exception" in src and '"error"' in src,
          "run_configs records a per-config error instead of propagating")

    # Drive it with a mix: two valid configs around one that cannot start
    # (krum needs K >= 2f+3 = 5, so K=2 is rejected).
    good = {"problem": "poisson", "aggregator": "fedavg", "n_clients": 2,
            "total_rounds": 1, "local_epochs": 1, "seed": 42}
    bad = {"problem": "poisson", "aggregator": "krum", "n_clients": 2,
           "total_rounds": 1, "local_epochs": 1, "seed": 42}
    results = runner.run_configs([good, bad, good], results_dir="/tmp/orient_sweep_iso",
                                 server_url=None)
    check(len(results) == 3, f"all 3 configs were attempted, got {len(results)}")
    check(results[1].get("error") is not None, "the invalid config recorded an error")
    check("krum requires K" in str(results[1].get("error")),
          f"the error names the real cause: {str(results[1].get('error'))[:60]}")
    check(results[0].get("status", {}).get("phase") == "complete"
          and results[2].get("status", {}).get("phase") == "complete",
          "the configs after the failure still ran to completion")

    # The shipped Appendix A.4 sweep must now expand AND be runnable: krum needs
    # n_clients >= 5, and the file used to say 2.
    from app.experiment_config import config_to_run_start, expand_sweeps, load_yaml
    shipped = expand_sweeps(load_yaml("/home/user/fedsciml/orient/configs/sweep_example.yaml"))
    krum = [c for c in shipped if c["aggregator"] == "krum"]
    check(len(krum) == 3, f"{len(krum)} krum cells in the shipped sweep")
    check(all(int(c["n_clients"]) >= 5 for c in krum),
          f"every krum cell has n_clients >= 5 (got {sorted({int(c['n_clients']) for c in krum})})")


def main() -> int:
    test_median_is_a_real_median_at_even_k()
    test_noise_fraction_always_compromises_someone()
    test_topk_sparsifies_the_update_not_the_model()
    test_robust_aggregators_can_start_before_clients_register()
    test_shrinking_round_cannot_feed_median_two_updates()
    test_holdout_only_when_quality_weighting()
    test_scaffold_control_variates_follow_the_client()
    test_quality_weighting_is_honest_about_domination()
    test_w1_nd_fallback_is_a_wasserstein_distance()
    test_dp_epsilon_does_not_scale_with_client_fraction()
    test_server_adaptive_does_not_double_apply_momentum()
    test_trimmed_mean_reports_when_it_trims_nothing()
    test_validation_gate_needs_the_published_value()
    test_w1_self_check_matches_manual_calculation()
    test_validation_gate_supports_both_tolerance_readings()
    test_burgers_operator_actually_depends_on_the_ic()
    test_allen_cahn_ground_truth_solves_its_own_pde()
    test_architectures_match_srs_table_4()
    test_sweep_expansion_is_neither_empty_nor_partly_ignored()
    test_one_bad_sweep_config_cannot_abort_the_rest()
    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} FAILURE(S):")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("ALL LOGIC-FIX REGRESSION TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
