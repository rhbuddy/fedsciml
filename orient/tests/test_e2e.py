"""End-to-end integration test.

Starts a **real** uvicorn server, launches **real** client processes against it,
runs a full federated job, and checks the global model actually improves.

Run from ``orient/``::

    .venv\\Scripts\\python.exe tests\\test_e2e.py
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import httpx

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
CLIENT = ROOT / "client"
BUNDLES = CLIENT / "examples" / "bundles"

VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
PY = str(VENV_PY) if VENV_PY.exists() else sys.executable

PROBLEM = os.environ.get("E2E_PROBLEM", "gramacy_lee")
ROUNDS = int(os.environ.get("E2E_ROUNDS", "30"))
LOCAL_EPOCHS = int(os.environ.get("E2E_LOCAL_EPOCHS", "10"))
AGGREGATOR = os.environ.get("E2E_AGGREGATOR", "fedavg")
WEIGHTING = os.environ.get("E2E_WEIGHTING", "data_size")
SERVER_LR = float(os.environ.get("E2E_SERVER_LR", "0.3"))
N_CLIENTS = int(os.environ.get("E2E_CLIENTS", "2"))


def discover_bundles(problem: str) -> List[str]:
    """Use EVERY client bundle generated for this problem.

    Using only a subset leaves part of the domain unseen, so the global model
    cannot improve and the test fails for the wrong reason.
    """
    directory = ROOT / "client" / "examples" / "bundles"
    if directory.exists():
        found = [
            path.name
            for path in directory.iterdir()
            if path.is_dir() and path.name.startswith(f"{problem}_client")
        ]
        if found:
            return sorted(found, key=lambda n: int(n.rsplit("client", 1)[1]))
    return [f"{problem}_client1", f"{problem}_client2"]


_env_bundles = os.environ.get("E2E_BUNDLES")
BUNDLE_NAMES: List[str] = (
    [b.strip() for b in _env_bundles.split(",")] if _env_bundles else discover_bundles(PROBLEM)
)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_health(url: str, timeout: float = 90.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(f"{url}/health", timeout=2.0).status_code == 200:
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.5)
    return False


def main() -> int:
    missing = [n for n in BUNDLE_NAMES if not (BUNDLES / n).exists()]
    if missing:
        print(f"Missing bundles {missing}. Run: python examples\\make_datasets.py")
        return 2

    port = free_port()
    url = f"http://127.0.0.1:{port}"
    env = {**os.environ, "ORIENT_HOST": "127.0.0.1", "ORIENT_PORT": str(port)}

    print(f"Starting server on {url} ...")
    logs = ROOT / "tests" / "_e2e_logs"
    logs.mkdir(exist_ok=True)
    server_log = (logs / "server.log").open("w", encoding="utf-8")
    client_logs = [(logs / f"client-{i}.log").open("w", encoding="utf-8") for i in range(1, 20)]
    # NOTE: never use subprocess.PIPE here without a reader — an undrained pipe
    # fills up and blocks the child process (which silently hangs the server).
    server = subprocess.Popen(
        [PY, "-m", "app.main"], cwd=BACKEND, env=env,
        stdout=server_log, stderr=subprocess.STDOUT, text=True,
    )
    clients: List[subprocess.Popen] = []
    try:
        if not wait_for_health(url):
            print("Server failed to start:")
            print((logs / "server.log").read_text(encoding="utf-8")[-3000:])
            return 1
        print("Server is up.\n")

        run = httpx.post(
            f"{url}/admin/run/start",
            json={
                "problem": PROBLEM,
                "aggregator": AGGREGATOR,
                "weighting": WEIGHTING,
                "total_rounds": ROUNDS,
                "local_epochs": LOCAL_EPOCHS,
                "learning_rate": 0.01,
                "aggregator_params": {"mu": 0.01, "server_lr": SERVER_LR, "trim_ratio": 0.1,
                                      "n_byzantine": 1, "multi_k": 1},
            },
            timeout=30.0,
        )
        run.raise_for_status()
        print(f"Run started: {PROBLEM} · {AGGREGATOR} · {WEIGHTING} · "
              f"{ROUNDS} rounds · {LOCAL_EPOCHS} local epochs\n")

        for index, bundle in enumerate(BUNDLE_NAMES, start=1):
            clients.append(
                subprocess.Popen(
                    [PY, "-m", "app.main", "--server", url, "--client-id", f"client-{index}",
                     "--dataset", str(BUNDLES / bundle), "--poll-interval", "0.5"],
                    cwd=CLIENT, env=env,
                    stdout=client_logs[index - 1], stderr=subprocess.STDOUT, text=True,
                )
            )
        print(f"Launched {len(clients)} client(s): {', '.join(BUNDLE_NAMES)}\n")

        deadline = time.time() + 600
        metrics: List[Dict[str, Any]] = []
        while time.time() < deadline:
            try:
                status = httpx.get(f"{url}/run/status", timeout=30.0).json()
            except Exception as exc:  # noqa: BLE001
                print(f"\n  status poll failed: {exc}")
                print("  server log tail:")
                print("  " + (logs / "server.log").read_text(encoding="utf-8")[-2000:])
                return 1
            metrics = status.get("metrics", [])
            done = len(metrics)
            print(f"\r  phase={status['phase']} round={status['round']}/{status['total_rounds']} "
                  f"completed={done} submitted={len(status['submitted_clients'])}", end="")
            if status["phase"] == "complete":
                print()
                break
            time.sleep(1.0)
        else:
            print("\nTimed out waiting for the run to finish.")
            return 1

        for proc in clients:
            proc.terminate()
        for proc in clients:
            try:
                proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                proc.kill()

        print(f"\n{'round':>6} {'clients':>8} {'mean_local_loss':>18} {'l2_relative_error':>20}")
        print("-" * 56)
        for record in metrics:
            print(f"{record['round']:>6} {record['n_clients']:>8} "
                  f"{record.get('mean_local_loss') or float('nan'):>18.6g} "
                  f"{record['l2_relative_error']:>20.6g}")

        problems: List[str] = []
        if len(metrics) != ROUNDS:
            problems.append(f"expected {ROUNDS} completed rounds, got {len(metrics)}")
        if metrics:
            first = metrics[0]["l2_relative_error"]
            last = metrics[-1]["l2_relative_error"]
            best = min(m["l2_relative_error"] for m in metrics)
            print(f"\nL2 relative error: first={first:.6g}  best={best:.6g}  last={last:.6g}")
            # The run must demonstrably learn something. Comparing only the last
            # round is too brittle: adaptive server optimizers (fedadam/yogi/...)
            # legitimately oscillate round to round.
            if not (best < first):
                problems.append(f"run never improved on the initial model ({best:.6g} >= {first:.6g})")
            if metrics[0]["n_clients"] != len(BUNDLE_NAMES):
                problems.append("first round did not aggregate all clients")
            if not all(m["l2_relative_error"] == m["l2_relative_error"] for m in metrics):
                problems.append("NaN appeared in the L2 error history")
        else:
            problems.append("no metrics were produced")

        print("=" * 60)
        if problems:
            print(f"E2E FAILED ({len(problems)} problem(s)):")
            for item in problems:
                print(f"  - {item}")
            return 1
        print("E2E TEST PASSED — full federated round trip works")
        return 0
    finally:
        for proc in clients:
            if proc.poll() is None:
                proc.kill()
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except Exception:  # noqa: BLE001
                server.kill()


if __name__ == "__main__":
    sys.exit(main())