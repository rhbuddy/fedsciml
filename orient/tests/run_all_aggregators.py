"""Run the end-to-end test once per aggregator and print a pass/fail summary.

Each child is launched with ``subprocess.run`` (which drains its pipes), so this
never deadlocks the way an undrained ``PIPE`` would.

    .venv\\Scripts\\python.exe tests\\run_all_aggregators.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
PY = str(VENV_PY) if VENV_PY.exists() else sys.executable

AGGREGATORS = ["fedavg", "fedprox", "fedadam", "fedagrad", "fedyogi", "median", "trimmed_mean", "krum"]
PROBLEM = os.environ.get("E2E_PROBLEM", "gramacy_lee")

# Server-side adaptive optimizers take a normalized step of ~server_lr per round, so
# they need more rounds AND a bigger server_lr than plain averaging to make progress.
# Krum needs K >= 2f+3 clients to be meaningful.
PLANS = {
    "fedavg": {"rounds": 12, "epochs": 5, "server_lr": 0.3, "clients": 2},
    "fedprox": {"rounds": 12, "epochs": 5, "server_lr": 0.3, "clients": 2},
    "fedadam": {"rounds": 30, "epochs": 5, "server_lr": 1.0, "clients": 2},
    "fedagrad": {"rounds": 30, "epochs": 5, "server_lr": 1.0, "clients": 2},
    "fedyogi": {"rounds": 30, "epochs": 5, "server_lr": 1.0, "clients": 2},
    "median": {"rounds": 12, "epochs": 5, "server_lr": 0.3, "clients": 2},
    "trimmed_mean": {"rounds": 12, "epochs": 5, "server_lr": 0.3, "clients": 2},
    "krum": {"rounds": 12, "epochs": 5, "server_lr": 0.3, "clients": 5},
}


def main() -> int:
    print(f"Running E2E for {len(AGGREGATORS)} aggregators\n", flush=True)
    summary = []
    for agg in AGGREGATORS:
        plan = PLANS[agg]
        bundles = ",".join(
            f"{PROBLEM}_client{i + 1}" for i in range(plan["clients"])
        )
        env = {
            **os.environ,
            "E2E_AGGREGATOR": agg,
            "E2E_ROUNDS": str(plan["rounds"]),
            "E2E_LOCAL_EPOCHS": str(plan["epochs"]),
            "E2E_SERVER_LR": str(plan["server_lr"]),
            "E2E_BUNDLES": bundles,
        }
        proc = subprocess.run(
            [PY, "tests/test_e2e.py"], cwd=ROOT, env=env,
            capture_output=True, text=True,
        )
        lines = [line for line in (proc.stdout or "").strip().splitlines() if line.strip()]
        tail = lines[-1] if lines else "(no stdout)"
        if proc.returncode != 0:
            err_lines = [line for line in (proc.stderr or "").strip().splitlines() if line.strip()]
            tail += "  ||  " + (err_lines[-1][:160] if err_lines else "(no stderr)")
            for line in lines:
                if "L2 relative error" in line or line.strip().startswith("round "):
                    print(f"      {line.strip()}", flush=True)
        summary.append((agg, proc.returncode))
        print(f"{agg:<14} exit={proc.returncode}  {tail}", flush=True)

    print("\n" + "=" * 62)
    failed = [agg for agg, code in summary if code != 0]
    for agg, code in summary:
        print(f"  {'PASS' if code == 0 else 'FAIL'}  {agg}")
    if failed:
        print(f"\n{len(failed)} aggregator(s) FAILED: {failed}")
        return 1
    print("\nALL AGGREGATORS PASS END-TO-END")
    return 0


if __name__ == "__main__":
    sys.exit(main())