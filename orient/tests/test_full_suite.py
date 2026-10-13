"""Run the complete Orient test suite and print a pass/fail summary.

Replaces the old Selenium-based harness (``test_all_selenium_fixed.py``) with a
pure-Python runner — no browser driver, no selenium import. Every suite runs as
its own subprocess, so one crashing script cannot take the rest of the run down
with it.

Run from ``orient/``::

    python tests/test_full_suite.py          # everything, incl. the live E2E
    python tests/test_full_suite.py --fast   # skip the live E2E (server+clients)

Suites run in fast-to-slow order so a broken import surfaces in seconds. The
exit code is 0 only when every suite passes.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
PY = str(VENV_PY) if VENV_PY.exists() else sys.executable

# (label, script, needs_e2e) — ordered fast → slow.
SUITES = [
    ("static imports", "test_static_imports.py", False),
    ("logic-fix regressions", "test_logic_fixes.py", False),
    ("server logic", "test_server_logic.py", False),
    ("client", "test_client.py", False),
    ("runtime", "test_runtime.py", False),
    ("ui smoke", "test_ui_smoke.py", False),
    ("e2e live", "test_e2e.py", True),
]


def check_entry_points() -> tuple[bool, str]:
    """Smoke-check the CLI entry points (replaces the old harness's test_entry)."""
    cwd = ROOT.parent
    problems = [
        (cwd / "server.py", ["--list-problems"]),
        (cwd / "client.py", ["--list-problems"]),
        (cwd / "client.py", ["--list-bundles"]),
    ]
    for script, args in problems:
        try:
            run = subprocess.run(
                [PY, str(script), *args],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(cwd),
            )
        except Exception as exc:  # noqa: BLE001
            return False, f"{script.name} {' '.join(args)} raised {exc}"
        if run.returncode != 0:
            return False, (
                f"{script.name} {' '.join(args)} exited {run.returncode}: "
                f"{(run.stderr or run.stdout)[-200:]}"
            )
        expected = "bundles" if "--list-bundles" in args else "poisson"
        if expected not in run.stdout.lower():
            return False, f"{script.name} {' '.join(args)} missing '{expected}'"
    return True, "server.py / client.py list flags OK"


def run_suite(label: str, script: str) -> tuple[bool, float, str]:
    start = time.time()
    try:
        run = subprocess.run(
            [PY, str(TESTS / script)],
            capture_output=True,
            text=True,
            timeout=1800,
            cwd=str(ROOT),
        )
    except subprocess.TimeoutExpired:
        return False, time.time() - start, "TIMEOUT after 1800s"
    except Exception as exc:  # noqa: BLE001
        return False, time.time() - start, f"raised {exc}"
    elapsed = time.time() - start
    tail = (run.stdout or "").strip().splitlines()
    summary = tail[-1] if tail else ""
    if run.returncode != 0:
        detail = "\n".join(tail[-8:])
        err = (run.stderr or "").strip()
        if err:
            detail += "\n" + "\n".join(err.splitlines()[-4:])
        return False, elapsed, detail or f"exit {run.returncode}"
    return True, elapsed, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="skip the live E2E suite (real server + client processes)",
    )
    args = parser.parse_args()

    results: list[tuple[str, bool, float, str]] = []

    print("=" * 60)
    print("ORIENT FULL SUITE (no selenium)")
    print("=" * 60)

    ok, detail = check_entry_points()
    results.append(("entry points", ok, 0.0, detail))
    print(f"\n--- entry points ---\n  {'PASS' if ok else 'FAIL'}  {detail}")

    for label, script, needs_e2e in SUITES:
        if needs_e2e and args.fast:
            results.append((label, True, 0.0, "skipped (--fast)"))
            print(f"\n--- {label} ---\n  SKIP  skipped (--fast)")
            continue
        print(f"\n--- {label} ({script}) ---", flush=True)
        ok, elapsed, detail = run_suite(label, script)
        results.append((label, ok, elapsed, detail))
        print(f"  {'PASS' if ok else 'FAIL'}  [{elapsed:.1f}s] {detail}", flush=True)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    failures = 0
    for label, ok, elapsed, detail in results:
        status = "PASS" if ok else "FAIL"
        if "skipped" in detail:
            status = "SKIP"
        if not ok:
            failures += 1
        print(f"  {status:4}  {label:<22} [{elapsed:6.1f}s]  {detail[:80]}")

    if failures:
        print(f"\n{failures} SUITE(S) FAILED")
        return 1
    print("\nFULL SUITE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
