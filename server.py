#!/usr/bin/env python3
"""
Orient Federated SciML — Server (single-file entry point)

Full access to the Server app from one file. Opening this file gives you
everything needed to run the federated server, dashboard, runner and
validation gate.

Usage:
  python server.py                          # start REST API on 0.0.0.0:8000
  python server.py --host 127.0.0.1 --port 8000
  python server.py --dashboard              # start Streamlit dashboard (needs streamlit)
  python server.py --config orient/configs/poisson_fedavg.yaml   # run experiment (in-memory)
  python server.py --config orient/configs/sweep_example.yaml --results-dir results
  python server.py --validation-gate poisson # reproduce paper FedAvg Poisson (FR-RUN-5)
  python server.py --list-problems
  python server.py --help

This file is a thin wrapper over orient/backend/app/* — all science logic
lives there, this file just makes it launchable from one place.

It ships in two identical copies (repo root and orient/) and works from
either location; see the ORIENT resolution below.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import pathlib
import sys

# ---- path setup: make `orient` importable regardless of cwd ----
HERE = pathlib.Path(__file__).resolve().parent
# This file lives either at the repo root (next to `orient/`) or inside `orient/`
# itself. Pick whichever layout actually holds the backend app.
ORIENT = HERE / "orient" if (HERE / "orient" / "backend" / "app").is_dir() else HERE
BACKEND = ORIENT / "backend"

# The backend app root must come FIRST: `orient/client` also contains a package
# named `app`, and if that one won, `from app.api import app` would resolve to the
# *client* package and fail with "No module named 'app.api'". The client path is
# deliberately NOT added here — runner.py adds it itself when it needs LocalTrainer.
for p in (ORIENT, BACKEND):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _load_backend(module: str, attr: str):
    """Import ``attr`` from backend/app/<module>.py.

    Tries the plain ``app.<module>`` name first (backend/ on sys.path) and then
    the fully qualified ``backend.app.<module>`` (orient/ on sys.path). Errors
    raised from inside the module propagate unchanged.
    """
    for candidate in (f"app.{module}", f"backend.app.{module}"):
        try:
            if importlib.util.find_spec(candidate) is None:
                continue
        except ImportError:
            continue
        return getattr(importlib.import_module(candidate), attr)
    raise ImportError(
        f"cannot import '{attr}' from backend/app/{module}.py "
        f"(looked under {BACKEND} and {ORIENT}). "
        f"Install deps: pip install -r orient/backend/requirements.txt"
    )


def _run_api(args: argparse.Namespace) -> int:
    try:
        import uvicorn  # noqa: F401  (fail fast with the friendly hint below)
        app = _load_backend("api", "app")
        settings = _load_backend("config", "settings")
    except ImportError as e:
        print(f"[server] import failed: {e}")
        print("Install deps: pip install -r orient/backend/requirements.txt")
        return 1

    host = args.host or settings.host
    port = args.port or settings.port
    print(f"[server] starting REST API on http://{host}:{port}  (docs: http://{host}:{port}/docs)")
    print(f"[server] results -> {settings.results_dir}  seed={settings.seed}")
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


def _run_dashboard(args: argparse.Namespace) -> int:
    """Run the dashboard, reporting failure honestly.

    ``python -m streamlit`` exits non-zero when Streamlit is missing, so the
    child's return code has to be propagated — a FileNotFoundError here would
    only fire if the interpreter itself were absent.
    """
    import subprocess

    dash = BACKEND / "ui" / "dashboard.py"
    if not dash.exists():
        print(f"[server] dashboard not found at {dash}")
        return 1
    if importlib.util.find_spec("streamlit") is None:
        print("[server] streamlit not installed: pip install streamlit pandas")
        return 1
    print(f"[server] launching dashboard: streamlit run {dash}")
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(dash),
        "--server.port", str(args.dashboard_port or 8501),
    ]
    return int(subprocess.run(cmd, check=False).returncode or 0)


def _run_runner(args: argparse.Namespace) -> int:
    try:
        runner_main = _load_backend("runner", "main")
    except ImportError as e:
        print(f"[server] runner import failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    argv: list[str] = []
    if args.config:
        argv += ["--config", args.config]
    if args.sweep:
        argv += ["--sweep", args.sweep]
    if args.results_dir:
        argv += ["--results-dir", args.results_dir]
    if args.server_url:
        argv += ["--server", args.server_url]
    if args.validation_gate is not None:
        # --validation-gate may be "" (default poisson) or a problem name
        val = args.validation_gate if isinstance(args.validation_gate, str) and args.validation_gate else "poisson"
        argv += ["--validation-gate", val]
        if args.expected_l2 is not None:
            argv += ["--expected-l2", str(args.expected_l2)]
        if args.gate_tolerance is not None:
            argv += ["--gate-tolerance", str(args.gate_tolerance)]
        if args.gate_mode is not None:
            argv += ["--gate-mode", str(args.gate_mode)]
    if args.list_problems:
        argv += ["--list-problems"]
    return int(runner_main(argv) or 0)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="server.py",
        description="Orient Server — single-file entry (REST API + runner + dashboard).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python server.py                                  # API\n"
            "  python server.py --config orient/configs/poisson_fedavg.yaml\n"
            "  python server.py --validation-gate poisson\n"
            "  python server.py --dashboard\n"
            "  python server.py --host 0.0.0.0 --port 8000\n"
        ),
    )
    # API
    p.add_argument("--host", type=str, default=None, help="Bind host (default from ORIENT_HOST or 0.0.0.0)")
    p.add_argument("--port", type=int, default=None, help="Bind port (default 8000)")
    # Dashboard
    p.add_argument("--dashboard", action="store_true", help="Launch Streamlit dashboard instead of API")
    p.add_argument("--dashboard-port", type=int, default=8501, help="Dashboard port")
    # Runner (delegated)
    p.add_argument("--config", type=str, default=None, help="YAML config for one run or sweep (FR-CFG-1)")
    p.add_argument("--sweep", type=str, default=None, help="Alias for --config (sweep YAML)")
    p.add_argument("--results-dir", type=str, default=None, help="Results root (default results)")
    p.add_argument("--server-url", type=str, default=None, help="If set, drive live server via HTTP instead of in-memory")
    p.add_argument("--validation-gate", nargs="?", const="poisson", default=None, help="Run validation gate (FR-RUN-5) for problem (default poisson)")
    p.add_argument("--expected-l2", type=float, default=None, help="Published relative L2; enables the FR-RUN-5 +/-tolerance comparison")
    p.add_argument("--gate-tolerance", type=float, default=0.05, help="FR-RUN-5 tolerance (default 0.05)")
    p.add_argument("--gate-mode", type=str, default=None, choices=["relative", "absolute"],
                   help="FR-RUN-5 tolerance reading: relative (default) or absolute difference")
    p.add_argument("--list-problems", action="store_true", help="List 10 benchmark problems and exit")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Mode dispatch: runner options take precedence over API/dashboard
    if args.config or args.sweep or args.validation_gate is not None or args.list_problems:
        return _run_runner(args)
    if args.dashboard:
        return _run_dashboard(args)
    return _run_api(args)


if __name__ == "__main__":
    raise SystemExit(main())
