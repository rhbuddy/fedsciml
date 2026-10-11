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
"""

from __future__ import annotations

import argparse
import pathlib
import sys

# ---- path setup: make `orient` importable regardless of cwd ----
ROOT = pathlib.Path(__file__).resolve().parent
ORIENT = ROOT / "orient"
BACKEND = ORIENT / "backend"
CLIENT = ORIENT / "client"
for p in (ROOT, ORIENT, BACKEND, CLIENT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _run_api(args: argparse.Namespace) -> int:
    try:
        import uvicorn
        try:
            from backend.app.api import app  # type: ignore  # when ORIENT on path
        except ImportError:
            from app.api import app  # type: ignore  # when BACKEND on path
        try:
            from backend.app.config import settings  # type: ignore
        except ImportError:
            from app.config import settings  # type: ignore
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
    import subprocess
    import pathlib
    dash = BACKEND / "ui" / "dashboard.py"
    if not dash.exists():
        print(f"[server] dashboard not found at {dash}")
        return 1
    print(f"[server] launching dashboard: streamlit run {dash}")
    cmd = [sys.executable, "-m", "streamlit", "run", str(dash), "--server.port", str(args.dashboard_port or 8501)]
    # also allow --server.address
    try:
        subprocess.run(cmd, check=False)
    except FileNotFoundError:
        print("streamlit not installed: pip install streamlit pandas")
        return 1
    return 0


def _run_runner(args: argparse.Namespace) -> int:
    try:
        try:
            from backend.app.runner import main as runner_main  # type: ignore
        except ImportError:
            from app.runner import main as runner_main  # type: ignore
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
