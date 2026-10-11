#!/usr/bin/env python3
"""
Orient Federated SciML — Client (single-file entry point)

Full access to the Client app from one file. Opening this file gives you
everything needed to run a federated client, generate datasets, or launch
the client UI.

Usage:
  # 1. Generate heterogeneous bundles (10 problems x K clients)
  python client.py --make-datasets --clients 5 --samples 1500

  # 2. Run a client (weights-only, data never leaves)
  python client.py --server http://127.0.0.1:8000 --client-id client-1 --dataset orient/client/examples/bundles/poisson_client1
  python client.py --server http://127.0.0.1:8000 --client-id client-2 --dataset orient/client/examples/bundles/poisson_client2

  # 3. Launch client UI (Streamlit)
  python client.py --ui
  python client.py --ui --server http://127.0.0.1:8000

  # 4. Helpers
  python client.py --list-problems
  python client.py --list-bundles
  python client.py --help

This file is a thin wrapper over orient/client/app/* — all training logic
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

# ---- path setup: make the client `app` package importable regardless of cwd ----
HERE = pathlib.Path(__file__).resolve().parent
# This file lives either at the repo root (next to `orient/`) or inside `orient/`
# itself. Pick whichever layout actually holds the client app.
ORIENT = HERE / "orient" if (HERE / "orient" / "client" / "app").is_dir() else HERE
CLIENT = ORIENT / "client"

# Put ONLY the client app root on sys.path. `orient/backend` also contains a
# package named `app`, so adding both makes `import app` ambiguous: whichever
# comes first wins, and the server's app silently shadows the client's.
#
# Likewise, do NOT import this app as `client.app.*`. The file `orient/client.py`
# is a top-level module named `client` (and `orient/client/` has no `__init__.py`),
# so `import client` binds to that file and `client.app` is never a package.
if str(CLIENT) not in sys.path:
    sys.path.insert(0, str(CLIENT))


def _load_client(module: str, attr: str):
    """Import ``attr`` from client/app/<module>.py.

    Errors from inside the module propagate unchanged — only a missing module is
    treated as "not this layout".
    """
    candidate = f"app.{module}"
    try:
        if importlib.util.find_spec(candidate) is None:
            raise ImportError(f"cannot find {candidate} under {CLIENT}")
    except ImportError as exc:
        raise ImportError(
            f"cannot import '{attr}' from {candidate} (looked under {CLIENT}). "
            f"Install deps: pip install -r orient/client/requirements.txt"
        ) from exc
    return getattr(importlib.import_module(candidate), attr)


def _list_problems() -> int:
    try:
        list_problems = _load_client("problems", "list_problems")
        print("Problems (10): " + ", ".join(list_problems()))
    except Exception as e:
        print(f"[client] failed to list problems: {e}")
        import traceback
        traceback.print_exc()
        return 1
    return 0


def _list_bundles() -> int:
    bundles = CLIENT / "examples" / "bundles"
    if not bundles.exists():
        print(f"[client] no bundles at {bundles} — run --make-datasets first")
        return 1
    names = sorted(p.name for p in bundles.iterdir() if p.is_dir())
    print(f"[client] {len(names)} bundles at {bundles}:")
    for n in names:
        print(f"  - {n}")
    return 0


def _make_datasets(args: argparse.Namespace) -> int:
    try:
        # import as a module (by path) so we can call its build() directly
        spec = importlib.util.spec_from_file_location(
            "orient_make_datasets", CLIENT / "examples" / "make_datasets.py"
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {CLIENT / 'examples' / 'make_datasets.py'}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
        # `if __name__ == "__main__"` does not fire for a module loaded by path,
        # so drive build() ourselves instead of its CLI.
        if not hasattr(mod, "build"):
            raise ImportError("make_datasets.py exposes no build()")
        out = pathlib.Path(args.out or (CLIENT / "examples" / "bundles"))
        mod.build(out, int(args.samples), int(args.clients))
        names = sorted(p.name for p in out.iterdir() if p.is_dir())
        print(f"[client] wrote {len(names)} bundles to {out}")
    except SystemExit as e:
        return int(e.code or 0)
    except Exception as e:
        print(f"[client] make-datasets failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    return 0


def _run_streamlit(script: pathlib.Path, port: int, label: str) -> int:
    """Run a Streamlit script, reporting failure honestly.

    ``python -m streamlit`` exits non-zero when Streamlit is missing, so the
    child's return code has to be propagated — a FileNotFoundError here would
    only fire if the interpreter itself were absent.
    """
    import subprocess

    if not script.exists():
        print(f"[{label}] UI not found at {script}")
        return 1
    if importlib.util.find_spec("streamlit") is None:
        print(f"[{label}] streamlit not installed: pip install streamlit pandas")
        return 1
    print(f"[{label}] launching UI: streamlit run {script}")
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(script),
        "--server.port", str(port),
    ]
    return int(subprocess.run(cmd, check=False).returncode or 0)


def _run_ui(args: argparse.Namespace) -> int:
    return _run_streamlit(CLIENT / "ui" / "app.py", args.ui_port or 8502, "client")


def _run_client(args: argparse.Namespace) -> int:
    # Delegate to client/app/main.py CLI
    try:
        client_main = _load_client("main", "main")
    except ImportError as e:
        print(f"[client] import failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    argv: list[str] = []
    # required
    if args.server:
        argv += ["--server", args.server]
    else:
        print("[client] --server is required (e.g. http://127.0.0.1:8000)")
        return 2
    if args.client_id:
        argv += ["--client-id", args.client_id]
    else:
        print("[client] --client-id is required (e.g. client-1)")
        return 2
    if args.dataset:
        argv += ["--dataset", args.dataset]
    else:
        print("[client] --dataset is required (e.g. orient/client/examples/bundles/poisson_client1)")
        return 2
    if args.problem:
        argv += ["--problem", args.problem]
    if args.batch_size is not None:
        argv += ["--batch-size", str(args.batch_size)]
    if args.poll_interval is not None:
        argv += ["--poll-interval", str(args.poll_interval)]
    if args.grad_clip is not None:
        argv += ["--grad-clip", str(args.grad_clip)]
    if args.once:
        argv += ["--once"]
    if args.max_wait is not None:
        argv += ["--max-wait", str(args.max_wait)]
    if args.verbose:
        argv += ["--verbose"]
    return int(client_main(argv) or 0)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="client.py",
        description="Orient Client — single-file entry (train + UI + datasets).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python client.py --make-datasets --clients 5\n"
            "  python client.py --server http://127.0.0.1:8000 --client-id client-1 --dataset orient/client/examples/bundles/poisson_client1\n"
            "  python client.py --ui\n"
        ),
    )
    # Mode selectors
    p.add_argument("--ui", action="store_true", help="Launch Streamlit client UI")
    p.add_argument("--ui-port", type=int, default=8502, help="UI port")
    p.add_argument("--make-datasets", action="store_true", help="Generate example bundles")
    p.add_argument("--list-problems", action="store_true", help="List 10 problems")
    p.add_argument("--list-bundles", action="store_true", help="List available bundles")
    # make-datasets options
    p.add_argument("--clients", type=int, default=5, help="Clients per problem for --make-datasets (default 5)")
    p.add_argument("--samples", type=int, default=1500, help="Samples per client (default 1500)")
    p.add_argument("--out", type=str, default=None, help="Output dir for bundles")
    # client run options (delegated)
    p.add_argument("--server", type=str, default=None, help="Server URL (e.g. http://127.0.0.1:8000)")
    p.add_argument("--client-id", type=str, default=None, help="Unique client id (e.g. client-1)")
    p.add_argument("--dataset", type=str, default=None, help="Path to dataset bundle directory")
    p.add_argument("--problem", type=str, default=None, help="Problem name override")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--poll-interval", type=float, default=None)
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--once", action="store_true")
    p.add_argument("--max-wait", type=float, default=None)
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_problems:
        return _list_problems()
    if args.list_bundles:
        return _list_bundles()
    if args.make_datasets:
        return _make_datasets(args)
    if args.ui:
        return _run_ui(args)
    # if any client-run flags present, delegate
    if args.server or args.dataset or args.client_id:
        return _run_client(args)

    parser.print_help()
    print("\nTip: start with --make-datasets then --server + --dataset, or --ui for browser UI.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
