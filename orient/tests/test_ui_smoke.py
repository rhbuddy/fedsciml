"""Smoke-test the Streamlit UIs by executing the scripts against a stub.

Streamlit only runs a script when a browser opens a websocket session, so a
script can appear to "work" (server boots, /healthz returns 200) while still
crashing the moment it is rendered. This injects a lightweight ``streamlit``
stub that records every call and then executes each UI script top-to-bottom,
catching NameError / AttributeError / ImportError / logic errors.

    .venv\\Scripts\\python.exe tests\\test_ui_smoke.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import types
from pathlib import Path

import httpx

# The UI labels contain characters like '▶' / '⚡'; a cp1252 Windows console
# cannot encode them, which would mask the real test result.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
PY = str(VENV_PY) if VENV_PY.exists() else sys.executable

failures: list[str] = []


def check(condition: bool, label: str) -> None:
    print(("  PASS  " if condition else "  FAIL  ") + label)
    if not condition:
        failures.append(label)


class Stop(Exception):
    """Raised by the stubbed st.stop()."""


class Ctx:
    """Stand-in for a Streamlit DeltaGenerator.

    Behaves as a context manager (``with c1:``) AND forwards unknown attributes
    to the stub module (``c1.selectbox(...)``), exactly like the real object.
    """

    _module = None

    def __getattr__(self, name):
        if name.startswith("_") or Ctx._module is None:
            raise AttributeError(name)
        return getattr(Ctx._module, name)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class Sidebar(Ctx):
    """``st.sidebar`` - a DeltaGenerator that is also a context manager."""


class SessionState(dict):
    """Supports both ``st.session_state["k"]`` and ``st.session_state.k``."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


def build_stub(server_url: str, calls: list[str]) -> types.ModuleType:
    st = types.ModuleType("streamlit")
    st.session_state = SessionState()

    def record(name):
        calls.append(name)

    for name in ("form", "expander", "container", "empty", "popover"):
        setattr(st, name, (lambda n: lambda *a, **k: (record(n), Ctx())[1])(name))

    def columns(spec, **kwargs):
        record("columns")
        n = spec if isinstance(spec, int) else len(spec)
        return [Ctx() for _ in range(n)]

    def tabs(labels):
        record("tabs")
        return tuple(Ctx() for _ in labels)

    st.columns, st.tabs = columns, tabs

    def text_input(label, value="", **k):
        record("text_input")
        return server_url if str(value).startswith("http") else value

    def number_input(label, min_value=None, max_value=None, value=0, **k):
        record("number_input")
        return value

    def slider(label, min_value=None, max_value=None, value=None, **k):
        record("slider")
        return value if value is not None else min_value

    def selectbox(label, options, index=0, **k):
        record("selectbox")
        return list(options)[index] if len(options) else None

    def button(label, **k):
        record("button")
        return False

    def form_submit_button(label, **k):
        record("form_submit_button")
        return False

    st.form_submit_button = form_submit_button

    setattr(st, "file_uploader", (lambda n: lambda *a, **k: (record(n), None)[1])("file_uploader"))

    def checkbox(label, value=False, **k):
        record("checkbox")
        return value

    def toggle(label, value=False, **k):
        record("toggle")
        return value

    def progress(*a, **k):
        record("progress")
        return Ctx()

    def stop():
        raise Stop()

    def rerun():
        record("rerun")

    st.stop, st.rerun = stop, rerun
    st.text_input, st.number_input, st.slider = text_input, number_input, slider
    st.selectbox, st.button, st.checkbox, st.toggle = selectbox, button, checkbox, toggle

    for name in (
        "set_page_config", "title", "caption", "header", "subheader", "markdown", "write",
        "info", "success", "warning", "error", "code", "dataframe", "line_chart",
        "scatter_chart", "area_chart", "bar_chart", "metric", "divider", "image", "json",
    ):
        setattr(st, name, (lambda n: lambda *a, **k: (record(n), None)[1])(name))

    st.radio = lambda label, options, index=0, **k: (record("radio"), list(options)[index])[1]

    record("sidebar")
    Ctx._module = st          # columns/tabs/exps now forward widget calls
    st.sidebar = Sidebar()
    return st


def run_script(script: Path, cwd: Path, server_url: str) -> tuple[bool, str, list[str]]:
    """Execute a UI script top-to-bottom with the stub installed."""
    calls: list[str] = []
    sys.modules["streamlit"] = build_stub(server_url, calls)
    code = compile(script.read_text(encoding="utf-8"), str(script), "exec")
    namespace = {"__name__": "__main__", "__file__": str(script)}
    cwd_before = Path.cwd()
    try:
        os.chdir(cwd)
        exec(code, namespace)  # noqa: S102 - deliberately executing the UI script
        return True, "", calls
    except Stop:
        return False, "script called st.stop() (server unreachable)", calls
    except Exception as exc:  # noqa: BLE001
        import traceback
        tb = traceback.format_exc().strip().splitlines()[-3:]
        return False, f"{type(exc).__name__}: {exc} | {' / '.join(t.strip() for t in tb)}", calls
    finally:
        os.chdir(cwd_before)


def wait_health(url: str, timeout: float = 90.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(f"{url}/health", timeout=2).status_code == 200:
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.5)
    return False


def main() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"

    server = subprocess.Popen(
        [PY, "-m", "app.main"], cwd=ROOT / "backend",
        env={**os.environ, "ORIENT_HOST": "127.0.0.1", "ORIENT_PORT": str(port)},
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        if not wait_health(url):
            print("Could not start the server for the dashboard test.")
            return 1

        print("\n=== backend/ui/dashboard.py ===")
        ok, err, calls = run_script(ROOT / "backend" / "ui" / "dashboard.py", ROOT / "backend", url)
        print(f"  widgets used: {sorted(set(calls))[:12]}")
        check(ok, f"dashboard.py executes without error{' -> ' + err if err else ''}")
        check("title" in calls, "dashboard rendered a title")

        print("\n=== client/ui/app.py ===")
        ok, err, calls = run_script(ROOT / "client" / "ui" / "app.py", ROOT / "client", url)
        print(f"  widgets used: {sorted(set(calls))[:12]}")
        check(ok, f"client app.py executes without error{' -> ' + err if err else ''}")
        check("text_input" in calls, "client UI rendered its sidebar")

        print("\n" + "=" * 60)
        if failures:
            print(f"{len(failures)} FAILURE(S):")
            for item in failures:
                print(f"  - {item}")
            return 1
        print("BOTH UI SCRIPTS EXECUTE CLEANLY")
        return 0
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except Exception:  # noqa: BLE001
            server.kill()


if __name__ == "__main__":
    sys.exit(main())