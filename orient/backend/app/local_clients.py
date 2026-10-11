"""Spawn and track local demo client processes.

This is a *development convenience*: it lets the dashboard start N local client
processes so a whole federated run can be driven from the browser. In a real
deployment clients live on other machines and start themselves — nothing here
is required for the protocol to work.

Clients are ordinary CLI clients (``client/app/main.py``); they still speak the
same HTTP protocol and only ever upload weights.
"""

from __future__ import annotations

import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

BACKEND_DIR = Path(__file__).resolve().parents[1]     # .../orient/backend
CLIENT_DIR = BACKEND_DIR.parent / "client"             # .../orient/client
BUNDLES_DIR = CLIENT_DIR / "examples" / "bundles"
LOG_DIR = CLIENT_DIR / "logs"


@dataclass
class SpawnedClient:
    client_id: str
    problem: str
    pid: int
    bundle: str
    proc: subprocess.Popen = field(repr=False, default=None)  # type: ignore[assignment]

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None


class LocalClients:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._clients: Dict[str, SpawnedClient] = {}

    # ------------------------------------------------------------------ helpers
    def available_problems(self) -> List[str]:
        if not BUNDLES_DIR.exists():
            return []
        names = set()
        for path in BUNDLES_DIR.iterdir():
            if path.is_dir() and "_client" in path.name:
                names.add(path.name.rsplit("_client", 1)[0])
        return sorted(names)

    def _reap(self) -> None:
        """Forget processes that have already exited."""
        for cid in [c for c, s in self._clients.items() if not s.alive()]:
            self._clients.pop(cid, None)

    def list(self) -> List[Dict[str, Any]]:
        with self._lock:
            self._reap()
            return [
                {"client_id": s.client_id, "problem": s.problem, "pid": s.pid,
                 "bundle": s.bundle, "alive": s.alive()}
                for s in self._clients.values()
            ]

    # ------------------------------------------------------------------- actions
    def _is_safe_server_url(self, url: str) -> bool:
        # SSRF guard: only allow http(s) to loopback or localhost when spawned locally
        # Settings.public_url is trusted (env), but if caller supplies arbitrary URL
        # we restrict to 127.0.0.1 / localhost to avoid probing internal metadata services
        try:
            from urllib.parse import urlparse
            p = urlparse(url)
            if p.scheme not in ("http", "https"):
                return False
            host = (p.hostname or "").lower()
            # allow localhost and 127.0.0.1 and ::1
            if host in ("127.0.0.1", "localhost", "::1"):
                return True
            # also allow if it exactly matches configured public_url host
            from .config import settings as _s
            from urllib.parse import urlparse as _up
            try:
                cfg_host = _up(_s.public_url).hostname or ""
                if host == cfg_host.lower():
                    return True
            except Exception:
                pass
            return False
        except Exception:
            return False

    def spawn(self, count: int, problem: str, server_url: str) -> Dict[str, Any]:
        count = max(1, min(int(count), 32))
        started: List[str] = []
        errors: List[str] = []
        # SSRF / injection guard
        if not self._is_safe_server_url(server_url):
            return {"started": [], "errors": [f"Blocked unsafe server_url '{server_url}' (only localhost/127.0.0.1 allowed)"], "running": self.list()}

        with self._lock:
            self._reap()
            # Global spawn cap: do not exceed 32 total concurrent demo clients
            if len(self._clients) + count > 32:
                return {"started": [], "errors": [f"Global demo limit 32 exceeded ({len(self._clients)} running + {count} requested)"], "running": self.list()}
            LOG_DIR.mkdir(parents=True, exist_ok=True, mode=0o750)

            for index in range(1, count + 1):
                client_id = f"client-{index}"
                bundle = BUNDLES_DIR / f"{problem}_client{index}"

                # Re-use an already-running slot instead of starting a duplicate.
                existing = self._clients.get(client_id)
                if existing is not None and existing.alive():
                    existing.proc.terminate()
                    self._clients.pop(client_id, None)

                if not bundle.exists():
                    errors.append(
                        f"{client_id}: missing bundle '{bundle.name}' "
                        f"(run: python examples/make_datasets.py --clients {count})"
                    )
                    continue

                log_path = LOG_DIR / f"ui_{client_id}.log"
                handle = log_path.open("w", encoding="utf-8")
                try:
                    proc = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
                        [sys.executable, "-m", "app.main",
                         "--server", server_url,
                         "--client-id", client_id,
                         "--dataset", str(bundle),
                         "--poll-interval", "0.5"],
                        cwd=str(CLIENT_DIR), stdout=handle, stderr=subprocess.STDOUT,
                    )
                except OSError as exc:
                    handle.close()
                    errors.append(f"{client_id}: could not start ({exc})")
                    continue

                self._clients[client_id] = SpawnedClient(
                    client_id=client_id, problem=problem, pid=proc.pid,
                    bundle=str(bundle), proc=proc,
                )
                started.append(client_id)

        return {"started": started, "errors": errors, "running": self.list()}

    def stop_all(self) -> int:
        with self._lock:
            count = 0
            for client_id, spawned in list(self._clients.items()):
                try:
                    if spawned.alive():
                        spawned.proc.terminate()
                        count += 1
                except Exception:  # noqa: BLE001
                    pass
                self._clients.pop(client_id, None)
            return count


local_clients = LocalClients()