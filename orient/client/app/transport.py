"""HTTP transport to the Orient server (weights travel as safetensors bytes)."""

from __future__ import annotations

from typing import Any, Dict, List

import httpx

from .protocol import (
    AssignmentResponse,
    ClientInfo,
    ClientRegistration,
    RunStatus,
)


class ServerConnectionError(RuntimeError):
    """Raised when the server cannot be reached."""


class ServerClient:
    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(base_url=self.base_url, timeout=timeout)

    # -- lifecycle -------------------------------------------------------------
    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "ServerClient":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        try:
            return self._http.request(method, url, **kwargs)
        except httpx.HTTPError as exc:  # network-level failure
            raise ServerConnectionError(
                f"Cannot reach server at {self.base_url}: {exc}"
            ) from exc

    # -- endpoints -------------------------------------------------------------
    def health(self) -> Dict[str, Any]:
        resp = self._request("GET", "/health")
        resp.raise_for_status()
        return resp.json()

    def register(self, registration: ClientRegistration) -> ClientInfo:
        resp = self._request("POST", "/clients/register", json=registration.model_dump())
        if resp.status_code >= 400:
            raise RuntimeError(f"Registration failed ({resp.status_code}): {resp.text}")
        return ClientInfo(**resp.json())

    def assignment(self, client_id: str) -> AssignmentResponse:
        resp = self._request("GET", f"/clients/{client_id}/assignment")
        if resp.status_code == 404:
            return AssignmentResponse(status="error", message=resp.text)
        resp.raise_for_status()
        return AssignmentResponse(**resp.json())

    def download_weights(self, client_id: str, round_no: int) -> bytes:
        resp = self._request(
            "GET", f"/clients/{client_id}/weights", params={"round": round_no}
        )
        resp.raise_for_status()
        return resp.content

    def upload_weights(
        self,
        client_id: str,
        round_no: int,
        n_samples: int,
        local_loss: float,
        data: bytes,
    ) -> AssignmentResponse:
        resp = self._request(
            "POST",
            f"/clients/{client_id}/update",
            params={"round": round_no, "n_samples": n_samples, "local_loss": local_loss},
            content=data,
            headers={"Content-Type": "application/octet-stream"},
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"Upload failed ({resp.status_code}): {resp.text}")
        return AssignmentResponse(**resp.json())

    def run_status(self) -> RunStatus:
        resp = self._request("GET", "/run/status")
        resp.raise_for_status()
        return RunStatus(**resp.json())

    def problems(self) -> List[str]:
        resp = self._request("GET", "/problems")
        resp.raise_for_status()
        return list(resp.json().get("problems", []))
