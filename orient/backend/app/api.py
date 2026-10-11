"""REST API for the Server app.

Endpoints (SRS v1.4 FR-SERVERAPP-4):
  GET  /health
  GET  /problems                      list available problems
  GET  /aggregators                   list available aggregators
  POST /clients/register              register a client
  GET  /clients                       list registered clients
  GET  /clients/{id}/assignment       poll for work (train | wait | done)
  GET  /clients/{id}/weights?round=R  download global weights (safetensors bytes)
  POST /clients/{id}/update?round=R   upload weights (safetensors bytes)
  GET  /run/status                    current run status + metrics
  POST /admin/run/start               start a federated run
  POST /admin/run/stop                stop the run
  POST /admin/run/force               aggregate with the updates received so far
  POST /admin/datasets/generate       generate bundles from UI (no CLI)
  GET  /admin/datasets/status         list bundles
  POST /admin/run/sweep               run YAML sweep from UI
  POST /admin/run/validation-gate     validation gate via UI
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import Response
from pydantic import BaseModel

from .aggregators import ALL_AGGREGATORS
from .config import settings
from .local_clients import local_clients
from .orchestrator import Federation
from .problems import get_problem, list_problems
from .protocol import (
    AssignmentResponse,
    ClientInfo,
    ClientRegistration,
    RunStartRequest,
    RunStatus,
    SpawnClientsRequest,
)
from .weighting import MODES as WEIGHTING_MODES

federation = Federation()

app = FastAPI(
    title="Orient Server",
    version="0.1.0",
    description="Federated SciML server: orchestrates rounds and aggregates client weights.",
)


# ---- UI-only helpers (no CLI) ----
class DatasetGenerateRequest(BaseModel):
    clients: int = 5
    samples: int = 1500
    problems: Optional[List[str]] = None  # None = all 10
    overwrite: bool = True


class SweepRequest(BaseModel):
    yaml_text: str
    results_dir: str = "results"
    use_http: bool = False  # if True drive live server, else in-memory


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "service": "orient-server", "phase": federation.status().phase}


@app.get("/problems")
def problems() -> Dict[str, List[str]]:
    return {"problems": list_problems()}


@app.get("/aggregators")
def aggregators() -> Dict[str, Any]:
    return {"aggregators": ALL_AGGREGATORS, "weighting_modes": WEIGHTING_MODES}


@app.post("/clients/register", response_model=ClientInfo)
def register(reg: ClientRegistration) -> ClientInfo:
    try:
        get_problem(reg.problem)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        return federation.register(reg)
    except ValueError as exc:
        # e.g. the client registered for a different problem than the active run
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/clients", response_model=List[ClientInfo])
def clients() -> List[ClientInfo]:
    return federation.list_clients()


@app.get("/clients/{client_id}/assignment", response_model=AssignmentResponse)
def assignment(client_id: str) -> AssignmentResponse:
    resp = federation.assignment(client_id)
    if resp.status == "error":
        raise HTTPException(status_code=404, detail=resp.message)
    return resp


@app.get("/clients/{client_id}/weights")
def weights(client_id: str, round_no: int = Query(..., alias="round")) -> Response:
    if not federation.is_registered(client_id):
        raise HTTPException(status_code=404, detail="Client not registered")
    try:
        data = federation.get_global_bytes(round_no)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return Response(
        content=data,
        media_type="application/octet-stream",
        headers={"X-Weights-Round": str(round_no)},
    )


@app.post("/clients/{client_id}/update", response_model=AssignmentResponse)
async def update(
    request: Request,
    client_id: str,
    round_no: int = Query(..., alias="round"),
    n_samples: int = Query(0),
    local_loss: float = Query(0.0),
) -> AssignmentResponse:
    payload = await request.body()
    if not payload:
        raise HTTPException(status_code=400, detail="Empty weight payload")
    resp = federation.submit_update(client_id, round_no, n_samples, local_loss, payload)
    if resp.status == "error":
        raise HTTPException(status_code=400, detail=resp.message)
    return resp


@app.get("/run/status", response_model=RunStatus)
def run_status() -> RunStatus:
    return federation.status()


@app.post("/admin/run/start", response_model=RunStatus)
def run_start(req: RunStartRequest) -> RunStatus:
    try:
        return federation.start_run(req)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/admin/run/stop", response_model=RunStatus)
def run_stop() -> RunStatus:
    return federation.stop_run()


@app.post("/admin/run/force", response_model=RunStatus)
def run_force() -> RunStatus:
    return federation.force_aggregate()


# --------------------------------------------------------------- local demo clients
# Development convenience: start/stop client processes on this machine so a whole
# run can be driven from the dashboard. Real deployments start their own clients.
@app.get("/admin/clients/local")
def local_client_status() -> Dict[str, Any]:
    return {
        "running": local_clients.list(),
        "problems": local_clients.available_problems(),
        "public_url": settings.public_url,
    }


@app.post("/admin/clients/local/spawn")
def local_client_spawn(req: SpawnClientsRequest) -> Dict[str, Any]:
    if req.problem not in local_clients.available_problems():
        raise HTTPException(
            status_code=400,
            detail=(
                f"No bundles for problem '{req.problem}'. "
                "Generate them first: python examples/make_datasets.py --clients 5"
            ),
        )
    return local_clients.spawn(req.count, req.problem, settings.public_url)


@app.post("/admin/clients/local/stop")
def local_client_stop() -> Dict[str, Any]:
    return {"stopped": local_clients.stop_all(), "running": local_clients.list()}


# ---- Dataset generation via UI (no CLI) ----
@app.get("/admin/datasets/status")
def dataset_status() -> Dict[str, Any]:
    from .local_clients import BUNDLES_DIR
    if not BUNDLES_DIR.exists():
        return {"bundle_count": 0, "bundles": [], "problems": [], "out_dir": str(BUNDLES_DIR)}
    bundles = sorted(p.name for p in BUNDLES_DIR.iterdir() if p.is_dir())
    probs = sorted({b.rsplit("_client", 1)[0] for b in bundles if "_client" in b})
    return {"bundle_count": len(bundles), "bundles": bundles, "problems": probs, "out_dir": str(BUNDLES_DIR)}


@app.post("/admin/datasets/generate")
def dataset_generate(req: DatasetGenerateRequest) -> Dict[str, Any]:
    clients = max(1, min(int(req.clients), 16))
    samples = max(100, min(int(req.samples), 10000))
    problems = req.problems  # None or list
    # Validate problems if given
    all_probs = set(list_problems())
    if problems:
        bad = [p for p in problems if p not in all_probs]
        if bad:
            raise HTTPException(status_code=400, detail=f"Unknown problems: {bad}. Options: {sorted(all_probs)}")
    try:
        from pathlib import Path as _P
        import shutil
        backend_dir = _P(__file__).resolve().parents[1]
        client_dir = backend_dir.parent / "client"
        out = client_dir / "examples" / "bundles"
        out.mkdir(parents=True, exist_ok=True)
        # Clean previous bundles to match requested client count exactly:
        # remove all existing problem_client* dirs before regenerating
        # If selective problems, only clean those problems
        to_clean = problems or sorted(all_probs)
        for prob in to_clean:
            for p in list(out.glob(f"{prob}_client*")):
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
        # Run as subprocess to avoid import collision (backend `app` vs client `app`)
        cmd = [sys.executable, str(client_dir / "examples" / "make_datasets.py"), "--clients", str(clients), "--samples", str(samples), "--out", str(out)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60, cwd=str(client_dir))
        if proc.returncode != 0:
            raise RuntimeError(f"make_datasets failed: {proc.stderr[:2000]} {proc.stdout[:1000]}")
        bundles = sorted(p.name for p in out.iterdir() if p.is_dir())
        if problems:
            wanted = set(problems)
            bundles = [b for b in bundles if b.rsplit("_client", 1)[0] in wanted]
        return {"ok": True, "clients": clients, "samples": samples, "problems": problems or sorted(all_probs), "bundle_count": len(bundles), "bundles": bundles, "out_dir": str(out), "stdout": proc.stdout[-600:]}
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        import traceback
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}\n{traceback.format_exc()[-1000:]}") from exc


@app.post("/admin/run/sweep")
def run_sweep(req: SweepRequest) -> Dict[str, Any]:
    # Run via subprocess to avoid import collision (runner imports client trainer)
    import tempfile
    import yaml as _yaml
    try:
        data = _yaml.safe_load(req.yaml_text)
        if not isinstance(data, dict):
            raise ValueError("YAML must be a mapping")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {exc}") from exc
    tmp = Path(tempfile.mktemp(suffix=".yaml", dir="/tmp"))
    tmp.write_text(req.yaml_text, encoding="utf-8")
    try:
        # Guard: estimate sweep size via expand_sweeps without running
        try:
            from .experiment_config import expand_sweeps, validate_config
            if "sweep" in data or any(isinstance(v, list) for v in data.values()):
                configs = expand_sweeps(data)
            else:
                configs = [validate_config(data)]
            if len(configs) > 32:
                raise HTTPException(status_code=400, detail=f"Sweep too large ({len(configs)} configs). Max 32 via UI; use CLI for larger.")
        except HTTPException:
            raise
        except Exception:
            pass  # let runner handle validation
        results_dir = req.results_dir or "results"
        # Use subprocess: python -m app.runner --config tmp --results-dir results_dir [--server url]
        cmd = [sys.executable, "-m", "app.runner", "--config", str(tmp), "--results-dir", results_dir]
        if req.use_http:
            from .config import settings as _settings
            cmd += ["--server", f"http://{_settings.host}:{_settings.port}"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=str(Path(__file__).resolve().parents[1]))
        if proc.returncode != 0:
            raise RuntimeError(f"Sweep failed (exit {proc.returncode}): {proc.stderr[-2000:]} {proc.stdout[-2000:]}")
        # Try to find summary json
        summary_path = Path(results_dir) / f"sweep_summary_{tmp.stem}.json"
        summary = []
        if summary_path.exists():
            try:
                import json as _json
                summary = _json.loads(summary_path.read_text())[:10]
            except: pass
        return {"ok": True, "exit_code": proc.returncode, "results_dir": results_dir, "stdout": proc.stdout[-3000:], "stderr": proc.stderr[-1000:], "summary_file": str(summary_path)}
    except HTTPException:
        raise
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Sweep timed out after 300s. Use CLI for larger sweeps.")
    except Exception as exc:  # noqa: BLE001
        import traceback
        raise HTTPException(status_code=500, detail=f"Sweep failed: {exc}\n{traceback.format_exc()[-1500:]}") from exc
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


@app.post("/admin/run/validation-gate")
def validation_gate_api(problem: str = "poisson") -> Dict[str, Any]:
    try:
        cmd = [sys.executable, "-m", "app.runner", "--validation-gate", problem]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=str(Path(__file__).resolve().parents[1]))
        out = (proc.stdout + "\n" + proc.stderr)[-4000:]
        return {"ok": proc.returncode == 0, "exit_code": proc.returncode, "problem": problem, "output": out}
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Validation gate timed out")
    except Exception as exc:  # noqa: BLE001
        import traceback
        raise HTTPException(status_code=500, detail=f"Gate failed: {exc}\n{traceback.format_exc()[-1000:]}") from exc
