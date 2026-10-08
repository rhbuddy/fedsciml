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
"""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import Response

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
