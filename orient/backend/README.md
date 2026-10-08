# Orient · Server App (backend)

The federated **server**: orchestrates rounds, receives client weight updates,
aggregates them, evaluates the global model, and stores artifacts.
Implements `SRS_FederatedSciML_Project.md` v1.4 §4.11 (`FR-SERVERAPP-1..6`).

## Run

```powershell
# from this directory, with the venv activated
python -m app.main              # REST API on http://127.0.0.1:8000
streamlit run ui\dashboard.py   # optional dashboard
```

Or use the helper: `.\run.ps1` · `.\run.ps1 dashboard`.

Interactive API docs: <http://127.0.0.1:8000/docs>

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `ORIENT_HOST` | `0.0.0.0` | bind address |
| `ORIENT_PORT` | `8000` | port |
| `ORIENT_RESULTS_DIR` | `results` | where run artifacts are written |
| `ORIENT_SEED` | `0` | recorded in each run's `config.json` |
| `ORIENT_ROUND_TIMEOUT` | `0` (off) | seconds before a round aggregates whatever arrived |

## Round engine

`orchestrator.Federation` holds the client registry, the global model and the
run state behind a single lock:

1. `start_run` builds the global model from the problem's `ModelSpec`, snapshots
   the currently-registered clients as `expected`, sets phase `collecting`.
2. A client polls `assignment()`. If it is in `expected` and has not yet
   reported this round, it gets `train` plus the `ModelSpec`/`ProblemSpec`,
   `local_epochs`, the optimizer, and the FedProx `mu`.
3. On upload, the payload is deserialized and its **keys/shapes are checked
   against the global model** before being accepted.
4. When every expected client has reported, `aggregate()` runs, the global model
   is replaced, the problem's `evaluate()` produces a relative L2 error, a
   metrics record is logged, and the round advances. At the final round the
   model is saved and the phase becomes `complete`.

If a client disappears mid-run the round would stall, so the operator can press
**⚡ Force aggregate** (or set `ORIENT_ROUND_TIMEOUT`).

## Modules

| File | Responsibility |
|---|---|
| `app/api.py` | FastAPI routes (thin — no science logic) |
| `app/orchestrator.py` | `Federation`: registry, round engine, aggregation trigger |
| `app/aggregators.py` | 9 aggregation algorithms (`fedavg/fedprox/fedadam/fedadagrad/fedyogi/scaffold` + `median/trimmed_mean/krum`) on `state_dict` tensors |
| `app/weighting.py` | `uniform` / `data_size` / `quality` client weights |
| `app/models.py` | pure-PyTorch model factory (shared with the client) |
| `app/problems.py` | problem registry: losses, residuals, analytic ground truth |
| `app/protocol.py` | Pydantic wire schemas (shared with the client) |
| `app/weights.py` | safetensors serialization — no pickle on the wire |
| `app/storage.py` | `results/<problem>/<aggregator>/<run_id>/` artifacts |
| `app/local_clients.py` | spawns/stops local demo client processes (dev convenience) |
| `app/config.py` | env-overridable settings |
| `ui/dashboard.py` | Streamlit control panel (thin layer over the API) |

## Artifacts (SRS §8.2 + FR-STORE)

```
results/<problem>/<aggregator>/<run_id>/
├── config.yaml              exact YAML snapshot (FR-CFG-6 reproducibility)
├── config.json              run configuration + parameter count + seed
├── metrics.jsonl            one JSON record per completed round
├── metrics.json             summary (best/final L2, histories)
├── final_model.safetensors  final global weights (wire)
├── server_final.pth         torch.save final (SRS)
├── server_best.pth          torch.save best (lowest L2)
├── loss.npz                 mean_local_loss curve
├── l2_error.npz             l2_relative_error curve
└── weight_divergence.npz    per-layer divergence (FR-EVAL-3)
```

**Runner:** `python -m app.runner --config configs/poisson_fedavg.yaml` (single) or `--validation-gate poisson` (gate) or sweep YAML → `sweep_summary_*.json` + comparison table.

**Metrics:** `W1` heterogeneity via `app/metrics.py` (`ot.emd2`/`scipy`), per-layer weight divergence, `compromised_clients` + `heterogeneity` logged per round.