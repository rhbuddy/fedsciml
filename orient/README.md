# Orient — Two-App Federated SciML

A **Server app** and a **Client app** for federated scientific machine learning,
built to `SRS_FederatedSciML_Project.md` **v1.4** (§4.10 – §4.13).

> Clients train on their **own private dataset** and upload **weights only**.
> Raw data never leaves the client machine.

---

## Architecture

```
 ┌──────────────────────── SERVER APP  (backend/) ────────────────────────┐
 │  FastAPI REST API · ClientRegistry · Federation round engine           │
 │  ui/dashboard.py  →  Streamlit control panel                           │
 │                                                                        │
 │  broadcast global weights  ──►  collect updates  ──►  aggregate  ──► next│
 └───────────▲──────────────────────────────────────────▲──────────────────┘
             │ GET  /clients/{id}/assignment             │
             │ GET  /clients/{id}/weights  (safetensors) │ POST /clients/{id}/update
 ┌───────────┴──────────────── CLIENT APP  (client/) ───┴──────────────────┐
 │  dataset bundle (private) · LocalTrainer · Streamlit UI / CLI           │
 │  shares: protocol.py · weights.py · models.py · problems.py (verbatim)  │
 └────────────────────────────────────────────────────────────────────────┘
```

Round loop: server publishes a round → each client polls its assignment,
downloads the global weights, trains locally, uploads its update → the server
aggregates once every expected client has reported and advances.

## Layout

```
orient/
├── backend/                     # ── Server app ────────────────────────
│   ├── app/
│   │   ├── api.py               #   FastAPI routes
│   │   ├── orchestrator.py      #   Federation state + round engine
│   │   ├── aggregators.py       #   fedavg fedprox fedadam fedagrad fedyogi
│   │   │                        #   median trimmed_mean krum
│   │   ├── weighting.py         #   uniform | data_size | quality
│   │   ├── models.py            #   pure-PyTorch model factory (MLP, DeepONet)
│   │   ├── problems.py          #   problem registry + losses + ground truth
│   │   ├── protocol.py          #   Pydantic wire schemas
│   │   ├── weights.py           #   safetensors serialization (no pickle)
│   │   ├── storage.py           #   results/<problem>/<aggregator>/<run_id>/
│   │   ├── config.py            #   env-overridable settings
│   │   └── main.py              #   uvicorn entry point
│   ├── ui/dashboard.py          #   Streamlit dashboard
│   ├── requirements.txt
│   └── run.ps1
├── client/                      # ── Client app ────────────────────────
│   ├── app/
│   │   ├── main.py              #   CLI entry point
│   │   ├── dataset.py           #   load/validate/save dataset bundles
│   │   ├── trainer.py           #   local training + optional FedProx term
│   │   ├── transport.py         #   HTTP client
│   │   ├── protocol.py  weights.py  models.py  problems.py   (shared copies)
│   │   └── ...
│   ├── ui/app.py                #   Streamlit client UI
│   ├── examples/make_datasets.py#   heterogeneous example bundles
│   ├── requirements.txt
│   └── run.ps1
├── tests/test_static_imports.py # stdlib-only import-graph check
├── federated-sciml-main/        # untouched copy of the upstream reference code
└── .gitignore
```

## Shared modules (must stay byte-identical)

`protocol.py`, `weights.py`, `models.py`, `problems.py` are **verbatim copies**
in both apps. They guarantee that the client rebuilds the exact same architecture
(the same `state_dict` keys/shapes) and computes the same loss as the server —
that is the compatibility contract (FR‑PROTO‑2).

Verify with:

```powershell
foreach ($m in 'protocol.py','weights.py','models.py','problems.py') {
  if ((Get-FileHash "backend\app\$m").Hash -eq (Get-FileHash "client\app\$m").Hash) { "$m OK" }
}
```

---

## Quickstart

### ⭐ Start everything from the UI (recommended)

One command brings up the server **and** both web UIs, then opens the dashboard
in your browser:

```powershell
cd D:\fedsciml\orient
.\start.ps1 -WithClientUi
```

```
  [ok] Server API      http://127.0.0.1:8000
  [ok] Dashboard       http://localhost:8501
  [ok] Client UI       http://localhost:8502
```

Then drive it from the browser:

1. **Dashboard → <http://localhost:8501>**
   *Run control* tab → choose problem, aggregator, rounds, local epochs →
   press **▶ Start run**.
2. **🧩 Local demo clients** (same tab, below the buttons)
   Opens N client processes *on this machine* that connect and train
   automatically — this is the quickest way to get a federated run going.
   Set the client count to match the bundles you generated.
3. **Metrics** tab → per-round L2 error; **Clients** tab → who has reported.

Prefer the client web UI instead? Open **<http://localhost:8502>** in **one
browser tab per client**. In each tab pick a *different* Client ID →
**Load dataset** → **Connect & register** → **▶ Start federated training**.

Stop everything with:

```powershell
.\start.ps1 -Stop
```

> Prefer the terminal? `client\launch_clients.ps1 -Count 5` starts N CLI clients
> instead of the client web UI.

### Two things that make a run look "broken" when it isn't

| Symptom | Cause | Fix |
|---|---|---|
| L2 error gets **worse**, local loss tiny | Too few clients — part of the domain has no owner, so the model overfits its slice | Start **all** clients that bundles were generated for |
| L2 error stays **flat** | Too few rounds; each round resets to the global model | Use **30–60 rounds** with **10+ local epochs** |

Sanity check: `curl.exe http://127.0.0.1:8000/clients` should list one entry per
client you started, and every metrics record should show that same `n_clients`.

<details>
<summary>Manual / command-line alternative</summary>

The steps below (environment → datasets → server → run → clients) reproduce the
same flow without the browser.
</details>

### 0. Environment

The checked-in-ready environment is **Python 3.14 + torch 2.14.1**, which is what
this project was validated on:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
pip install -r client\requirements.txt
```

> The upstream `federated-sciml-main/requirements.txt` pins `torch==2.6.0`,
> which does **not** support Python 3.14. Use **Python 3.11/3.12** if you need
> those exact pins; otherwise the unpinned ranges above work on 3.14.

### 0b. Verify the build (optional but recommended)

```powershell
.\.venv\Scripts\python.exe tests\test_static_imports.py   # import graph
.\.venv\Scripts\python.exe tests\test_runtime.py          # aggregators, models, problems
.\.venv\Scripts\python.exe tests\test_e2e.py              # real server + 2 real clients
```

### 1. Generate example datasets (creates *heterogeneous* local datasets)

```powershell
cd orient\client
python examples\make_datasets.py
```

This writes two clients per problem, each seeing a **different slice**
(e.g. `poisson_client1` only sees `x ∈ [0, π/2]`, `poisson_client2` only sees
`x ∈ [π/2, π]`).

Use `--clients N` for a larger federation (needed for Krum, which requires
K ≥ 2f + 3):

```powershell
python examples\make_datasets.py --clients 5
```

### 2. Start the Server app

```powershell
cd orient\backend
python -m app.main            # or: .\run.ps1
```

- REST API: <http://127.0.0.1:8000>
- Interactive API docs: <http://127.0.0.1:8000/docs>

Optional — the dashboard in a second terminal:

```powershell
cd orient\backend
streamlit run ui\dashboard.py   # or: .\run.ps1 dashboard
```

### 3. Start a run

Either press **▶ Start run** in the dashboard, or:

```powershell
curl.exe -X POST http://127.0.0.1:8000/admin/run/start -H "Content-Type: application/json" `
  -d '{\"problem\":\"poisson\",\"aggregator\":\"fedavg\",\"total_rounds\":10,\"local_epochs\":5,\"learning_rate\":0.001}'
```

### 4. Start the Client apps

Each client needs its own terminal (they can be on different machines — just
point `--server` at the host):

```powershell
cd orient\client
python -m app.main --server http://127.0.0.1:8000 --client-id client-1 --dataset examples\bundles\poisson_client1
python -m app.main --server http://127.0.0.1:8000 --client-id client-2 --dataset examples\bundles\poisson_client2
```

Or the Streamlit client UI (load a bundle, click *Connect & register*, then
*▶ Start federated training*):

```powershell
streamlit run ui\app.py         # or: .\run.ps1 ui
```

### 5. Watch it

The dashboard shows the round counter, which clients have reported, and the
global model's relative L2 error per round. Artifacts land in
`backend/results/<problem>/<aggregator>/<run_id>/`:
`config.json`, `metrics.jsonl`, `final_model.safetensors`.

---

## Problems

| Name | Family | Model | Data | Ground truth |
|---|---|---|---|---|
| `gramacy_lee` | supervised | MLP `[1,32,32,32,1]` | `x_train, y_train` | `sin(10πx)/(2x) + (x−1)⁴`, x∈[0.5,2.5] |
| `schaffer` | supervised | MLP `[2,48,48,48,1]` | `x_train, y_train` | Schaffer N.2 on [−2,2]² |
| `poisson` | **pinn** | MLP `[1,20,20,20,1]` + hard-constraint transform | `x_train` (collocation) | `u(x)=x+(1/8)sin8x+Σ(1/i)sin(ix)` on [0,π] |
| `antiderivative` | **operator** | DeepONet (50 sensors, [64,64] branch/trunk) | `branch_train, trunk_train, y_train` | truncated Fourier series |

`poisson` reproduces the paper's setup: `u(0)=0` and `u(π)=π` are enforced
*exactly* by the output transform `u = x + tanh(x)·tanh(π−x)·ŷ`, so the loss is
the pure PDE residual `−u″ − f`.

## Aggregators

| Group | Algorithms |
|---|---|
| Standard | `fedavg`, `fedprox`, `fedadam`, `fedagrad`, `fedyogi` |
| Byzantine-robust | `median`, `trimmed_mean`, `krum` (multi‑Krum) |

Weighting: `uniform` · `data_size` (the paper's default) · `quality`.
All operate purely on `state_dict` tensors, so they are architecture-agnostic.

> **Client-count requirements.** `median` and `trimmed_mean` work with any K ≥ 2.
> **Krum needs K ≥ 2f + 3** (so 5 clients for f = 1) — with fewer clients the two
> candidates are always mutually nearest and Krum degenerates into "pick one".
> The server logs a warning when this is violated.
> Generate enough clients with `python examples/make_datasets.py --clients 5`.
>
> **Tuning.** The server-adaptive methods (`fedadam`/`fedagrad`/`fedyogi`) take a
> step of roughly `server_lr` per round, so they need **more rounds** and a larger
> `server_lr` than plain averaging to make progress; they also oscillate more.

## REST API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | liveness + current phase |
| `GET` | `/problems` | available problems |
| `GET` | `/aggregators` | available aggregators + weighting modes |
| `POST` | `/clients/register` | register a client (`client_id`, `problem`, `n_samples`) |
| `GET` | `/clients` | list registered clients |
| `GET` | `/clients/{id}/assignment` | poll → `train` \| `wait` \| `done` \| `error` |
| `GET` | `/clients/{id}/weights?round=R` | download global weights (safetensors bytes) |
| `POST` | `/clients/{id}/update?round=R` | upload weights + `n_samples`, `local_loss` |
| `GET` | `/run/status` | phase, round, expected/submitted clients, metrics |
| `POST` | `/admin/run/start` | start a run |
| `POST` | `/admin/run/stop` | stop the run |
| `POST` | `/admin/run/force` | aggregate with the updates received so far |
| `GET` | `/admin/clients/local` | locally spawned clients + available bundle problems |
| `POST` | `/admin/clients/local/spawn` | start N local demo clients (`count`, `problem`) |
| `POST` | `/admin/clients/local/stop` | stop the locally spawned clients |

## Number of clients

The federation is **not limited to a fixed number of clients**. Every count is
derived at runtime (`len(states)`, `len(n_samples)`, the `expected` list), so K
is set purely by how many clients register. Verified end-to-end with **2, 5 and
8** clients.

```powershell
python examples/make_datasets.py --clients 8   # generate N clients per problem
python -m app.main --client-id client7 --dataset examples\bundles\gramacy_lee_client7
```

Clients are dynamic at runtime too: `expected` is re-snapshotted every round, so
a client that joins mid-run is picked up from the next round.

Practical notes when scaling up:
- Rounds are **synchronous** — a round completes when every expected client has
  reported, so its duration is bounded by the slowest client. Use
  `POST /admin/run/force` or `ORIENT_ROUND_TIMEOUT` if a client disappears.
- Krum scores all pairs of updates, so its cost is O(K²) — negligible for
  hundreds of clients.
- Per-round CPU scales with K: each client is a separate process.

---

## Client dataset bundle (§4.13)

```
mydata/
├── dataset.json     {"problem": "poisson", "n_samples": 1500, "domain": {"interval": [0, 1.57]}}
└── data.npz         x_train            (supervised & pinn)
                     branch_train, trunk_train, y_train   (operator)
```

Required arrays are validated against the problem registry, so an incompatible
bundle is rejected with a clear error before training starts.

## References

- Zhang, Liu, Weng & Lu — *Federated scientific machine learning for approximating
  functions and solving differential equations with data heterogeneity*, IEEE
  TNNLS 2025 (arXiv:2410.13141). The Poisson / Gramacy-Lee / Schaffer /
  Antiderivative setups follow this paper.
- **Flower** — round-based FL architecture and strategy set (FedAvg, FedProx,
  FedAdam/FedAdagrad/FedYogi, FedMedian, FedTrimmedAvg, Krum/MultiKrum).
  <https://flower.ai/docs/framework/explanation-flower-architecture.html>
- **Reddi et al.** — *Adaptive Federated Optimization* (arXiv:2003.00295), the
  server-side adaptive update used by FedAdam/FedAdagrad/FedYogi.
- **safetensors** — `save()`/`load()` byte API used for the weight wire format
  (no pickle). <https://huggingface.co/docs/safetensors/main/en/api/torch>

## Current status

**Validated on Python 3.14 + torch 2.14.1:**

| Test | Covers |
|---|---|
| `tests/test_static_imports.py` | 63 imports resolve; shared modules byte-identical |
| `tests/test_runtime.py` | 33 checks — safetensors round-trip, all 8 aggregators (incl. median/Krum rejecting Byzantine updates), 3 weighting modes, all 4 problems, Poisson hard constraints exact (`u(0)=0`, `u(π)=π`) |
| `tests/test_server_logic.py` | run-config validation, registration guards, seed reproducibility, round engine, weighting robustness |
| `tests/test_client.py` | FedProx proximal term is differentiable, dataset bundle validation, all shipped bundles load |
| `tests/test_e2e.py` | real uvicorn server + real client processes |
| `tests/run_all_aggregators.py` | the E2E test for **all 8 aggregators** (per-aggregator budgets; Krum with 5 clients) — **all 8 pass** |

Observed convergence: `gramacy_lee`, FedAvg, 60 rounds × 10 epochs → L2
**0.6225 → 0.2109**; `poisson` PINN → residual **23.7 → 2.2**.

### Bugs found and fixed during review

- **FedProx silently degenerated to FedAvg** — the proximal term was `.detach()`-ed,
  so it contributed no gradient. Now it stays in the autograd graph.
- **Seed was recorded but never applied**, so runs were not reproducible
  (`torch.manual_seed`/`np.random.seed` are now set before model creation).
- **A client registering for a different problem was added to `expected`**, which
  stalled the round for everyone; registration now returns `409`.
- **A bad aggregator/weighting/round count was only detected mid-run**, leaving the
  run stuck in `collecting`; now validated at `start_run`.
- **`/clients/{id}/weights` did not check registration** — any id could download
  the global model.
- **`quality` weighting exploded** (`1/loss` → ~1e12) when a local loss was ~0.
- **Dataset bundles were not shape-validated**, despite FR-CLIENTAPP-8.
- **Missing `None` guards** on `model_spec` in the client UI.

Not yet done (next steps):
- **SCAFFOLD** is not implemented — it needs persistent per-client control
  variates, which does not fit the stateless poll-based client yet.
- **Byzantine / noisy-client injection** (`noise_sim.py`) is not wired in; the
  robust aggregators are implemented and selectable, but nothing corrupts an
  update yet.
- **Wasserstein (W1) heterogeneity metric** is not computed yet.
- Models here are pure-PyTorch (`models.py`); DeepXDE-based PINN/DeepONet models
  from `federated-sciml-main/` are not yet wrapped by an adapter.
- The Streamlit UIs were written but not yet launched interactively (they need
  `streamlit`, which is listed in both `requirements.txt` files).