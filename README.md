# Federated SciML — *Orient*

Two-app framework for **federated scientific machine learning**: clients train on
their own private data and upload **weights only**. Raw data never leaves the
client machine.

Built to `SRS_FederatedSciML_Project.md` **v1.4**.

> **Everything lives in [`orient/`](orient/).** This file is the front door;
> [`orient/README.md`](orient/README.md) has the full architecture, API table,
> problem suite, and troubleshooting.

---

## What it does

Solves differential equations and approximates functions across a federation of
heterogeneous clients — supervised regression, PINNs, and neural operators —
without ever pooling the underlying data.

```
 ┌──────────────────── SERVER APP  (orient/backend/) ────────────────────┐
 │  FastAPI REST API · ClientRegistry · Federation round engine          │
 │  ui/dashboard.py  →  Streamlit control panel                          │
 │                                                                       │
 │  broadcast global weights ─► collect updates ─► aggregate ─► next round│
 └───────────▲───────────────────────────────────────────▲───────────────┘
             │ GET  /clients/{id}/assignment              │
             │ GET  /clients/{id}/weights  (safetensors)  │ POST /clients/{id}/update
 ┌───────────┴──────────────── CLIENT APP (orient/client/)┴───────────────┐
 │  dataset bundle (private) · LocalTrainer · Streamlit UI / CLI          │
 │  shared verbatim: protocol.py · weights.py · models.py · problems.py   │
 └────────────────────────────────────────────────────────────────────────┘
```

Weights cross the wire as **safetensors** — no pickle.

---

## Quickstart

```bash
cd orient

python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
pip install -r client/requirements.txt

# 1. heterogeneous example datasets (10 problems x N clients)
cd client && python examples/make_datasets.py --clients 5 && cd ..

# 2. server
cd backend && python -m app.main && cd ..      # REST API on http://127.0.0.1:8000

# 3. clients — one process each, run concurrently
cd client
python -m app.main --server http://127.0.0.1:8000 --client-id client-1 \
  --dataset examples/bundles/poisson_client1
python -m app.main --server http://127.0.0.1:8000 --client-id client-2 \
  --dataset examples/bundles/poisson_client2
```

Interactive API docs at <http://127.0.0.1:8000/docs>.

**Prefer the browser?** `.\start.ps1 -WithClientUi` (Windows) brings up the API,
the dashboard on :8501, and the client UI on :8502, then opens the dashboard.
`.\start.ps1 -Stop` tears it all down.

> ⚠️ **Clients must launch concurrently.** A synchronous round completes only
> when every expected client has reported, so starting them one at a time with
> `--once` hangs in `wait`. See the troubleshooting table in
> [`orient/README.md`](orient/README.md#two-things-that-make-a-run-look-broken-when-it-isnt).

---

## Features

### 10 problems — SRS §9.1, Table 4

| Family | Problems |
|---|---|
| Supervised | `gramacy_lee`, `schaffer` |
| PINN | `poisson`, `helmholtz`, `allen_cahn`, `inverse_ns`, `inverse_dr` |
| Operator | `antiderivative`, `burgers`, `diffusion_reaction` |

Each ships `sample_dataset()`, an autograd PDE residual where applicable, and an
`evaluate()` relative-L2 against a known ground truth. `poisson` enforces its
boundary conditions through a hard output transform rather than a penalty term.

### 10 aggregation algorithms — SRS §4.4

| Group | Algorithms |
|---|---|
| Standard | `fedavg`, `fedprox`, `fedadam`, `fedadagrad`, `fedyogi`, `feddyn`, `scaffold` |
| Byzantine-robust | `median`, `trimmed_mean`, `krum` (multi-Krum) |

Weighting: `uniform` · `data_size` (the paper's default) · `quality`.
All aggregators operate purely on `state_dict` tensors, so they are
architecture-agnostic. `fedagrad` is accepted as an alias for `fedadagrad`.

> **Client-count requirements.** `median`/`trimmed_mean` need K ≥ 2.
> **Krum needs K ≥ 2f + 3** — 5 clients for f = 1. With fewer, the two candidates
> are always mutually nearest and Krum degenerates into "pick one". Generate
> enough with `--clients 5`.

### Framework v2 extensions

- **Async participation** — sample a `client_fraction` per round with
  `round_timeout`, resampled deterministically from `seed + round`
- **Compression** — int8 quantization and top-k sparsification of the *update*
- **Differential privacy** — per-gradient Gaussian noise after norm clipping,
  with ε tracked per round

---

## Layout

```
.
├── orient/
│   ├── backend/          # Server app — FastAPI + orchestrator + Streamlit dashboard
│   │   ├── app/          #   api, orchestrator, aggregators, weighting, models,
│   │   │                 #   problems, protocol, weights, storage, runner, metrics
│   │   └── ui/dashboard.py
│   ├── client/           # Client app — LocalTrainer, transport, CLI + Streamlit UI
│   │   └── examples/     #   make_datasets.py + 50 ready-made bundles
│   ├── configs/          # 10 YAML run configs (incl. sweep_example.yaml)
│   ├── tests/            # 9 test suites + run_all_aggregators.py
│   └── federated-sciml-main/   # untouched upstream reference implementation
├── SRS_FederatedSciML_Project.md   # the spec this is built to
├── SRS_SOURCE_CODE_AUDIT.md        # audit findings and fixes
├── literature_survey.md
└── federated_sciml_notes.md
```

### Shared modules must stay byte-identical

`protocol.py`, `weights.py`, `models.py`, `problems.py` are **verbatim copies**
in both apps. That is the compatibility contract (FR-PROTO-2): it guarantees the
client rebuilds the exact same architecture and computes the same loss as the
server. `tests/test_static_imports.py` fails the build if they drift.

---

## Tests

Nine suites, all stdlib-driven — no pytest required.

```bash
cd orient
python tests/test_static_imports.py   # import graph; shared modules byte-identical
python tests/test_runtime.py          # all 10 aggregators, all 10 problems, safetensors
python tests/test_server_logic.py     # run validation, registration guards, round engine
python tests/test_client.py           # FedProx proximal term, dataset validation, 50 bundles
python tests/test_logic_fixes.py      # regression tests for the 19 logic errors found
python tests/test_ui_smoke.py         # both Streamlit UIs against a live server
python tests/test_e2e.py              # real uvicorn server + real client processes
python tests/test_all_selenium_fixed.py  # 134-check roll-up: every module, both UIs, HTTP/DOM, E2E
python tests/run_all_aggregators.py   # E2E for all 10 aggregators
```

`test_all_selenium_fixed.py` is the roll-up — it exercises every module, both
UIs, the HTTP API, and an end-to-end run, and reports `134 PASS, 0 FAIL`.
Despite the name it drives **no browser**: its DOM assertions fetch pages with
`httpx` and parse them with BeautifulSoup, so no Selenium or WebDriver is needed.

---

## Environment

Validated on **Python 3.14 + torch 2.14.1**. The unpinned ranges in
`backend/requirements.txt` and `client/requirements.txt` work there.

The upstream `orient/federated-sciml-main/requirements.txt` pins `torch==2.6.0`,
which does **not** support Python 3.14 — use Python 3.11/3.12 if you need those
exact pins.

The roll-up suite `tests/test_all_selenium_fixed.py` imports `beautifulsoup4`
for its DOM assertions. It is **not** listed in either `requirements.txt`, so
install it separately — otherwise the suite reports `125 PASS, 2 FAIL` instead
of `134 PASS, 0 FAIL`:

```bash
pip install beautifulsoup4
```

---

## References

- Zhang, Liu, Weng & Lu — *Federated scientific machine learning for
  approximating functions and solving differential equations with data
  heterogeneity*, IEEE TNNLS 2025 ([arXiv:2410.13141](https://arxiv.org/abs/2410.13141)).
  The Poisson / Gramacy-Lee / Schaffer / Antiderivative setups follow this paper.
- **Flower** — round-based FL architecture and strategy set.
  <https://flower.ai/docs/framework/explanation-flower-architecture.html>
- Reddi et al. — *Adaptive Federated Optimization*
  ([arXiv:2003.00295](https://arxiv.org/abs/2003.00295)), the server-side
  adaptive update behind FedAdam/FedAdagrad/FedYogi.
- **safetensors** — weight wire format.
  <https://huggingface.co/docs/safetensors/main/en/api/torch>

---

## Docs

| Document | Contents |
|---|---|
| [`orient/README.md`](orient/README.md) | Full architecture, REST API table, problem suite, troubleshooting |
| [`orient/backend/README.md`](orient/backend/README.md) | Server app: env vars, round engine |
| [`orient/client/README.md`](orient/client/README.md) | Client app: bundles, CLI flags |
| [`SRS_FederatedSciML_Project.md`](SRS_FederatedSciML_Project.md) | The software requirements specification |
| [`SRS_SOURCE_CODE_AUDIT.md`](SRS_SOURCE_CODE_AUDIT.md) | Audit findings and their fixes |
