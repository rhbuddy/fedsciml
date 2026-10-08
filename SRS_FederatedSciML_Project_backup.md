# Software Requirements Specification (SRS)

## A Robust Multi-Algorithm Federated Learning Framework for Scientific Machine Learning

**Project:** Final Year Project  
**Based on:** "Federated scientific machine learning for approximating functions and solving differential equations with data heterogeneity" (Zhang, Liu, Weng, Lu — IEEE TNNLS 2025 / arXiv:2410.13141)  
**Base Codebase:** `lu-group/federated-sciml` (local: `federated-sciml-main/`)  
**Document Version:** 1.4 (adds two-app client/server distributed deployment — see §4.10–§4.13). Previous: 1.3.

---

## 1. Introduction

### 1.1 Purpose
This document specifies the requirements for a **robust, multi-algorithm federated learning framework for scientific machine learning (SciML)**. The system extends the original federated SciML codebase, which supports only a single aggregation algorithm (FedAvg) with hardcoded client counts, into a **configurable, reproducible benchmark framework** supporting multiple standard and Byzantine-robust aggregation algorithms under heterogeneous, noisy, and malicious client conditions.

The deliverable is intended to serve as a final year capstone project, contributing:
1. A reusable federated aggregation framework (`src/federated/`).
2. Nine aggregation algorithms across two categories (standard and robust).
3. A benchmarking capability that answers: *"Which aggregation algorithm is most robust to data heterogeneity, noise, and malicious clients in physics-informed and operator learning?"*

**Motivation (quoted from the base paper's conclusion, §7):** *"This study considers the FedAvg method in aggregating local model parameters. Still, the effect of different weights on local models and even adaptive updating algorithms can be investigated in future work."* — Zhang, Liu, Weng, Lu (IEEE TNNLS 2025). This project implements exactly that stated future work, and extends it with Byzantine-robust aggregation for noisy/malicious clients.

### 1.2 Scope
**In scope:**
- A pluggable aggregation framework supporting arbitrary client counts K.
- Standard aggregators: FedAvg, FedProx, FedAdam, FedAdagrad, FedYogi, SCAFFOLD.
- Robust aggregators: Median, Trimmed Mean, Krum.
- Configurable aggregation weighting: uniform, data-size, quality-based.
- Noisy/malicious client simulation.
- Config-driven experiment runner (YAML), fixed-seed reproducibility, structured result logging.
- Benchmark suite across the existing 10 problems (FedFuncApprox, FedPINN, FedDeepONet).
- A **two-app distributed deployment**: a **Server app** (REST orchestration of federated rounds) and a **Client app** (trains locally on a client's own private dataset and uploads weights only).
- Optional lightweight visualization dashboard (Phase 3).

**Out of scope:**
- New physics problems or new PDE formulations (existing problems reused as benchmark suite).
- Heterogeneous model architectures across clients (feature-space heterogeneity); requires alignment/distillation techniques and is explicitly excluded.
- Image (CNN/U-Net) support; noted as future extension.
- Differential-privacy guarantees/enforcement, secure aggregation, and real-time streaming infrastructure — all noted as future extensions (the two-app client/server exchanges weights over plain HTTP/REST, without cryptographic protection).

### 1.3 Definitions and Acronyms

| Term | Definition |
|---|---|
| **FL** | Federated Learning — training a shared model across distributed clients without exchanging raw data |
| **SciML** | Scientific Machine Learning |
| **PINN** | Physics-Informed Neural Network — model whose loss includes PDE residuals |
| **DeepONet** | Deep Operator Network — learns a mapping between function spaces (branch + trunk networks) |
| **FedAvg** | Federated Averaging — server averages client weights uniformly |
| **FedProx** | FedAvg with a proximal term (`μ`) penalizing client drift |
| **FedAdam / FedAdagrad / FedYogi** | Server-side adaptive optimizers applied to the aggregated update |
| **SCAFFOLD** | Uses control variates to correct client gradient bias |
| **Krum** | Byzantine-robust aggregation keeping the update closest to all others |
| **W1** | 1-Wasserstein distance — quantitative measure of non-IIDness between client data distributions |
| **Non-IID** | Non-independent and identically distributed data (heterogeneous client data) |
| **Global model** | The aggregated model maintained by the server |
| **Client** | A participant holding private local data and training the shared architecture locally |

### 1.4 References
1. Zhang, H., Liu, L., Weng, K., & Lu, L. *Federated scientific machine learning for approximating functions and solving differential equations with data heterogeneity.* IEEE TNNLS, 36(10), 18104–18117, 2025.
2. McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data* (FedAvg), 2017.
3. Li et al., *Federated Optimization in Heterogeneous Networks* (FedProx), 2020.
4. Reddi et al., *Adaptive Federated Optimization*, 2021.
5. Karimireddy et al., *SCAFFOLD: Stochastic Controlled Averaging for Federated Learning*, 2020.
6. Blanchard et al., *Machine Learning with Adversaries: Byzantine-Tolerant Gradient Descent* (Krum), 2017.
7. Lu, L., Meng, X., Mao, Z., & Karniadakis, G. E., *DeepXDE*, 2021 (core dependency).
8. Lu et al., *DeepONet*, 2021.

### 1.5 Publication Plan
This project is designed to be publishable as:
1. **Primary venue:** IEEE conference (IEEE ICMLA, IEEE IJCNN, or similar)
2. **Alternative venue:** Workshop paper at NeurIPS/ICML FL workshop
3. **Contribution type:** Applied benchmark paper (algorithm comparison)
4. **Novelty:** First systematic benchmark of FL aggregation algorithms on SciML (PINN/DeepONet) under controlled heterogeneity

**Paper structure:** Introduction → Related Work → Methodology → Experiments (9 algorithms × 3 problems) → Analysis (trade-off curves) → Conclusion

**Target length:** 8–10 pages (IEEE format)

### 1.6 Naming Conventions
| Context | Convention | Example |
|---------|------------|---------|
| Config files (YAML) | snake_case | `fedavg`, `fedprox`, `trimmed_mean` |
| Code (Python) | snake_case | `fed_avg()`, `fed_prox()` |
| Documentation | CamelCase | FedAvg, FedProx, TrimmedMean |
| File names | snake_case | `aggregators.py`, `noise_sim.py` |
| Requirement IDs | SCREAMING_SNAKE | FR-AGG-1, NFR-PERF-1 |

---

## 2. Overall Description

### 2.1 Product Perspective
The system is a **major extension** of the existing `lu-group/federated-sciml` repository. It preserves the validated scientific components (physics problem definitions, PDE loss formulations, data generation, partitioning logic, and benchmark baselines) while replacing the duplicated, hardcoded FedAvg training loops with a **configurable federated framework**.

```
Original codebase (reused unchanged):
  data/                    → data generation + partitioning (heterogeneity control)
  src/FedFuncApprox/       → 1D Gramacy & Lee, 2D Schaffer problems
  src/FedPINN/             → Poisson, Helmholtz, Allen-Cahn, inverse NS, inverse DR
  src/FedDeepONet/         → Antiderivative, Burgers, Diffusion-reaction, Cavity flow (cavity flow code exists in the repo but is NOT part of the paper's reported experiments nor the 10-problem benchmark suite)

New framework (this project's contribution):
  src/federated/           → aggregation framework (aggregators, client, server, runner)
  src/federated/protocol/  → wire schemas (ModelSpec, ProblemSpec, ClientUpdate, ...)
  src/federated/models/    → shared model factory (ModelSpec → architecture)
  src/federated/io/        → weight serialization (safetensors)
  src/federated/problems/  → problem registry (name → residual/transform)
  src/apps/server/         → Server app (FastAPI REST orchestration of federated rounds)
  src/apps/client/         → Client app (local dataset → local training → weights-only upload)
  configs/                 → YAML experiment definitions
  results/                 → reproducible artifacts (weights, metrics, configs)
  dashboard.py             → optional visualization layer
```

### 2.2 Product Functions (Summary)
- Run federated training for any of the 10 existing problems with any supported aggregation algorithm and arbitrary client count.
- Simulate data heterogeneity (controlled via partitioning, measured via W1).
- Simulate noisy and malicious clients with configurable fraction.
- Compute evaluation metrics: relative L2 error, weight divergence, robustness accuracy.
- Persist reproducible results: config snapshot, model weights, metrics, plots.
- Compare algorithms across heterogeneity and noise conditions via a benchmark runner.

### 2.3 User Characteristics
| Role | Description |
|---|---|
| **Researcher (primary)** | Runs benchmark experiments, compares algorithms, analyzes accuracy under heterogeneity/noise |
| **Supervisor / Reviewer** | Evaluates correctness of design, scope, and experimental protocol |
| **Client (runtime user)** | Runs the **Client app**: trains locally on its own private dataset and uploads weights only (raw data never leaves the client) |
| **Server operator** | Runs the **Server app**: starts/stops federated runs and monitors round status |

### 2.4 Assumptions and Dependencies
**Assumptions:**
- All clients use the same model architecture (weight tensors of identical shape) — required for weight averaging.
- Client data represents the same physical field/operator under different distributions (covariance shift). Feature-space heterogeneity is out of scope.
- Clients participate with full communication rounds by default; partial participation is an extension.
- Environment: Python, PyTorch, DeepXDE, numpy, scipy, POT, scikit-learn.

**Project-level assumptions:**
- The project is intended to be carried out by a single final-year student.
- The schedule assumes prior familiarity with Python and PyTorch.
- Additional learning time may be required for DeepXDE and federated learning concepts; such time is included in Phase 1 of the development plan.

**Dependencies:**
- `deepxde==1.13.1`, `torch`, `numpy`, `scipy`, `matplotlib`, `scikit-learn`, `scikit-optimize`, `skopt`, `POT` (Python Optimal Transport).
- Two-app stack: `fastapi`, `uvicorn`, `pydantic`, `pydantic-settings`, `httpx`, `safetensors`, `pyyaml`.
- Fix the missing `spaces.py` dependency (currently imported by antiderivative files but absent from the repo) — either vendor the module or provide a local `spaces` implementation.

### 2.5 Domain Applicability
| Domain | Applicability |
|---|---|
| Physics / engineering (fluid, thermal, structural) | ✅ Full — the current benchmark suite |
| Medical-physics (cardiac, biomechanics, hemodynamics) | ✅ Natural fit — same PINN approach |
| Medical imaging (segmentation, MRI/X-ray) | ⚠️ Requires CNN/U-Net backbone (future work) |
| Generic image data | ⚠️ Requires CNN + image partitioning (future work) |

The federated aggregation layer is domain-agnostic (operates on weight tensors); only the model family is domain-specific.

---

## 3. System Architecture

### 3.1 Architecture Diagram
```
┌──────────────────────────────────────────────────────────────────────────┐
│                          CONFIG LAYER (YAML)                             │
│   problem | n_clients | aggregator | heterogeneity | noise_mode | seed    │
└──────────────────────────────────────────────────────────────────────────┘
                                      │ feeds
┌──────────────────────────────────────────────────────────────────────────┐
│                            DATA LAYER                                     │
│   data_gen_*.py ──▶ Ground truth (Burgers, Poisson, antiderivative...)     │
│   data_assignment.py ──▶ Partition ──▶ Client subdomains + W1 control      │
│   noise_sim.py ──▶ inject noisy/malicious clients (NEW)                    │
└───────────────┬──────────────────────────────────────────────────────────┘
                │  assigns data slices
┌───────────────▼──────────────────────────────────────────────────────────┐
│                          CLIENT LAYER (K clients)                        │
│   ┌───────────────┐  ┌───────────────┐              ┌───────────────┐     │
│   │  Client 1     │  │  Client 2     │    ...       │  Client K     │     │
│   │ local data ✓  │  │ local data ✓  │              │ local data ✓  │     │
│   │ model (FNN/   │  │ model (FNN/   │              │ model (FNN/   │     │
│   │ PINN/DeepONet)│  │ PINN/DeepONet)│              │ PINN/DeepONet)│     │
│   │ local Adam ✓  │  │ local Adam ✓  │              │ local Adam ✓  │     │
│   └──────┬────────┘  └──────┬────────┘              └──────┬────────┘     │
│          └─────────── WEIGHTS ONLY (no data leaves) ──────┘              │
└───────────────┬──────────────────────────────────────────────────────────┘
                │  upload weights / download global model
┌───────────────▼──────────────────────────────────────────────────────────┐
│                          SERVER / AGGREGATION LAYER                      │
│   ┌──────────────────────────────────────────────────────────────────┐   │
│   │  aggregators.py                                                   │   │
│   │  STANDARD: FedAvg | FedProx | FedAdam | FedAdagrad | FedYogi     │   │
│   │           | SCAFFOLD                                              │   │
│   │  ROBUST (NEW): Median | Trimmed Mean | Krum                      │   │
│   │  WEIGHTING: uniform | data-size | quality                        │   │
│   └──────────────────────────────────────────────────────────────────┘   │
│   server.py: broadcast → collect → aggregate → global model θ*           │
└───────────────┬──────────────────────────────────────────────────────────┘
                │  global model + metrics
┌───────────────▼──────────────────────────────────────────────────────────┐
│                         EVALUATION LAYER                                 │
│   l2_error.py (vs centralized & extrapolation baselines)                 │
│   weight_divergence.py (per-layer, any architecture)                     │
│   robustness.py (accuracy under noise fraction)                          │
└───────────────┬──────────────────────────────────────────────────────────┘
                │  results
┌───────────────▼──────────────────────────────────────────────────────────┐
│                         STORAGE LAYER                                    │
│   results/<problem>/<algorithm>/  {config.yaml, server_best.pth,         │
│                                    loss.npz, metrics.json}               │
└──────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Federated Training Loop
```
        START
          │
          ▼
    [CONFIG: load YAML]
          │
          ▼
    [DATA: generate + partition into K subdomains, set W1, inject noise]
          │
          ▼
  ┌─────────────── FEDERATED ROUND (repeat G times) ───────────────┐
  │                                                                 │
  │   for each client k (in parallel):                              │
  │      model_k.train(local_epochs)  ← local data, local Adam     │
  │            │                                                    │
  │            ▼                                                    │
  │   upload weights θ₁...θ_K  (NOT data)                          │
  │            │                                                    │
  │            ▼                                                    │
  │   SERVER: θ* ← aggregate(θ₁..θ_K, aggregator, weighting)       │
  │            ├─ standard: fedavg / fedprox(μ) / fedadam / ...     │
  │            └─ robust:   median / trimmed_mean / krum            │
  │            │                                                    │
  │            ▼                                                    │
  │   broadcast θ* back to all clients (load_state_dict)           │
  │            │                                                    │
  │            ▼                                                    │
  │   every N rounds: evaluate L2 error + weight divergence        │
  └───────────────────────────────┬─────────────────────────────────┘
                                  │
                                  ▼
                        [SAVE: config, weights, metrics]
                                  │
                                  ▼
                        [COMPARE: algorithms × heterogeneity × noise]
                                  │
                                  ▼
                         [PLOTS + DASHBOARD (optional)]
```

### 3.3 Module Map
```
federated-sciml-main/
├── data/                        (KEEP — unchanged)
│   ├── data_gen_*.py
│   └── data_assignment.py
├── src/
│   ├── federated/               (NEW — this project's contribution)
│   │   ├── config.py            # load/validate YAML configs
│   │   ├── client.py            # local training wrapper
│   │   ├── server.py            # broadcast/collect/aggregate round loop
│   │   ├── aggregators.py       # fedavg, fedprox, fedadam, fedadagrad, fedyogi, scaffold
│   │   ├── robust.py            # median, trimmed_mean, krum
│   │   ├── noise_sim.py         # noisy/adversarial client injection
│   │   ├── weighting.py         # uniform / data-size / quality weights
│   │   ├── metrics.py           # l2_error, weight_divergence (architecture-agnostic)
│   │   ├── logging.py           # structured result storage
│   │   └── runner.py            # single-command experiment + sweep runner
│   ├── FedFuncApprox/           (keep models; route training through framework)
│   ├── FedPINN/                 (keep models; route training through framework)
│   └── FedDeepONet/             (keep models; route training through framework)
├── configs/                     (NEW — YAML experiment definitions + sweeps)
├── results/                     (NEW — artifacts)
└── dashboard.py                 (OPTIONAL — Phase 3, Streamlit)
```

---

## 4. Functional Requirements

### 4.1 Configuration (FR-CFG)
| ID | Requirement |
|---|---|
| FR-CFG-1 | The system SHALL load experiment configuration from a YAML file. |
| FR-CFG-2 | The system SHALL support configuration of: `problem`, `model_family`, `n_clients`, `aggregator`, `aggregation_weights`, `local_epochs`, `global_rounds`, `heterogeneity`, `learning_rate`, `seed`, `noise_mode`, `noise_fraction`, `gradient_clip`, `max_norm`. |
| FR-CFG-3 | The system SHALL validate config values (e.g., algorithm name exists, K ≥ 2, noise_fraction ∈ [0,1]). |
| FR-CFG-4 | The system SHALL support algorithm-specific hyperparameters: `mu` (FedProx), `server_lr` (FedAdam/Adagrad/Yogi), `beta1`, `beta2`. |
| FR-CFG-5 | The system SHALL support sweep syntax: lists of values for any parameter produce multiple runs. |
| FR-CFG-6 | The system SHALL save a copy of the exact content alongside each run's results (reproducibility). |

### 4.2 Data Layer (FR-DATA)
| ID | Requirement |
|---|---|
| FR-DATA-1 | The system SHALL reuse existing data generation scripts (antiderivative, Burgers, diffusion-reaction) and function approximation datasets (Gramacy & Lee, Schaffer). |
| FR-DATA-2 | The system SHALL partition data across K clients via 1D subdomain partition (n subdomains per client, alternating assignment), 2D x-partition or xy-partition (n² subdomains, alternating assignment), or Chebyshev functional-space partition for operator learning (M=10 terms; forward / middle / inverse directions per the paper). |
| FR-DATA-3 | The system SHALL compute and record the 1-Wasserstein distance (W1) between client distributions as the heterogeneity metric — direct W1 for K=2, mean pairwise W1 for K≥3 (per the paper's Eq. 4). |
| FR-DATA-4 | The system SHALL support heterogeneity levels (low/medium/high) via partition granularity (`n_pieces`, `n_terms`). |
| FR-DATA-5 | The system SHALL resolve the missing `spaces.py` dependency so operator-learning problems run out of the box. |
| FR-DATA-6 | The system SHALL validate that data partitions cover the entire domain (no gaps, no overlaps) and log partition statistics (min/max points per client, total coverage). |
| FR-DATA-7 | The system SHALL verify W1 computation against manual calculation for a simple test case (e.g., 2 uniform distributions) during validation gate. |

### 4.3 Client Layer (FR-CLIENT)
| ID | Requirement |
|---|---|
| FR-CLIENT-1 | The system SHALL create K client model instances of identical architecture for a given problem. |
| FR-CLIENT-2 | Each client SHALL train locally on its own data slice for `local_epochs` using the configured local optimizer (default: Adam; configurable: Adam, SGD, Lion). |
| FR-CLIENT-3 | Clients SHALL communicate only model weights, never raw data. |
| FR-CLIENT-4 | Clients SHALL be able to load the aggregated global model (broadcast) before the next round. |
| FR-CLIENT-5 | The system SHALL support weighting modes: (a) uniform: `w_k = 1/K`, (b) data-size: `w_k = N_k / N`, (c) quality: `w_k = (1/L_k) / Σ(1/L_i)` where `L_k` is the client's local validation loss (lower loss → higher weight). |
| FR-CLIENT-6 | The system SHALL preserve the paper's hard boundary constraint enforcement (output transforms) for PINN examples, matching the paper's default (tanh activation, Adam, LR 0.001 unless specified otherwise). |
| FR-CLIENT-7 | The system SHALL detect NaN/Inf gradients during local training and exclude the corrupted client from aggregation for that round. |
| FR-CLIENT-8 | The system SHALL log excluded clients with timestamps and round numbers for debugging. |
| FR-CLIENT-9 | The system SHALL support gradient clipping via config: `gradient_clip: none / value / norm` with configurable `max_norm` (default: 1.0) or `clip_value` (default: 0.5). |

### 4.4 Aggregation Layer (FR-AGG)
| ID | Requirement |
|---|---|
| FR-AGG-1 | The system SHALL generalize aggregation to an arbitrary number of clients K (no hardcoded K=2/3). |
| FR-AGG-2 | **FedAvg** (matches the paper's FedAvg-Adam): server aggregates via `θ* = Σ_k (Nk/N) · θ_k` — **data-size proportional weighting** is the paper's default; uniform weighting is configurable. Local clients use Adam; server applies plain averaging (equivalent to FedOPT with Client-OPT=Adam, Server-OPT=GD). |
| FR-AGG-3 | **FedProx**: add proximal term `(μ/2)‖θ_k − θ*‖²` to each client's local objective. |
| FR-AGG-4 | **FedAdam**: server applies Adam-style update (momentum + adaptive LR) to the aggregated pseudo-gradient. |
| FR-AGG-5 | **FedAdagrad**: server applies Adagrad-style adaptive LR. |
| FR-AGG-6 | **FedYogi**: server applies Yogi-style adaptive LR. |
| FR-AGG-7 | **SCAFFOLD**: clients maintain control variates; server corrects updates toward the true gradient. |
| FR-AGG-8 | **Median**: element-wise median across client weight tensors. |
| FR-AGG-9 | **Trimmed Mean**: drop extreme clients (top/bottom fraction) then average. |
| FR-AGG-10 | **Krum**: select the single update with minimum squared distance to all others. |
| FR-AGG-11 | All aggregators SHALL expose a common interface (drop-in via `aggregator:` config). |
| FR-AGG-12 | Aggregation weighting SHALL support: (a) uniform: all clients equal, (b) data-size proportional: `w_k = N_k/N`, (c) quality-based: inverse validation loss `w_k = (1/L_k) / Σ(1/L_i)` computed on a 20% held-out shard from each client's local data. |
| FR-AGG-13 | The system SHALL support per-aggregator local optimizer overrides (e.g., SCAFFOLD uses SGD, FedAvg uses Adam). |

### 4.5 Noise / Malicious Client Simulation (FR-NOISE)
| ID | Requirement |
|---|---|
| FR-NOISE-1 | The system SHALL support `noise_mode: none / noisy / adversarial`. |
| FR-NOISE-2 | In `noisy` mode, a configurable fraction of clients SHALL have their local training corrupted (e.g., high noise / random weights). |
| FR-NOISE-3 | In `adversarial` mode, a configurable fraction of clients SHALL submit maliciously manipulated weights (e.g., inverted/randomized). |
| FR-NOISE-4 | The system SHALL record which clients were compromised for analysis. |

### 4.6 Evaluation (FR-EVAL)
| ID | Requirement |
|---|---|
| FR-EVAL-1 | The system SHALL compute relative L2 error of the federated model on a held-out test set. |
| FR-EVAL-2 | The system SHALL compute centralized (all-data) and extrapolation (no-collaboration) baselines for comparison. |
| FR-EVAL-3 | The system SHALL compute per-layer weight divergence between federated and centralized models, in an architecture-agnostic manner (iterate state_dict keys generically, not hardcoded layer names). |
| FR-EVAL-4 | The system SHALL produce comparison tables across algorithms, heterogeneity levels, and noise fractions. |

### 4.7 Storage & Logging (FR-STORE)
| ID | Requirement |
|---|---|
| FR-STORE-1 | The system SHALL persist per-run artifacts under `results/<problem>/<algorithm>/<run_id>/`. |
| FR-STORE-2 | Each run SHALL store: `config.yaml`, `server_best.pth` (weights), `loss.npz`, `l2_error.npz`, `weight_divergence.npz`, `metrics.json`. |
| FR-STORE-3 | The system SHALL save the final global model weights in a reloadable format (`torch.save` / DeepXDE checkpoint). |
| FR-STORE-4 | The system SHALL support loading saved weights for inference, comparison, resume, and warm-start. |
| FR-STORE-5 | The system SHALL set and record random seeds (`random`, `numpy`, `torch`) for reproducibility. |

### 4.8 Benchmark Runner (FR-RUN)
| ID | Requirement |
|---|---|
| FR-RUN-1 | The system SHALL run a single experiment from a YAML config with one command. |
| FR-RUN-2 | The system SHALL run multi-dimensional sweeps (algorithm × heterogeneity × noise) and aggregate results into a comparison table. |
| FR-RUN-3 | The system SHALL reuse the existing 10 problems as the benchmark suite (2 function approximation + 5 FedPINN + 3 FedDeepONet, matching the paper's reported experiments). |
| FR-RUN-4 | The system SHALL run the full sweep matrix on the 3 deep-benchmark problems (Poisson, Antiderivative, Schaffer) and reduced-configuration spot-checks on the remaining 7. |
| FR-RUN-5 | The system SHALL provide a validation-gate command that reproduces the paper's FedAvg Poisson result within ±5% relative L2 error (absolute difference) of the published value before other aggregators are enabled. |

### 4.9 Dashboard (FR-DASH) — Phase 3, optional
| ID | Requirement |
|---|---|
| FR-DASH-1 | The system SHALL provide a lightweight web dashboard (Streamlit/Gradio) to select a config, launch a run, and view loss/L2/weight-divergence curves. |
| FR-DASH-2 | The dashboard SHALL display algorithm-vs-algorithm comparison plots. |
| FR-DASH-3 | The dashboard SHALL be a thin layer over the runner, with no science logic embedded. |

### 4.10 Client App (FR-CLIENTAPP) — Two-App Deployment
| ID | Requirement |
|---|---|
| FR-CLIENTAPP-1 | The system SHALL provide a standalone **Client app** that a participating client runs on its own machine to train locally and produce model weights. |
| FR-CLIENTAPP-2 | The Client app SHALL load the client's private dataset from a standard directory bundle (§4.13) and SHALL NOT transmit raw data to the server. |
| FR-CLIENTAPP-3 | The Client app SHALL register with the server (`/clients/register`) supplying `client_id`, `problem`, `n_samples`, and local domain information. |
| FR-CLIENTAPP-4 | The Client app SHALL fetch the current global weights and `ModelSpec`, build an identical architecture, and load the weights before training. |
| FR-CLIENTAPP-5 | The Client app SHALL train for the configured `local_epochs` using the local optimizer (Adam) and optional gradient clipping. |
| FR-CLIENTAPP-6 | The Client app SHALL upload **only** the trained weight payload (safetensors) plus metadata (`n_samples`, `local_loss`) via `/clients/{id}/update`. |
| FR-CLIENTAPP-7 | The Client app SHALL run headlessly via CLI, log round progress, and be usable without modifying the server code. |
| FR-CLIENTAPP-8 | The Client app SHALL reject a dataset bundle whose `problem` is unknown or whose arrays do not match the `ProblemSpec`. |
| FR-CLIENTAPP-9 | Client datasets SHALL remain on the client; only weights leave the machine. |

### 4.11 Server App (FR-SERVERAPP) — Two-App Deployment
| ID | Requirement |
|---|---|
| FR-SERVERAPP-1 | The system SHALL provide a standalone **Server app** (FastAPI) exposing REST endpoints for registration, model distribution, update collection, and run control. |
| FR-SERVERAPP-2 | The Server app SHALL maintain a client registry and a per-round update store. |
| FR-SERVERAPP-3 | The Server app SHALL drive synchronous rounds: broadcast global model → collect client updates → aggregate → advance round. |
| FR-SERVERAPP-4 | The Server app SHALL expose endpoints: `/health`, `/clients/register`, `/clients/{id}/assignment`, `/clients/{id}/weights`, `/clients/{id}/update`, `/run/status`, `/admin/run/start`, `/admin/run/stop`. |
| FR-SERVERAPP-5 | The Server app SHALL reuse the existing aggregation engine (`aggregators.py`, `robust.py`) and SHALL NOT reimplement aggregation. |
| FR-SERVERAPP-6 | The Server app SHALL persist results through the logging layer (`results/<problem>/<aggregator>/`). |

### 4.12 Wire Protocol (FR-PROTO)
| ID | Requirement |
|---|---|
| FR-PROTO-1 | The system SHALL define versioned Pydantic schemas: `ModelSpec`, `ProblemSpec`, `ClientRegistration`, `AssignmentResponse`, `ClientUpdate`. |
| FR-PROTO-2 | `ModelSpec` SHALL fully determine the architecture so both apps produce identical `state_dict` keys and shapes. |
| FR-PROTO-3 | `ProblemSpec` SHALL reference problems by name from a fixed registry; no arbitrary code is transmitted. |
| FR-PROTO-4 | Weight payloads SHALL use safetensors (no pickle). |
| FR-PROTO-5 | The same protocol SHALL serve both the in-memory (simulation) and HTTP (distributed) transports. |

### 4.13 Client Dataset Format
Each client supplies a self-contained bundle that the Client app validates against the `ProblemSpec`:

```
mydata/
├── dataset.json      # {"problem": "...", "n_samples": N, "domain": {...}}
└── data.npz          # arrays keyed by problem family:
                      #   PINN (e.g. Poisson):  x_train [, u_train]
                      #   DeepONet:             branch_train, trunk_train, y_train
                      #   FuncApprox:           X, y
```

---

## 5. Algorithm Selection Guide (Design Decision)

### 5.1 Selection Table
| Data situation | Symptoms | Recommended algorithm | Why |
|---|---|---|---|
| Similar clients | Low W1, same physics, balanced data | **FedAvg** | Simple, near-optimal when clients are alike |
| Mild heterogeneity | Small domain differences | **FedAdam** | Server momentum smooths convergence |
| Strong heterogeneity | High W1, different subdomains/physics | **FedProx** | Proximal term stops client drift |
| Extreme heterogeneity | Disjoint domains, very different regimes | **SCAFFOLD** | Control variates correct client bias — most robust to non-IID |
| Unequal data volumes | 1k vs 2k vs 10k per client | **Weighted averaging** (data-size) | Mimics centralized training |
| One noisy client | Single faulty sensor/participant | **Trimmed mean** | Drops extreme updates |
| One malicious client | Deliberate attack (flipped weights) | **Krum** | Keeps update closest to all others |
| Several compromised clients | >1 bad participant | **Median** | Element-wise median resists multiple outliers |
| Unknown quality, mixed | Can't trust anyone's reliability | **Median / Krum + validation** | No trust assumptions needed |
| Privacy-sensitive data | Need basic gradient protection | **Gradient clipping** + robust aggregation | Bounds gradient sensitivity |

### 5.2 Decision Flow
```
START: how different are the clients (W1)?
│
├─ LOW  → FedAvg / FedAdam
├─ HIGH → FedProx → still bad? → SCAFFOLD
│
THEN: is anyone untrustworthy?
│
├─ NO  → keep the standard choice
├─ YES, 1 bad   → Trimmed mean / Krum
└─ YES, several → Median
│
THEN: do data volumes differ a lot?
└─ YES → data-size weighted averaging
```

### 5.3 Key Claims to be Validated by the Benchmark
1. No single algorithm is best — the winner depends on the data situation.
2. Robust algorithms (median/Krum) trade clean-data accuracy for resilience; quantify the trade-off.
3. The correct choice is data-driven, hence the config-driven sweep is the right tool.

---

## 6. Non-Functional Requirements

### 6.1 Reproducibility (NFR-REP)
- NFR-REP-1: All runs SHALL be reproducible given the same config and seed.
- NFR-REP-2: The system SHALL record environment/dependency versions.

### 6.2 Performance (NFR-PERF)
- NFR-PERF-1: Aggregation cost SHALL scale linearly with K and model size.
- NFR-PERF-2: The framework SHALL not materially slow down the local training compared to the original scripts (aggregation is a small fraction of round time).
- NFR-PERF-3: GPU usage SHALL be supported when available (no hard GPU assumptions).

### 6.3 Maintainability (NFR-MNT)
- NFR-MNT-1: The framework SHALL eliminate the original copy-paste duplication (~15 copies of the loop → one shared implementation).
- NFR-MNT-2: Aggregators SHALL be added by implementing one common interface.
- NFR-MNT-3: Weight-divergence computation SHALL be architecture-agnostic (iterate `state_dict` keys).

### 6.4 Usability (NFR-USE)
- NFR-USE-1: A user SHALL be able to run an experiment with a single command: `python -m src.federated.runner --config configs/burgers_fedprox.yaml`.
- NFR-USE-2: Config files SHALL be human-readable and documented.

### 6.5 Security / Privacy (NFR-SEC)
- NFR-SEC-1: Raw client data SHALL never leave a client (by design of the FL protocol).
- NFR-SEC-2: The system SHALL NOT claim formal privacy guarantees (no DP in this version); DP is documented as future work.
- NFR-SEC-3: Robust aggregation SHALL mitigate the effect of malicious clients on the global model.
- NFR-SEC-4: No secrets, keys, or credentials SHALL be stored in the repository.
- NFR-SEC-5: Gradient clipping SHALL be available as a lightweight privacy mechanism to bound gradient sensitivity (clip_value or max_norm configurable).

### 6.6 Portability (NFR-PORT)
- NFR-PORT-1: The system SHALL run on CPU-only and GPU environments.
- NFR-PORT-2: The system SHALL pin dependency versions (per existing `requirements.txt`) to avoid drift.

### 6.7 Recommended Development Hardware
- **OS:** Ubuntu / Linux / Windows
- **Runtime:** Python 3.x
- **GPU:** CUDA-capable NVIDIA GPU (recommended); CPU-only execution supported
- **RAM:** 16 GB minimum; 24 GB recommended
- **Disk:** approximately 20–30 GB free space (dependencies, datasets, and results)

---

## 7. System Interfaces & External Dependencies

### 7.1 External Software Interfaces
| Interface | Direction | Protocol | Purpose |
|-----------|-----------|----------|---------|
| DeepXDE | Internal | Python API | PINN/DeepONet model definition, PDE loss computation |
| PyTorch | Internal | Python API | Neural network training, autograd, optimizer |
| POT (Python Optimal Transport) | Internal | Python API | Wasserstein distance computation |
| NumPy/SciPy | Internal | Python API | Data generation, numerical operations |
| Matplotlib | Internal | Python API | Plotting results |
| YAML parser | Internal | File I/O | Configuration loading |
| File system | Output | File I/O | Results storage (weights, metrics, configs) |

### 7.2 Hardware Interfaces
| Interface | Requirement |
|-----------|-------------|
| CPU | Required (all computations must work on CPU) |
| CUDA GPU | Optional (accelerates training 2-10x) |
| RAM | Minimum 16GB, recommended 24GB |
| Disk | 20-30GB for dependencies, datasets, and results |

### 7.3 Communication Interfaces
| Interface | Protocol | Purpose |
|-----------|----------|---------|
| Client-Server (distributed) | HTTP/REST (FastAPI) | Weight exchange in federated training (two-app deployment) |
| Client-Server (local simulation) | In-memory (single machine) | Single-process runner / tests |
| Dashboard | HTTP/REST (future) | Visualization (Phase 3) |

### 7.4 Data Format Interfaces
| Format | Purpose | Standard |
|--------|---------|----------|
| YAML | Configuration files | YAML 1.2 |
| .pth / .pt | Model weights (local persistence) | PyTorch serialization |
| .safetensors | Model weights (wire transfer between apps) | Safetensors (no pickle) |
| .npz | Numerical arrays | NumPy compressed |
| dataset.json | Client dataset manifest | JSON (see §4.13) |
| .json | Metrics and logs | JSON |
| .txt | Datasets | Plain text (space-separated) |

### 7.5 Missing Dependencies Resolution
| Dependency | Status | Resolution |
|------------|--------|------------|
| `spaces.py` | Missing from repo | Create minimal implementation in Phase 1 (§11.1) |
| `deepxde==1.13.1` | Pinned version | Verify compatibility with PyTorch version |
| `POT` | External | Install via pip, verify W1 computation matches paper |

---


## 8. Data Requirements

### 8.1 Input Data
| Dataset | Source | Used by |
|---|---|---|
| Gramacy & Lee train/test `.txt` | `data/` | FedFuncApprox 1D |
| Schaffer function samples | generated | FedFuncApprox 2D |
| Antiderivative operator data | `data_gen_antid_100.py` (GRF / Chebyshev) | FedDeepONet |
| Burgers solutions | `data_gen_burgers_101_101.py` | FedDeepONet |
| Diffusion-reaction data | `data_gen_dr_101_101.py` | FedDeepONet |
| PDE domain data (Poisson, Helmholtz, Allen-Cahn, NS, DR) | generated in-script | FedPINN |

### 8.2 Output Data (per run)
```
results/<problem>/<algorithm>/<run_id>/
├── config.yaml              # exact run config
├── server_best.pth          # best global model weights
├── server_final.pth         # final global model weights
├── loss.npz                 # training loss curve
├── l2_error.npz             # L2 error over rounds
├── weight_divergence.npz    # per-layer divergence vs centralized
└── metrics.json             # L2, W1, runtime, noise info, summary
└── plots/                   # optional PNG figures
```

### 8.3 Storage Schema (naming)
- Run directory: `results/<problem>/<aggregator>/<run_id>` where `run_id` encodes `{aggregator}-{heterogeneity}-{noise_mode}-{noise_fraction}-{seed}`.

---

## 9. Benchmark Matrix & Test Cases

### 9.1 Benchmark Matrix
```
problems      = {GramacyLee_1D, Schaffer_2D, Poisson, Helmholtz, AllenCahn,
                  InverseNS, InverseDR, Antiderivative, Burgers, DiffusionReaction}
algorithms    = {fedavg, fedprox, fedadam, fedadagrad, fedyogi, scaffold,
                  median, trimmed_mean, krum}
heterogeneity = {low, medium, high}
noise_fraction= {0.0, 0.1, 0.2, 0.3}   (noisy and adversarial modes)
seeds         = {≥3 runs per configuration}
```

**Benchmark feasibility note:** The complete matrix above represents the intended evaluation. Preliminary pilot runs on a subset will be used to estimate per-run runtime and total resource cost. If computational resources are limited, representative subsets of problems, heterogeneity levels, and noise fractions may be selected while preserving a fair comparison across all algorithms (i.e., every algorithm is evaluated under identical conditions on the chosen subset).

**Deep-benchmark strategy (fixed decision):** Depth over breadth. The **full sweep matrix** is run on **3 representative problems — one per model family**:
- **Poisson** (FedPINN)
- **Antiderivative** (FedDeepONet)
- **Schaffer** (FedFuncApprox)

All 9 algorithms × heterogeneity levels × noise fractions are evaluated in full on these three. The remaining 7 problems are spot-checked with a reduced configuration (e.g., FedAvg vs FedProx vs Krum, medium heterogeneity, clean data) as evidence of generality, time permitting.

**Validation gate (fixed decision):** Before any algorithm beyond FedAvg is added, the framework MUST reproduce the paper's FedAvg result on the Poisson problem within a tolerance (relative L2 error within a defined tolerance of the published value on the same setting; the exact tolerance is set in Phase 1 from the paper's reported result). This gate must pass before proceeding to Phase 2; it protects against the refactor silently breaking the working physics.

**Runtime Estimates (NEW):**

| Problem | Model Size | GPU Time/Run | CPU Time/Run | Total Runs (Deep Benchmark) |
|---------|------------|--------------|--------------|----------------------------|
| Poisson (FedPINN) | FNN-20×3 | ~5 min | ~30 min | 324 |
| Antiderivative (FedDeepONet) | DeepONet-40×2 | ~15 min | ~2 hours | 324 |
| Schaffer (FedFuncApprox) | FNN-64×3 | ~10 min | ~1 hour | 324 |

**Total Deep Benchmark Time:**
- **GPU (A100):** 3 × 324 × avg(10 min) ≈ **162 hours** (~7 days continuous, unattended)
- **CPU only:** 3 × 324 × avg(1 hour) ≈ **972 hours** (40+ days) — NOT feasible

**Recommendation:** GPU is required for full benchmark (162 hours ≈ 7 days continuous). CPU-only should use reduced matrix (1 problem × 9 algos × 2 heterogeneity × 2 noise × 3 seeds = 108 runs ≈ 108 hours).

**Spot-check Runtime (remaining 7 problems):**
- 7 × 3 algos × 1 heterogeneity × 1 noise × 3 seeds = 63 runs
- GPU: ~10 hours
- CPU: ~63 hours (feasible over 3 days)



**Paper reference settings (Table 4, arXiv:2410.13141) for reproduction:**

| Example | Width | Depth | Activation | Local epochs | Global epochs |
|---|---|---|---|---|---|
| Gramacy & Lee | 3 | 64 | tanh | 5 | 3000 |
| Schaffer | 3 | 64 | tanh | 5 | 3000 |
| 1D Poisson | 3 | 20 | tanh | 5 | 1000 |
| 2D Helmholtz | 3 | 64 | sine | 5 | 2000 |
| Allen-Cahn | 3 | 64 | sine | 5 | 10000 |
| Inverse Navier-Stokes | 6 | 50 | tanh | 1 | 40000 |
| Inverse Diffusion-reaction | 3 | 20 | tanh | 5 | 20000 |
| Antiderivative (DeepONet) | 2 | 40 | ReLU | 5 | 10000 |
| Diffusion-reaction (DeepONet) | 3 | 100 | ReLU | 5 | 10000 |
| Burgers' (DeepONet) | 2 | 64 | ReLU | 5 | 10000 |

Default training: hyperbolic tangent activation and Adam optimizer with learning rate 0.001 (unless listed otherwise). These settings SHALL be used as the default in the framework's configs so the validation gate and benchmark match the paper's reported results.

### 9.2 Test Cases (functional verification)

| Test | Input | Expected | Edge case |
|---|---|---|---|
| K generalization | n_clients = 2, 3, 5 | aggregation generalizes | K not dividing n_blocks |
| Heterogeneity sweep | n_pieces / n_terms | L2 decreases with W1 | W1→0 matches centralized |
| FedProx μ sweep | μ values | stability | μ too large → slow convergence |
| FedAdam server LR | server_lr values | convergence | server_lr too large → divergence |
| Noise: 1 bad client | noisy, 1 client | FedAvg degrades; median/Krum recover | noise_fraction → 0.5 |
| Noise: adversarial | adversarial mode | Krum rejects outlier | full adversarial majority |
| Weighted aggregation | data volumes 1k vs 2k | size-weighted ≈ centralized | extreme imbalance |
| NaN gradient | corrupted client | detected and skipped | NaN propagates in FedAvg |
| Resume | interrupted run | reload weights and continue | — |
| Weight loading | saved `.pth` | inference matches saved metrics | architecture mismatch |
| K=1 | n_clients=1 | Error: K must be ≥2 | Minimum clients |
| K=100 | n_clients=100 | Aggregation scales linearly | Resource limits |
| Empty client | 0 data points | Detected and skipped | Client with no data |
| All adversarial | noise_fraction=1.0 | System fails gracefully | Full adversarial |
| Single data point | N=1 per client | Overfitting expected | Minimum data |
| Mismatched architecture | Different layer sizes | Error: architecture mismatch | Model compatibility |
| Interrupted training | Kill process mid-round | Resume from last checkpoint | Crash recovery |
| Very large K (1000) | n_clients=1000 | Memory/performance test | Scalability |

### 9.3 Acceptance Criteria
1. All 9 aggregators produce valid (finite, converging) training on at least 3 problems.
2. Median/Krum maintain accuracy (L2) when ≤1 client is noisy, where FedAvg degrades.
3. Results are reproducible across ≥3 seeded runs (low variance).
4. The framework runs the full sweep matrix without manual code edits.
5. **Validation gate passed:** the framework's FedAvg reproduces the paper's FedAvg result on the Poisson problem within the defined tolerance before other algorithms are integrated.

---

## 10. Expected Contributions & Findings

### 10.1 Engineering Contributions
1. `src/federated/` — a reusable, config-driven, K-agnostic federated aggregation framework.
2. Nine aggregation algorithms (6 standard + 3 robust).
3. Noise/malicious client simulation and architecture-agnostic weight-divergence metrics.
4. Reproducibility infrastructure (seeds, YAML configs, structured artifact storage).

### 10.2 Research Contributions
1. Benchmark of FL aggregation algorithms on physics-informed and operator-learning problems under controlled W1 heterogeneity.
2. Benchmark of robustness to noisy/malicious clients in federated SciML.
3. New empirical evidence: which aggregation algorithm is most robust to data heterogeneity in PINN/DeepONet — an explicitly stated open problem in the original paper.

### 10.3 Expected Findings (hypotheses)
1. Under clean, low-heterogeneity data, FedAvg/FedAdam achieve the best accuracy.
2. Under high heterogeneity, FedProx/SCAFFOLD outperform FedAvg.
3. Under a single noisy/malicious client, FedAvg collapses while median/Krum maintain accuracy (with a small clean-data accuracy trade-off).
4. Data-size weighted averaging approaches centralized accuracy when client volumes are unequal.
5. **Baseline reproduction (from the paper):** smaller W1 → smaller L2 error for all tasks; federated outperforms the no-communication extrapolation baseline; as W1 → 0.005, federated converges toward centralized performance.
6. **Known nuance from the paper (§7):** weight divergence tracks error change for function approximation and FedPINN, but this trend is **not evident for FedDeepONet** — the benchmark should not assume a monotonic weight-divergence correlation for operator learning.
7. **Known nuance from the paper (§6):** FedDeepONet accuracy is **insensitive to communication frequency** (local epochs E from 1 to 1000 at fixed total iterations) — relevant when comparing aggregators that add communication cost (e.g., SCAFFOLD).

### 10.4 Explicit Deliverable: Robustness–Accuracy Trade-off Curves
A primary output of the benchmark SHALL be a set of trade-off curves — for each problem and algorithm, **accuracy (relative L2 error) plotted against noise_fraction** (0.0 → 0.3). These curves directly quantify the central research question: how much clean-data accuracy each algorithm gives up in exchange for resilience to noisy/malicious clients. They SHALL be presented for the 3 deep-benchmark problems and used in the write-up as the headline comparison figure.

---

## 11. Development Plan (Timeline)

| Phase | Weeks | Deliverable |
|---|---|---|
| **1. Framework core** | 1–3 | `src/federated/` (config, client, server, runner), arbitrary K, seeds, YAML configs, logging; resolve `spaces.py`; run the **validation gate** (FedAvg reproduces paper's Poisson result) |
| **2. Standard aggregators** | 4–6 | FedAvg, FedProx, FedAdam, FedAdagrad, FedYogi, SCAFFOLD; validate on Poisson, Allen-Cahn, antiderivative; run **noise-level pilot** (noise_fraction sweep on one problem) to confirm the robustness effect is visible |
| **3. Robustness** | 7–8 | `robust.py` (median, trimmed_mean, krum), `noise_sim.py`, weighting modes |
| **3.5 Client & Server apps** | 8–9 | `src/apps/server/` (FastAPI REST orchestration) + `src/apps/client/` (dataset loader, local trainer, weights-only upload); protocol schemas (`src/federated/protocol/`); end-to-end client↔server round-trip test |
| **4. Benchmark** | 9–11 | **Deep benchmark on 3 representative problems** (Poisson, Antiderivative, Schaffer): 9 algorithms × {heterogeneity, noise} × seeds; spot-check remaining 7 problems (FedAvg vs FedProx vs Krum); produce trade-off curves |
| **5. Write-up** | 12 | Report, contributions, defense materials; trade-off curve as an explicit deliverable |
| **6. Dashboard (optional)** | 12+ | Streamlit visualization layer |

### 11.1 Project Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| **Inherited repository dependency issues** (e.g., missing `spaces.py`, DeepXDE version pinning) | Blocks operator-learning problems | Vendor/fix missing modules in Phase 1; pin and verify dependencies early |
| **Long experiment runtime** (full sweep matrix is large) | Benchmark may exceed schedule | **Deep-benchmark strategy fixed in §9.1** (full matrix on 3 problems, spot-checks on 7); pilot runs first; noise_fraction sweep early to validate the effect |
| **GPU availability** | Slow training on CPU-only hardware | Framework supports CPU fallback; scale problem sizes/pilot runs to available hardware |
| **Reproducibility risks** (no seeds in original code) | Results hard to defend | Fixed seeds, config snapshots, and environment version recording from Phase 1 |
| **Scope creep** (image support, DP, heterogeneous features) | Schedule overrun | All such items are explicitly out of scope or future work in Section 12 |

### 11.2 Rollback Strategy
If the validation gate fails or critical blockers arise:

| Scenario | Action | Time Budget | Escalation |
|----------|--------|-------------|------------|
| Validation gate fails by <10% | Debug training loop, check data partitioning | 2 days | — |
| Validation gate fails by 10-20% | Check DeepXDE version, loss function, activation | 3 days | Supervisor review |
| Validation gate fails by >20% | Fallback: use DeepXDE's built-in training, document discrepancy | 1 day | Proceed with caveat |
| SCAFFOLD implementation blocks | Skip SCAFFOLD, focus on 8 other aggregators | — | Document as limitation |
| spaces.py cannot be resolved | Implement Chebyshev sampling inline in data_gen scripts | 1 day | — |
| GPU unavailable | Run reduced benchmark (1 problem, fewer seeds) | — | Document limitation |

**Decision rule:** If any critical blocker consumes >5 days, escalate to supervisor and adjust scope.

### 11.3 Go/No-Go Checklist
Before proceeding to Phase 2, verify ALL items:

| Check | Status | Notes |
|-------|--------|-------|
| Section 7 exists in SRS | ☐ | System Interfaces added |
| No duplicate requirement IDs | ☐ | NFR-SEC-3 fixed |
| Validation gate tolerance defined (±5%) | ☐ | FR-RUN-5 updated |
| spaces.py resolved | ☐ | Phase 1 deliverable |
| Runtime estimates added | ☐ | Section 9 updated |
| Example YAML configs in Appendix | ☐ | Appendix A added |
| SCAFFOLD complexity acknowledged | ☐ | Phase 2 timeline updated |
| Rollback strategy documented | ☐ | Section 11.2 added |
| NaN error handling specified | ☐ | FR-CLIENT-7 added |
| Data partition validation specified | ☐ | FR-DATA-6 added |

**Decision:** If any ☐ remains unchecked, do NOT proceed to Phase 2. Fix the issue first.

---

## 12. Future Work & Extensions
1. **Differential privacy** — Gaussian noise on clipped gradients, ε accounting (RDP); enables a privacy-compliant deployment story. Note: gradient clipping (FR-CLIENT-9) is already supported as the first step toward DP.
2. **Heterogeneous features across clients** — feature alignment / FedMD-style knowledge distillation for clients measuring different physical quantities.
3. **Image/CNN support** — U-Net/CNN backbones, image partitioning, image-to-physics operators (medical imaging).
4. **Personalized FL** — Per-FedAvg / FedBN with per-client heads on shared physics backbone.
5. **Partial participation & asynchronous FL** — client sampling (C<100%), communication-cost accounting.
6. **Continuous federated learning** — model updates under distribution drift and growing participant pools (network effect).
7. **Communication-efficient FL** — gradient quantization (4-bit/2-bit), sparse communication (top-k%), one-shot FL with knowledge distillation, adaptive compression ratios, model pruning for edge deployment.

---


---

## Appendix A: Example YAML Configurations

### A.1 FedAvg on Poisson (Validation Gate)
```yaml
problem: poisson
model_family: fnn
n_clients: 2
aggregator: fedavg
aggregation_weights: uniform
local_epochs: 5
global_rounds: 1000
heterogeneity:
  mode: 1d_partition
  n_pieces: 10
learning_rate: 0.001
optimizer: adam
seed: 42
noise_mode: none
noise_fraction: 0.0
output_dir: results/poisson/fedavg/
```

### A.2 FedProx with High Heterogeneity
```yaml
problem: poisson
model_family: fnn
n_clients: 2
aggregator: fedprox
mu: 0.01
aggregation_weights: uniform
local_epochs: 5
global_rounds: 1000
heterogeneity:
  mode: 1d_partition
  n_pieces: 2  # High heterogeneity
learning_rate: 0.001
optimizer: adam
seed: 42
noise_mode: none
noise_fraction: 0.0
output_dir: results/poisson/fedprox_highhet/
```

### A.3 Krum with Adversarial Clients
```yaml
problem: schaffer
model_family: fnn
n_clients: 5
aggregator: krum
aggregation_weights: uniform
local_epochs: 5
global_rounds: 3000
heterogeneity:
  mode: xy_partition
  n_pieces: 5
learning_rate: 0.001
optimizer: adam
seed: 42
noise_mode: adversarial
noise_fraction: 0.2  # 1 out of 5 clients is malicious (0.2 × 5 = 1)
output_dir: results/schaffer/krum_adversarial/
```

### A.4 Performance Sweep (Optimizer Comparison)
```yaml
sweep:
  problem: [poisson, antiderivative, schaffer]
  aggregator: [fedavg, fedprox, krum]
  optimizer: [adam, lion]
  n_clients: 2
  heterogeneity:
    mode: 1d_partition
    n_pieces: 10
  seeds: [42, 43, 44]
```

---

## Appendix A.5: Gradient Clipping Configuration

Gradient clipping bounds gradient magnitude for **training stability and outlier mitigation**. 
Note: clipping alone does NOT provide formal privacy guarantees — differential privacy 
requires calibrated noise addition on top of clipping (documented as future work in §12.1).

```yaml
# Example: FedAvg with gradient clipping for training stability
problem: poisson
aggregator: fedavg
n_clients: 2
local_epochs: 5
global_rounds: 1000

# Gradient clipping (training stability / outlier bounding)
gradient_clip: norm        # options: none, value, norm
max_norm: 1.0              # for norm clipping (L2 norm)
# clip_value: 0.5          # for value clipping (per-element)

# Combined with robust aggregation for stronger robustness
# aggregator: krum         # Byzantine-robust
# noise_mode: adversarial  # Test against attacks
```

**Gradient Clipping Options:**

| Option | What It Does | When to Use |
|--------|--------------|-------------|
| `none` | No clipping | Default, clean data |
| `value` | Clip each gradient element to `[-clip_value, clip_value]` | Bound outlier gradients |
| `norm` | Clip L2 norm of gradient vector to `max_norm` | Better for federated averaging |

**Note:** Gradient clipping is a precursor to differential privacy (§12.1). 
When combined with calibrated Gaussian noise, it enables ε-differential privacy. 
This combination is NOT implemented in the current version.

---

## Appendix A.6: Weighting Modes

```yaml
# Example: FedAvg with quality-based weighting
problem: poisson
aggregator: fedavg
n_clients: 5
aggregation_weights: quality   # options: uniform, data_size, quality
local_epochs: 5
global_rounds: 1000
```

**Weighting Modes:**

| Mode | Formula | When to Use |
|------|---------|-------------|
| `uniform` | `w_k = 1/K` | Default, equal client trust |
| `data_size` | `w_k = N_k / N` | Unequal data volumes |
| `quality` | `w_k = (1/L_k) / Σ(1/L_i)` | Varying client data quality |

**Quality Weighting Details:**
- `L_k` = validation loss of client k's local model
- Validation set = 20% held-out shard from each client's local data
- Lower validation loss → higher weight in aggregation
- Computed fresh each round after local training

---

*End of SRS — Version 1.4*