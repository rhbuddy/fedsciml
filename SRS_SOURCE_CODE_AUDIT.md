# SRS-Based Source Code Audit — `orient/`

Date: 2026-10-05

## Follow-up fix pass applied

After this audit, I applied a bug-fix pass to `orient/` covering several of the high-priority findings:

- canonicalized FedAdagrad to the SRS name `fedadagrad` while preserving legacy alias `fedagrad`;
- changed default aggregation weighting to `data_size`;
- added non-finite tensor/loss/gradient validation on both server and client paths;
- prevented single-client aggregation and made clients wait until at least two clients are available;
- fixed client registration/run-start behavior so clients from a previous/different problem or completed run do not stall new runs;
- aligned the currently implemented Gramacy-Lee, Schaffer, and Antiderivative model specs with the SRS paper defaults;
- made result run directories collision-resistant;
- added stricter dataset validation for NaN/Inf arrays.

The larger research-framework gaps noted below still remain, especially SCAFFOLD, the full 10-problem registry, YAML sweeps, W1 metrics, noisy/adversarial simulation, baselines, and weight-divergence artifacts.

## Scope

I reviewed the `orient/` source code against:

- `SRS_FederatedSciML_Project.md` v1.4
- `federated_sciml_notes.md`
- the extracted main-paper text in `fed reference/00_main_paper.txt` / `fed reference/federated_sciml_main file.txt` corresponding to `federated_sciml.pdf`
- reference summaries under `fed reference/`

The focus was whether the current source implements the SRS requirements and remains aligned with the FedSciML paper: FedAvg-Adam, FedPINN, FedDeepONet, data heterogeneity via W1, and the 10-problem benchmark plan.

## Checks run

```bash
find orient -maxdepth 4 -type f | sort
python tests/test_static_imports.py   # from orient/
diff -u orient/backend/app/{protocol,weights,models,problems}.py orient/client/app/...
```

Result:

- Static import check passed: `All relative imports resolve to defined names. OK`.
- Shared client/server modules are byte-identical for `protocol.py`, `weights.py`, `models.py`, and `problems.py`.
- Runtime tests could not be executed in this sandbox because required packages are not installed (`torch`, `numpy`, `fastapi`, `pydantic`, `safetensors`, `pytest`, etc.).

## Overall conclusion

`orient/` is a good **two-app federated learning prototype** and implements much of SRS §§4.10–4.13: a FastAPI server, standalone client, safetensors weight exchange, Pydantic protocol schemas, model-spec compatibility checks, and private local dataset bundles.

However, it is **not yet compliant with the full SRS v1.4 benchmark framework**. The current implementation is closer to a Phase-1/Phase-2 distributed demo than the full robust multi-algorithm SciML benchmark specified in the SRS.

Major gaps:

1. SCAFFOLD is not implemented, so the SRS-required 9 aggregators are incomplete.
2. The adaptive optimizer is named `fedagrad` in code instead of the SRS name `fedadagrad`.
3. Only 4 problems are registered in the new app layer, while the SRS requires the 10 paper problems.
4. YAML config loading, sweep runner, validation-gate command, W1 heterogeneity metrics, noisy/adversarial client simulation, centralized/extrapolation baselines, and weight-divergence outputs are not implemented in the new framework.
5. Several paper/SRS default model settings are simplified or mismatched, especially Gramacy-Lee, Schaffer, and Antiderivative.
6. Storage artifacts do not match SRS §8.2: code writes `config.json`, `metrics.jsonl`, and `final_model.safetensors`, but not `config.yaml`, `server_best.pth`, `loss.npz`, `l2_error.npz`, `weight_divergence.npz`, or `metrics.json`.

---

## High-priority findings

### 1. Aggregator list is incomplete / naming mismatch

**SRS requirements:** FR-AGG-4 through FR-AGG-10 require:

- `fedavg`
- `fedprox`
- `fedadam`
- `fedadagrad`
- `fedyogi`
- `scaffold`
- `median`
- `trimmed_mean`
- `krum`

**Current source:** `orient/backend/app/aggregators.py`

```python
STANDARD_AGGREGATORS = ["fedavg", "fedprox", "fedadam", "fedagrad", "fedyogi"]
ROBUST_AGGREGATORS = ["median", "trimmed_mean", "krum"]
```

Status: **partial**

Issues:

- `scaffold` is missing.
- SRS says `fedadagrad`; code exposes `fedagrad`.
- README and tests also use `fedagrad`.

Recommended fix:

- Add an alias so both `fedadagrad` and `fedagrad` work, but document `fedadagrad` as the SRS-compliant name.
- Add SCAFFOLD only after designing persistent client/server control variate storage.

---

### 2. Paper/SRS default FedAvg weighting mismatch

**SRS requirement:** FR-AGG-2 says FedAvg should default to data-size proportional weighting:

```text
θ* = Σ_k (Nk/N) · θ_k
```

**Current source:** `RunStartRequest` in `orient/backend/app/protocol.py` defaults to:

```python
weighting: str = Field(default="uniform", description="uniform | data_size | quality")
```

Status: **partial**

The `data_size` mode exists and works, but the default does not match the SRS/paper default.

Recommended fix:

- Change default from `uniform` to `data_size`, or make documentation explicit that `uniform` is a demo default and `data_size` must be selected for paper/SRS reproduction.

---

### 3. Only 4 of the 10 required paper problems are integrated

**SRS requirement:** FR-RUN-3 requires the existing 10 problems:

- Function approximation: Gramacy-Lee, Schaffer
- FedPINN: Poisson, Helmholtz, Allen-Cahn, Inverse Navier-Stokes, Inverse Diffusion-Reaction
- FedDeepONet: Antiderivative, Burgers, Diffusion-Reaction

**Current source:** `orient/backend/app/problems.py` / `orient/client/app/problems.py`

```python
_PROBLEM_CLASSES = [GramacyLee1D, Schaffer2D, Poisson1D, Antiderivative]
```

Status: **partial**

Implemented in new app layer:

- `gramacy_lee`
- `schaffer`
- `poisson`
- `antiderivative`

Missing from new app layer:

- Helmholtz
- Allen-Cahn
- Inverse Navier-Stokes
- Inverse Diffusion-Reaction
- Burgers DeepONet
- Diffusion-Reaction DeepONet

Important note: the untouched upstream code under `orient/federated-sciml-main/` contains many of these original DeepXDE scripts, but they are not wrapped by the new server/client/problem registry.

Recommended fix:

- Add problem adapters in `problems.py` or a new problem package that wraps the upstream DeepXDE implementations.
- Keep `ModelSpec`/`ProblemSpec` deterministic so client and server still build matching models.

---

### 4. SRS YAML configuration and sweep runner are missing

**SRS requirements:** FR-CFG-1..6 and FR-RUN-1..5 require:

- loading a YAML experiment config
- validating config values
- sweep syntax over lists
- saving exact config content beside results
- one-command single experiment runner
- multi-dimensional benchmark sweeps
- validation-gate command for FedAvg Poisson reproduction

**Current source:**

- Server accepts JSON body through `/admin/run/start`.
- Config settings are environment variables in `backend/app/config.py`.
- No `configs/` directory, YAML loader, sweep runner, or validation-gate command exists in `orient/`.

Status: **not implemented**

Recommended fix:

- Add `backend/app/experiment_config.py` or a shared `runner.py`.
- Add `orient/configs/*.yaml` examples matching Appendix A of the SRS.
- Add commands such as:
  - `python -m app.runner --config configs/poisson_fedavg.yaml`
  - `python -m app.runner --sweep configs/deep_benchmark.yaml`
  - `python -m app.runner --validation-gate poisson`

---

### 5. Data heterogeneity/W1 requirements are not implemented

**SRS requirements:** FR-DATA-2..7 require:

- specific 1D/2D/Chebyshev partitioning schemes
- heterogeneity levels low/medium/high
- W1 computation and recording
- partition coverage validation and stats
- W1 manual validation test

**Current source:** `orient/client/examples/make_datasets.py`

The demo generator creates heterogeneous bundles by slicing domains or selecting Fourier modes, but:

- no W1 is computed
- no heterogeneity level abstraction exists
- no `n_pieces` / `n_terms` configuration exists
- no partition coverage report is written
- operator partitioning is simplified random Fourier-mode selection, not the paper/SRS Chebyshev forward/middle/inverse setup

Status: **partial demo only**

Recommended fix:

- Add `data_partition.py` with SRS partitioning methods.
- Add `metrics.py` with W1 computation.
- Record W1 and partition stats in run config/metrics.

---

### 6. Noise / malicious-client simulation is missing

**SRS requirements:** FR-NOISE-1..4 require:

- `noise_mode: none / noisy / adversarial`
- configurable compromised fraction
- random/noisy/malicious weight corruption
- logging compromised clients

**Current source:**

- Robust aggregators exist: `median`, `trimmed_mean`, `krum`.
- No noise/adversarial update injection exists in the new backend/client path.
- No run field exists for `noise_mode` or `noise_fraction`.

Status: **not implemented**

Recommended fix:

- Add a server-side `noise_sim.py` for simulation mode, or a client-side flag to intentionally corrupt selected clients.
- Log compromised clients in each metrics record.

---

### 7. NaN/Inf gradient and corrupted-update handling is incomplete

**SRS requirements:** FR-CLIENT-7 and FR-CLIENT-8 require detecting NaN/Inf gradients during local training, excluding corrupted clients, and logging exclusions with timestamps/rounds.

**Current source:**

- `LocalTrainer.train()` does not check gradients for NaN/Inf before optimizer step.
- Server validates state-dict keys/shapes but does not appear to reject non-finite tensor values.
- No exclusion log is written.

Status: **not implemented / partial shape validation only**

Recommended fix:

- In `trainer.py`, check `torch.isfinite(loss)` and gradient tensors after `loss.backward()`.
- In `orchestrator.submit_update()`, reject state dicts containing NaN/Inf.
- Add exclusion records to `metrics.jsonl`.

---

### 8. Evaluation and benchmark outputs are incomplete

**SRS requirements:** FR-EVAL-1..4 and §8.2 require:

- relative L2 error
- centralized baseline
- extrapolation baseline
- per-layer weight divergence
- comparison tables
- `loss.npz`, `l2_error.npz`, `weight_divergence.npz`, `metrics.json`

**Current source:**

- `Problem.evaluate()` returns `l2_relative_error` for registered problems.
- No centralized baseline or extrapolation baseline is computed by the new app.
- No generic weight-divergence module exists.
- Storage writes `config.json`, `metrics.jsonl`, `final_model.safetensors` only.

Status: **partial**

Recommended fix:

- Add `metrics.py` with relative L2, weight divergence, and baseline comparisons.
- Extend `storage.py` to write the SRS-required artifact set.
- Add a post-processing command to generate algorithm comparison tables.

---

### 9. Model/paper setting mismatches

From the SRS Table 4 / paper notes, default architecture settings should match the paper for reproduction.

Current app-layer model specs:

| Problem | SRS/paper expectation | Current new app code | Status |
|---|---|---|---|
| Gramacy-Lee | width 64, depth 3, tanh | `[1,32,32,32,1]` tanh | mismatch |
| Schaffer | width 64, depth 3, tanh | `[2,48,48,48,1]` tanh | mismatch |
| Poisson | `[1,20,20,20,1]`, tanh, hard constraint | matches closely | good |
| Antiderivative | DeepONet width 40, depth 2, ReLU per SRS table | branch/trunk `[*,64,64]`, tanh | mismatch |

Status: **partial**

Recommended fix:

- Separate `demo` model specs from `paper_reproduction` model specs, or update defaults to match the SRS.
- Ensure validation-gate Poisson uses the exact paper settings.

---

### 10. Optimizer configurability is incomplete

**SRS requirement:** FR-CLIENT-2 and FR-AGG-13 mention configurable local optimizers such as Adam, SGD, Lion and per-aggregator overrides.

**Current source:**

- `AssignmentResponse.optimizer` exists.
- Server always sends `OptimizerSpec(name="adam", lr=...)`.
- Client always builds `torch.optim.Adam`, ignoring `optimizer_spec.name`.

Status: **partial**

Recommended fix:

- Implement optimizer factory supporting at least Adam and SGD.
- Only add Lion if dependency/version supports it or document it as unsupported.

---

### 11. Gradient clipping is only CLI-based, not full SRS config mode

**SRS requirement:** FR-CLIENT-9 requires `gradient_clip: none / value / norm` with `max_norm` or `clip_value`.

**Current source:**

- CLI has `--grad-clip` as a max norm.
- No value clipping mode.
- No server/run config field for clipping.

Status: **partial**

Recommended fix:

- Add clipping fields to `RunStartRequest` and `AssignmentResponse`.
- Implement both norm and value clipping.

---

## Requirement traceability summary

| SRS area | Status | Evidence |
|---|---:|---|
| FR-CFG: YAML config/sweeps | Not implemented | JSON REST only; env settings only |
| FR-DATA: reuse data scripts/partition/W1 | Partial | upstream scripts present; new app has simple demo bundles; no W1 |
| FR-CLIENT: K clients, local training, weights-only | Partial/good | client CLI and trainer exist; optimizer/NaN/clip modes incomplete |
| FR-AGG: 9 aggregators | Partial | 8 implemented; no SCAFFOLD; `fedadagrad` naming mismatch |
| FR-NOISE | Not implemented | no `noise_mode`/`noise_fraction` pipeline |
| FR-EVAL | Partial | relative L2 only; no baselines/divergence/tables |
| FR-STORE | Partial | SRS directory shape mostly yes; artifact names/content differ |
| FR-RUN | Not implemented | no YAML runner, sweeps, validation gate |
| FR-DASH optional | Partial | Streamlit dashboard/client UI exist, but not runtime-verified here |
| FR-CLIENTAPP | Mostly implemented | standalone CLI, dataset bundles, register/poll/train/upload |
| FR-SERVERAPP | Mostly implemented | FastAPI endpoints and synchronous rounds exist |
| FR-PROTO | Mostly implemented | Pydantic specs and safetensors; shared modules identical |
| Dataset format §4.13 | Mostly implemented | `dataset.json` + `data.npz`; array validation exists |
| NFR reproducibility | Partial | seed recorded/applied for server model; no full config snapshot/sweep reproducibility |
| NFR security/privacy | Partial | weights-only, safetensors; plain HTTP as SRS allows; no DP/secure aggregation |

---

## Positive findings

The following parts are in good shape:

- Clear two-app split: `orient/backend` and `orient/client`.
- REST endpoints match the SRS server-app endpoint list, plus useful extras.
- Client sends weights only; raw arrays remain local.
- `safetensors` is used instead of pickle for the wire format.
- Shared protocol/model/problem/weight modules are identical between client and server.
- Server validates uploaded state-dict keys/shapes before accepting updates.
- FedProx proximal term is implemented client-side and remains in the autograd graph.
- Server-side adaptive aggregation for FedAdam/FedAdagrad/FedYogi style methods exists.
- K is dynamic; aggregation code is not hardcoded to 2 or 3 clients.
- Dataset bundles are validated for required arrays, 2-D shape, emptiness, numeric dtype, and basic sample alignment.

---

## Recommended implementation order

1. **SRS naming/default quick fixes**
   - Add `fedadagrad` alias.
   - Consider changing default weighting to `data_size`.
   - Reject K < 2 at run start unless explicitly in demo mode.

2. **Robustness safety checks**
   - Detect non-finite loss/gradients in the client.
   - Reject non-finite uploaded tensors on the server.
   - Log excluded clients.

3. **Config and runner foundation**
   - Add YAML config loader and validator.
   - Add single-run CLI.
   - Save exact YAML config into results.

4. **Metrics required by the paper/SRS**
   - Add W1 heterogeneity metric.
   - Add weight divergence.
   - Add partition stats.

5. **Problem coverage**
   - Wrap remaining 6 paper problems from `federated-sciml-main/`.
   - Align model specs with the SRS paper settings.

6. **Benchmark features**
   - Add noisy/adversarial simulation.
   - Add sweeps and comparison tables.
   - Add Poisson FedAvg validation-gate command.

7. **SCAFFOLD**
   - Implement after runner/protocol are stable because it needs persistent client/server control variates.

## Bottom line

The current `orient/` code is a solid distributed FL demo and a partial implementation of the SRS two-app deployment. It should not yet be presented as the complete SRS v1.4 project because the main benchmark/research requirements — full 10-problem suite, 9 aggregators including SCAFFOLD, W1 heterogeneity, noise/adversarial tests, YAML sweeps, validation gate, baselines, and weight-divergence analysis — are still missing or only partially implemented.
