# Orient · Client App

The federated **client**: loads a **private** local dataset, downloads the
global model, trains locally, and uploads **weights only**. Raw data never
leaves the machine.
Implements `SRS_FederatedSciML_Project.md` v1.4 §4.10 (`FR-CLIENTAPP-1..9`)
and the dataset bundle format in §4.13.

## Run the CLI

```powershell
# from this directory, with the venv activated
python -m app.main --server http://127.0.0.1:8000 --client-id client-1 --dataset examples\bundles\poisson_client1
```

| Flag | Default | Meaning |
|---|---|---|
| `--server` | *required* | server base URL |
| `--client-id` | *required* | unique id for this client |
| `--dataset` | *required* | path to a dataset bundle directory |
| `--problem` | from `dataset.json` | problem name override |
| `--batch-size` | 256 | local minibatch size |
| `--poll-interval` | 2.0 | seconds between assignment polls |
| `--grad-clip` | off | max gradient norm (lightweight privacy step) |
| `--once` | off | train a single round, then exit |
| `--max-wait` | 0 (forever) | give up waiting after N seconds |

The model and the Adam optimizer persist across rounds, so momentum carries
over instead of being reset every round.

## Run the UI

```powershell
streamlit run ui\app.py        # or .\run.ps1 ui
```

Load a bundle (example or your own files), press **Connect & register**, then
**▶ Start federated training**. The Train tab shows the round, a progress bar,
and a live chart of the local loss; the Data preview tab shows the arrays and a
scatter/line plot of the inputs.

## Build a dataset bundle

```
mydata/
├── dataset.json   {"problem": "poisson", "n_samples": 1500, "domain": {"interval": [0.0, 1.5708]}}
└── data.npz       x_train                                (supervised & pinn)
                   branch_train, trunk_train, y_train     (operator)
```

Requirements by family:

| Problem | Required arrays |
|---|---|
| `gramacy_lee`, `schaffer` | `x_train` (n, d), `y_train` (n, 1) |
| `poisson` | `x_train` (n, 1) collocation points |
| `antiderivative` | `branch_train` (n, 50), `trunk_train` (m, 1), `y_train` (n, m) |

The bundle is validated against the problem registry on load, so a mismatched
bundle fails immediately with a clear message rather than silently training
garbage.

Generate ready-made heterogeneous examples (each client gets a different slice
of the domain / different Fourier modes):

```powershell
python examples\make_datasets.py
```

## Modules

| File | Responsibility |
|---|---|
| `app/main.py` | CLI entry point + the register → poll → train → upload loop |
| `app/dataset.py` | load / validate / save dataset bundles |
| `app/trainer.py` | local training, Adam, optional FedProx term, gradient clipping |
| `app/transport.py` | HTTP client for the server REST API |
| `app/protocol.py` `app/weights.py` `app/models.py` `app/problems.py` | **verbatim copies** of the server's — keep them identical |
| `ui/app.py` | Streamlit UI |
| `examples/make_datasets.py` | generates heterogeneous example bundles |

> The four shared modules guarantee the client rebuilds the exact same
> architecture (identical `state_dict` keys/shapes) and computes the same loss
> as the server. If you edit one, copy it to the other.