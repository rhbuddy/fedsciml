"""Orient Client — Streamlit UI.

Minimal, clean front-end over ``client/app``: load a private dataset bundle,
register with the server, and run the federated training loop while watching
the local loss. Raw data never leaves the machine; only weights are uploaded.

Run (from the ``client/`` directory)::

    streamlit run ui/app.py
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.dataset import load_bundle  # noqa: E402
from app.protocol import ClientRegistration  # noqa: E402
from app.trainer import LocalTrainer  # noqa: E402
from app.transport import ServerClient, ServerConnectionError  # noqa: E402
from app.weights import bytes_to_state_dict, state_dict_to_bytes  # noqa: E402

EXAMPLES_DIR = ROOT / "examples" / "bundles"

st.set_page_config(page_title="Orient Client", page_icon="🧠", layout="wide")

# --------------------------------------------------------------------------- state
defaults = {"bundle": None, "logs": [], "history": [], "connected": False, "rounds_done": 0}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)


def log(message: str) -> None:
    st.session_state.logs.append(f"[{time.strftime('%H:%M:%S')}] {message}")


# --------------------------------------------------------------------------- header
st.title("Orient · Client")
st.caption(
    "Train on your **private** dataset and share **weights only** — "
    "raw data never leaves this machine."
)

# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Connection")
    server_url = st.text_input("Server URL", "http://127.0.0.1:8000")
    client_id = st.text_input("Client ID", "client-1")

    st.header("Local training")
    batch_size = st.number_input("Batch size", 16, 4096, 256, step=16)
    poll_interval = st.slider("Poll interval (s)", 0.5, 10.0, 2.0, step=0.5)
    grad_clip = st.number_input("Gradient clip norm (0 = off)", 0.0, 100.0, 0.0, step=0.5)

    st.header("Dataset source")
    source = st.radio("Where is your dataset?", ["Example bundle", "Upload files"], index=0)

    manifest_file = arrays_file = example_name = None
    if source == "Example bundle":
        available = sorted(p.name for p in EXAMPLES_DIR.iterdir()) if EXAMPLES_DIR.exists() else []
        if available:
            example_name = st.selectbox("Example bundles", available)
        else:
            st.info("No example bundles yet. Run:\n\n`python examples/make_datasets.py`")
    else:
        st.caption("Expects `dataset.json` + `data.npz` (§4.13).")
        manifest_file = st.file_uploader("dataset.json", type="json")
        arrays_file = st.file_uploader("data.npz", type="npz")

    if st.button("Load dataset", use_container_width=True):
        try:
            if source == "Example bundle":
                if not example_name:
                    raise ValueError("No example bundle selected")
                bundle = load_bundle(EXAMPLES_DIR / example_name)
            else:
                if not (manifest_file and arrays_file):
                    raise ValueError("Please upload both dataset.json and data.npz")
                tmp = Path(tempfile.mkdtemp(prefix="orient-dataset-"))
                (tmp / "dataset.json").write_bytes(manifest_file.getvalue())
                (tmp / "data.npz").write_bytes(arrays_file.getvalue())
                bundle = load_bundle(tmp)
            st.session_state.bundle = bundle
            st.session_state.logs = []
            log(f"Loaded dataset '{bundle.problem}' with {bundle.size} samples")
            st.success("Dataset loaded.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not load dataset: {exc}")

# --------------------------------------------------------------------------- main
bundle = st.session_state.bundle
col1, col2, col3, col4 = st.columns(4)
col1.metric("Problem", bundle.problem if bundle else "—")
col2.metric("Local samples", f"{bundle.size:,}" if bundle else "—")
col3.metric("Rounds trained", st.session_state.rounds_done)
col4.metric("Status", "connected" if st.session_state.connected else "offline")

tab_train, tab_preview, tab_log = st.tabs(["Train", "Data preview", "Log"])

with tab_preview:
    if bundle is None:
        st.info("Load a dataset to preview it.")
    else:
        arrays = bundle.arrays
        st.write("**Arrays**")
        st.dataframe(
            pd.DataFrame(
                [{"array": k, "shape": str(v.shape), "dtype": str(v.dtype)} for k, v in arrays.items()]
            ),
            use_container_width=True,
            hide_index=True,
        )
        plot_key = next((k for k in ("x_train", "branch_train") if k in arrays), None)
        if plot_key is not None and arrays[plot_key].ndim == 2 and arrays[plot_key].shape[1] in (1, 2):
            cols = ["x"] if arrays[plot_key].shape[1] == 1 else ["x1", "x2"]
            df = pd.DataFrame(np.asarray(arrays[plot_key]), columns=cols)
            st.write(f"**{plot_key}** — {bundle.size:,} points")
            if len(cols) == 2:
                st.scatter_chart(df, x="x1", y="x2", height=320)
            else:
                st.line_chart(df.set_index("x"), height=320)

with tab_train:
    if bundle is None:
        st.info("Load a dataset first (sidebar).")
    else:
        if st.button("Connect & register", disabled=st.session_state.connected):
            try:
                with ServerClient(server_url) as server:
                    server.health()
                    info = server.register(
                        ClientRegistration(
                            client_id=client_id,
                            problem=bundle.problem,
                            n_samples=bundle.size,
                            domain=bundle.domain,
                        )
                    )
                st.session_state.connected = True
                log(f"Registered '{info.client_id}' with {server_url}")
                st.success(f"Registered as **{info.client_id}**.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not connect: {exc}")

        start = st.button(
            "▶ Start federated training",
            type="primary",
            disabled=not st.session_state.connected,
            use_container_width=True,
        )

        status = st.empty()
        progress = st.progress(0.0, text="idle")
        chart = st.empty()

        if start:
            trainer = LocalTrainer(bundle.problem)
            try:
                with ServerClient(server_url) as server:
                    for _ in range(2000):  # safety bound
                        resp = server.assignment(client_id)
                        if resp.status in ("done", "error"):
                            if resp.status == "error":
                                status.error(f"Server error: {resp.message}")
                                log(f"ERROR {resp.message}")
                                break
                            if st.session_state.rounds_done == 0:
                                # A previous run already finished; keep waiting so we
                                # do not leave a stale entry blocking the next run.
                                status.info(
                                    "Waiting for a new run to start "
                                    "(previous run already completed)."
                                )
                                time.sleep(poll_interval)
                                continue
                            status.success(resp.message)
                            break
                        if resp.status == "wait":
                            status.info(resp.message or "Waiting for the server…")
                            time.sleep(poll_interval)
                            continue

                        if resp.model_spec is None or resp.problem_spec is None or resp.optimizer is None:
                            status.error("Malformed assignment from server.")
                            log(f"ERROR malformed assignment: {resp.model_dump()}")
                            break

                        trainer.assert_spec_compatible(resp.model_spec)
                        raw = server.download_weights(client_id, resp.round)
                        trainer.load_weights(bytes_to_state_dict(raw))

                        status.info(
                            f"Round **{resp.round}/{resp.total_rounds}** — "
                            f"training {resp.local_epochs} epoch(s) on {bundle.size:,} samples"
                        )
                        stats = trainer.train(
                            bundle.arrays,
                            epochs=resp.local_epochs,
                            optimizer_spec=resp.optimizer,
                            batch_size=int(batch_size),
                            prox_mu=resp.prox_mu,
                            grad_clip=float(grad_clip) or None,
                        )
                        payload = state_dict_to_bytes(trainer.state_dict())
                        server.upload_weights(
                            client_id, resp.round, bundle.size, stats["loss"], payload
                        )

                        st.session_state.rounds_done += 1
                        st.session_state.history.append(
                            {"round": resp.round, "local_loss": stats["loss"]}
                        )
                        progress.progress(
                            min(1.0, resp.round / max(1, resp.total_rounds)),
                            text=f"round {resp.round}/{resp.total_rounds} · loss {stats['loss']:.4g}",
                        )
                        chart.line_chart(
                            pd.DataFrame(st.session_state.history).set_index("round")
                        )
                        log(
                            f"round {resp.round}: loss={stats['loss']:.6g} "
                            f"({len(payload)} bytes uploaded)"
                        )
            except ServerConnectionError as exc:
                st.error(str(exc))
                st.session_state.connected = False
            except Exception as exc:  # noqa: BLE001
                st.error(f"Training failed: {exc}")

with tab_log:
    if st.session_state.logs:
        st.code("\n".join(st.session_state.logs), language="text")
    else:
        st.info("No activity yet.")


