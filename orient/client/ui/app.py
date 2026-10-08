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

st.set_page_config(page_title="Orient Client", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
  .client-header {font-size:2rem; font-weight:700; background: linear-gradient(90deg,#0ea5e9,#6366f1); -webkit-background-clip:text; -webkit-text-fill-color:transparent;}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------- state
defaults = {"bundle": None, "logs": [], "history": [], "connected": False, "rounds_done": 0}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)


def log(message: str) -> None:
    st.session_state.logs.append(f"[{time.strftime('%H:%M:%S')}] {message}")


# --------------------------------------------------------------------------- header
st.markdown('<div class="client-header">🧠 Orient · Client</div>', unsafe_allow_html=True)
st.caption(
    "Train on your **private** dataset and share **weights only** — "
    "raw data never leaves this machine. Supports all 10 benchmark problems & 9 aggregators (SRS v1.4)."
)
st.caption("Privacy: dataset bundle stays local → only `safetensors` weights uploaded. Robust to noisy/adversarial peers via server-side median/krum/trimmed_mean.")

# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Connection")
    server_url = st.text_input("Server URL", "http://127.0.0.1:8000")
    client_id = st.text_input("Client ID", "client-1", help="Unique ID (client-1..K). For Krum: K≥5 recommended")

    st.header("Local training")
    batch_size = st.number_input("Batch size", 16, 4096, 256, step=16)
    poll_interval = st.slider("Poll interval (s)", 0.5, 10.0, 2.0, step=0.5)
    grad_clip = st.number_input("Gradient clip norm (0 = off)", 0.0, 100.0, 0.0, step=0.5, help="Local clipping; server may also enforce norm/value clipping (FR-CLIENT-9)")
    st.caption("Follows server assignment: `gradient_clip: none|norm|value` from YAML (e.g., `poisson_fedavg.yaml`).")

    st.header("Dataset source")
    source = st.radio("Where is your dataset?", ["Example bundle", "Upload files"], index=0)

    manifest_file = arrays_file = example_name = None
    if source == "Example bundle":
        available = sorted(p.name for p in EXAMPLES_DIR.iterdir()) if EXAMPLES_DIR.exists() else []
        if available:
            # Group by problem for nicer UX
            probs = sorted({n.split("_client")[0] for n in available})
            sel_prob = st.selectbox("Filter by problem", ["all"] + probs)
            filtered = [n for n in available if sel_prob == "all" or n.startswith(sel_prob)]
            example_name = st.selectbox(f"Bundles ({len(filtered)} shown)", filtered)
            if example_name:
                # show bundle preview hint
                try:
                    tmp_b = load_bundle(EXAMPLES_DIR / example_name)
                    st.caption(f"→ `{tmp_b.problem}` · {tmp_b.size} samples · {', '.join(tmp_b.arrays.keys())}")
                except Exception:
                    pass
        else:
            st.info("No example bundles yet. Run:\n\n`python examples/make_datasets.py --clients 5`  (creates all 10 problems)")
    else:
        st.caption("Expects `dataset.json` + `data.npz` (§4.13). Validation per FR-CLIENTAPP-8.")
        manifest_file = st.file_uploader("dataset.json", type="json")
        arrays_file = st.file_uploader("data.npz", type="npz")

    if st.button("Load dataset", use_container_width=True, type="primary"):
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
            st.session_state.history = []
            st.session_state.rounds_done = 0
            log(f"Loaded dataset '{bundle.problem}' with {bundle.size} samples ({EXAMPLES_DIR / example_name if example_name else 'upload'})")
            st.success(f"✓ Dataset loaded: **{bundle.problem}** · {bundle.size} samples")
            st.toast(f"Loaded {bundle.problem} bundle", icon="✅")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not load dataset: {exc}")

# --------------------------------------------------------------------------- main
bundle = st.session_state.bundle
col1, col2, col3, col4 = st.columns(4)
col1.metric("Problem", bundle.problem if bundle else "—", help="10 problems: Gramacy, Schaffer, Poisson, Helmholtz, Allen-Cahn, Inverse NS/DR, Antiderivative, Burgers, Diffusion-Reaction")
col2.metric("Local samples", f"{bundle.size:,}" if bundle else "—")
col3.metric("Rounds trained", st.session_state.rounds_done)
col4.metric("Status", "● connected" if st.session_state.connected else "○ offline")

if bundle:
    st.caption(f"Bundle: `{bundle.path}` · arrays: {', '.join(f'{k}{list(v.shape)}' for k,v in bundle.arrays.items())}  ·  domain: {bundle.domain}")

tab_train, tab_preview, tab_log = st.tabs(["🏋️ Train", "🔍 Data preview", "📝 Log"])

with tab_preview:
    if bundle is None:
        st.info("Load a dataset to preview it.  Supports: `x_train,y_train` (supervised/PINN) and `branch_train,trunk_train,y_train` (operator).")
        st.markdown("**Generate all 10:** `python examples/make_datasets.py --clients 5`  — heterogeneous slices per SRS §4.2 (W1 heterogeneity)")
    else:
        arrays = bundle.arrays
        st.write("**Arrays**")
        st.dataframe(
            pd.DataFrame(
                [{"array": k, "shape": str(v.shape), "dtype": str(v.dtype), "mean": f"{float(np.mean(v)):.3g}", "finite": bool(np.isfinite(v).all())} for k, v in arrays.items()]
            ),
            use_container_width=True,
            hide_index=True,
        )
        # W1 heterogeneity hint
        if "x_train" in arrays and arrays["x_train"].shape[1] <= 2:
            st.caption(f"Local domain approx: x∈[{float(np.min(arrays['x_train'])):.2f}, {float(np.max(arrays['x_train'])):.2f}] — heterogeneity tuned via `n_pieces` (SRS FR-DATA-4)")
        plot_key = next((k for k in ("x_train", "branch_train") if k in arrays), None)
        if plot_key is not None and arrays[plot_key].ndim == 2 and arrays[plot_key].shape[1] in (1, 2):
            cols = ["x"] if arrays[plot_key].shape[1] == 1 else ["x1", "x2"]
            df = pd.DataFrame(np.asarray(arrays[plot_key]), columns=cols)
            # sample for chart
            if len(df) > 500:
                df = df.sample(500, random_state=0)
            st.write(f"**{plot_key}** — {bundle.size:,} points (sample 500 shown)")
            if len(cols) == 2:
                st.scatter_chart(df, x="x1", y="x2", height=320)
            else:
                st.line_chart(df.set_index("x"), height=320)
        if "branch_train" in arrays:
            st.caption("Operator problem: branch input = function sensors, trunk = evaluation points, y = operator output.  Heterogeneity via functional-space (Chebyshev/Fourier) subsets (SRS §4.2).")

with tab_train:
    if bundle is None:
        st.info("Load a dataset first (sidebar).")
    else:
        c1, c2 = st.columns([1,1])
        with c1:
            if st.button("🔌 Connect & register", disabled=st.session_state.connected, use_container_width=True):
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
                    log(f"Registered '{info.client_id}' with {server_url} on problem {bundle.problem}")
                    st.success(f"✓ Registered as **{info.client_id}** on **{bundle.problem}**")
                    st.toast("Connected", icon="🔗")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Could not connect: {exc}")
        with c2:
            if st.session_state.connected:
                st.success("Ready — press Start below when server has started a run")
            else:
                st.caption("Start server & run first: `POST /admin/run/start` or Dashboard → Start run")

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
                    for _ in range(3000):  # safety bound
                        resp = server.assignment(client_id)
                        if resp.status in ("done", "error"):
                            if resp.status == "error":
                                status.error(f"Server error: {resp.message}")
                                log(f"ERROR {resp.message}")
                                break
                            if st.session_state.rounds_done == 0:
                                status.info(
                                    "Waiting for a new run to start "
                                    "(previous run already completed)."
                                )
                                time.sleep(poll_interval)
                                continue
                            status.success(f"✓ {resp.message} — {st.session_state.rounds_done} rounds")
                            progress.progress(1.0, text="complete")
                            st.balloons()
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
                            f"**Round {resp.round}/{resp.total_rounds}** — "
                            f"{resp.optimizer.name} lr={resp.optimizer.lr:.1e} · "
                            f"prox_mu={resp.prox_mu:.3g} · clip={resp.gradient_clip} · "
                            f"training {resp.local_epochs} epoch(s) on {bundle.size:,} samples"
                        )
                        stats = trainer.train(
                            bundle.arrays,
                            epochs=resp.local_epochs,
                            optimizer_spec=resp.optimizer,
                            batch_size=int(batch_size),
                            prox_mu=resp.prox_mu,
                            grad_clip=float(grad_clip) or None,
                            gradient_clip=resp.gradient_clip,
                            max_norm=resp.max_norm,
                            clip_value=resp.clip_value,
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
                            text=f"round {resp.round}/{resp.total_rounds} · loss {stats['loss']:.4g} · {len(payload)//1024}KB uploaded",
                        )
                        chart.line_chart(
                            pd.DataFrame(st.session_state.history).set_index("round"), color="#0ea5e9"
                        )
                        log(
                            f"round {resp.round}: loss={stats['loss']:.6g} "
                            f"({len(payload)} bytes uploaded · {resp.gradient_clip})"
                        )
            except ServerConnectionError as exc:
                st.error(str(exc))
                st.session_state.connected = False
            except Exception as exc:  # noqa: BLE001
                st.error(f"Training failed: {exc}")

with tab_log:
    if st.session_state.logs:
        st.code("\n".join(st.session_state.logs[-100:]), language="text")
        st.download_button("⬇ Download log", "\n".join(st.session_state.logs).encode(), "client.log")
    else:
        st.info("No activity yet.  Load dataset → Connect → Start training.  Raw data never leaves this machine (FR-CLIENTAPP-9).")
