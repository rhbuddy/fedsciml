"""Orient Server — Streamlit dashboard.

Controls a federated run and visualises round metrics. Thin layer over the
server REST API (SRS v1.4 FR-DASH-3: no science logic embedded).

Run (from the ``backend/`` directory)::

    streamlit run ui/dashboard.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Orient Server", page_icon="🛰️", layout="wide")

st.title("Orient · Server Dashboard")
st.caption("Start a federated run, watch clients report in, and track the global model's accuracy.")

# --------------------------------------------------------------------------- api
server_url = st.sidebar.text_input("Server URL", "http://127.0.0.1:8000").rstrip("/")
auto_refresh = st.sidebar.toggle("Auto-refresh (every 2 s)", value=False)


def api(method: str, path: str, **kwargs):
    resp = httpx.request(method, f"{server_url}{path}", timeout=60.0, **kwargs)
    resp.raise_for_status()
    return resp.json()


try:
    health = api("GET", "/health")
    st.sidebar.success(f"Online · {health.get('phase', '?')}")
except Exception as exc:  # noqa: BLE001
    st.sidebar.error(f"Server unreachable: {exc}")
    st.stop()

problems = api("GET", "/problems").get("problems", [])
meta = api("GET", "/aggregators")
aggregators = meta.get("aggregators", [])
weighting_modes = meta.get("weighting_modes", ["uniform"])

tab_run, tab_metrics, tab_clients = st.tabs(["Run control", "Metrics", "Clients"])

# ----------------------------------------------------------------------- run control
with tab_run:
    with st.form("run_config"):
        c1, c2, c3 = st.columns(3)
        problem = c1.selectbox("Problem", problems)
        aggregator = c2.selectbox("Aggregator", aggregators)
        weighting = c3.selectbox("Weighting", weighting_modes)

        c4, c5, c6 = st.columns(3)
        total_rounds = c4.number_input("Rounds", 1, 2000, 10)
        local_epochs = c5.number_input("Local epochs", 1, 1000, 5)
        learning_rate = c6.number_input("Client learning rate", 1e-6, 1.0, 1e-3, format="%.6f")

        with st.expander("Aggregator hyper-parameters"):
            p1, p2, p3 = st.columns(3)
            mu = p1.number_input("FedProx mu", 0.0, 10.0, 0.01, step=0.01)
            server_lr = p2.number_input("Server LR (FedAdam/Adagrad/Yogi)", 1e-6, 10.0, 0.1, format="%.6f")
            tau = p3.number_input("tau", 1e-6, 1.0, 1e-3, format="%.6f")
            p4, p5, p6 = st.columns(3)
            beta1 = p4.number_input("beta1", 0.0, 1.0, 0.9)
            beta2 = p5.number_input("beta2", 0.0, 1.0, 0.99)
            trim_ratio = p6.number_input("Trim ratio (trimmed mean)", 0.0, 0.49, 0.1)
            p7, p8 = st.columns(2)
            n_byzantine = p7.number_input("Krum: n_byzantine", 0, 50, 1)
            multi_k = p8.number_input("Krum: multi_k", 1, 50, 1)

        submitted = st.form_submit_button("▶ Start run", type="primary", use_container_width=True)

    if submitted:
        payload = {
            "problem": problem,
            "aggregator": aggregator,
            "weighting": weighting,
            "total_rounds": int(total_rounds),
            "local_epochs": int(local_epochs),
            "learning_rate": float(learning_rate),
            "aggregator_params": {
                "mu": float(mu),
                "server_lr": float(server_lr),
                "tau": float(tau),
                "beta1": float(beta1),
                "beta2": float(beta2),
                "trim_ratio": float(trim_ratio),
                "n_byzantine": int(n_byzantine),
                "multi_k": int(multi_k),
            },
        }
        try:
            api("POST", "/admin/run/start", json=payload)
            st.success(f"Run started: {problem} · {aggregator} · {int(total_rounds)} rounds")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not start the run: {exc}")

    b1, b2 = st.columns(2)
    if b1.button("⏹ Stop run", use_container_width=True):
        try:
            api("POST", "/admin/run/stop")
            st.warning("Run stopped.")
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
    if b2.button("⚡ Force aggregate", use_container_width=True):
        try:
            api("POST", "/admin/run/force")
            st.info("Aggregated with the updates received so far.")
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))

    st.divider()
    with st.expander("🧩 Local demo clients", expanded=False):
        st.caption(
            "Starts N client processes **on this machine** so you can drive a whole run "
            "from the browser. In a real deployment each participant runs their own "
            "client (Client UI, port 8502) — the protocol is identical either way."
        )
        local = api("GET", "/admin/clients/local")
        problems = local.get("problems", [])
        if not problems:
            st.warning(
                "No dataset bundles found. Generate them first:\n\n"
                "`python examples/make_datasets.py --clients 5`"
            )
        else:
            lc1, lc2, lc3 = st.columns([2, 2, 3])
            lc_problem = lc1.selectbox("Problem", problems, key="lc_problem")
            lc_count = lc2.number_input("Clients", 1, 16, 3, key="lc_count")
            if lc3.button("▶ Start clients", use_container_width=True, key="lc_start"):
                try:
                    res = api(
                        "POST", "/admin/clients/local/spawn",
                        json={"count": int(lc_count), "problem": lc_problem},
                    )
                    if res.get("started"):
                        st.success(f"Started: {', '.join(res['started'])}")
                    for err in res.get("errors", []):
                        st.warning(err)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Could not start clients: {exc}")

            running = local.get("running", [])
            if running:
                st.write(
                    "**Running:** "
                    + ", ".join(f"`{r['client_id']}` (pid {r['pid']})" for r in running)
                )
            if st.button("⏹ Stop clients", use_container_width=True, key="lc_stop"):
                res = api("POST", "/admin/clients/local/stop")
                st.info(f"Stopped {res.get('stopped', 0)} client process(es).")

# --------------------------------------------------------------------------- status
status = api("GET", "/run/status")

with tab_metrics:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Phase", status.get("phase", "—"))
    m2.metric("Round", f"{status.get('round', 0)}/{status.get('total_rounds', 0)}")
    m3.metric(
        "Submitted",
        f"{len(status.get('submitted_clients', []))}/{len(status.get('expected_clients', []))}",
    )
    m4.metric("Problem · Aggregator", f"{status.get('problem') or '—'} · {status.get('aggregator') or '—'}")

    st.caption(
        f"Expected: {', '.join(status.get('expected_clients', [])) or '—'}  ·  "
        f"Submitted: {', '.join(status.get('submitted_clients', [])) or '—'}"
    )

    metrics = status.get("metrics", [])
    if metrics:
        df = pd.DataFrame(metrics)
        columns = [c for c in ("round", "l2_relative_error", "mean_local_loss", "n_clients") if c in df.columns]
        if columns:
            st.line_chart(df[columns].set_index("round"))
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No completed rounds yet. Start a run and connect clients.")

with tab_clients:
    clients = api("GET", "/clients")
    if clients:
        st.dataframe(pd.DataFrame(clients), use_container_width=True, hide_index=True)
    else:
        st.info("No clients registered yet. Start a client app and press 'Connect & register'.")

if auto_refresh:
    time.sleep(2)
    st.rerun()

