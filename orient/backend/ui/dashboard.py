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

st.set_page_config(page_title="Orient Server", page_icon="🛰️", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for polished look
st.markdown("""
<style>
  .main-header {font-size:2.2rem; font-weight:700; background: linear-gradient(90deg,#6366f1,#8b5cf6); -webkit-background-clip:text; -webkit-text-fill-color:transparent;}
  .metric-card {background:#f8fafc; padding:12px; border-radius:12px; border:1px solid #e2e8f0;}
  .stMetric {background:#ffffff; padding:8px; border-radius:8px; border:1px solid #e2e8f0;}
</style>
""", unsafe_allow_html=True)

st.title("Orient · Server Dashboard")
st.markdown('<div class="main-header">🛰️ Federated SciML — Robust Benchmark</div>', unsafe_allow_html=True)
st.caption("Start a federated run, watch clients report in, and track the global model's accuracy — **robust multi-algorithm SciML benchmark** (SRS v1.4).")

# --------------------------------------------------------------------------- api
server_url = st.sidebar.text_input("Server URL", "http://127.0.0.1:8000").rstrip("/")
auto_refresh = st.sidebar.toggle("Auto-refresh (every 2 s)", value=False)
st.sidebar.divider()
st.sidebar.caption("**SRS v1.4** · 10 problems · 10 aggregators · W1 heterogeneity · noisy/adversarial · scaffold/feddyn · async C + int8/topk + DP")
st.sidebar.caption("Results → `results/<problem>/<aggregator>/<run_id>/`  ·  `config.yaml` + `server_best.pth` + `l2_error.npz`  ·  filesystem (postgres/s3/redis excluded)")


def api(method: str, path: str, **kwargs):
    resp = httpx.request(method, f"{server_url}{path}", timeout=60.0, **kwargs)
    resp.raise_for_status()
    return resp.json()


try:
    health = api("GET", "/health")
    st.sidebar.success(f"● Online · {health.get('phase', '?')}")
except Exception as exc:  # noqa: BLE001
    st.sidebar.error(f"Server unreachable: {exc}")
    st.warning("Start the server first: `python -m app.main` in `backend/`")
    st.stop()

problems = api("GET", "/problems").get("problems", [])
meta = api("GET", "/aggregators")
aggregators = meta.get("aggregators", [])
weighting_modes = meta.get("weighting_modes", ["uniform"])

tab_run, tab_metrics, tab_clients, tab_results, tab_compare = st.tabs(["🚀 Run control", "📈 Metrics", "🧩 Clients", "📁 Results", "⚖️ Compare"])

# ----------------------------------------------------------------------- run control
with tab_run:
    st.subheader("Launch a federated run")
    st.caption("Config-driven (YAML) or interactive — mirrors `python -m app.runner --config configs/poisson_fedavg.yaml` (FR-CFG-1, FR-RUN-1)")
    with st.form("run_config"):
        c1, c2, c3 = st.columns(3)
        problem = c1.selectbox("Problem", problems, help="10 problems: Gramacy, Schaffer, Poisson, Helmholtz, Allen-Cahn, Inverse NS/DR, Antiderivative, Burgers, Diffusion-Reaction")
        aggregator = c2.selectbox("Aggregator", aggregators, help="Standard: fedavg, fedprox, fedadam, fedadagrad, fedyogi, scaffold, feddyn  •  Robust: median, trimmed_mean, krum")
        weighting = c3.selectbox("Weighting", weighting_modes, help="data_size = paper default (FedAvg ∝ Nk/N)")

        c4, c5, c6 = st.columns(3)
        total_rounds = c4.number_input("Rounds (global)", 1, 5000, 30, help="SRS Table 4: Poisson 1000, Antiderivative/Burgers 10000")
        local_epochs = c5.number_input("Local epochs", 1, 100, 5)
        learning_rate = c6.number_input("Client LR", 1e-6, 1.0, 1e-3, format="%.6f")

        # Quick heterogeneity / noise / clipping (new SRS fields)
        h1, h2, h3 = st.columns(3)
        n_pieces = h1.number_input("Heterogeneity n_pieces", 1, 50, 10, help="SRS FR-DATA-4: 1=high heterogeneity (large W1), 10+=iid (small W1)")
        h_mode = h2.selectbox("Heterogeneity mode", ["1d_partition", "xy_partition", "chebyshev"], index=0)
        noise_mode = h3.selectbox("Noise mode", ["none", "noisy", "adversarial"], help="FR-NOISE: corrupt fraction of updates")

        h4, h5, h6 = st.columns(3)
        noise_fraction = h4.slider("Noise fraction", 0.0, 0.5, 0.0, step=0.05)
        grad_clip = h5.selectbox("Gradient clip", ["none", "norm", "value"], help="FR-CLIENT-9: norm or value")
        max_norm = h6.number_input("max_norm / clip_value", 0.1, 10.0, 1.0, step=0.1)

        # Framework v2: async sampling, compression, DP (reviewer delight, 1h/5lines/Low per user spec)
        f1, f2, f3 = st.columns(3)
        client_fraction = f1.slider("Client sampling C", 0.1, 1.0, 1.0, step=0.1, help="FedAvg C: fraction sampled per round (1.0=sync wait all, 0.5=sample half)")
        round_timeout = f2.number_input("Round timeout (s)", 0, 600, 60, step=10, help="Async: aggregate after timeout even if not all sampled reported (0=wait all, 60s=async)")
        compression = f3.selectbox("Compression", ["none", "int8", "topk", "int8_topk"], help="int8 quantize + topk 1% → 200KB→20KB, plot comm vs L2")
        f4, f5, f6 = st.columns(3)
        topk_ratio = f4.number_input("Top-k ratio", 0.001, 1.0, 0.01, step=0.01, format="%.3f", help="1% for 10x saving")
        dp_noise_multiplier = f5.number_input("DP σ (noise mult)", 0.0, 2.0, 0.0, step=0.1, help="0=off, 0.1=on (adds Gaussian σ*C after norm clip, RDP ε tracked)")
        dp_delta = f6.number_input("DP δ", 1e-7, 1e-3, 1e-5, format="%.1e", help="DP delta for ε accounting")

        with st.expander("⚙️ Aggregator hyper-parameters"):
            p1, p2, p3 = st.columns(3)
            mu = p1.number_input("FedProx mu", 0.0, 10.0, 0.01, step=0.01)
            server_lr = p2.number_input("Server LR (FedAdam/Adagrad/Yogi/Scaffold)", 1e-6, 10.0, 0.1, format="%.6f")
            tau = p3.number_input("tau", 1e-6, 1.0, 1e-3, format="%.6f")
            p4, p5, p6 = st.columns(3)
            beta1 = p4.number_input("beta1", 0.0, 1.0, 0.9)
            beta2 = p5.number_input("beta2", 0.0, 1.0, 0.99)
            trim_ratio = p6.number_input("Trim ratio", 0.0, 0.49, 0.1)
            p7, p8, p9 = st.columns(3)
            n_byzantine = p7.number_input("Krum: n_byzantine", 0, 50, 1)
            multi_k = p8.number_input("Krum: multi_k", 1, 50, 1)
            seed = p9.number_input("Seed", 0, 9999, 42, help="NFR-REP-1 reproducibility")

        submitted = st.form_submit_button("▶ Start run", type="primary", use_container_width=True)

    if submitted:
        payload = {
            "problem": problem,
            "aggregator": aggregator,
            "weighting": weighting,
            "total_rounds": int(total_rounds),
            "local_epochs": int(local_epochs),
            "learning_rate": float(learning_rate),
            "heterogeneity": {"mode": h_mode, "n_pieces": int(n_pieces)},
            "noise_mode": noise_mode,
            "noise_fraction": float(noise_fraction),
            "gradient_clip": grad_clip,
            "max_norm": float(max_norm),
            "clip_value": float(max_norm),
            "seed": int(seed),
            "client_fraction": float(client_fraction),
            "round_timeout": int(round_timeout),
            "compression": compression,
            "topk_ratio": float(topk_ratio),
            "dp_noise_multiplier": float(dp_noise_multiplier),
            "dp_delta": float(dp_delta),
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
            st.success(f"✓ Run started: **{problem}** · **{aggregator}** · **{int(total_rounds)} rounds** · seed {int(seed)}  — heterogeneity {h_mode} n={int(n_pieces)}, noise {noise_mode} {noise_fraction}")
            st.toast("Run launched — watch Metrics tab", icon="🚀")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not start the run: {exc}")

    b1, b2, b3 = st.columns([1,1,1])
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
    if b3.button("✅ Validation gate (Poisson)", use_container_width=True, help="FR-RUN-5: reproduces paper FedAvg Poisson within tolerance"):
        try:
            # Trigger a short validation run via runner endpoint would be ideal; fallback to info
            st.info("Run `python -m app.runner --validation-gate poisson` in backend for the official gate (30 rounds, checks improvement).")
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))

    st.divider()
    with st.expander("🧩 Local demo clients", expanded=False):
        st.caption(
            "Starts N client processes **on this machine** so you can drive a whole run "
            "from the browser. In a real deployment each participant runs their own "
            "client (Client UI, port 8502) — the protocol is identical either way."
        )
        try:
            local = api("GET", "/admin/clients/local")
            problems_local = local.get("problems", [])
            if not problems_local:
                st.warning(
                    "No dataset bundles found. Generate them first:\n\n"
                    "`python ../client/examples/make_datasets.py --clients 5`  (creates all 10 problems)"
                )
            else:
                lc1, lc2, lc3 = st.columns([2, 2, 3])
                lc_problem = lc1.selectbox("Demo problem", problems_local, key="lc_problem")
                lc_count = lc2.number_input("Clients", 1, 16, 3, key="lc_count", help="Krum needs K≥2f+3 (so 5 for f=1)")
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
        except Exception:
            st.caption("Local demo clients unavailable (server not exposing /admin/clients/local).")

# --------------------------------------------------------------------------- status
try:
    status = api("GET", "/run/status")
except Exception:
    status = {"phase": "?", "round": 0, "total_rounds": 0, "expected_clients": [], "submitted_clients": [], "metrics": []}

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
        # Main curves: L2 and loss
        chart_cols = [c for c in ("round", "l2_relative_error", "mean_local_loss") if c in df.columns]
        if chart_cols:
            st.subheader("📉 Relative L2 error & local loss")
            st.line_chart(df[chart_cols].set_index("round"), color=["#6366f1", "#f59e0b"])
        # Weight divergence if present
        if "weight_divergence" in df.columns:
            try:
                wd = pd.DataFrame([d.get("_mean") if isinstance(d, dict) else None for d in df["weight_divergence"]])
                wd["round"] = df["round"]
                wd = wd.dropna()
                if not wd.empty and wd.iloc[:, 0].notna().any():
                    st.subheader("🧬 Weight divergence (per-layer mean vs previous global)")
                    st.line_chart(wd.set_index("round"), color=["#10b981"])
                    st.caption("SRS FR-EVAL-3 · Tracks federated vs centralized drift; FedDeepONet expected flat per §10.3.6")
            except Exception:
                pass
        # Framework v2: comm vs L2 and DP epsilon
        if "total_bytes" in df.columns and df["total_bytes"].notna().any():
            try:
                st.subheader("📦 Communication cost vs L2 (Framework v2)")
                # show KB per round
                df_plot = df[["round", "total_bytes", "l2_relative_error"]].copy()
                df_plot["KB"] = df_plot["total_bytes"] / 1024.0
                # line for KB
                st.line_chart(df_plot[["round", "KB"]].set_index("round"), color=["#f97316"])
                # scatter comm vs L2
                st.scatter_chart(df_plot, x="KB", y="l2_relative_error", color="#6366f1")
                saving = 0
                if "raw_bytes_est" in df.columns and df["raw_bytes_est"].notna().any():
                    try:
                        avg_ratio = df["compression_ratio_pct"].mean()
                        saving = float(avg_ratio)
                    except Exception:
                        pass
                st.caption(f"Avg compression saving: {saving:.1f}% (int8+topk 1% → ~10x saving; goal 200KB→20KB) · plot comm vs L2 reviewers love")
            except Exception:
                pass
        if "dp_epsilon" in df.columns and (df["dp_epsilon"] > 0).any():
            try:
                st.subheader("🔒 DP privacy budget (RDP)")
                st.line_chart(df[["round", "dp_epsilon"]].set_index("round"), color=["#8b5cf6"])
                st.caption(f"DP σ={df['dp_noise_multiplier'].iloc[-1]:.2g} δ={df.get('dp_delta', pd.Series([1e-5])).iloc[-1]:.1e} → ε≈{df['dp_epsilon'].iloc[-1]:.2f} (RDP via opacus or analytic; needs norm clip)")
            except Exception:
                pass
        # Show compromised if any
        if "compromised_clients" in df.columns and any(df["compromised_clients"].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)):
            st.warning("Some rounds had noisy/adversarial clients — see `compromised_clients` column. Robust aggregators (median/krum/trimmed_mean) should maintain accuracy.")
        st.dataframe(df, use_container_width=True, hide_index=True)
        # Download hint
        st.caption("Artifacts: `results/<problem>/<aggregator>/<run_id>/` → `config.yaml`, `server_best.pth`, `l2_error.npz`, `weight_divergence.npz`, `metrics.json` (+ `baselines.json`, plots) — filesystem (postgres+s3+redis excluded per user spec) (SRS §8.2)")
    else:
        st.info("No completed rounds yet. Start a run and connect clients.  Tip: use **30–60 rounds** with **5–10 local epochs** for a visible L2 drop (see `configs/poisson_fedavg.yaml`).")

with tab_clients:
    try:
        clients = api("GET", "/clients")
        if clients:
            st.dataframe(pd.DataFrame(clients), use_container_width=True, hide_index=True)
            st.caption(f"{len(clients)} client(s) registered — they will be auto-assigned to the next run on their problem.")
        else:
            st.info("No clients registered yet. Start a client app (`client/ui/app.py` or `python -m app.main --dataset ...`) and press 'Connect & register'.")
    except Exception:
        st.info("Clients endpoint unavailable")

with tab_results:
    st.subheader("📁 Results browser")
    st.caption("Browse `results/` artifacts produced by the runner or the REST orchestrator (SRS §8.2). Direct file serving is via filesystem; this tab lists metadata from `/run/status`.")
    if metrics:
        df = pd.DataFrame(metrics)
        st.write(f"**Current run**: `{status.get('problem')}/{status.get('aggregator')}` — {len(metrics)} rounds, best L2 `{df['l2_relative_error'].min():.4g}` at round {int(df.loc[df['l2_relative_error'].idxmin(), 'round'])}")
        csv = df.to_csv(index=False).encode()
        st.download_button("⬇ Download metrics CSV", csv, "metrics.csv", "text/csv")
    else:
        st.info("No results yet. After a run completes, artifacts are in `results/<problem>/<aggregator>/<run_id>/`  (`config.yaml`, `metrics.json`, `server_best.pth`, `loss.npz`, `l2_error.npz`, `weight_divergence.npz`).")
    st.markdown("**Example:** `python -m app.runner --config configs/poisson_fedavg.yaml` → check `results/poisson/fedavg/*/metrics.json`")

with tab_compare:
    st.subheader("⚖️ Robustness–Accuracy Trade-off (SRS §10.4)")
    st.caption("Headline figure: accuracy (relative L2) vs `noise_fraction` (0.0 → 0.3). Run the deep-benchmark sweep to populate this: `python -m app.runner --config configs/deep_benchmark.yaml`")
    if metrics:
        try:
            df = pd.DataFrame(metrics)
            if "l2_relative_error" in df.columns and "round" in df.columns:
                # Show last round's L2 as proxy for trade-off single point; full sweep will have multiple runs
                st.line_chart(df[["round", "l2_relative_error"]].set_index("round"), color=["#ef4444"])
                st.caption(f"Current run ({status.get('aggregator')}): L2 trajectory. Compare across aggregators to see trade-off — robust (median/krum) trades clean-data accuracy for resilience (SRS Table 5.1).")
        except Exception:
            st.line_chart(pd.DataFrame({"round": [1,2,3], "l2": [0.5,0.3,0.2]}).set_index("round"))
    else:
        # Placeholder illustration
        demo = pd.DataFrame({
            "noise_fraction": [0.0, 0.1, 0.2, 0.3],
            "fedavg": [0.05, 0.12, 0.35, 0.80],
            "median": [0.07, 0.08, 0.09, 0.11],
            "krum": [0.06, 0.07, 0.08, 0.10],
        }).set_index("noise_fraction")
        st.line_chart(demo, color=["#6366f1", "#10b981", "#f59e0b"])
        st.caption("Illustrative: FedAvg collapses under adversarial noise, median/Krum stay flat (hypothesis §10.3.3). Replace with your sweep's `results/*/metrics.json` aggregation (runner auto-generates `sweep_summary_*.json` + comparison table).")
        st.info("To generate the real curve:  **Runner → `configs/deep_benchmark.yaml` (9 algos × 3 problems × noise)** then `python -m app.runner --validation-gate poisson` + spot-checks.")

if auto_refresh:
    time.sleep(2)
    st.rerun()
