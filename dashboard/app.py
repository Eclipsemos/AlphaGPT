"""Streamlit dashboard for Binance factor research only."""

from __future__ import annotations

import hmac
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Streamlit puts the script directory, rather than the repository root, first
# on sys.path. Resolve imports from the repository regardless of launch cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from dashboard.data_service import DashboardService
from dashboard.mining_jobs import ACTIVE_STATUSES, MiningJobConfig, MiningJobManager
from dashboard.visualizer import plot_coverage, plot_market_scatter
from model_core.binance_batch import parse_seeds
from model_core.binance_features import (
    BINANCE_FEATURE_CODE_VERSION,
    BINANCE_FEATURE_DEFINITIONS,
    BINANCE_FEATURE_WARMUPS,
)
from model_core.vocab import BINANCE_FORMULA_VOCAB


st.set_page_config(
    page_title="AlphaGPT Binance Factor Research",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_service() -> DashboardService:
    return DashboardService()


svc = get_service()
job_manager = MiningJobManager(project_root=PROJECT_ROOT, python_executable=sys.executable)


def _control_authorized() -> bool:
    configured = os.getenv("DASHBOARD_CONTROL_TOKEN", "")
    if not configured:
        st.warning("Mining controls are disabled until DASHBOARD_CONTROL_TOKEN is configured.")
        return False
    if st.session_state.get("mining_control_authorized"):
        return True
    with st.form("mining-control-auth"):
        supplied = st.text_input("Control token", type="password")
        submitted = st.form_submit_button("Unlock mining controls")
    if submitted:
        if hmac.compare_digest(supplied, configured):
            st.session_state["mining_control_authorized"] = True
            st.rerun()
        else:
            st.error("Invalid control token.")
    return False


def _elapsed_time(state: dict) -> str:
    started = state.get("started_at") or state.get("created_at")
    finished = state.get("finished_at")
    if not started:
        return "n/a"
    try:
        start_time = datetime.fromisoformat(started)
        end_time = datetime.fromisoformat(finished) if finished else datetime.now(timezone.utc)
    except (TypeError, ValueError):
        return "n/a"
    seconds = max(0, int((end_time - start_time).total_seconds()))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


@st.fragment(run_every=2.0)
def render_job_monitor() -> None:
    jobs = job_manager.list_jobs()
    if not jobs:
        st.info("No dashboard mining jobs yet.")
        return

    labels = {
        item["job_id"]: f"{item['job_id']} · {item.get('status', 'unknown')}"
        for item in jobs
    }
    selected_job_id = st.selectbox(
        "Mining job",
        list(labels),
        format_func=lambda value: labels[value],
        key="mining-job-selection",
    )
    state = job_manager.get_job(selected_job_id)
    progress = state.get("progress", {})
    percent = max(0.0, min(100.0, float(progress.get("percent", 0.0))))

    status_col, phase_col, elapsed_col, device_col = st.columns(4)
    status_col.metric("Status", str(state.get("status", "unknown")).upper())
    phase_col.metric("Phase", str(progress.get("phase", "n/a")).replace("_", " "))
    elapsed_col.metric("Elapsed", _elapsed_time(state))
    device_col.metric("Progress", f"{percent:.1f}%")
    st.progress(percent / 100.0, text=progress.get("message", "Working"))

    if progress.get("phase") == "mining":
        seed_col, step_col, candidate_col, score_col = st.columns(4)
        seed_col.metric(
            "Seed",
            f"{progress.get('seed_index', 0)}/{progress.get('seed_count', 0)}",
            help=f"Current random seed: {progress.get('seed', 'n/a')}",
        )
        step_col.metric("Step", f"{progress.get('step', 0)}/{progress.get('steps', 0)}")
        candidate_col.metric("Candidates", progress.get("unique_candidate_count", 0))
        score = progress.get("best_validation_score")
        score_col.metric("Best validation IC", f"{score:.4f}" if isinstance(score, (int, float)) else "n/a")

    action_col, path_col = st.columns([1, 4])
    if state.get("status") in {"queued", "running"}:
        if action_col.button("Stop", type="secondary", key=f"stop-{selected_job_id}"):
            try:
                job_manager.stop_job(selected_job_id)
                st.rerun(scope="fragment")
            except RuntimeError as exc:
                st.error(str(exc))
    elif state.get("status") in {"failed", "stopped"}:
        if action_col.button("Resume", key=f"resume-{selected_job_id}"):
            try:
                job_manager.resume_job(selected_job_id)
                st.rerun(scope="fragment")
            except RuntimeError as exc:
                st.error(str(exc))
    path_col.code(state.get("output_dir", ""), language=None)

    if state.get("error"):
        st.error(state["error"])
    with st.expander("Worker log", expanded=state.get("status") == "failed"):
        log = job_manager.log_tail(selected_job_id)
        st.code(log or "Waiting for worker output...", language=None, height=320)


st.title("AlphaGPT · Binance Spot Factor Research")
st.caption(
    "历史研究界面：数据快照、因子特征、公式挖掘和离线评估。"
    "不包含钱包、账户、模拟盘、订单或实盘功能。"
)

snapshots = svc.list_snapshots()
if snapshots.empty:
    st.warning("尚未找到 Binance dataset snapshot。先运行 data_pipeline.run_binance_pipeline。")
    st.stop()

snapshot_options = snapshots["snapshot_id"].tolist()
selected_snapshot = st.sidebar.selectbox("Dataset snapshot", snapshot_options)
st.sidebar.caption("研究范围：Binance Spot / USDT / 1h / public data")
if st.sidebar.button("Refresh"):
    st.cache_resource.clear()
    st.rerun()

payload = svc.snapshot_payload(selected_snapshot)
status = svc.get_data_status(selected_snapshot)
coverage = svc.get_snapshot_coverage(selected_snapshot)
market = svc.get_market_overview()
runs = svc.latest_research_runs()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Symbols", len(payload.get("symbols", [])))
col2.metric("Bars in DB", f"{int(status.get('bar_count', 0)):,}")
coverage_ratio = (
    coverage["bar_count"].sum() / max(1, coverage["expected_bar_count"].sum())
    if not coverage.empty
    else None
)
col3.metric("Snapshot coverage", f"{coverage_ratio:.2%}" if coverage_ratio is not None else "n/a")
col4.metric("Latest 1h bar", str(status.get("latest_bar") or "n/a"))

st.info(
    f"Snapshot `{selected_snapshot}` · {payload.get('start_time', 'n/a')} → "
    f"{payload.get('end_time', 'n/a')} · source `{payload.get('source', 'n/a')}` · "
    f"code `{payload.get('code_version', 'n/a')}`"
)

tab_data, tab_features, tab_mining, tab_runs = st.tabs(
    ["Dataset", "Factors & Features", "Mining", "Research Runs"]
)

with tab_data:
    st.subheader("Coverage and quality")
    if coverage.empty:
        st.warning("No coverage rows found for this snapshot.")
    else:
        st.plotly_chart(plot_coverage(coverage), width="stretch")
        st.dataframe(coverage, width="stretch", hide_index=True)
    st.subheader("Latest Binance market bars")
    if market.empty:
        st.warning("No Binance market bars found in DB.")
    else:
        st.plotly_chart(plot_market_scatter(market), width="stretch")
        st.dataframe(market, width="stretch", hide_index=True)

with tab_features:
    st.subheader("Feature schema")
    feature_metadata = {
        "feature_schema_version": BINANCE_FEATURE_CODE_VERSION,
        "feature_names": list(BINANCE_FORMULA_VOCAB.feature_names),
        "feature_definitions": BINANCE_FEATURE_DEFINITIONS,
        "feature_warmups": dict(
            zip(BINANCE_FORMULA_VOCAB.feature_names, BINANCE_FEATURE_WARMUPS)
        ),
    }
    for run in runs:
        if run["snapshot_id"] == selected_snapshot:
            artifacts = svc.load_run_artifacts(run["path"])
            saved_metadata = artifacts.get("batch", {}).get("research_metadata", {})
            if saved_metadata:
                feature_metadata = saved_metadata
                break
    st.write(f"Feature version: `{feature_metadata.get('feature_schema_version', 'n/a')}`")
    st.json(
        {
            "feature_names": feature_metadata.get("feature_names", []),
            "feature_definitions": feature_metadata.get("feature_definitions", {}),
            "feature_warmups": feature_metadata.get("feature_warmups", {}),
            "normalization": feature_metadata.get("normalization", "available after mining"),
        }
    )

with tab_mining:
    st.subheader("GPU factor mining")
    selected_symbol_count = len(payload.get("symbols", []))
    if selected_symbol_count < 10:
        st.error(
            f"Selected snapshot has {selected_symbol_count} symbols. "
            "Mining requires at least 10 valid cross-sectional symbols; rebuild "
            "the dataset with the default 20-symbol universe."
        )
    if _control_authorized():
        lock_col, scope_col = st.columns([1, 4])
        if lock_col.button("Lock controls"):
            st.session_state["mining_control_authorized"] = False
            st.rerun()
        scope_col.caption("Binance Spot · public historical data · factor research only")

        active_jobs = [
            job for job in job_manager.list_jobs() if job.get("status") in ACTIVE_STATUSES
        ]
        with st.form("start-mining-job"):
            config_left, config_middle, config_right = st.columns(3)
            seeds_text = config_left.text_input("Seeds", value="1,2,3,4,5")
            steps = config_left.number_input("Steps per seed", 1, 100_000, 1000, 100)
            batch_size = config_left.selectbox(
                "Formula batch size", [1024, 2048, 4096, 8192, 16384], index=3
            )

            windows = config_middle.number_input("Walk-forward windows", 1, 20, 4)
            shortlist_size = config_middle.number_input("Candidate shortlist", 1, 100, 25)
            max_positions = config_middle.number_input("Maximum positions", 1, 100, 10)
            weighting = config_middle.segmented_control(
                "Historical weighting", ["equal", "risk"], default="equal"
            )
            config_middle.caption("Minimum valid cross-section: 10 symbols")

            rebalance_hours = config_right.number_input("Rebalance hours", 1, 168, 24)
            taker_fee_bps = config_right.number_input("Assumed fee (bps)", 0.0, 100.0, 10.0, 1.0)
            slippage_bps = config_right.number_input("Assumed slippage (bps)", 0.0, 100.0, 5.0, 1.0)
            use_lord = config_right.toggle("LoRD regularization", value=True)

            submitted = st.form_submit_button(
                "Start mining batch",
                type="primary",
                disabled=bool(active_jobs) or selected_symbol_count < 10,
            )
        if active_jobs:
            st.info(f"Job `{active_jobs[0]['job_id']}` is active. Only one GPU batch can run at a time.")
        if submitted:
            try:
                config = MiningJobConfig(
                    snapshot_id=selected_snapshot,
                    seeds=tuple(parse_seeds(seeds_text)),
                    steps=int(steps),
                    batch_size=int(batch_size),
                    windows=int(windows),
                    shortlist_size=int(shortlist_size),
                    max_positions=int(max_positions),
                    weighting=str(weighting),
                    rebalance_hours=int(rebalance_hours),
                    risk_lookback_hours=24,
                    taker_fee_bps=float(taker_fee_bps),
                    slippage_bps=float(slippage_bps),
                    minimum_cross_section=10,
                    use_lord_regularization=bool(use_lord),
                )
                job_manager.start_job(config)
                st.rerun()
            except (ValueError, RuntimeError) as exc:
                st.error(str(exc))

        st.divider()
        render_job_monitor()

with tab_runs:
    st.subheader("Factor mining batches")
    if not runs:
        st.info("No Binance research runs found under runs/binance/.")
    else:
        st.dataframe(pd.DataFrame(runs), width="stretch", hide_index=True)
        choices = [run["path"] for run in runs]
        selected_run = st.selectbox("Run artifact", choices)
        artifacts = svc.load_run_artifacts(selected_run)
        decision = artifacts.get("decision", {})
        if decision:
            status_label = decision.get("status", "unknown").upper()
            st.metric("Research decision", status_label)
            st.caption(decision.get("disclaimer", ""))
            st.json(decision)
        evaluation = artifacts.get("evaluation", {})
        if evaluation:
            st.subheader("Final test report (selected formula only)")
            st.json(
                {
                    "formula": evaluation.get("formula"),
                    "test": evaluation.get("splits", {}).get("test", {}),
                    "baselines": evaluation.get("test_baselines", {}),
                    "cost_sensitivity": evaluation.get("test_cost_sensitivity", {}),
                    "regimes": evaluation.get("test_regimes", {}),
                    "robustness": evaluation.get("test_robustness", {}),
                }
            )

st.divider()
st.caption(
    "研究状态不是盈利或生产可用性承诺。所有成本、权重和收益数字仅用于历史统计；"
    "AlphaGPT 不提交订单、不维护虚拟余额，也不连接 Binance 私有 API。"
)
