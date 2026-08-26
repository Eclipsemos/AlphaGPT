"""Streamlit dashboard for Binance factor research only."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from data_service import DashboardService
from visualizer import plot_coverage, plot_market_scatter
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
st.title("AlphaGPT · Binance Spot Factor Research")
st.caption(
    "只读历史研究界面：数据快照、因子特征、公式挖掘和离线评估。"
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

tab_data, tab_features, tab_runs = st.tabs(["Dataset", "Factors & Features", "Research Runs"])

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
                }
            )

st.divider()
st.caption(
    "研究状态不是盈利或生产可用性承诺。所有成本、权重和收益数字仅用于历史统计；"
    "AlphaGPT 不提交订单、不维护虚拟余额，也不连接 Binance 私有 API。"
)
