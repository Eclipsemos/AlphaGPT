import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_market_scatter(market_df: pd.DataFrame):
    if market_df.empty:
        return go.Figure()
    frame = market_df.copy()
    frame["quote_volume"] = frame["quote_volume"].clip(lower=1)
    frame["close"] = frame["close"].clip(lower=1e-12)
    return px.scatter(
        frame,
        x="quote_volume",
        y="close",
        size="trade_count",
        color="symbol",
        hover_name="symbol",
        log_x=True,
        log_y=True,
        title="Binance Spot: latest quote volume vs close",
        template="plotly_dark",
    )


def plot_coverage(coverage_df: pd.DataFrame):
    if coverage_df.empty:
        return go.Figure()
    frame = coverage_df.copy()
    frame["coverage"] = frame["bar_count"] / frame["expected_bar_count"].clip(lower=1)
    return px.bar(
        frame,
        x="symbol",
        y="coverage",
        title="Snapshot bar coverage",
        labels={"coverage": "coverage ratio"},
        range_y=[0, 1.05],
        template="plotly_dark",
    )
