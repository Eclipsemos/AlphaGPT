"""Worker entry point for a dashboard-launched Binance research batch."""

from __future__ import annotations

import argparse
import os
import signal
import traceback
from pathlib import Path
from typing import Any

from dashboard.mining_jobs import MiningJobConfig, MiningJobManager, utc_now


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one managed Binance factor-mining job")
    parser.add_argument("--job-id", required=True)
    return parser


def run_job(job_id: str, *, project_root: str | Path | None = None) -> None:
    manager = MiningJobManager(project_root=project_root)
    state = manager.get_job(job_id)
    config = MiningJobConfig.from_dict(state["config"])

    def handle_stop(_signum: int, _frame: Any) -> None:
        manager.update_job(
            job_id,
            status="stopped",
            finished_at=utc_now(),
            updated_at=utc_now(),
            progress={"phase": "stopped", "message": "Stopped by dashboard user"},
        )
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, handle_stop)
    manager.update_job(
        job_id,
        status="running",
        started_at=state.get("started_at") or utc_now(),
        updated_at=utc_now(),
        pid=os.getpid(),
        process_start_ticks=manager._process_start_ticks(os.getpid()),
    )

    try:
        # Importing torch can take several seconds, so expose running state first.
        from model_core.binance_batch import run_batch
        from model_core.binance_engine import BinanceMiningConfig
        from model_core.binance_evaluation import BinanceEvaluationConfig

        total_cost = config.taker_fee_bps + config.slippage_bps
        result = run_batch(
            snapshot_id=config.snapshot_id,
            seeds=config.seeds,
            output_dir=state["output_dir"],
            mining_config=BinanceMiningConfig(
                steps=config.steps,
                batch_size=config.batch_size,
                minimum_cross_section=config.minimum_cross_section,
            ),
            evaluation_config=BinanceEvaluationConfig(
                max_positions=config.max_positions,
                weighting=config.weighting,
                rebalance_hours=config.rebalance_hours,
                risk_lookback_hours=config.risk_lookback_hours,
                taker_fee_bps=config.taker_fee_bps,
                slippage_bps=config.slippage_bps,
                portfolio_notional_usd=config.portfolio_notional_usd,
                minimum_quote_volume_usd=config.minimum_quote_volume_usd,
                minimum_cross_section=config.minimum_cross_section,
            ),
            window_count=config.windows,
            shortlist_size=config.shortlist_size,
            cost_scenarios=(0.0, total_cost, total_cost * 2.0),
            resume=bool(state.get("resume_requested")),
            use_lord_regularization=config.use_lord_regularization,
            progress_callback=lambda event: manager.update_progress(job_id, event),
        )
        manager.update_job(
            job_id,
            status="complete",
            finished_at=utc_now(),
            updated_at=utc_now(),
            progress={
                "phase": "complete",
                "message": "Mining and historical evaluation complete",
                "percent": 100.0,
            },
            decision=result.get("decision", {}).get("status", "unknown"),
        )
    except SystemExit:
        raise
    except BaseException as exc:
        traceback.print_exc()
        manager.update_job(
            job_id,
            status="failed",
            finished_at=utc_now(),
            updated_at=utc_now(),
            error=f"{type(exc).__name__}: {exc}",
            progress={"phase": "failed", "message": str(exc)},
        )
        raise


def main() -> None:
    args = _parser().parse_args()
    run_job(args.job_id)


if __name__ == "__main__":
    main()
