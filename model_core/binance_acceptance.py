"""Offline end-to-end acceptance check for Binance factor research."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

import torch

from .binance_batch import (
    BATCH_REPORT_VERSION,
    DECISION_REPORT_VERSION,
    parse_seeds,
    run_batch,
    write_json,
)
from .binance_data_loader import BinanceDataLoader
from .binance_engine import BinanceMiningConfig
from .binance_evaluation import BinanceEvaluationConfig
from .binance_features import BinanceFeatureEngineer
from .config import ModelConfig
from .data_loader import compute_forward_returns
from .formula_artifact import FORMULA_ARTIFACT_VERSION


FIXTURE_VERSION = "binance-factor-acceptance-fixture-v1"


def build_fixture_loader(length: int = 240) -> BinanceDataLoader:
    if length < 120:
        raise ValueError("Acceptance fixture requires at least 120 hours")
    device = ModelConfig.DEVICE
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    time_index = torch.arange(length, dtype=torch.float32, device=device)
    bases = torch.tensor([90_000.0, 3_000.0, 600.0], device=device).unsqueeze(1)
    slopes = torch.tensor([0.00035, 0.00025, 0.00020], device=device).unsqueeze(1)
    phases = torch.tensor([0.0, 1.3, 2.6], device=device).unsqueeze(1)
    log_close = torch.log(bases) + slopes * time_index + 0.012 * torch.sin(
        time_index / 9 + phases
    )
    close = torch.exp(log_close)
    open_price = close * (1 + 0.001 * torch.sin(time_index / 5 + phases))
    high = torch.maximum(open_price, close) * 1.004
    low = torch.minimum(open_price, close) * 0.996
    base_volume = (1000 + 30 * torch.cos(time_index / 7 + phases)).clamp_min(10)
    quote_volume = base_volume * close
    trade_count = (5000 + 100 * torch.sin(time_index / 6 + phases)).clamp_min(1)
    taker_buy_quote = quote_volume * (0.5 + 0.1 * torch.sin(time_index / 11 + phases))
    raw = {
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "base_volume": base_volume,
        "quote_volume": quote_volume,
        "trade_count": trade_count,
        "taker_buy_quote_volume": taker_buy_quote,
    }
    observed = torch.ones((len(symbols), length), dtype=torch.bool, device=device)
    fit_end = int(length * ModelConfig.TRAIN_RATIO)
    features = BinanceFeatureEngineer.compute(raw, observed, fit_end=fit_end)
    target, target_valid = compute_forward_returns(open_price, observed, return_valid=True)

    loader = BinanceDataLoader.__new__(BinanceDataLoader)
    loader.engine = None
    fixture_spec = json.dumps(
        {"version": FIXTURE_VERSION, "symbols": symbols, "length": length},
        sort_keys=True,
        separators=(",", ":"),
    )
    loader.snapshot_id = hashlib.sha256(fixture_spec.encode("ascii")).hexdigest()
    loader.requested_symbols = symbols
    loader.snapshot = {
        "schema_version": "binance-dataset-v1",
        "code_version": FIXTURE_VERSION,
    }
    loader.symbols = symbols
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    loader.times = [start + timedelta(hours=index) for index in range(length)]
    loader.raw_data_cache = raw
    loader.observed_mask = observed
    loader.feature_valid = features.valid
    loader.signal_valid = features.valid.all(dim=1)
    loader.feat_tensor = features.values
    loader.feature_normalization = features.normalization.as_dict()
    loader.target_ret = target
    loader.target_valid = target_valid & loader.signal_valid
    loader.splits = loader._build_splits(length)
    loader.train_feat_tensor = None
    loader.validation_feat_tensor = None
    loader.test_feat_tensor = None
    loader.train_raw_data_cache = None
    loader.validation_raw_data_cache = None
    loader.test_raw_data_cache = None
    loader.train_target_ret = None
    loader.validation_target_ret = None
    loader.test_target_ret = None
    loader.train_target_valid = None
    loader.validation_target_valid = None
    loader.test_target_valid = None
    loader.train_signal_valid = None
    loader.validation_signal_valid = None
    loader.test_signal_valid = None
    loader._assign_split_views()
    return loader


def verify_acceptance_artifacts(output_dir: str | Path) -> dict[str, Any]:
    directory = Path(output_dir)
    required = {
        "manifest": "experiment_manifest.json",
        "batch": "batch_report.json",
        "aggregation": "candidate_aggregation.json",
        "walk_forward": "walk_forward_report.json",
        "formula": "selected_formula.json",
        "evaluation": "final_evaluation_report.json",
        "decision": "decision_report.json",
    }
    payloads = {}
    for name, filename in required.items():
        path = directory / filename
        if not path.is_file():
            raise RuntimeError(f"Acceptance artifact is missing: {path}")
        payloads[name] = json.loads(path.read_text())
    checks = {
        "manifest_complete": payloads["manifest"].get("status") == "complete",
        "batch_version": payloads["batch"].get("report_version") == BATCH_REPORT_VERSION,
        "formula_version": payloads["formula"].get("artifact_version")
        == FORMULA_ARTIFACT_VERSION,
        "evaluation_version": payloads["evaluation"].get("report_version")
        == "binance-factor-evaluation-v1",
        "decision_version": payloads["decision"].get("report_version")
        == DECISION_REPORT_VERSION,
        "selection_test_isolation": payloads["walk_forward"].get("test_was_accessed") is False,
        "decision_test_isolation": payloads["decision"].get(
            "test_was_used_for_candidate_selection"
        )
        is False,
        "decision_status": payloads["decision"].get("status")
        in {"reject", "research-only", "promising"},
        "all_splits_present": set(payloads["evaluation"].get("splits", {}))
        == {"train", "validation", "test"},
        "baselines_present": set(payloads["evaluation"].get("test_baselines", {}))
        == {
            "equal_weight_cross_section",
            "btcusdt_reference",
            "cross_sectional_momentum",
            "random_rank",
        },
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Acceptance schema checks failed: {failed}")
    return {
        "report_version": "binance-factor-acceptance-v1",
        "status": "passed",
        "fixture_version": FIXTURE_VERSION,
        "checks": checks,
        "decision": payloads["decision"]["status"],
        "output_dir": str(directory),
    }


def default_output_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("runs") / "binance" / f"acceptance-{stamp}"


def run_acceptance(
    output_dir: str | Path,
    *,
    seeds: Sequence[int] = (1, 2),
    steps: int = 1,
    batch_size: int = 512,
) -> dict[str, Any]:
    loader = build_fixture_loader()
    run_batch(
        snapshot_id=loader.snapshot_id,
        symbols=loader.symbols,
        seeds=seeds,
        output_dir=output_dir,
        mining_config=BinanceMiningConfig(steps=steps, batch_size=batch_size),
        evaluation_config=BinanceEvaluationConfig(
            max_positions=2,
            rebalance_hours=24,
            taker_fee_bps=10,
            slippage_bps=5,
            portfolio_notional_usd=10_000,
        ),
        window_count=2,
        shortlist_size=5,
        cost_scenarios=(0.0, 15.0, 30.0),
        use_lord_regularization=False,
        loader_override=loader,
    )
    result = verify_acceptance_artifacts(output_dir)
    write_json(Path(output_dir) / "acceptance_report.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the offline Binance factor-research acceptance workflow"
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seeds", type=parse_seeds, default=[1, 2])
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run_acceptance(
            args.output_dir or default_output_dir(),
            seeds=args.seeds,
            steps=args.steps,
            batch_size=args.batch_size,
        )
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
