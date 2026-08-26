"""Evaluate a versioned Binance formula artifact on an immutable snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .binance_data_loader import BinanceDataLoader
from .binance_evaluation import (
    BinanceEvaluationConfig,
    BinanceFactorEvaluator,
    baseline_reports,
    cost_sensitivity_reports,
)
from .formula_artifact import validate_formula_artifact
from .vm import StackVM
from .vocab import BINANCE_FORMULA_VOCAB


def parse_cost_scenarios(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("cost scenarios must be non-negative comma-separated bps")
    return values


def validate_research_metadata(artifact: dict, loader: BinanceDataLoader) -> None:
    actual = loader.research_metadata
    saved = artifact.get("research_metadata")
    if not isinstance(saved, dict):
        raise ValueError("Formula artifact has no research metadata")
    for key in (
        "market",
        "dataset_snapshot_id",
        "symbols",
        "dataset_schema_version",
        "dataset_code_version",
        "feature_schema_version",
        "formula_vocab_version",
        "feature_names",
        "feature_definitions",
        "feature_warmups",
        "normalization",
    ):
        if saved.get(key) != actual.get(key):
            raise ValueError(f"Formula artifact metadata mismatch for {key}")


def run(
    formula_path: str | Path,
    snapshot_id: str,
    output_path: str | Path,
    config: BinanceEvaluationConfig,
    cost_scenarios: Sequence[float],
    symbols: list[str] | None = None,
) -> dict:
    artifact = json.loads(Path(formula_path).read_text())
    loader = BinanceDataLoader(snapshot_id, symbols=symbols)
    loader.load_data()
    report = evaluate_artifact(artifact, loader, config, cost_scenarios)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def evaluate_artifact(
    artifact: dict,
    loader: BinanceDataLoader,
    config: BinanceEvaluationConfig,
    cost_scenarios: Sequence[float],
) -> dict:
    formula = validate_formula_artifact(artifact, BINANCE_FORMULA_VOCAB)
    validate_research_metadata(artifact, loader)
    vm = StackVM(BINANCE_FORMULA_VOCAB)
    if not vm.is_valid_formula(formula):
        raise ValueError("Formula artifact does not contain a valid stack expression")
    evaluator = BinanceFactorEvaluator(config)
    evaluations = {}
    splits = (
        (
            "train",
            loader.train_feat_tensor,
            loader.train_raw_data_cache,
            loader.train_target_ret,
            loader.train_target_valid,
            loader.train_signal_valid,
        ),
        (
            "validation",
            loader.validation_feat_tensor,
            loader.validation_raw_data_cache,
            loader.validation_target_ret,
            loader.validation_target_valid,
            loader.validation_signal_valid,
        ),
        (
            "test",
            loader.test_feat_tensor,
            loader.test_raw_data_cache,
            loader.test_target_ret,
            loader.test_target_valid,
            loader.test_signal_valid,
        ),
    )
    test_factors = None
    for name, features, raw, target, return_valid, signal_valid in splits:
        factors = vm.execute(formula, features)
        if factors is None:
            raise ValueError(f"Formula is invalid for the {name} Binance tensor")
        evaluations[name] = evaluator.evaluate(
            factors,
            raw,
            target,
            return_valid,
            signal_valid,
            loader.symbols,
        ).as_dict()
        if name == "test":
            test_factors = factors
    baselines = baseline_reports(
        loader.test_raw_data_cache,
        loader.test_target_ret,
        loader.test_target_valid,
        loader.test_signal_valid,
        loader.symbols,
        config,
    )
    sensitivity = cost_sensitivity_reports(
        test_factors,
        loader.test_raw_data_cache,
        loader.test_target_ret,
        loader.test_target_valid,
        loader.test_signal_valid,
        loader.symbols,
        config,
        cost_scenarios,
    )
    report = {
        "report_version": "binance-factor-evaluation-v1",
        "market": "binance-spot",
        "dataset_snapshot_id": loader.snapshot_id,
        "formula": formula,
        "formula_vocab_version": BINANCE_FORMULA_VOCAB.version,
        "research_metadata": loader.research_metadata,
        "evaluation_config": config.__dict__,
        "splits": evaluations,
        "test_baselines": {name: value.as_dict() for name, value in baselines.items()},
        "test_cost_sensitivity": {
            name: value.as_dict() for name, value in sensitivity.items()
        },
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a Binance factor artifact (research only)")
    parser.add_argument("--formula", required=True)
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--symbols", default=None, help="Comma-separated snapshot symbol subset")
    parser.add_argument("--output", default="binance_evaluation_report.json")
    parser.add_argument("--max-positions", type=int, default=10)
    parser.add_argument("--weighting", choices=("equal", "risk"), default="equal")
    parser.add_argument("--rebalance-hours", type=int, default=24)
    parser.add_argument("--risk-lookback-hours", type=int, default=24)
    parser.add_argument("--taker-fee-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--portfolio-notional-usd", type=float, default=100_000.0)
    parser.add_argument("--cost-scenarios", type=parse_cost_scenarios, default=[0.0, 15.0, 30.0])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = BinanceEvaluationConfig(
        max_positions=args.max_positions,
        weighting=args.weighting,
        rebalance_hours=args.rebalance_hours,
        risk_lookback_hours=args.risk_lookback_hours,
        taker_fee_bps=args.taker_fee_bps,
        slippage_bps=args.slippage_bps,
        portfolio_notional_usd=args.portfolio_notional_usd,
    )
    report = run(
        args.formula,
        args.snapshot_id,
        args.output,
        config,
        args.cost_scenarios,
        [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
        if args.symbols
        else None,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
