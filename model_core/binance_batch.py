"""Reproducible multi-seed Binance Spot factor research workflow.

Candidate discovery and ranking use train/validation data only. The selected
formula crosses the final-test boundary exactly once, after its identity has
been persisted. This module performs historical research and has no account,
position, fill, or order state.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from .binance_data_loader import BinanceDataLoader
from .binance_engine import BinanceAlphaEngine, BinanceMiningConfig, parse_symbols
from .binance_evaluation import BinanceEvaluationConfig, BinanceFactorEvaluator
from .evaluate_binance import run as run_final_evaluation
from .formula_artifact import build_formula_artifact
from .formula_canonical import canonical_formula
from .vm import StackVM
from .vocab import BINANCE_FORMULA_VOCAB


BATCH_REPORT_VERSION = "binance-factor-batch-v1"
DECISION_REPORT_VERSION = "binance-factor-decision-v1"
DEFAULT_SEEDS = (1, 2, 3, 4, 5)


@dataclass(frozen=True)
class DecisionThresholds:
    minimum_validation_ic: float = 0.0
    minimum_positive_rolling_fraction: float = 0.5
    minimum_median_rolling_sharpe: float = 0.0
    maximum_drawdown: float = 0.5
    maximum_volume_participation: float = 0.01
    minimum_seed_support: int = 2

    def __post_init__(self) -> None:
        if not 0 <= self.minimum_positive_rolling_fraction <= 1:
            raise ValueError("minimum_positive_rolling_fraction must be between zero and one")
        if not 0 < self.maximum_drawdown <= 1:
            raise ValueError("maximum_drawdown must be in (0, 1]")
        if self.maximum_volume_participation <= 0:
            raise ValueError("maximum_volume_participation must be positive")
        if self.minimum_seed_support <= 0:
            raise ValueError("minimum_seed_support must be positive")


def parse_seeds(value: str) -> list[int]:
    seeds: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            seeds.append(int(item))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid seed: {item!r}") from exc
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must be unique")
    return seeds


def parse_nonnegative_floats(value: str) -> list[float]:
    try:
        values = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("cost scenarios must be comma-separated numbers") from exc
    if not values or any(not math.isfinite(item) or item < 0 for item in values):
        raise argparse.ArgumentTypeError("cost scenarios must be finite and non-negative")
    return values


def default_output_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("runs") / "binance" / stamp


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)


def confidence_summary(values: Iterable[float]) -> dict[str, float | int]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"count": 0, "mean": 0.0, "std": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    mean = statistics.fmean(finite)
    std = statistics.stdev(finite) if len(finite) > 1 else 0.0
    margin = 1.96 * std / math.sqrt(len(finite))
    return {
        "count": len(finite),
        "mean": mean,
        "std": std,
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
    }


def aggregate_candidates(seed_candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for run in seed_candidates:
        seed = int(run["seed"])
        for raw in run["candidates"]:
            formula = [int(token) for token in raw["formula"]]
            canonical = canonical_formula(formula, BINANCE_FORMULA_VOCAB)
            group = grouped.setdefault(
                canonical,
                {
                    "canonical_formula": canonical,
                    "formula": formula,
                    "formula_length": len(formula),
                    "occurrences": [],
                },
            )
            if tuple(formula) < tuple(group["formula"]):
                group["formula"] = formula
            group["occurrences"].append(
                {
                    "seed": seed,
                    "train_score": float(raw["train_score"]),
                    "validation_score": float(raw["validation_score"]),
                    "first_seen_step": int(raw["first_seen_step"]),
                }
            )
    result: list[dict[str, Any]] = []
    for group in grouped.values():
        occurrences = group["occurrences"]
        validation = [item["validation_score"] for item in occurrences]
        train = [item["train_score"] for item in occurrences]
        group["seeds"] = sorted({item["seed"] for item in occurrences})
        group["seed_support"] = len(group["seeds"])
        group["occurrence_count"] = len(occurrences)
        group["train_score"] = confidence_summary(train)
        group["validation_score"] = confidence_summary(validation)
        result.append(group)
    return sorted(
        result,
        key=lambda item: (
            -float(item["validation_score"]["mean"]),
            -int(item["seed_support"]),
            int(item["formula_length"]),
            str(item["canonical_formula"]),
        ),
    )


def walk_forward_windows(length: int, count: int) -> dict[str, list[tuple[int, int]]]:
    if length < 2:
        raise ValueError("walk-forward evaluation requires at least two validation hours")
    if count <= 0:
        raise ValueError("walk-forward window count must be positive")
    count = min(count, length)
    boundaries = [index * length // count for index in range(count + 1)]
    rolling = [
        (boundaries[index], boundaries[index + 1])
        for index in range(count)
        if boundaries[index + 1] > boundaries[index]
    ]
    anchored = [(0, end) for _, end in rolling]
    return {"rolling": rolling, "anchored": anchored}


def _slice_raw(raw: dict[str, torch.Tensor], start: int, end: int) -> dict[str, torch.Tensor]:
    return {key: value[:, start:end] for key, value in raw.items()}


def _report_summary(reports: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        return {"valid_window_count": 0, "failed_window_count": 0}
    metrics = (
        "score",
        "cumulative_return",
        "sharpe",
        "max_drawdown",
        "mean_rank_ic",
        "total_turnover",
        "maximum_volume_participation",
    )
    valid = [item for item in reports if "metrics" in item]
    summary: dict[str, Any] = {
        "valid_window_count": len(valid),
        "failed_window_count": len(reports) - len(valid),
    }
    for metric in metrics:
        values = [float(item["metrics"][metric]) for item in valid]
        summary[metric] = confidence_summary(values)
        summary[metric]["median"] = statistics.median(values) if values else 0.0
        summary[metric]["minimum"] = min(values) if values else 0.0
        summary[metric]["maximum"] = max(values) if values else 0.0
    summary["positive_sharpe_fraction"] = (
        sum(float(item["metrics"]["sharpe"]) > 0 for item in valid) / len(valid)
        if valid
        else 0.0
    )
    return summary


def evaluate_validation_walk_forward(
    candidate: dict[str, Any],
    loader: BinanceDataLoader,
    config: BinanceEvaluationConfig,
    window_count: int,
) -> dict[str, Any]:
    """Evaluate one candidate without reading any test split attribute."""
    vm = StackVM(BINANCE_FORMULA_VOCAB)
    factors = vm.execute(candidate["formula"], loader.validation_feat_tensor)
    if factors is None:
        raise ValueError("Candidate formula is invalid on the validation feature tensor")
    length = int(factors.shape[-1])
    schedule = walk_forward_windows(length, window_count)
    evaluator = BinanceFactorEvaluator(config)
    modes: dict[str, Any] = {}
    absolute_start = int(loader.splits.validation.start)
    validation_times = loader.times[loader.splits.validation]
    for mode, windows in schedule.items():
        reports: list[dict[str, Any]] = []
        for index, (start, end) in enumerate(windows):
            entry: dict[str, Any] = {
                "window": index,
                "relative_start": start,
                "relative_end": end,
                "absolute_start": absolute_start + start,
                "absolute_end": absolute_start + end,
                "start_time": validation_times[start].isoformat(),
                "end_time_exclusive": (
                    validation_times[end].isoformat()
                    if end < len(validation_times)
                    else (loader.times[absolute_start + end - 1] + timedelta(hours=1)).isoformat()
                ),
            }
            try:
                report = evaluator.evaluate(
                    factors[:, start:end],
                    _slice_raw(loader.validation_raw_data_cache, start, end),
                    loader.validation_target_ret[:, start:end],
                    loader.validation_target_valid[:, start:end],
                    loader.validation_signal_valid[:, start:end],
                    loader.symbols,
                )
                entry["metrics"] = report.as_dict()
            except ValueError as exc:
                entry["error"] = str(exc)
            reports.append(entry)
        modes[mode] = {"windows": reports, "summary": _report_summary(reports)}
    return modes


def rank_walk_forward_candidates(candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(item: dict[str, Any]) -> tuple[Any, ...]:
        rolling = item["walk_forward"]["rolling"]["summary"]
        valid = int(rolling["valid_window_count"])
        expected = valid + int(rolling["failed_window_count"])
        complete = int(valid == expected and valid > 0)
        sharpe = rolling.get("sharpe", {})
        score = rolling.get("score", {})
        return (
            -complete,
            -float(rolling.get("positive_sharpe_fraction", 0.0)),
            -float(sharpe.get("median", -math.inf)),
            -float(score.get("minimum", -math.inf)),
            -float(item["validation_score"]["mean"]),
            -int(item["seed_support"]),
            int(item["formula_length"]),
            str(item["canonical_formula"]),
        )

    return sorted(candidates, key=key)


def build_decision_report(
    selected: dict[str, Any],
    final_evaluation: dict[str, Any],
    thresholds: DecisionThresholds,
) -> dict[str, Any]:
    rolling = selected["walk_forward"]["rolling"]["summary"]
    rolling_sharpe = rolling.get("sharpe", {})
    rolling_drawdown = rolling.get("max_drawdown", {})
    test = final_evaluation["splits"]["test"]
    baselines = final_evaluation["test_baselines"]
    sensitivity = final_evaluation["test_cost_sensitivity"]
    highest_cost_key = max(sensitivity, key=lambda key: float(key.removesuffix("_bps")))
    comparable_baseline_score = max(float(value["score"]) for value in baselines.values())

    def gate(actual: float | int, operator: str, threshold: float | int, passed: bool) -> dict[str, Any]:
        return {"actual": actual, "operator": operator, "threshold": threshold, "passed": passed}

    pretest = {
        "validation_ic": gate(
            float(selected["validation_score"]["mean"]),
            ">",
            thresholds.minimum_validation_ic,
            float(selected["validation_score"]["mean"]) > thresholds.minimum_validation_ic,
        ),
        "positive_rolling_fraction": gate(
            float(rolling.get("positive_sharpe_fraction", 0.0)),
            ">=",
            thresholds.minimum_positive_rolling_fraction,
            float(rolling.get("positive_sharpe_fraction", 0.0))
            >= thresholds.minimum_positive_rolling_fraction,
        ),
        "median_rolling_sharpe": gate(
            float(rolling_sharpe.get("median", 0.0)),
            ">",
            thresholds.minimum_median_rolling_sharpe,
            float(rolling_sharpe.get("median", 0.0))
            > thresholds.minimum_median_rolling_sharpe,
        ),
        "rolling_drawdown": gate(
            float(rolling_drawdown.get("minimum", -1.0)),
            ">=",
            -thresholds.maximum_drawdown,
            float(rolling_drawdown.get("minimum", -1.0)) >= -thresholds.maximum_drawdown,
        ),
        "cross_seed_support": gate(
            int(selected["seed_support"]),
            ">=",
            thresholds.minimum_seed_support,
            int(selected["seed_support"]) >= thresholds.minimum_seed_support,
        ),
    }
    final = {
        "test_return": gate(float(test["cumulative_return"]), ">", 0.0, float(test["cumulative_return"]) > 0),
        "test_sharpe": gate(float(test["sharpe"]), ">", 0.0, float(test["sharpe"]) > 0),
        "test_drawdown": gate(
            float(test["max_drawdown"]),
            ">=",
            -thresholds.maximum_drawdown,
            float(test["max_drawdown"]) >= -thresholds.maximum_drawdown,
        ),
        "capacity_proxy": gate(
            float(test["maximum_volume_participation"]),
            "<=",
            thresholds.maximum_volume_participation,
            float(test["maximum_volume_participation"])
            <= thresholds.maximum_volume_participation,
        ),
        "highest_cost_return": gate(
            float(sensitivity[highest_cost_key]["cumulative_return"]),
            ">",
            0.0,
            float(sensitivity[highest_cost_key]["cumulative_return"]) > 0,
        ),
        "baseline_score": gate(
            float(test["score"]),
            ">",
            comparable_baseline_score,
            float(test["score"]) > comparable_baseline_score,
        ),
    }
    failed_pretest = [name for name, value in pretest.items() if not value["passed"]]
    failed_final = [name for name, value in final.items() if not value["passed"]]
    if failed_pretest:
        status = "reject"
    elif failed_final:
        status = "research-only"
    else:
        status = "promising"
    return {
        "report_version": DECISION_REPORT_VERSION,
        "status": status,
        "formula": selected["formula"],
        "canonical_formula": selected["canonical_formula"],
        "pretest_gates": pretest,
        "final_test_gates": final,
        "failed_gates": {"pretest": failed_pretest, "final_test": failed_final},
        "highest_cost_scenario": highest_cost_key,
        "test_was_used_for_candidate_selection": False,
        "disclaimer": "Historical factor-research status only; this is not a trading or profitability claim.",
    }


def experiment_environment() -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=repository,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unknown", None
    packages = {}
    for name in ("torch", "numpy", "pandas", "sqlalchemy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "missing"
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "torch_device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _load_completed_seed(
    seed_dir: Path,
    metadata: dict[str, Any],
    seed: int,
    mining_config: BinanceMiningConfig,
) -> list[dict[str, Any]] | None:
    candidates_path = seed_dir / "candidates.json"
    if not candidates_path.is_file():
        return None
    payload = json.loads(candidates_path.read_text())
    if payload.get("research_metadata") != metadata:
        raise ValueError(f"Completed seed artifact is incompatible: {candidates_path}")
    if payload.get("seed") != seed:
        raise ValueError(f"Completed seed artifact has the wrong seed: {candidates_path}")
    if payload.get("mining_config") != asdict(mining_config):
        raise ValueError(f"Completed seed artifact has an incompatible mining config: {candidates_path}")
    if payload.get("test_was_accessed") is not False:
        raise ValueError(f"Completed seed artifact does not prove test isolation: {candidates_path}")
    return payload.get("candidates")


def run_batch(
    *,
    snapshot_id: str,
    seeds: Sequence[int],
    output_dir: str | Path,
    mining_config: BinanceMiningConfig,
    evaluation_config: BinanceEvaluationConfig,
    symbols: list[str] | None = None,
    window_count: int = 4,
    shortlist_size: int = 25,
    cost_scenarios: Sequence[float] = (0.0, 15.0, 30.0),
    thresholds: DecisionThresholds | None = None,
    resume: bool = False,
    use_lord_regularization: bool = True,
) -> dict[str, Any]:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Batch seeds must be non-empty and unique")
    if window_count <= 0 or shortlist_size <= 0:
        raise ValueError("window_count and shortlist_size must be positive")
    if any(value < 0 for value in cost_scenarios):
        raise ValueError("cost scenarios cannot be negative")
    thresholds = thresholds or DecisionThresholds()
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()) and not resume:
        raise ValueError(f"Output directory is not empty; use --resume or choose another path: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc)
    monotonic_start = time.monotonic()
    loader = BinanceDataLoader(snapshot_id, symbols=symbols)
    loader.load_data()
    if len(loader.symbols) < 2:
        raise ValueError("Binance batch research requires at least two symbols")
    if "BTCUSDT" not in loader.symbols:
        raise ValueError("BTCUSDT must be present for the fixed reference baseline")
    manifest = {
        "report_version": BATCH_REPORT_VERSION,
        "status": "running",
        "created_at": started_at.isoformat(),
        "command": list(sys.argv),
        "market": "binance-spot",
        "interval": "1h",
        "snapshot_id": loader.snapshot_id,
        "symbols": loader.symbols,
        "seeds": [int(seed) for seed in seeds],
        "mining_config": asdict(mining_config),
        "evaluation_config": asdict(evaluation_config),
        "window_count": window_count,
        "shortlist_size": shortlist_size,
        "cost_scenarios_bps": [float(value) for value in cost_scenarios],
        "decision_thresholds": asdict(thresholds),
        "environment": experiment_environment(),
    }
    manifest_path = destination / "experiment_manifest.json"
    write_json(manifest_path, manifest)
    seed_payloads: list[dict[str, Any]] = []
    seed_runs: list[dict[str, Any]] = []
    try:
        for seed in seeds:
            seed_dir = destination / f"seed_{int(seed)}"
            completed = (
                _load_completed_seed(seed_dir, loader.research_metadata, int(seed), mining_config)
                if resume
                else None
            )
            resumed = False
            if completed is None:
                engine = BinanceAlphaEngine(
                    loader.snapshot_id,
                    symbols=loader.symbols,
                    seed=int(seed),
                    output_dir=seed_dir,
                    config=mining_config,
                    use_lord_regularization=use_lord_regularization,
                )
                resumed = resume and engine.checkpoint_path.is_file()
                engine.train(resume=resumed)
                completed = json.loads((seed_dir / "candidates.json").read_text())["candidates"]
            seed_payloads.append({"seed": int(seed), "candidates": completed})
            seed_runs.append(
                {
                    "seed": int(seed),
                    "directory": str(seed_dir),
                    "resumed_from_checkpoint": resumed,
                    "candidate_count": len(completed),
                    "best_formula": str(seed_dir / "best_formula.json"),
                    "candidates": str(seed_dir / "candidates.json"),
                    "history": str(seed_dir / "training_history.json"),
                    "checkpoint": str(seed_dir / "binance_training_checkpoint.pt"),
                }
            )

        aggregated = aggregate_candidates(seed_payloads)
        if not aggregated:
            raise RuntimeError("No valid Binance candidates were produced by any seed")
        write_json(
            destination / "candidate_aggregation.json",
            {
                "research_metadata": loader.research_metadata,
                "candidate_count": len(aggregated),
                "candidates": aggregated,
                "test_was_accessed": False,
            },
        )
        shortlist = aggregated[: min(shortlist_size, len(aggregated))]
        evaluated = []
        for candidate in shortlist:
            value = dict(candidate)
            value["walk_forward"] = evaluate_validation_walk_forward(
                value, loader, evaluation_config, window_count
            )
            evaluated.append(value)
        ranked = rank_walk_forward_candidates(evaluated)
        selected = ranked[0]
        walk_forward_report = {
            "research_metadata": loader.research_metadata,
            "selection_data": "validation-only",
            "test_was_accessed": False,
            "candidate_count": len(ranked),
            "candidates": ranked,
        }
        write_json(destination / "walk_forward_report.json", walk_forward_report)

        artifact = build_formula_artifact(
            selected["formula"], BINANCE_FORMULA_VOCAB, loader.research_metadata
        )
        artifact["canonical_formula"] = selected["canonical_formula"]
        artifact["discovery"] = {
            "engine": BATCH_REPORT_VERSION,
            "seeds": selected["seeds"],
            "seed_support": selected["seed_support"],
            "mining_config": asdict(mining_config),
        }
        artifact["selection"] = {
            "criterion": "validation_ic_then_rolling_walk_forward",
            "validation_score": selected["validation_score"],
            "rolling_summary": selected["walk_forward"]["rolling"]["summary"],
            "test_was_accessed": False,
        }
        selected_path = destination / "selected_formula.json"
        write_json(selected_path, artifact)

        final_path = destination / "final_evaluation_report.json"
        final_evaluation = run_final_evaluation(
            selected_path,
            loader.snapshot_id,
            final_path,
            evaluation_config,
            cost_scenarios,
            loader.symbols,
        )
        decision = build_decision_report(selected, final_evaluation, thresholds)
        write_json(destination / "decision_report.json", decision)
        seed_winner_scores = [
            max(item["validation_score"] for item in payload["candidates"])
            for payload in seed_payloads
            if payload["candidates"]
        ]
        finished_at = datetime.now(timezone.utc)
        result = {
            "report_version": BATCH_REPORT_VERSION,
            "status": "complete",
            "created_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "wall_clock_seconds": time.monotonic() - monotonic_start,
            "market": "binance-spot",
            "interval": "1h",
            "research_metadata": loader.research_metadata,
            "seeds": [int(seed) for seed in seeds],
            "seed_runs": seed_runs,
            "cross_seed_stability": {
                "seed_winner_validation_score": confidence_summary(seed_winner_scores),
                "unique_candidate_count": len(aggregated),
                "multi_seed_candidate_count": sum(item["seed_support"] > 1 for item in aggregated),
                "selected_seed_support": selected["seed_support"],
            },
            "selected_formula": str(selected_path),
            "selection": {
                "canonical_formula": selected["canonical_formula"],
                "validation_score": selected["validation_score"],
                "walk_forward": selected["walk_forward"],
                "test_was_accessed": False,
            },
            "final_evaluation": str(final_path),
            "decision_report": str(destination / "decision_report.json"),
            "decision": decision,
            "artifacts": {
                "manifest": str(manifest_path),
                "candidate_aggregation": str(destination / "candidate_aggregation.json"),
                "walk_forward": str(destination / "walk_forward_report.json"),
            },
        }
        write_json(destination / "batch_report.json", result)
        manifest.update(
            {
                "status": "complete",
                "finished_at": finished_at.isoformat(),
                "wall_clock_seconds": result["wall_clock_seconds"],
                "batch_report": str(destination / "batch_report.json"),
            }
        )
        write_json(manifest_path, manifest)
        return result
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "wall_clock_seconds": time.monotonic() - monotonic_start,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        write_json(manifest_path, manifest)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mine and evaluate Binance Spot factors across independent seeds (research only)"
    )
    parser.add_argument("--market", choices=("binance-spot",), default="binance-spot")
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--interval", choices=("1h",), default="1h")
    parser.add_argument("--symbols", type=parse_symbols, default=None)
    parser.add_argument("--seeds", type=parse_seeds, default=list(DEFAULT_SEEDS))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--windows", type=int, default=4)
    parser.add_argument("--shortlist-size", type=int, default=25)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-lord", action="store_true")
    parser.add_argument("--max-positions", type=int, default=10)
    parser.add_argument("--weighting", choices=("equal", "risk"), default="equal")
    parser.add_argument("--rebalance-hours", type=int, default=24)
    parser.add_argument("--risk-lookback-hours", type=int, default=24)
    parser.add_argument("--taker-fee-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--portfolio-notional-usd", type=float, default=100_000.0)
    parser.add_argument("--cost-scenarios", type=parse_nonnegative_floats, default=[0.0, 15.0, 30.0])
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run_batch(
            snapshot_id=args.snapshot_id,
            symbols=args.symbols,
            seeds=args.seeds,
            output_dir=args.output_dir or default_output_dir(),
            mining_config=BinanceMiningConfig(steps=args.steps, batch_size=args.batch_size),
            evaluation_config=BinanceEvaluationConfig(
                max_positions=args.max_positions,
                weighting=args.weighting,
                rebalance_hours=args.rebalance_hours,
                risk_lookback_hours=args.risk_lookback_hours,
                taker_fee_bps=args.taker_fee_bps,
                slippage_bps=args.slippage_bps,
                portfolio_notional_usd=args.portfolio_notional_usd,
            ),
            window_count=args.windows,
            shortlist_size=args.shortlist_size,
            cost_scenarios=args.cost_scenarios,
            resume=args.resume,
            use_lord_regularization=not args.no_lord,
        )
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps({"batch_report": report["artifacts"], "decision": report["decision"]}, indent=2))


if __name__ == "__main__":
    main()
