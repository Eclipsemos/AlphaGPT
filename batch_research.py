"""Run the repeatable, non-trading AlphaGPT research workflow.

The workflow is intentionally conservative: refresh data (unless skipped),
train one independent model per seed, summarize all formulas on the test
split, and run walk-forward evaluation for every formula.  All artifacts are
written below one timestamped directory so runs cannot overwrite one another.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from data_pipeline.data_manager import DataManager
from model_core.config import ModelConfig
from model_core.data_loader import CryptoDataLoader
from model_core.engine import AlphaEngine
from model_core.multi_seed import run as multi_seed_run
from model_core.walk_forward import run as walk_forward_run


def parse_seeds(value: str) -> list[int]:
    """Parse a comma-separated seed list while rejecting malformed input."""
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refresh data, train multiple seeds, and evaluate formulas (no trading)."
    )
    parser.add_argument(
        "--seeds",
        type=parse_seeds,
        default=[1, 2, 3],
        help="Comma-separated random seeds (default: 1,2,3)",
    )
    parser.add_argument("--steps", type=int, default=None, help="Training steps per seed")
    parser.add_argument("--batch-size", type=int, default=None, help="Formula batch size per step")
    parser.add_argument("--windows", type=int, default=4, help="Walk-forward windows per formula")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Root directory for this batch (default: runs/<UTC timestamp>)",
    )
    parser.add_argument("--skip-refresh", action="store_true", help="Use existing PostgreSQL data")
    parser.add_argument("--tokens", type=int, default=500, help="Maximum tokens loaded for reports")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.steps is not None and args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.windows <= 0:
        raise ValueError("--windows must be positive")
    if args.tokens <= 0:
        raise ValueError("--tokens must be positive")


async def _refresh_data() -> None:
    manager = DataManager()
    try:
        await manager.initialize()
        await manager.pipeline_sync_daily()
    finally:
        await manager.close()


def _default_output_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("runs") / stamp


def _quality_report(limit_tokens: int) -> dict:
    loader = CryptoDataLoader()
    loader.load_data(limit_tokens=limit_tokens)
    return loader.quality_report.as_dict()


def run_batch(
    *,
    seeds: Sequence[int],
    output_dir: Path,
    steps: int | None = None,
    batch_size: int | None = None,
    windows: int = 4,
    skip_refresh: bool = False,
    tokens: int = 500,
) -> dict:
    """Execute one complete batch and return the JSON-serializable report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not skip_refresh:
        asyncio.run(_refresh_data())

    if steps is not None:
        ModelConfig.TRAIN_STEPS = steps
    if batch_size is not None:
        ModelConfig.BATCH_SIZE = batch_size

    seed_runs: list[dict] = []
    formula_paths: list[str] = []
    for seed in seeds:
        seed_dir = output_dir / f"seed_{seed}"
        print(f"\n=== Training seed {seed} -> {seed_dir} ===")
        engine = AlphaEngine(seed=seed, output_dir=seed_dir)
        engine.train()
        formula_path = seed_dir / "best_meme_strategy.json"
        if not formula_path.is_file():
            raise RuntimeError(f"training seed {seed} did not produce {formula_path}")
        formula_paths.append(str(formula_path))
        seed_runs.append(
            {
                "seed": seed,
                "output_dir": str(seed_dir),
                "formula": str(formula_path),
                "training_history": str(seed_dir / "training_history.json"),
                "evaluation": str(seed_dir / "evaluation_report.json"),
                "checkpoint": str(seed_dir / ModelConfig.CHECKPOINT_PATH),
            }
        )

    multi_seed_path = output_dir / "multi_seed_report.json"
    multi_seed_result = multi_seed_run(formula_paths, str(multi_seed_path))

    walk_forward_results: list[dict] = []
    for run in seed_runs:
        report_path = Path(run["output_dir"]) / "walk_forward_report.json"
        walk_forward_run(run["formula"], str(report_path), windows)
        with report_path.open() as handle:
            walk_forward_results.append({"seed": run["seed"], "report": json.load(handle)})

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seeds": list(seeds),
        "skip_refresh": skip_refresh,
        "steps": ModelConfig.TRAIN_STEPS,
        "batch_size": ModelConfig.BATCH_SIZE,
        "walk_forward_windows": windows,
        "tokens": tokens,
        "output_dir": str(output_dir),
        "runs": seed_runs,
        "multi_seed": {"path": str(multi_seed_path), "result": multi_seed_result},
        "walk_forward": walk_forward_results,
        "data_quality": _quality_report(tokens),
    }
    report_path = output_dir / "batch_report.json"
    with report_path.open("w") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nBatch complete. Report: {report_path}")
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_args(args)
        output_dir = args.output_dir or _default_output_dir()
        run_batch(
            seeds=args.seeds,
            output_dir=output_dir,
            steps=args.steps,
            batch_size=args.batch_size,
            windows=args.windows,
            skip_refresh=args.skip_refresh,
            tokens=args.tokens,
        )
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
