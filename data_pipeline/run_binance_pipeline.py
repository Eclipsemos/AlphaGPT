"""Build or refresh an immutable Binance Spot research dataset.

This command only reads public market data. It never loads credentials or
calls Binance account, order, wallet, Testnet, or Futures endpoints.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from loguru import logger

from .binance_contracts import dataset_snapshot_id, dataset_snapshot_payload
from .binance_archive import (
    BinanceSpotArchiveProvider,
    aggregate_archive_checksum,
    missing_bar_ranges,
)
from .config import Config
from .db_manager import DBManager
from .binance_quality import inspect_symbol_bars
from .providers.binance_spot import BinanceSpotProvider


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest a read-only Binance Spot research snapshot")
    parser.add_argument("--start-time", type=parse_datetime, help="UTC ISO timestamp; defaults to end minus history days")
    parser.add_argument("--end-time", type=parse_datetime, help="UTC ISO timestamp; defaults to the latest completed hour")
    parser.add_argument("--history-days", type=int, default=None)
    parser.add_argument("--max-symbols", type=int, default=None)
    parser.add_argument("--minimum-symbols", type=int, default=None)
    parser.add_argument("--candidate-multiplier", type=int, default=2)
    parser.add_argument("--base-url", default=Config.BINANCE_BASE_URL)
    parser.add_argument(
        "--source",
        choices=("rest", "archive"),
        default="rest",
        help="Bar source: public REST or verified data.binance.vision monthly archives",
    )
    parser.add_argument("--archive-base-url", default=Config.BINANCE_ARCHIVE_BASE_URL)
    parser.add_argument("--output", type=Path, default=Path("runs/binance_latest/dataset_report.json"))
    parser.add_argument("--code-version", default=None)
    parser.add_argument("--minimum-coverage", type=float, default=Config.BINANCE_MIN_COVERAGE)
    parser.add_argument("--dry-run", action="store_true", help="Discover and report symbols without downloading bars")
    return parser


def parse_datetime(value: str) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("timestamp must include a timezone, for example 2026-01-01T00:00:00Z")
    return parsed.astimezone(UTC)


def completed_hour(value: datetime) -> datetime:
    return value.astimezone(UTC).replace(minute=0, second=0, microsecond=0)


def current_code_version() -> str:
    configured = os.getenv("ALPHAGPT_CODE_VERSION")
    if configured:
        return configured
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "working-tree"


async def run(args: argparse.Namespace) -> dict[str, object]:
    rules = Config.BINANCE_RULES
    if (
        args.history_days is not None
        or args.max_symbols is not None
        or args.minimum_symbols is not None
    ):
        rules = replace(
            rules,
            history_days=args.history_days if args.history_days is not None else rules.history_days,
            max_symbols=args.max_symbols if args.max_symbols is not None else rules.max_symbols,
            minimum_symbols=(
                args.minimum_symbols
                if args.minimum_symbols is not None
                else rules.minimum_symbols
            ),
        )
    end_time = completed_hour(args.end_time or datetime.now(UTC))
    start_time = completed_hour(args.start_time or (end_time - timedelta(days=rules.history_days)))
    if start_time >= end_time:
        raise ValueError("start-time must precede end-time")
    if end_time - start_time < timedelta(days=rules.history_days):
        raise ValueError(
            f"Binance snapshot must cover at least {rules.history_days} days; "
            "use a frozen fixture for shorter smoke tests"
        )

    expected_bar_count = int((end_time - start_time).total_seconds() // 3600)
    if not 0 < args.minimum_coverage <= 1:
        raise ValueError("minimum-coverage must be in (0, 1]")
    db = DBManager()
    await db.connect()
    try:
        await db.init_schema()
        async with BinanceSpotProvider(args.base_url) as provider:
            instruments = await provider.discover_instruments(
                rules,
                candidate_multiplier=args.candidate_multiplier,
                as_of=end_time,
            )
            instruments = instruments[: rules.max_symbols]
            if not instruments:
                raise RuntimeError("Binance discovery returned no instruments matching the configured rules")
            if len(instruments) < rules.minimum_symbols:
                raise RuntimeError(
                    f"Binance discovery returned {len(instruments)} instruments, below the "
                    f"configured minimum of {rules.minimum_symbols}; lower the volume "
                    "threshold or collect a broader universe"
                )
            await db.upsert_market_instruments(instruments)
            symbols = [item.symbol for item in instruments]
            canonical = dataset_snapshot_payload(
                rules,
                symbols,
                start_time,
                end_time,
                source=("binance-spot-archive+rest" if args.source == "archive" else provider.source_name),
                code_version=args.code_version or current_code_version(),
                instruments=instruments,
            )
            snapshot_id = dataset_snapshot_id(canonical)
            retrieved_at = datetime.now(UTC)
            coverage: list[dict[str, object]] = []
            downloaded_bars = 0
            quality_reports: list[dict[str, object]] = []
            if args.dry_run:
                coverage = [
                    {
                        "symbol": instrument.symbol,
                        "requested_start_time": start_time,
                        "requested_end_time": end_time,
                        "response_start_time": None,
                        "response_end_time": None,
                        "bar_count": 0,
                        "expected_bar_count": expected_bar_count,
                        "archive_checksum": None,
                        "source_metadata": {"dry_run": True, "provider": provider.source_name},
                        "retrieved_at": retrieved_at,
                    }
                    for instrument in instruments
                ]
            else:
                archive_provider = None
                if args.source == "archive":
                    archive_provider = BinanceSpotArchiveProvider(args.archive_base_url)
                    await archive_provider.__aenter__()
                try:
                    for index, instrument in enumerate(instruments, start=1):
                        latest = await db.latest_market_bar_time(
                            instrument.symbol,
                            rules.interval,
                            before=end_time,
                        )
                        fetch_start = start_time
                        if latest is not None:
                            fetch_start = max(start_time, latest.astimezone(UTC) + timedelta(hours=1))
                        logger.info(
                            "Fetching {}/{}: {} from {}",
                            index,
                            len(instruments),
                            instrument.symbol,
                            fetch_start.isoformat(),
                        )
                        archive_metadata: list[dict[str, object]] = []
                        if args.source == "archive" and fetch_start < end_time:
                            archive_bars, archive_metadata = await archive_provider.monthly_bars(
                                instrument.symbol, rules.interval, fetch_start, end_time
                            )
                            bars_by_time = {bar.open_time: bar for bar in archive_bars}
                            for gap_start, gap_end in missing_bar_ranges(
                                archive_bars, fetch_start, end_time
                            ):
                                rest_bars = await provider.klines(
                                    instrument.symbol, rules.interval, gap_start, gap_end
                                )
                                bars_by_time.update({bar.open_time: bar for bar in rest_bars})
                            bars = [bars_by_time[key] for key in sorted(bars_by_time)]
                        elif args.source == "archive":
                            bars = []
                        else:
                            bars = (
                                await provider.klines(instrument.symbol, rules.interval, fetch_start, end_time)
                                if fetch_start < end_time
                                else []
                            )
                        await db.upsert_market_bars(bars)
                        downloaded_bars += len(bars)
                        stored = await db.market_bar_coverage(
                            instrument.symbol,
                            rules.interval,
                            start_time,
                            end_time,
                        )
                        stored_bars = await db.fetch_market_bars(
                            instrument.symbol,
                            rules.interval,
                            start_time,
                            end_time,
                        )
                        quality = inspect_symbol_bars(
                            instrument,
                            stored_bars,
                            start_time,
                            end_time,
                            minimum_coverage=args.minimum_coverage,
                        )
                        quality_reports.append(quality.as_dict())
                        coverage.append(
                            {
                                "symbol": instrument.symbol,
                                "requested_start_time": start_time,
                                "requested_end_time": end_time,
                                "response_start_time": stored["response_start_time"],
                                "response_end_time": stored["response_end_time"],
                                "bar_count": stored["bar_count"],
                                "expected_bar_count": quality.expected_bar_count,
                                "archive_checksum": aggregate_archive_checksum(archive_metadata),
                                "source_metadata": {
                                    "provider": (
                                        provider.source_name
                                        if args.source == "rest"
                                        else "binance-spot-archive+rest"
                                    ),
                                    "provider_status": provider.status,
                                    "archives": archive_metadata,
                                    "quality": quality.as_dict(),
                                },
                                "retrieved_at": retrieved_at,
                            }
                        )
                finally:
                    if archive_provider is not None:
                        await archive_provider.__aexit__(None, None, None)
            if not args.dry_run:
                quality_passed = all(item["accepted"] for item in quality_reports)
                if quality_passed:
                    await db.create_dataset_snapshot(
                        snapshot_id,
                        canonical,
                        canonical["symbols"],
                        coverage,
                    )
            else:
                quality_passed = None
            report = {
                "snapshot_id": snapshot_id,
                "dry_run": args.dry_run,
                "symbols": symbols,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "interval": rules.interval,
                "downloaded_bars": downloaded_bars,
                "stored_bars": sum(int(item["bar_count"]) for item in coverage),
                "expected_bars": expected_bar_count * len(symbols),
                "quality_passed": quality_passed,
                "quality": quality_reports,
                "snapshot_created": bool(quality_passed),
                "coverage": coverage,
                "provider_status": provider.status,
                "canonical_payload": canonical,
            }
    finally:
        await db.close()
    return report


def json_default(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Cannot encode {type(value).__name__}")


def main() -> None:
    args = build_parser().parse_args()
    report = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, default=json_default) + "\n")
    if report["quality_passed"] is False:
        logger.error("Binance dataset failed quality gates; no research snapshot was created")
        raise SystemExit(2)
    if report["dry_run"]:
        logger.success("Binance discovery complete: {} symbols", len(report["symbols"]))
    else:
        logger.success(
            "Binance snapshot {} complete: {} symbols, {} bars",
            report["snapshot_id"],
            len(report["symbols"]),
            report["stored_bars"],
        )


if __name__ == "__main__":
    main()
