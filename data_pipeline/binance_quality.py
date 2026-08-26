"""Deterministic quality gates for Binance research bars."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, Sequence

from .binance_contracts import BinanceBar, BinanceInstrument


@dataclass(frozen=True)
class SymbolQualityReport:
    symbol: str
    accepted: bool
    bar_count: int
    expected_bar_count: int
    coverage_ratio: float
    effective_start_time: str
    effective_end_time: str
    requested_end_time: str
    first_open_time: str | None
    last_open_time: str | None
    duplicate_rows: int
    non_monotonic_rows: int
    misaligned_rows: int
    out_of_range_rows: int
    pre_listing_rows: int
    post_delisting_rows: int
    incomplete_rows: int
    invalid_ohlc_rows: int
    nonpositive_price_rows: int
    negative_volume_rows: int
    invalid_trade_count_rows: int
    invalid_taker_volume_rows: int
    invalid_identity_rows: int
    missing_bar_count: int
    missing_bar_examples: tuple[str, ...]
    period_coverage: tuple[dict[str, Any], ...]
    reasons: tuple[str, ...]
    fingerprint: str

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["missing_bar_examples"] = list(self.missing_bar_examples)
        value["reasons"] = list(self.reasons)
        value["period_coverage"] = list(self.period_coverage)
        return value


def inspect_symbol_bars(
    instrument: BinanceInstrument,
    bars: Sequence[BinanceBar],
    start_time: datetime,
    end_time: datetime,
    *,
    minimum_coverage: float,
) -> SymbolQualityReport:
    if not 0 < minimum_coverage <= 1:
        raise ValueError("minimum_coverage must be in (0, 1]")
    if start_time.tzinfo is None or end_time.tzinfo is None:
        raise ValueError("quality bounds must be timezone-aware")
    start_time = start_time.astimezone(UTC)
    end_time = end_time.astimezone(UTC)
    if start_time >= end_time:
        raise ValueError("quality start_time must precede end_time")

    effective_start = start_time
    if instrument.onboard_time is not None:
        onboard = instrument.onboard_time.astimezone(UTC)
        listing_hour = onboard.replace(minute=0, second=0, microsecond=0)
        effective_start = max(effective_start, listing_hour)
    effective_end = end_time
    if instrument.offboard_time is not None:
        offboard = instrument.offboard_time.astimezone(UTC)
        delisting_hour = offboard.replace(minute=0, second=0, microsecond=0)
        effective_end = min(effective_end, delisting_hour)
    expected_times = set(hourly_grid(effective_start, effective_end))
    expected_count = len(expected_times)
    observed_times = [bar.open_time.astimezone(UTC) for bar in bars]
    unique_times = set(observed_times)
    duplicate_rows = len(observed_times) - len(unique_times)
    non_monotonic_rows = sum(
        current <= previous for previous, current in zip(observed_times, observed_times[1:])
    )
    misaligned_rows = sum(
        value.minute != 0 or value.second != 0 or value.microsecond != 0
        for value in observed_times
    )
    out_of_range_rows = sum(value < start_time or value >= end_time for value in observed_times)
    pre_listing_rows = 0
    if instrument.onboard_time is not None:
        listing_hour = instrument.onboard_time.astimezone(UTC).replace(minute=0, second=0, microsecond=0)
        pre_listing_rows = sum(value < listing_hour for value in observed_times)
    post_delisting_rows = 0
    if instrument.offboard_time is not None:
        delisting_hour = instrument.offboard_time.astimezone(UTC).replace(
            minute=0, second=0, microsecond=0
        )
        post_delisting_rows = sum(value >= delisting_hour for value in observed_times)
    incomplete_rows = sum(
        bar.close_time.astimezone(UTC) >= end_time
        or bar.close_time <= bar.open_time
        or bar.close_time.astimezone(UTC) > bar.open_time.astimezone(UTC) + timedelta(hours=1)
        for bar in bars
    )
    invalid_identity_rows = sum(
        bar.symbol.upper() != instrument.symbol.upper()
        or bar.interval != "1h"
        for bar in bars
    )
    nonpositive_price_rows = sum(
        any(value <= 0 for value in (bar.open, bar.high, bar.low, bar.close))
        for bar in bars
    )
    invalid_ohlc_rows = sum(
        bar.low > min(bar.open, bar.close)
        or bar.high < max(bar.open, bar.close)
        or bar.low > bar.high
        for bar in bars
    )
    negative_volume_rows = sum(
        any(
            value < 0
            for value in (
                bar.base_volume,
                bar.quote_volume,
                bar.taker_buy_base_volume,
                bar.taker_buy_quote_volume,
            )
        )
        for bar in bars
    )
    invalid_trade_count_rows = sum(bar.trade_count < 0 for bar in bars)
    invalid_taker_volume_rows = sum(
        bar.taker_buy_base_volume > bar.base_volume
        or bar.taker_buy_quote_volume > bar.quote_volume
        for bar in bars
    )
    missing = sorted(expected_times - unique_times)
    observed_in_range = len(expected_times & unique_times)
    coverage_ratio = observed_in_range / expected_count if expected_count else 0.0
    expected_by_period: dict[str, set[datetime]] = {}
    for value in expected_times:
        expected_by_period.setdefault(value.strftime("%Y-%m"), set()).add(value)
    period_coverage = tuple(
        {
            "period": period,
            "bar_count": len(values & unique_times),
            "expected_bar_count": len(values),
            "missing_bar_count": len(values - unique_times),
            "coverage_ratio": len(values & unique_times) / len(values),
        }
        for period, values in sorted(expected_by_period.items())
    )

    failures = {
        "duplicate bars": duplicate_rows,
        "non-monotonic bars": non_monotonic_rows,
        "misaligned bars": misaligned_rows,
        "out-of-range bars": out_of_range_rows,
        "pre-listing bars": pre_listing_rows,
        "post-delisting bars": post_delisting_rows,
        "incomplete bars": incomplete_rows,
        "invalid OHLC bars": invalid_ohlc_rows,
        "nonpositive-price bars": nonpositive_price_rows,
        "negative-volume bars": negative_volume_rows,
        "invalid trade-count bars": invalid_trade_count_rows,
        "invalid taker-volume bars": invalid_taker_volume_rows,
        "invalid symbol or interval bars": invalid_identity_rows,
    }
    reasons = [f"{count} {label}" for label, count in failures.items() if count]
    if expected_count == 0:
        reasons.append("no expected bars after applying listing boundary")
    elif coverage_ratio < minimum_coverage:
        reasons.append(
            f"coverage {coverage_ratio:.6f} is below minimum {minimum_coverage:.6f}"
        )
    failed_periods = [
        item for item in period_coverage if item["coverage_ratio"] < minimum_coverage
    ]
    if failed_periods:
        periods = ", ".join(str(item["period"]) for item in failed_periods)
        reasons.append(f"period coverage below minimum for: {periods}")
    return SymbolQualityReport(
        symbol=instrument.symbol,
        accepted=not reasons,
        bar_count=observed_in_range,
        expected_bar_count=expected_count,
        coverage_ratio=coverage_ratio,
        effective_start_time=effective_start.isoformat(),
        effective_end_time=effective_end.isoformat(),
        requested_end_time=end_time.isoformat(),
        first_open_time=min(unique_times).isoformat() if unique_times else None,
        last_open_time=max(unique_times).isoformat() if unique_times else None,
        duplicate_rows=duplicate_rows,
        non_monotonic_rows=non_monotonic_rows,
        misaligned_rows=misaligned_rows,
        out_of_range_rows=out_of_range_rows,
        pre_listing_rows=pre_listing_rows,
        post_delisting_rows=post_delisting_rows,
        incomplete_rows=incomplete_rows,
        invalid_ohlc_rows=invalid_ohlc_rows,
        nonpositive_price_rows=nonpositive_price_rows,
        negative_volume_rows=negative_volume_rows,
        invalid_trade_count_rows=invalid_trade_count_rows,
        invalid_taker_volume_rows=invalid_taker_volume_rows,
        invalid_identity_rows=invalid_identity_rows,
        missing_bar_count=len(missing),
        missing_bar_examples=tuple(value.isoformat() for value in missing[:20]),
        period_coverage=period_coverage,
        reasons=tuple(reasons),
        fingerprint=bars_fingerprint(bars),
    )


def hourly_grid(start_time: datetime, end_time: datetime):
    cursor = start_time.astimezone(UTC)
    while cursor < end_time:
        yield cursor
        cursor += timedelta(hours=1)


def bars_fingerprint(bars: Sequence[BinanceBar]) -> str:
    payload = [
        {
            "venue": bar.venue,
            "market_type": bar.market_type,
            "symbol": bar.symbol,
            "interval": bar.interval,
            "open_time": bar.open_time.astimezone(UTC).isoformat(),
            "close_time": bar.close_time.astimezone(UTC).isoformat(),
            "open": decimal_text(bar.open),
            "high": decimal_text(bar.high),
            "low": decimal_text(bar.low),
            "close": decimal_text(bar.close),
            "base_volume": decimal_text(bar.base_volume),
            "quote_volume": decimal_text(bar.quote_volume),
            "trade_count": bar.trade_count,
            "taker_buy_base_volume": decimal_text(bar.taker_buy_base_volume),
            "taker_buy_quote_volume": decimal_text(bar.taker_buy_quote_volume),
            "source": bar.source,
        }
        for bar in sorted(bars, key=lambda item: (item.symbol, item.open_time))
    ]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def decimal_text(value: Decimal) -> str:
    return format(value, "f")
