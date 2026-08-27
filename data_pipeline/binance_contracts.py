"""Versioned contracts for read-only Binance Spot research datasets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Sequence


BINANCE_DATASET_SCHEMA_VERSION = "binance-spot-v1"
BINANCE_FEATURE_SCHEMA_VERSION = "binance-features-v1"


@dataclass(frozen=True)
class BinanceUniverseRules:
    venue: str = "binance"
    market_type: str = "spot"
    quote_asset: str = "USDT"
    interval: str = "1h"
    history_days: int = 365
    max_symbols: int = 50
    minimum_symbols: int = 20
    minimum_listing_days: int = 30
    minimum_quote_volume: Decimal = Decimal("10000000")
    excluded_base_assets: tuple[str, ...] = (
        "USDC",
        "FDUSD",
        "TUSD",
        "USDP",
        "DAI",
        "USD1",
        "RLUSD",
        "EURI",
        "USDE",
        "USDS",
        "USDD",
        "PYUSD",
        "GUSD",
        "EURC",
        "AEUR",
        "EUR",
        "TRY",
    )
    excluded_symbol_suffixes: tuple[str, ...] = ("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT")

    def __post_init__(self) -> None:
        if self.venue != "binance" or self.market_type != "spot":
            raise ValueError("The first Binance dataset release supports Binance Spot only")
        if self.quote_asset != "USDT":
            raise ValueError("The first Binance dataset release supports USDT markets only")
        if self.interval != "1h":
            raise ValueError("The canonical Binance research interval is 1h")
        if self.history_days < 365:
            raise ValueError("Binance research history must cover at least one year")
        if self.max_symbols <= 0 or self.minimum_symbols <= 0 or self.minimum_listing_days <= 0:
            raise ValueError("Universe limits must be positive")
        if self.minimum_quote_volume <= 0:
            raise ValueError("minimum_quote_volume must be positive")

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["minimum_quote_volume"] = str(self.minimum_quote_volume)
        value["excluded_base_assets"] = list(self.excluded_base_assets)
        value["excluded_symbol_suffixes"] = list(self.excluded_symbol_suffixes)
        return value


@dataclass(frozen=True)
class BinanceInstrument:
    symbol: str
    status: str
    base_asset: str
    quote_asset: str
    onboard_time: datetime | None
    offboard_time: datetime | None
    quantity_step: Decimal
    minimum_quantity: Decimal
    minimum_notional: Decimal
    tick_size: Decimal
    quote_volume: Decimal
    raw_filters: dict[str, Any]
    venue: str = "binance"
    market_type: str = "spot"


@dataclass(frozen=True)
class BinanceBar:
    symbol: str
    interval: str
    open_time: datetime
    close_time: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    base_volume: Decimal
    quote_volume: Decimal
    trade_count: int
    taker_buy_base_volume: Decimal
    taker_buy_quote_volume: Decimal
    source: str
    venue: str = "binance"
    market_type: str = "spot"


def dataset_snapshot_payload(
    rules: BinanceUniverseRules,
    symbols: Sequence[str],
    start_time: datetime,
    end_time: datetime,
    *,
    source: str,
    code_version: str,
    instruments: Sequence[BinanceInstrument] | None = None,
) -> dict[str, Any]:
    if start_time.tzinfo is None or end_time.tzinfo is None:
        raise ValueError("Dataset snapshot times must be timezone-aware")
    if start_time >= end_time:
        raise ValueError("Dataset snapshot start_time must precede end_time")
    normalized_symbols = sorted({str(symbol).upper() for symbol in symbols})
    if not normalized_symbols:
        raise ValueError("A dataset snapshot requires at least one symbol")
    payload = {
        "schema_version": BINANCE_DATASET_SCHEMA_VERSION,
        "feature_schema_version": BINANCE_FEATURE_SCHEMA_VERSION,
        "rules": rules.as_dict(),
        "symbols": normalized_symbols,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "source": source,
        "code_version": code_version,
    }
    if instruments is not None:
        instrument_symbols = {item.symbol.upper() for item in instruments}
        if instrument_symbols != set(normalized_symbols):
            raise ValueError("Snapshot instruments must match snapshot symbols")
        payload["instruments"] = [
            {
                "selection_rank": rank,
                "venue": item.venue,
                "market_type": item.market_type,
                "symbol": item.symbol,
                "status": item.status,
                "base_asset": item.base_asset,
                "quote_asset": item.quote_asset,
                "onboard_time": item.onboard_time.isoformat() if item.onboard_time else None,
                "offboard_time": item.offboard_time.isoformat() if item.offboard_time else None,
                "quantity_step": str(item.quantity_step),
                "minimum_quantity": str(item.minimum_quantity),
                "minimum_notional": str(item.minimum_notional),
                "tick_size": str(item.tick_size),
                "quote_volume": str(item.quote_volume),
                "raw_filters": item.raw_filters,
            }
            for rank, item in enumerate(instruments, start=1)
        ]
    return payload


def dataset_snapshot_id(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()
