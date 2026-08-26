"""Read-only Binance Spot public market-data provider."""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from email.utils import parsedate_to_datetime
from typing import Any, Awaitable, Callable

import aiohttp

from ..binance_contracts import BinanceBar, BinanceInstrument, BinanceUniverseRules


INTERVAL_MILLISECONDS = {"1h": 60 * 60 * 1000}


class BinancePublicAPIError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(f"Binance public API {status}: {message}")
        self.status = status
        self.message = message


class BinanceSpotProvider:
    source_name = "binance-spot-rest"

    def __init__(
        self,
        base_url: str = "https://data-api.binance.vision",
        *,
        session: aiohttp.ClientSession | None = None,
        max_attempts: int = 5,
        request_spacing_seconds: float = 0.05,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        self.base_url = base_url.rstrip("/")
        self._session = session
        self._owns_session = session is None
        self.max_attempts = max_attempts
        self.request_spacing_seconds = request_spacing_seconds
        self._sleep = sleep
        self._monotonic = monotonic
        self._cooldown_until = 0.0
        self._request_lock = asyncio.Lock()
        self.request_count = 0
        self.rate_limit_count = 0
        self.last_status: int | None = None

    async def __aenter__(self) -> "BinanceSpotProvider":
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30), trust_env=True)
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._owns_session and self._session is not None:
            await self._session.close()
            self._session = None

    @property
    def status(self) -> dict[str, Any]:
        return {
            "requests": self.request_count,
            "rate_limits": self.rate_limit_count,
            "last_status": self.last_status,
            "cooldown_seconds": max(0.0, self._cooldown_until - self._monotonic()),
        }

    async def exchange_info(self) -> dict[str, Any]:
        value = await self._request_json("/api/v3/exchangeInfo")
        if not isinstance(value, dict):
            raise BinancePublicAPIError(200, "exchangeInfo returned a non-object payload")
        return value

    async def tickers_24h(self) -> list[dict[str, Any]]:
        value = await self._request_json("/api/v3/ticker/24hr")
        if not isinstance(value, list):
            raise BinancePublicAPIError(200, "ticker/24hr returned a non-list payload")
        return value

    async def discover_instruments(
        self,
        rules: BinanceUniverseRules,
        *,
        candidate_multiplier: int = 2,
        as_of: datetime | None = None,
    ) -> list[BinanceInstrument]:
        if candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be positive")
        reference_time = as_of or datetime.now(UTC)
        if reference_time.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        reference_time = reference_time.astimezone(UTC)
        exchange_info, tickers = await asyncio.gather(self.exchange_info(), self.tickers_24h())
        quote_volume = {
            str(row.get("symbol", "")): Decimal(str(row.get("quoteVolume", "0")))
            for row in tickers
            if isinstance(row, dict)
        }
        candidates = [
            instrument
            for row in exchange_info.get("symbols", [])
            if (instrument := parse_instrument(row, quote_volume.get(str(row.get("symbol", "")), Decimal("0"))))
            and instrument_matches_rules(instrument, rules, check_listing_age=False)
        ]
        candidates.sort(key=lambda item: (-item.quote_volume, item.symbol))

        selected: list[BinanceInstrument] = []
        target_count = rules.max_symbols * candidate_multiplier
        for instrument in candidates:
            if instrument.onboard_time is None:
                listed = await self.listing_time(instrument.symbol)
                instrument = replace(instrument, onboard_time=listed)
            if instrument_matches_rules(instrument, rules, as_of=reference_time):
                selected.append(instrument)
            if len(selected) >= target_count:
                break
        return selected

    async def listing_time(self, symbol: str) -> datetime | None:
        payload = await self._request_json(
            "/api/v3/klines",
            {"symbol": symbol, "interval": "1h", "startTime": 0, "limit": 1},
        )
        if not isinstance(payload, list) or not payload:
            return None
        return timestamp_to_datetime(int(payload[0][0]))

    async def klines(
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: datetime,
    ) -> list[BinanceBar]:
        interval_ms = INTERVAL_MILLISECONDS.get(interval)
        if interval_ms is None:
            raise ValueError(f"Unsupported Binance interval: {interval}")
        if start_time.tzinfo is None or end_time.tzinfo is None:
            raise ValueError("Kline bounds must be timezone-aware")
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)
        if start_ms >= end_ms:
            raise ValueError("Kline start_time must precede end_time")

        bars: list[BinanceBar] = []
        cursor = start_ms
        while cursor < end_ms:
            payload = await self._request_json(
                "/api/v3/klines",
                {
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": cursor,
                    "endTime": end_ms - 1,
                    "limit": 1000,
                },
            )
            if not isinstance(payload, list):
                raise BinancePublicAPIError(200, f"klines returned a non-list payload for {symbol}")
            page = [parse_kline(symbol, interval, row, self.source_name) for row in payload]
            page = [bar for bar in page if start_time <= bar.open_time < end_time and bar.close_time < end_time]
            if not page:
                break
            if bars and page[0].open_time <= bars[-1].open_time:
                page = [bar for bar in page if bar.open_time > bars[-1].open_time]
            if not page:
                break
            bars.extend(page)
            next_cursor = int(page[-1].open_time.timestamp() * 1000) + interval_ms
            if next_cursor <= cursor:
                raise RuntimeError(f"Binance kline pagination did not advance for {symbol}")
            cursor = next_cursor
            if len(payload) < 1000:
                break
        return bars

    async def _request_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        if self._session is None:
            raise RuntimeError("BinanceSpotProvider must be used as an async context manager")
        for attempt in range(self.max_attempts):
            async with self._request_lock:
                cooldown = self._cooldown_until - self._monotonic()
                if cooldown > 0:
                    await self._sleep(cooldown)
                async with self._session.get(f"{self.base_url}{path}", params=params) as response:
                    self.request_count += 1
                    self.last_status = response.status
                    try:
                        payload = await response.json(content_type=None)
                    except (ValueError, aiohttp.ContentTypeError):
                        payload = {"msg": await response.text()}
                    if response.status not in {418, 429}:
                        if response.status >= 400:
                            message = payload.get("msg", payload) if isinstance(payload, dict) else payload
                            raise BinancePublicAPIError(response.status, str(message))
                        if self.request_spacing_seconds > 0:
                            await self._sleep(self.request_spacing_seconds)
                        return payload
                    self.rate_limit_count += 1
                    delay = retry_delay_seconds(response.status, response.headers, payload, attempt)
                    self._cooldown_until = max(self._cooldown_until, self._monotonic() + delay)
            if attempt + 1 < self.max_attempts:
                await self._sleep(max(0.0, self._cooldown_until - self._monotonic()))
        raise BinancePublicAPIError(self.last_status or 429, "rate limit persisted after retries")


def timestamp_to_datetime(value: int) -> datetime:
    milliseconds = value // 1000 if value > 10_000_000_000_000 else value
    return datetime.fromtimestamp(milliseconds / 1000, UTC)


def parse_instrument(row: dict[str, Any], quote_volume: Decimal) -> BinanceInstrument | None:
    try:
        filters = {item["filterType"]: item for item in row.get("filters", [])}
        lot = filters.get("LOT_SIZE", {})
        notional = filters.get("NOTIONAL") or filters.get("MIN_NOTIONAL") or {}
        price_filter = filters.get("PRICE_FILTER", {})
        onboard_value = row.get("onboardDate")
        offboard_value = row.get("offboardDate") or row.get("delistTime")
        return BinanceInstrument(
            symbol=str(row["symbol"]).upper(),
            status=str(row.get("status", "UNKNOWN")),
            base_asset=str(row["baseAsset"]).upper(),
            quote_asset=str(row["quoteAsset"]).upper(),
            onboard_time=timestamp_to_datetime(int(onboard_value)) if onboard_value else None,
            offboard_time=timestamp_to_datetime(int(offboard_value)) if offboard_value else None,
            quantity_step=Decimal(str(lot.get("stepSize", "0"))),
            minimum_quantity=Decimal(str(lot.get("minQty", "0"))),
            minimum_notional=Decimal(str(notional.get("minNotional", "0"))),
            tick_size=Decimal(str(price_filter.get("tickSize", "0"))),
            quote_volume=quote_volume,
            raw_filters=filters,
        )
    except (KeyError, TypeError, ValueError, ArithmeticError):
        return None


def instrument_matches_rules(
    instrument: BinanceInstrument,
    rules: BinanceUniverseRules,
    *,
    as_of: datetime | None = None,
    check_listing_age: bool = True,
) -> bool:
    if instrument.status != "TRADING" or instrument.quote_asset != rules.quote_asset:
        return False
    if instrument.base_asset in rules.excluded_base_assets:
        return False
    if any(instrument.symbol.endswith(suffix) for suffix in rules.excluded_symbol_suffixes):
        return False
    if instrument.quote_volume < rules.minimum_quote_volume:
        return False
    if check_listing_age:
        if instrument.onboard_time is None:
            return False
        reference_time = as_of or datetime.now(UTC)
        if reference_time.tzinfo is None or instrument.onboard_time.tzinfo is None:
            raise ValueError("listing age timestamps must be timezone-aware")
        reference_time = reference_time.astimezone(UTC)
        onboard_time = instrument.onboard_time.astimezone(UTC)
        if onboard_time > reference_time:
            return False
        age_seconds = (reference_time - onboard_time).total_seconds()
        if age_seconds < rules.minimum_listing_days * 24 * 60 * 60:
            return False
    return True


def parse_kline(symbol: str, interval: str, row: list[Any], source: str) -> BinanceBar:
    if len(row) < 11:
        raise ValueError(f"Invalid Binance kline row for {symbol}: {row}")
    return BinanceBar(
        symbol=symbol.upper(),
        interval=interval,
        open_time=timestamp_to_datetime(int(row[0])),
        close_time=timestamp_to_datetime(int(row[6])),
        open=Decimal(str(row[1])),
        high=Decimal(str(row[2])),
        low=Decimal(str(row[3])),
        close=Decimal(str(row[4])),
        base_volume=Decimal(str(row[5])),
        quote_volume=Decimal(str(row[7])),
        trade_count=int(row[8]),
        taker_buy_base_volume=Decimal(str(row[9])),
        taker_buy_quote_volume=Decimal(str(row[10])),
        source=source,
    )


def retry_delay_seconds(
    status: int,
    headers: Any,
    payload: Any,
    attempt: int,
) -> float:
    retry_after = headers.get("Retry-After") if headers else None
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            try:
                return max(0.0, parsedate_to_datetime(retry_after).timestamp() - time.time())
            except (TypeError, ValueError, OverflowError):
                pass
    message = str(payload.get("msg", "")) if isinstance(payload, dict) else str(payload)
    ban_timestamp = re.search(r"until\s+(\d{13,16})", message)
    if ban_timestamp:
        until = int(ban_timestamp.group(1))
        until_ms = until // 1000 if until > 10_000_000_000_000 else until
        return max(0.0, until_ms / 1000 - time.time())
    base = 120.0 if status == 418 else 30.0
    return min(15 * 60.0, base * (2**attempt))
