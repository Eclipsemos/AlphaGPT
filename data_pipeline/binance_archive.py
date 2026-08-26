"""Verified monthly Binance Spot kline archive reader.

The archive host is static public data and does not expose account or order
endpoints. Files are verified before their CSV rows are parsed.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
import zipfile
from datetime import UTC, datetime, timedelta
from typing import Any, Iterable

import aiohttp

from .binance_contracts import BinanceBar
from .providers.binance_spot import parse_kline


ARCHIVE_BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"


class BinanceArchiveError(RuntimeError):
    pass


class BinanceArchiveNotFound(BinanceArchiveError):
    pass


class BinanceSpotArchiveProvider:
    source_name = "binance-spot-archive"

    def __init__(
        self,
        base_url: str = ARCHIVE_BASE_URL,
        *,
        session: aiohttp.ClientSession | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self._session = session
        self._owns_session = session is None
        self.download_count = 0
        self.checksum_count = 0

    async def __aenter__(self) -> "BinanceSpotArchiveProvider":
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120), trust_env=True)
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._owns_session and self._session is not None:
            await self._session.close()
            self._session = None

    async def monthly_bars(
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: datetime,
    ) -> tuple[list[BinanceBar], list[dict[str, Any]]]:
        if interval != "1h":
            raise ValueError("Binance archive ingestion currently supports 1h only")
        if start_time.tzinfo is None or end_time.tzinfo is None:
            raise ValueError("Archive bounds must be timezone-aware")
        start_time = start_time.astimezone(UTC)
        end_time = end_time.astimezone(UTC)
        if start_time >= end_time:
            raise ValueError("Archive start_time must precede end_time")
        bars: dict[datetime, BinanceBar] = {}
        metadata: list[dict[str, Any]] = []
        for year, month in iter_complete_months(start_time, end_time):
            filename = f"{symbol.upper()}-{interval}-{year:04d}-{month:02d}.zip"
            archive_url = f"{self.base_url}/{symbol.upper()}/{interval}/{filename}"
            checksum_url = f"{archive_url}.CHECKSUM"
            try:
                content, checksum = await self._download_verified(archive_url, checksum_url, filename)
            except BinanceArchiveNotFound:
                metadata.append(
                    {
                        "url": archive_url,
                        "checksum_url": checksum_url,
                        "archive": filename,
                        "available": False,
                        "bar_count": 0,
                    }
                )
                continue
            month_bars = parse_archive_zip(symbol, interval, content, self.source_name)
            selected = [
                bar
                for bar in month_bars
                if start_time <= bar.open_time < end_time and bar.close_time < end_time
            ]
            bars.update({bar.open_time: bar for bar in selected})
            metadata.append(
                {
                    "url": archive_url,
                    "checksum_url": checksum_url,
                    "archive": filename,
                    "available": True,
                    "archive_checksum": checksum,
                    "bar_count": len(selected),
                }
            )
        return [bars[key] for key in sorted(bars)], metadata

    async def _download_verified(self, archive_url: str, checksum_url: str, filename: str) -> tuple[bytes, str]:
        if self._session is None:
            raise RuntimeError("BinanceSpotArchiveProvider must be used as an async context manager")
        checksum_text = await self._get_text(checksum_url)
        expected = parse_checksum(checksum_text, filename)
        content = await self._get_bytes(archive_url)
        actual = hashlib.sha256(content).hexdigest()
        if actual.lower() != expected.lower():
            raise BinanceArchiveError(
                f"SHA-256 mismatch for {filename}: expected {expected}, got {actual}"
            )
        return content, actual

    async def _get_text(self, url: str) -> str:
        if self._session is None:
            raise RuntimeError("BinanceSpotArchiveProvider must be used as an async context manager")
        async with self._session.get(url) as response:
            self.checksum_count += 1
            if response.status == 404:
                raise BinanceArchiveNotFound(f"Archive checksum not found: {url}")
            if response.status >= 400:
                raise BinanceArchiveError(f"Archive checksum request failed ({response.status}): {url}")
            return await response.text()

    async def _get_bytes(self, url: str) -> bytes:
        if self._session is None:
            raise RuntimeError("BinanceSpotArchiveProvider must be used as an async context manager")
        async with self._session.get(url) as response:
            self.download_count += 1
            if response.status == 404:
                raise BinanceArchiveNotFound(f"Archive not found: {url}")
            if response.status >= 400:
                raise BinanceArchiveError(f"Archive request failed ({response.status}): {url}")
            return await response.read()


def iter_complete_months(start_time: datetime, end_time: datetime):
    start_time = start_time.astimezone(UTC)
    end_time = end_time.astimezone(UTC)
    cursor = datetime(start_time.year, start_time.month, 1, tzinfo=UTC)
    if cursor < start_time:
        cursor = next_month(cursor)
    while next_month(cursor) <= end_time:
        yield cursor.year, cursor.month
        cursor = next_month(cursor)


def next_month(value: datetime) -> datetime:
    if value.month == 12:
        return datetime(value.year + 1, 1, 1, tzinfo=UTC)
    return datetime(value.year, value.month + 1, 1, tzinfo=UTC)


def missing_bar_ranges(
    bars: Iterable[BinanceBar],
    start_time: datetime,
    end_time: datetime,
) -> list[tuple[datetime, datetime]]:
    present = {bar.open_time.astimezone(UTC) for bar in bars}
    ranges: list[tuple[datetime, datetime]] = []
    cursor = start_time.astimezone(UTC)
    missing_start: datetime | None = None
    while cursor < end_time:
        if cursor not in present and missing_start is None:
            missing_start = cursor
        elif cursor in present and missing_start is not None:
            ranges.append((missing_start, cursor))
            missing_start = None
        cursor += timedelta(hours=1)
    if missing_start is not None:
        ranges.append((missing_start, end_time.astimezone(UTC)))
    return ranges


def aggregate_archive_checksum(metadata: list[dict[str, Any]]) -> str | None:
    if not metadata:
        return None
    canonical = "\n".join(
        f"{item['archive']}:{item['archive_checksum']}"
        for item in sorted(metadata, key=lambda row: str(row["archive"]))
        if item.get("archive_checksum")
    )
    return hashlib.sha256(canonical.encode("ascii")).hexdigest() if canonical else None


def parse_checksum(value: str, filename: str) -> str:
    for line in value.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        candidate = parts[0].lower()
        if re.fullmatch(r"[0-9a-f]{64}", candidate) and (len(parts) == 1 or filename in parts[-1]):
            return candidate
    raise BinanceArchiveError(f"No SHA-256 checksum found for {filename}")


def parse_archive_zip(symbol: str, interval: str, content: bytes, source: str) -> list[BinanceBar]:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            names = [name for name in archive.namelist() if not name.endswith("/")]
            if len(names) != 1:
                raise BinanceArchiveError(f"Expected one CSV in archive, found {names}")
            raw = archive.read(names[0]).decode("utf-8-sig")
    except (OSError, zipfile.BadZipFile, UnicodeDecodeError) as exc:
        raise BinanceArchiveError(f"Invalid Binance archive for {symbol}: {exc}") from exc
    bars: list[BinanceBar] = []
    for line_number, fields in enumerate(csv.reader(io.StringIO(raw)), start=1):
        if not fields or str(fields[0]).strip().lower().startswith("open time"):
            continue
        fields = [field.strip() for field in fields]
        try:
            bars.append(parse_kline(symbol, interval, fields, source))
        except (TypeError, ValueError, ArithmeticError) as exc:
            raise BinanceArchiveError(f"Invalid kline at {names[0]}:{line_number}: {exc}") from exc
    return bars
