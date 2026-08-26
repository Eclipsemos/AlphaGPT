import asyncio
import unittest
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from data_pipeline.binance_contracts import BinanceBar, BinanceInstrument, BinanceUniverseRules
from data_pipeline.db_manager import DBManager
from data_pipeline.providers.binance_spot import (
    BinanceSpotProvider,
    instrument_matches_rules,
    parse_instrument,
    parse_kline,
    retry_delay_seconds,
    timestamp_to_datetime,
)


class FakeResponse:
    def __init__(self, status, payload, headers=None):
        self.status = status
        self._payload = payload
        self.headers = headers or {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None

    async def json(self, **_):
        return self._payload

    async def text(self):
        return str(self._payload)


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None):
        self.calls.append((url, params))
        if not self.responses:
            raise AssertionError("unexpected HTTP request")
        return self.responses.pop(0)


class FakeClock:
    def __init__(self):
        self.value = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.value

    async def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.value += seconds


class FakeConnection:
    def __init__(self):
        self.executemany_calls = []
        self.execute_calls = []
        self.transaction_count = 0

    async def executemany(self, query, args):
        self.executemany_calls.append((query, list(args)))

    async def execute(self, query, *args):
        self.execute_calls.append((query, args))

    def transaction(self):
        connection = self

        class Transaction:
            async def __aenter__(self):
                connection.transaction_count += 1
                return self

            async def __aexit__(self, *_):
                return None

        return Transaction()


class FakeAcquire:
    def __init__(self, connection):
        self.connection = connection

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, *_):
        return None


class FakePool:
    def __init__(self, connection):
        self.connection = connection

    def acquire(self):
        return FakeAcquire(self.connection)


def instrument(symbol="BTCUSDT", onboard=None, volume="20000000"):
    row = {
        "symbol": symbol,
        "status": "TRADING",
        "baseAsset": symbol.removesuffix("USDT"),
        "quoteAsset": "USDT",
        "filters": [
            {"filterType": "LOT_SIZE", "stepSize": "0.00001000", "minQty": "0.00001000"},
            {"filterType": "MIN_NOTIONAL", "minNotional": "5"},
            {"filterType": "PRICE_FILTER", "tickSize": "0.01000000"},
        ],
    }
    if onboard is not None:
        row["onboardDate"] = int(onboard.timestamp() * 1000)
    return row


def kline(open_time, *, close_time=None, close="101"):
    close_time = close_time or open_time + timedelta(minutes=59, seconds=59)
    return [
        int(open_time.timestamp() * 1000),
        "100",
        "102",
        "99",
        close,
        "12.5",
        int(close_time.timestamp() * 1000),
        "1260",
        42,
        "6.25",
        "630",
    ]


class BinanceProviderTests(unittest.IsolatedAsyncioTestCase):
    def test_timestamp_parser_accepts_milliseconds_and_microseconds(self):
        expected = datetime(2025, 1, 1, tzinfo=UTC)
        milliseconds = int(expected.timestamp() * 1000)
        microseconds = milliseconds * 1000
        self.assertEqual(timestamp_to_datetime(milliseconds), expected)
        self.assertEqual(timestamp_to_datetime(microseconds), expected)

    def test_listing_age_and_universe_exclusions(self):
        as_of = datetime(2026, 1, 1, tzinfo=UTC)
        rules = BinanceUniverseRules(max_symbols=2)
        old = parse_instrument(instrument(onboard=datetime(2024, 1, 1, tzinfo=UTC)), Decimal("20000000"))
        recent = parse_instrument(instrument("NEWUSDT", onboard=datetime(2025, 12, 15, tzinfo=UTC)), Decimal("20000000"))
        leveraged = parse_instrument(instrument("BTCUPUSDT", onboard=datetime(2024, 1, 1, tzinfo=UTC)), Decimal("20000000"))
        self.assertTrue(instrument_matches_rules(old, rules, as_of=as_of))
        self.assertFalse(instrument_matches_rules(recent, rules, as_of=as_of))
        self.assertFalse(instrument_matches_rules(leveraged, rules, as_of=as_of))

    async def test_discovery_uses_onboard_date_and_volume_order(self):
        as_of = datetime(2026, 1, 1, tzinfo=UTC)
        session = FakeSession([
            FakeResponse(200, {"symbols": [
                instrument("ETHUSDT", as_of - timedelta(days=400)),
                instrument("BTCUSDT", as_of - timedelta(days=500)),
            ]}),
            FakeResponse(200, [
                {"symbol": "ETHUSDT", "quoteVolume": "20000000"},
                {"symbol": "BTCUSDT", "quoteVolume": "50000000"},
            ]),
        ])
        rules = BinanceUniverseRules(max_symbols=1)
        async with BinanceSpotProvider(session=session, request_spacing_seconds=0) as provider:
            found = await provider.discover_instruments(rules, candidate_multiplier=1, as_of=as_of)
        self.assertEqual([item.symbol for item in found], ["BTCUSDT"])
        self.assertEqual(len(session.calls), 2)

    async def test_klines_drop_open_current_bar_and_deduplicate_pages(self):
        start = datetime(2026, 1, 1, tzinfo=UTC)
        end = start + timedelta(hours=2)
        first_page = [kline(start), kline(start + timedelta(hours=1), close_time=end + timedelta(minutes=1))]
        session = FakeSession([FakeResponse(200, first_page)])
        async with BinanceSpotProvider(session=session, request_spacing_seconds=0) as provider:
            bars = await provider.klines("BTCUSDT", "1h", start, end)
        self.assertEqual(len(bars), 1)
        self.assertEqual(bars[0].open_time, start)
        self.assertEqual(bars[0].close, Decimal("101"))

    async def test_kline_pagination_advances_and_removes_duplicate_boundary(self):
        start = datetime(2026, 1, 1, tzinfo=UTC)
        first = [kline(start + timedelta(hours=index)) for index in range(1000)]
        second = [kline(start + timedelta(hours=999)), kline(start + timedelta(hours=1000))]
        session = FakeSession([FakeResponse(200, first), FakeResponse(200, second)])
        async with BinanceSpotProvider(session=session, request_spacing_seconds=0) as provider:
            bars = await provider.klines("BTCUSDT", "1h", start, start + timedelta(hours=1002))
        self.assertEqual(len(bars), 1001)
        self.assertEqual(len({bar.open_time for bar in bars}), 1001)
        self.assertEqual(bars[-1].open_time, start + timedelta(hours=1000))

    def test_parse_kline_rejects_short_rows_and_preserves_decimal_values(self):
        start = datetime(2026, 1, 1, tzinfo=UTC)
        parsed = parse_kline("BTCUSDT", "1h", kline(start), "fixture")
        self.assertIsInstance(parsed, BinanceBar)
        self.assertEqual(parsed.trade_count, 42)
        self.assertEqual(parsed.base_volume, Decimal("12.5"))
        with self.assertRaises(ValueError):
            parse_kline("BTCUSDT", "1h", [1, 2], "fixture")

    def test_rate_limit_delay_honors_retry_after_ban_and_backoff(self):
        self.assertEqual(retry_delay_seconds(429, {"Retry-After": "2"}, {}, 0), 2.0)
        self.assertEqual(retry_delay_seconds(418, {}, {"msg": "banned until 1700000000000"}, 0), 0.0)
        self.assertEqual(retry_delay_seconds(429, {}, {}, 2), 120.0)

    async def test_request_retries_after_shared_rate_limit_cooldown(self):
        clock = FakeClock()
        session = FakeSession([
            FakeResponse(429, {"msg": "too many requests"}, {"Retry-After": "2"}),
            FakeResponse(200, {"serverTime": 1}),
        ])
        async with BinanceSpotProvider(
            session=session,
            request_spacing_seconds=0,
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        ) as provider:
            payload = await provider._request_json("/api/v3/time")
        self.assertEqual(payload, {"serverTime": 1})
        self.assertEqual(provider.request_count, 2)
        self.assertEqual(provider.rate_limit_count, 1)
        self.assertEqual(clock.sleeps, [2.0])


class BinanceDBTests(unittest.IsolatedAsyncioTestCase):
    async def test_upserts_use_idempotent_conflict_queries(self):
        connection = FakeConnection()
        db = DBManager()
        db.pool = FakePool(connection)
        onboard = datetime(2025, 1, 1, tzinfo=UTC)
        item = parse_instrument(instrument(onboard=onboard), Decimal("20000000"))
        bar = parse_kline("BTCUSDT", "1h", kline(onboard), "fixture")
        self.assertEqual(await db.upsert_market_instruments([item]), 1)
        self.assertEqual(await db.upsert_market_bars([bar]), 1)
        self.assertIn("ON CONFLICT (venue, market_type, symbol)", connection.executemany_calls[0][0])
        self.assertIn("ON CONFLICT (venue, market_type, symbol, interval, open_time)", connection.executemany_calls[1][0])


if __name__ == "__main__":
    unittest.main()
