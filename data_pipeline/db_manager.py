import json
from collections.abc import Sequence
from datetime import datetime
from typing import Any

import asyncpg
from loguru import logger

from .binance_contracts import BinanceBar, BinanceInstrument
from .config import Config


def binance_schema_statements() -> tuple[str, ...]:
    """Return the additive Binance research schema used by ``init_schema``."""
    return (
        """
        CREATE TABLE IF NOT EXISTS market_instruments (
            venue TEXT NOT NULL,
            market_type TEXT NOT NULL,
            symbol TEXT NOT NULL,
            status TEXT NOT NULL,
            base_asset TEXT NOT NULL,
            quote_asset TEXT NOT NULL,
            onboard_time TIMESTAMPTZ,
            quantity_step NUMERIC NOT NULL,
            minimum_quantity NUMERIC NOT NULL,
            minimum_notional NUMERIC NOT NULL,
            tick_size NUMERIC NOT NULL,
            quote_volume NUMERIC NOT NULL,
            raw_filters JSONB NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (venue, market_type, symbol)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS market_bars (
            venue TEXT NOT NULL,
            market_type TEXT NOT NULL,
            symbol TEXT NOT NULL,
            interval TEXT NOT NULL,
            open_time TIMESTAMPTZ NOT NULL,
            close_time TIMESTAMPTZ NOT NULL,
            open NUMERIC NOT NULL,
            high NUMERIC NOT NULL,
            low NUMERIC NOT NULL,
            close NUMERIC NOT NULL,
            base_volume NUMERIC NOT NULL,
            quote_volume NUMERIC NOT NULL,
            trade_count BIGINT NOT NULL,
            taker_buy_base_volume NUMERIC NOT NULL,
            taker_buy_quote_volume NUMERIC NOT NULL,
            source TEXT NOT NULL,
            retrieved_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (venue, market_type, symbol, interval, open_time),
            FOREIGN KEY (venue, market_type, symbol)
                REFERENCES market_instruments (venue, market_type, symbol)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS dataset_snapshots (
            snapshot_id TEXT PRIMARY KEY,
            schema_version TEXT NOT NULL,
            feature_schema_version TEXT NOT NULL,
            venue TEXT NOT NULL,
            market_type TEXT NOT NULL,
            interval TEXT NOT NULL,
            start_time TIMESTAMPTZ NOT NULL,
            end_time TIMESTAMPTZ NOT NULL,
            source TEXT NOT NULL,
            code_version TEXT NOT NULL,
            rules JSONB NOT NULL,
            payload JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS dataset_snapshot_instruments (
            snapshot_id TEXT NOT NULL REFERENCES dataset_snapshots(snapshot_id),
            symbol TEXT NOT NULL,
            rank INTEGER NOT NULL,
            PRIMARY KEY (snapshot_id, symbol),
            UNIQUE (snapshot_id, rank)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS dataset_snapshot_coverage (
            snapshot_id TEXT NOT NULL REFERENCES dataset_snapshots(snapshot_id),
            symbol TEXT NOT NULL,
            requested_start_time TIMESTAMPTZ NOT NULL,
            requested_end_time TIMESTAMPTZ NOT NULL,
            response_start_time TIMESTAMPTZ,
            response_end_time TIMESTAMPTZ,
            bar_count BIGINT NOT NULL,
            expected_bar_count BIGINT NOT NULL,
            archive_checksum TEXT,
            source_metadata JSONB NOT NULL,
            retrieved_at TIMESTAMPTZ NOT NULL,
            PRIMARY KEY (snapshot_id, symbol)
        );
        """,
        "CREATE INDEX IF NOT EXISTS idx_market_bars_lookup ON market_bars (symbol, interval, open_time);",
    )

class DBManager:
    def __init__(self):
        self.pool = None

    async def connect(self):
        if not self.pool:
            self.pool = await asyncpg.create_pool(dsn=Config.DB_DSN)
            logger.info("Database connection established.")

    async def close(self):
        if self.pool:
            await self.pool.close()
            self.pool = None

    async def init_schema(self):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS tokens (
                    address TEXT PRIMARY KEY,
                    symbol TEXT,
                    name TEXT,
                    decimals INT,
                    chain TEXT,
                    last_updated TIMESTAMP DEFAULT NOW()
                );
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv (
                    time TIMESTAMP NOT NULL,
                    address TEXT NOT NULL,
                    open DOUBLE PRECISION,
                    high DOUBLE PRECISION,
                    low DOUBLE PRECISION,
                    close DOUBLE PRECISION,
                    volume DOUBLE PRECISION,
                    liquidity DOUBLE PRECISION, 
                    fdv DOUBLE PRECISION,
                    source TEXT,
                    PRIMARY KEY (time, address)
                );
            """)
            
            try:
                await conn.execute("SELECT create_hypertable('ohlcv', 'time', if_not_exists => TRUE);")
                logger.info("Converted ohlcv to Hypertable.")
            except Exception:
                logger.warning("TimescaleDB extension not found, using standard Postgres.")

            await conn.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_address ON ohlcv (address);")

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS token_snapshots (
                    snapshot_time TIMESTAMP NOT NULL,
                    address TEXT NOT NULL,
                    symbol TEXT,
                    name TEXT,
                    decimals INT,
                    liquidity DOUBLE PRECISION,
                    fdv DOUBLE PRECISION,
                    rank INT,
                    PRIMARY KEY (snapshot_time, address)
                );
            """)
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_token_snapshots_address ON token_snapshots (address);")

            for statement in binance_schema_statements():
                await conn.execute(statement)

    async def upsert_market_instruments(self, instruments: Sequence[BinanceInstrument]) -> int:
        if not instruments:
            return 0
        records = [
            (
                item.venue,
                item.market_type,
                item.symbol,
                item.status,
                item.base_asset,
                item.quote_asset,
                item.onboard_time,
                item.quantity_step,
                item.minimum_quantity,
                item.minimum_notional,
                item.tick_size,
                item.quote_volume,
                json.dumps(item.raw_filters, sort_keys=True, separators=(",", ":")),
            )
            for item in instruments
        ]
        async with self.pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO market_instruments (
                    venue, market_type, symbol, status, base_asset, quote_asset,
                    onboard_time, quantity_step, minimum_quantity,
                    minimum_notional, tick_size, quote_volume, raw_filters
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13::jsonb
                )
                ON CONFLICT (venue, market_type, symbol) DO UPDATE SET
                    status = EXCLUDED.status,
                    base_asset = EXCLUDED.base_asset,
                    quote_asset = EXCLUDED.quote_asset,
                    onboard_time = EXCLUDED.onboard_time,
                    quantity_step = EXCLUDED.quantity_step,
                    minimum_quantity = EXCLUDED.minimum_quantity,
                    minimum_notional = EXCLUDED.minimum_notional,
                    tick_size = EXCLUDED.tick_size,
                    quote_volume = EXCLUDED.quote_volume,
                    raw_filters = EXCLUDED.raw_filters,
                    updated_at = NOW()
                """,
                records,
            )
        return len(records)

    async def upsert_market_bars(self, bars: Sequence[BinanceBar]) -> int:
        if not bars:
            return 0
        records = [
            (
                item.venue,
                item.market_type,
                item.symbol,
                item.interval,
                item.open_time,
                item.close_time,
                item.open,
                item.high,
                item.low,
                item.close,
                item.base_volume,
                item.quote_volume,
                item.trade_count,
                item.taker_buy_base_volume,
                item.taker_buy_quote_volume,
                item.source,
            )
            for item in bars
        ]
        async with self.pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO market_bars (
                    venue, market_type, symbol, interval, open_time, close_time,
                    open, high, low, close, base_volume, quote_volume,
                    trade_count, taker_buy_base_volume, taker_buy_quote_volume,
                    source
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12,
                    $13, $14, $15, $16
                )
                ON CONFLICT (venue, market_type, symbol, interval, open_time)
                DO UPDATE SET
                    close_time = EXCLUDED.close_time,
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    base_volume = EXCLUDED.base_volume,
                    quote_volume = EXCLUDED.quote_volume,
                    trade_count = EXCLUDED.trade_count,
                    taker_buy_base_volume = EXCLUDED.taker_buy_base_volume,
                    taker_buy_quote_volume = EXCLUDED.taker_buy_quote_volume,
                    source = EXCLUDED.source,
                    retrieved_at = NOW()
                """,
                records,
            )
        return len(records)

    async def latest_market_bar_time(
        self,
        symbol: str,
        interval: str,
        *,
        before: datetime,
    ) -> datetime | None:
        async with self.pool.acquire() as conn:
            return await conn.fetchval(
                """
                SELECT MAX(open_time)
                FROM market_bars
                WHERE venue = 'binance' AND market_type = 'spot'
                    AND symbol = $1 AND interval = $2 AND open_time < $3
                """,
                symbol,
                interval,
                before,
            )

    async def market_bar_coverage(
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: datetime,
    ) -> dict[str, Any]:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT MIN(open_time) AS response_start_time,
                       MAX(open_time) AS response_end_time,
                       COUNT(*) AS bar_count
                FROM market_bars
                WHERE venue = 'binance' AND market_type = 'spot'
                    AND symbol = $1 AND interval = $2
                    AND open_time >= $3 AND open_time < $4
                """,
                symbol,
                interval,
                start_time,
                end_time,
            )
        return dict(row)

    async def create_dataset_snapshot(
        self,
        snapshot_id: str,
        payload: dict[str, Any],
        symbols: Sequence[str],
        coverage: Sequence[dict[str, Any]],
    ) -> None:
        encoded_rules = json.dumps(payload["rules"], sort_keys=True, separators=(",", ":"))
        encoded_payload = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO dataset_snapshots (
                        snapshot_id, schema_version, feature_schema_version,
                        venue, market_type, interval, start_time, end_time,
                        source, code_version, rules, payload
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                        $11::jsonb, $12::jsonb
                    ) ON CONFLICT (snapshot_id) DO NOTHING
                    """,
                    snapshot_id,
                    payload["schema_version"],
                    payload["feature_schema_version"],
                    payload["rules"]["venue"],
                    payload["rules"]["market_type"],
                    payload["rules"]["interval"],
                    datetime.fromisoformat(payload["start_time"]),
                    datetime.fromisoformat(payload["end_time"]),
                    payload["source"],
                    payload["code_version"],
                    encoded_rules,
                    encoded_payload,
                )
                await conn.executemany(
                    """
                    INSERT INTO dataset_snapshot_instruments (snapshot_id, symbol, rank)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (snapshot_id, symbol) DO NOTHING
                    """,
                    [(snapshot_id, symbol, rank) for rank, symbol in enumerate(symbols, start=1)],
                )
                await conn.executemany(
                    """
                    INSERT INTO dataset_snapshot_coverage (
                        snapshot_id, symbol, requested_start_time,
                        requested_end_time, response_start_time,
                        response_end_time, bar_count, expected_bar_count,
                        archive_checksum, source_metadata, retrieved_at
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb, $11
                    ) ON CONFLICT (snapshot_id, symbol) DO NOTHING
                    """,
                    [
                        (
                            snapshot_id,
                            item["symbol"],
                            item["requested_start_time"],
                            item["requested_end_time"],
                            item["response_start_time"],
                            item["response_end_time"],
                            item["bar_count"],
                            item["expected_bar_count"],
                            item.get("archive_checksum"),
                            json.dumps(item["source_metadata"], sort_keys=True, separators=(",", ":")),
                            item["retrieved_at"],
                        )
                        for item in coverage
                    ],
                )

    async def upsert_tokens(self, tokens):
        if not tokens: return
        async with self.pool.acquire() as conn:
            # tokens: list of (address, symbol, name, decimals, chain)
            await conn.executemany("""
                INSERT INTO tokens (address, symbol, name, decimals, chain, last_updated)
                VALUES ($1, $2, $3, $4, $5, NOW())
                ON CONFLICT (address) DO UPDATE 
                SET symbol = EXCLUDED.symbol, last_updated = NOW();
            """, tokens)

    async def batch_insert_ohlcv(self, records):
        if not records: return
        async with self.pool.acquire() as conn:
            try:
                await conn.copy_records_to_table(
                    'ohlcv',
                    records=records,
                    columns=['time', 'address', 'open', 'high', 'low', 'close', 
                             'volume', 'liquidity', 'fdv', 'source'],
                    timeout=60
                )
            except asyncpg.UniqueViolationError:
                pass # 忽略重复
            except Exception as e:
                logger.error(f"Batch insert error: {e}")

    async def insert_token_snapshot(self, snapshot_time, tokens):
        if not tokens:
            return
        records = [
            (
                snapshot_time,
                token["address"],
                token.get("symbol"),
                token.get("name"),
                token.get("decimals", 6),
                float(token.get("liquidity") or 0),
                float(token.get("fdv") or 0),
                index,
            )
            for index, token in enumerate(tokens)
        ]
        async with self.pool.acquire() as conn:
            await conn.executemany("""
                INSERT INTO token_snapshots
                    (snapshot_time, address, symbol, name, decimals, liquidity, fdv, rank)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (snapshot_time, address) DO UPDATE SET
                    liquidity = EXCLUDED.liquidity,
                    fdv = EXCLUDED.fdv,
                    rank = EXCLUDED.rank;
            """, records)
