import asyncpg
from loguru import logger
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
