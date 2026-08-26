# Binance Factor Research Contract

The first Binance release is deliberately limited to public Binance Spot data
and factor mining. It never reads an account, places an order, or imports API
credentials.

## Canonical Dataset

- Venue: `binance`
- Market type: `spot`
- Quote asset: `USDT`
- Bar interval: `1h`
- History: at least 365 days
- Universe size: at most 50 symbols
- Minimum listing age: 30 days
- Minimum 24-hour quote volume: 10,000,000 USDT
- Exclusions: stable quote-like base assets and leveraged-token suffixes

The universe is sorted deterministically after filtering and persisted in an
immutable dataset snapshot. A snapshot ID is the SHA-256 of its canonical JSON
payload, including rules, symbols, time range, source, schema versions, and code
version.

## Storage

Binance data uses `market_instruments`, `market_bars`, `dataset_snapshots`, and
`dataset_snapshot_instruments`. The existing Solana tables remain unchanged.
Binance symbols must never be stored as Solana addresses, and Binance volume
must never be mapped to Solana liquidity or FDV fields.

All timestamps are timezone-aware UTC values. Numeric market values use
PostgreSQL `NUMERIC` so exchange filters and source values are not silently
rounded before feature tensors are built.

The default REST host is Binance's public market-data endpoint,
`data-api.binance.vision`. No Binance API key is read or sent.

Discover the current universe without downloading bars:

```bash
python -m data_pipeline.run_binance_pipeline --dry-run
```

Build or resume the canonical dataset:

```bash
python -m data_pipeline.run_binance_pipeline
```

Use verified monthly archives for the complete calendar months and public REST
for the leading/trailing partial months:

```bash
python -m data_pipeline.run_binance_pipeline --source archive
```

Every downloaded ZIP is checked against its official `.CHECKSUM` SHA-256 before
parsing. Verification failure aborts the snapshot; it is never silently
accepted or replaced. Archive URLs, individual checksums, and a deterministic
aggregate checksum are stored in snapshot coverage metadata.

The command writes `runs/binance_latest/dataset_report.json`. A rerun resumes
after the latest stored bar for each symbol, while primary-key upserts make
overlapping responses idempotent.

The report includes a quality result for every symbol. A research snapshot is
created only when every symbol meets `BINANCE_MIN_COVERAGE` (default `0.995`)
and passes timestamp, gap, OHLC, volume, listing-boundary, and completed-bar
checks. Failed runs retain raw bars and write diagnostics, but do not create a
dataset snapshot for factor research.

## Known Universe Limitation

The first snapshot is selected from the symbols returned by the current public
`exchangeInfo` and `ticker/24hr` responses. Freezing that response makes future
runs reproducible, but it cannot reconstruct symbols that were already delisted
before AlphaGPT began collecting snapshots. Historical studies using the first
snapshot therefore retain current-universe survivorship bias. Accumulating
immutable point-in-time snapshots reduces this limitation prospectively; reports
must not claim that the initial snapshot is survivorship-bias free.

## Feature Vocabulary

`binance-formula-v1` uses 11 ordered features: `RET_1H`, `RANGE`, `ATR_14`,
`CLOSE_POSITION`, `MOMENTUM_24H`, `REALIZED_VOL_24H`, `LOG_BASE_VOLUME`,
`LOG_QUOTE_VOLUME`, `QUOTE_VOLUME_CHANGE`, `LOG_TRADE_COUNT`, and
`TAKER_BUY_IMBALANCE`.

Features are causal. Rolling features remain invalid through their warmup and
after missing input bars; the loader writes zero only as a tensor placeholder
and keeps the corresponding feature/label mask false. It never forward-fills
market values or labels. Per-symbol median/MAD normalization is fitted on the
training split and then applied unchanged to validation and test data.

Saved Binance formulas use a versioned artifact contract containing the market,
formula vocabulary, token names, dataset snapshot ID, feature names, warmup
rules, and normalization state. A legacy Solana formula or bare token list is
not compatible with `binance-formula-v1`.
