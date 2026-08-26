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

The command writes `runs/binance_latest/dataset_report.json`. A rerun resumes
after the latest stored bar for each symbol, while primary-key upserts make
overlapping responses idempotent.
