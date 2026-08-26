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
