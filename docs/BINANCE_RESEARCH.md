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

## Historical Evaluation

Binance factors are evaluated cross-sectionally. A factor observed after hour
`t` ranks the available symbols and forms research weights no earlier than the
`t+1` open; its label is the simple return from the `t+1` open to the `t+2`
open. The evaluator supports equal or inverse-risk weights, a maximum selected
symbol count, and an hourly rebalance cadence.

Taker fees and slippage are configurable assumptions charged only when weights
change, including a terminal exit at the evaluation boundary. They do not
represent submitted orders, account balances, fills, or an execution engine.
Reports include fixed-1h annualized volatility/Sharpe, drawdown, exposure,
turnover, cost attribution, quote-volume participation as a capacity proxy,
rank IC, and per-symbol contributions.

Evaluate a versioned formula artifact:

```bash
python -m model_core.evaluate_binance \
  --formula runs/binance/<run>/best_formula.json \
  --snapshot-id <snapshot-id> \
  --weighting equal \
  --max-positions 10 \
  --rebalance-hours 24 \
  --taker-fee-bps 10 \
  --slippage-bps 5
```

The report always includes equal-weight cross-section, BTCUSDT reference,
cross-sectional momentum, and seeded random-rank baselines, plus configurable
cost-sensitivity scenarios. These are historical research statistics, not a
claim of future profitability or production readiness.

## Multi-Seed Mining And Selection

Run the full research workflow with five independent discovery seeds:

```bash
python -m model_core.binance_batch \
  --snapshot-id <snapshot-id> \
  --seeds 1,2,3,4,5 \
  --steps 1000 \
  --batch-size 8192 \
  --windows 4
```

The miner rewards formulas with train-split cross-sectional IC. Each seed may
send a bounded candidate pool to validation; canonical RPN representations
merge commutative equivalents such as `A + B` and `B + A`. Shortlisting uses
validation IC and final selection uses anchored and disjoint rolling windows
within validation. No selection function receives a test tensor.

After the selected formula artifact is written, the workflow performs one
final test evaluation and compares it with fixed baselines and cost scenarios.
The decision report uses three statuses:

- `reject`: at least one validation or walk-forward pre-test gate failed.
- `research-only`: pre-test gates passed but at least one final gate failed.
- `promising`: all configured gates passed; still not a profitability claim.

A positive test return alone cannot produce `promising`. Cross-seed support,
validation IC, rolling-window stability, drawdown, baseline score, assumed
costs, and the quote-volume capacity proxy are independently recorded gates.

The experiment manifest records the exact command, Git revision and dirty
state, Python/package versions, CUDA/device details, and elapsed time. Resume
accepts only artifacts with identical snapshot metadata, universe, seed, and
mining configuration.

## Research Boundary

Historical factor evaluation computes hypothetical weights, turnover, costs,
and returns so formulas can be compared on the same data. Those arrays are not
an account model: they do not persist cash, balances, positions, fills, or
orders. AlphaGPT does not connect to a private Binance endpoint and does not
provide a simulated or live trading service.

## Offline Acceptance

Run the deterministic synthetic dataset through short formula mining,
validation selection, final evaluation, and artifact-schema checks:

```bash
python -m model_core.binance_acceptance
```

This command requires neither network access nor PostgreSQL. Its fixture is
not research evidence; it exists only to verify that the complete workflow and
artifact contracts function on a new machine.
