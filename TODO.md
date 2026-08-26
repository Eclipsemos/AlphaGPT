# TODO

## Scope Lock

AlphaGPT is a factor-mining and historical research tool for Binance Spot
markets. It produces data snapshots, candidate formulas, and reproducible
evaluation reports. It does not operate a portfolio or connect to a trading
account.

The following are permanently excluded from this roadmap: paper/simulated
trading services, Binance Testnet/Demo, API keys and private account APIs,
wallets, order submission, live execution, and Binance Futures in the first
release.

## Delivery Order

Work in order: B1 public data -> B2 data quality -> B3 features -> B4
historical factor evaluation -> B5 mining workflow -> B6 documentation and
research dashboard. A later phase must not hide an incomplete earlier phase.

## Completed Research Foundation

- [x] Add explicit time-based train/validation/test splits.
- [x] Add walk-forward evaluation.
- [x] Prevent future data from entering feature normalization and split labels.
- [x] Add data-quality checks for missing candles, zero prices, duplicate rows, and timestamp gaps.
- [x] Exclude invalid forward-return labels and compound log-return labels as simple returns.
- [x] Report cumulative return, volatility, Sharpe, maximum drawdown, turnover, trade count, win rate, and fees.
- [x] Add buy-and-hold, liquidity, momentum, and random baselines.
- [x] Save checkpoints, resume interrupted training, and record deterministic seeds.
- [x] Validate generated formula trees before evaluation.
- [x] Add saved-formula evaluation, multi-seed summaries, and confidence intervals.
- [x] Add a one-command workflow for data refresh, multi-seed mining, and evaluation.
- [x] Add automated tests for return alignment, data quality, VM operators, backtest accounting, and database queries.

## B0: Define the Binance Research Dataset

- [x] Limit the first release to Binance Spot USDT markets; do not include Futures.
- [x] Choose the canonical bar interval and history range (`1h`, at least one year).
- [x] Define deterministic universe rules: `TRADING` status, USDT quote asset, minimum listing age, minimum quote volume, and exclusions.
- [x] Define a versioned instrument schema using `venue`, `market_type`, `symbol`, `base_asset`, `quote_asset`, and exchange filters.
- [x] Define a versioned bar schema using `symbol`, `interval`, UTC open time, OHLC, base volume, quote volume, trade count, and taker-buy volume.
- [x] Define a `dataset_snapshot_id` contract that records universe rules, symbols, time range, interval, source, and code version.

## B1: Build the Binance Public-Data Pipeline

- [x] Add a read-only Binance Spot provider for `exchangeInfo`, `ticker/24hr`, and paginated `klines`.
- [ ] Reuse the timestamp normalization, retry, pacing, and integrity-check patterns from `mmtick` without importing its trading code.
- [ ] Add optional ingestion from `data.binance.vision` with SHA-256 verification for reproducible bulk history.
- [x] Normalize Binance timestamps to UTC and handle both millisecond and microsecond archive timestamps.
- [x] Store Binance instruments and bars in new tables; do not overload Solana `address`, `liquidity`, or `fdv` fields.
- [x] Make REST ingestion idempotent and safe to resume after interruption.
- [x] Persist REST source metadata, retrieval time, requested range, and response coverage; reserve archive checksum storage for archive ingestion.
- [x] Add rate-limit handling for HTTP 429/418, `Retry-After`, exponential backoff, and a shared cooldown.
- [x] Add one command to build or incrementally refresh a Binance dataset snapshot.

## B2: Validate Dataset Quality and Reproducibility

- [ ] Check primary-key uniqueness, monotonic timestamps, expected cadence, and OHLC invariants.
- [ ] Reject negative prices/volumes and record missing or incomplete bars; never silently forward-fill research labels.
- [ ] Detect symbol listing/delisting boundaries and exclude pre-listing or post-delisting periods.
- [ ] Measure per-symbol and per-period coverage and fail snapshots below a configured threshold.
- [ ] Save immutable universe snapshots to prevent survivorship bias from today's symbol list.
- [ ] Add tests for Binance response parsing, pagination boundaries, timestamp units, duplicate pages, gaps, retries, and idempotent inserts.
- [ ] Add a small frozen Binance fixture with an expected dataset fingerprint for deterministic integration tests.

## B3: Generalize Features for Binance

- [ ] Replace Solana-specific liquidity/FDV features with exchange-neutral Binance features.
- [ ] Define the first Binance feature vocabulary: returns, range/ATR, close position, momentum, realized volatility, base volume, quote volume, volume change, trade count, and taker-buy imbalance.
- [ ] Fit every normalization parameter on the training split only and apply it unchanged to validation/test data.
- [ ] Add warmup masks for rolling features so early incomplete windows cannot become valid samples.
- [ ] Record feature names, formulas, normalization state, and feature-code version in every run artifact.
- [ ] Version formula vocabularies so old Solana formulas cannot be silently evaluated against Binance feature indices.
- [ ] Add unit tests for each Binance feature and explicit no-future-data tests.

## B4: Historical Factor Evaluation (No Trading Simulator)

- [ ] Replace Solana liquidity gates and the fixed `0.6%` fee with configurable research cost assumptions for sensitivity analysis; do not build an order or account simulator.
- [ ] Execute signals no earlier than the next bar open and test signal/entry/exit alignment.
- [ ] Define a deterministic cross-sectional evaluation protocol: ranking, maximum selected symbols, equal/risk weights, and rebalance cadence.
- [ ] Attribute turnover and assumed costs to historical factor results; do not model order routing, balances, or execution.
- [ ] Annualize volatility and Sharpe from the configured bar interval instead of the number of bars in the evaluation slice.
- [ ] Add exposure, capacity proxy, turnover, cost attribution, and per-symbol contribution to research reports.
- [ ] Add Binance baselines: equal-weight cross-section, BTCUSDT reference return, cross-sectional momentum, and random rank.
- [ ] Add hand-calculated golden tests for compounding, fees, rebalances, missing symbols, and delisting exits.

## B5: Run the Binance Mining Workflow

- [ ] Add `--market binance-spot`, dataset snapshot, interval, and universe options to research commands.
- [ ] Keep Solana and Binance run directories and report metadata explicitly separated.
- [ ] Reject startup when the requested dataset snapshot, feature vocabulary, or cost model is missing or incompatible.
- [ ] Run smoke mining on a small universe before full GPU batches.
- [ ] Run at least five independent seeds for candidate discovery and reserve the test split until final comparison.
- [ ] Run anchored and rolling walk-forward evaluation across multiple market regimes.
- [ ] Deduplicate semantically equivalent formulas before multi-seed aggregation.
- [ ] Rank candidates by validation and walk-forward criteria, never by test performance alone.
- [ ] Generate a batch decision report with explicit `reject`, `research-only`, or `promising` status and reasons.

## B6: Documentation and Dashboard

- [ ] Rewrite setup documentation for the Binance public-data research workflow; no API key should be required for the MVP.
- [ ] Document dataset provenance, supported interval, universe rules, costs, feature vocabulary, and known biases.
- [ ] Update the dashboard to select a market/dataset snapshot and show Binance symbols instead of Solana addresses.
- [ ] Show dataset coverage, gaps, freshness, feature version, cost assumptions, and baseline comparisons.
- [ ] Remove or hide wallet, SOL balance, Birdeye status, portfolio, and execution controls from the research dashboard.
- [ ] Add an end-to-end acceptance command: build fixture dataset, mine a short batch, evaluate, and verify report schema.

## Explicitly Out of Scope

- [x] Binance API keys, account balances, or private account endpoints.
- [x] Binance Spot Testnet or Futures Demo integration.
- [x] Paper or simulated trading services.
- [x] Wallet integration, order creation, cancellation, or execution.
- [x] Binance Futures, leverage, margin, funding, or liquidation modeling in the first release.
- [x] Solana/Jupiter live execution.
- [x] Claims that a mined formula is profitable or production-ready.
