# TODO: Binance Spot Factor Research

## Scope Lock

AlphaGPT is a factor-mining and historical research tool for Binance Spot
markets. The first release covers public USDT-market data at the `1h`
interval. Its outputs are immutable datasets, versioned factor formulas, and
reproducible historical evaluation reports. It does not operate a portfolio,
connect to a trading account, or claim that a formula will remain profitable.

The following are permanently excluded from this roadmap: paper/simulated
trading services, Binance Testnet/Demo, API keys and private account APIs,
wallets, order submission, live execution, and Binance Futures. Historical
cost assumptions are evaluation inputs only; they are not an execution model.

## Research Deliverables

Every completed research batch must produce all of the following:

- An immutable `dataset_snapshot_id` with provenance, universe, interval, time range, and quality results.
- One or more versioned formula artifacts tied to the exact dataset, feature vocabulary, normalization state, code version, and seed.
- Validation and walk-forward results with baselines, assumed costs, turnover, capacity proxy, and uncertainty across seeds.
- A final untouched-test report for candidates selected without using test results.
- An explicit `reject`, `research-only`, or `promising` research decision with machine-readable reasons.

None of these artifacts authorizes or performs a trade.

## Delivery Order

Work in order: B1 public data -> B2 data quality -> B3 features -> B4
historical factor evaluation -> B5 mining workflow -> B6 documentation and
research dashboard. A later phase must not hide an incomplete earlier phase.
The current milestone is B5; B0-B4 are usable research foundations.

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
- [x] Reuse the timestamp normalization, retry, pacing, and integrity-check patterns from `mmtick` without importing its trading code.
- [x] Add optional ingestion from `data.binance.vision` with SHA-256 verification for reproducible bulk history.
- [x] Normalize Binance timestamps to UTC and handle both millisecond and microsecond archive timestamps.
- [x] Store Binance instruments and bars in new tables; do not overload Solana `address`, `liquidity`, or `fdv` fields.
- [x] Make REST ingestion idempotent and safe to resume after interruption.
- [x] Persist REST source metadata, retrieval time, requested range, and response coverage; reserve archive checksum storage for archive ingestion.
- [x] Add rate-limit handling for HTTP 429/418, `Retry-After`, exponential backoff, and a shared cooldown.
- [x] Add one command to build or incrementally refresh a Binance dataset snapshot.

## B2: Validate Dataset Quality and Reproducibility

- [x] Check primary-key uniqueness, monotonic timestamps, expected cadence, and OHLC invariants.
- [x] Reject negative prices/volumes and record missing or incomplete bars; never silently forward-fill research labels.
- [x] Detect symbol listing/delisting boundaries and exclude pre-listing or post-delisting periods.
- [x] Measure per-symbol and per-period coverage and fail snapshots below a configured threshold.
- [x] Save immutable point-in-time universe snapshots and explicitly flag the initial current-universe survivorship limitation.
- [x] Add tests for Binance response parsing, pagination boundaries, timestamp units, duplicate pages, gaps, retries, and idempotent inserts.
- [x] Add a small frozen Binance fixture with an expected dataset fingerprint for deterministic integration tests.

## B3: Generalize Features for Binance

- [x] Replace Solana-specific liquidity/FDV features with exchange-neutral Binance features.
- [x] Define the first Binance feature vocabulary: returns, range/ATR, close position, momentum, realized volatility, base volume, quote volume, volume change, trade count, and taker-buy imbalance.
- [x] Fit every normalization parameter on the training split only and apply it unchanged to validation/test data.
- [x] Add warmup masks for rolling features so early incomplete windows cannot become valid samples.
- [x] Record feature names, formulas, normalization state, and feature-code version in formula artifacts.
- [x] Version formula vocabularies so old Solana formulas cannot be silently evaluated against Binance feature indices.
- [x] Add unit tests for each Binance feature and explicit no-future-data tests.

## B4: Historical Factor Evaluation (No Trading Simulator)

- [x] Replace Solana liquidity gates and the fixed `0.6%` fee with configurable research cost assumptions for sensitivity analysis; do not build an order or account simulator.
- [x] Execute signals no earlier than the next bar open and test signal/entry/exit alignment.
- [x] Define a deterministic cross-sectional evaluation protocol: ranking, maximum selected symbols, equal/risk weights, and rebalance cadence.
- [x] Attribute turnover and assumed costs to historical factor results; do not model order routing, balances, or execution.
- [x] Annualize volatility and Sharpe from the configured bar interval instead of the number of bars in the evaluation slice.
- [x] Add exposure, capacity proxy, turnover, cost attribution, and per-symbol contribution to research reports.
- [x] Add Binance baselines: equal-weight cross-section, BTCUSDT reference return, cross-sectional momentum, and random rank.
- [x] Add hand-calculated golden tests for compounding, fees, rebalances, missing symbols, and delisting exits.

## B5: Run the Binance Mining Workflow

- [x] Add a Binance-specific GPU miner using train-only cross-sectional IC rewards.
- [x] Select candidate formulas on validation data without exposing test tensors during mining.
- [x] Save versioned candidate artifacts with snapshot, feature, vocabulary, normalization, seed, and code metadata.
- [x] Save checkpoints and RNG state so an interrupted CUDA run can resume reproducibly.
- [x] Deduplicate commutative and otherwise canonical-equivalent formulas before candidate aggregation.
- [x] Run smoke mining on a small Binance universe and verify that the test split is not accessed.
- [x] Add `--market binance-spot`, dataset snapshot, interval, and universe options to research commands.
- [x] Keep Solana and Binance run directories and report metadata explicitly separated.
- [x] Reject startup when the requested dataset snapshot, feature vocabulary, or cost model is missing or incompatible.
- [x] Run at least five independent seeds for candidate discovery and reserve the test split until final comparison.
- [x] Run anchored and rolling walk-forward evaluation across sequential validation windows.
- [ ] Add named market-regime slices without choosing regimes from future data.
- [x] Rank candidates by validation and walk-forward criteria, never by test performance alone.
- [x] Generate a batch decision report with explicit `reject`, `research-only`, or `promising` status and reasons.
- [x] Add confidence intervals and cross-seed stability statistics to the batch report.
- [x] Add a one-command Binance batch workflow: snapshot validation, mining, aggregation, walk-forward evaluation, and final report.

### B5 Acceptance Gate

- [x] A five-seed fixture batch completes from one command and can resume after interruption.
- [x] Re-running the same snapshot and seeds reproduces candidate identities and materially identical metrics.
- [x] Candidate selection code cannot read test tensors before the final evaluation stage.
- [x] The report rejects candidates that fail validation, window stability, baseline, cost, or capacity criteria.
- [x] A positive test return alone can never produce `promising` status.

## B6: Documentation and Dashboard

- [x] Rewrite setup documentation for the Binance public-data research workflow; no API key should be required for the MVP.
- [x] Document dataset provenance, supported interval, universe rules, costs, feature vocabulary, and known biases.
- [x] Update the dashboard to select a Binance dataset snapshot and show symbols instead of Solana addresses.
- [x] Show dataset coverage, gaps, freshness, feature version, cost assumptions, and baseline comparisons.
- [x] Remove wallet, SOL balance, Birdeye status, portfolio, and execution controls from the research dashboard.
- [ ] Add an end-to-end acceptance command: build fixture dataset, mine a short batch, evaluate, and verify report schema.
- [x] Document the exact boundary between historical factor evaluation and prohibited simulated/live trading.

## B7: Research Hardening

- [x] Add experiment manifests containing CLI arguments, package versions, Git commit, CUDA/device details, and wall-clock time.
- [ ] Add leakage checks for universe construction, normalization, rolling features, labels, candidate selection, and report aggregation.
- [ ] Add robustness tests across fees, slippage, rebalance cadence, maximum positions, weighting, and liquidity thresholds.
- [ ] Add regime slices for trend, drawdown, high volatility, and low volatility without selecting regimes using future data.
- [ ] Measure formula complexity and reject unstable formulas whose results depend on a few symbols or time periods.
- [ ] Add deterministic fixture CI that requires no Binance API key, wallet, database account, or network access.

## Definition of Done

The Binance factor-research MVP is complete only when:

- [ ] Public `1h` Spot data can be built into a quality-gated, immutable snapshot from one documented command.
- [ ] A resumable five-seed mining batch produces canonical, versioned candidate formulas.
- [ ] Validation and walk-forward selection happens before exactly one final untouched-test evaluation.
- [ ] Results include baselines, realistic cost sensitivity, turnover, drawdown, exposure, capacity, and stability diagnostics.
- [ ] The workflow is reproducible from a frozen fixture and documented for a new machine.
- [ ] The dashboard contains research data only and exposes no wallet, portfolio, order, simulated-trading, or live-trading controls.

## Explicitly Out of Scope

- [x] Binance API keys, account balances, or private account endpoints.
- [x] Binance Spot Testnet or Futures Demo integration.
- [x] Paper or simulated trading services.
- [x] Wallet integration, order creation, cancellation, or execution.
- [x] Binance Futures, leverage, margin, funding, or liquidation modeling in the first release.
- [x] Solana/Jupiter live execution.
- [x] Claims that a mined formula is profitable or production-ready.
- [x] Persistent virtual balances, fills, positions, PnL ledgers, or other simulated-account state.
