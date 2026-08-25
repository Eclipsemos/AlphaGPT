# TODO

This list is ordered by research reliability, not by visual polish.

## P0: Make Results Trustworthy

- [x] Add explicit time-based train/validation/test splits.
- [x] Add walk-forward evaluation.
- [x] Prevent future data from entering feature normalization and split labels.
- [x] Record token-selection snapshots so the research universe is reproducible over time.
- [x] Add data-quality checks for missing candles, zero prices, duplicate rows, and stale liquidity/FDV values.
- [x] Produce a backtest report with cumulative return, volatility, Sharpe, maximum drawdown, turnover, trade count, win rate, and fees.
- [x] Add baseline comparisons: buy-and-hold, liquidity filter, momentum, and random formula.

## P1: Improve the Research Loop

- [x] Save model checkpoints and resume interrupted training.
- [x] Add deterministic seeds and record the full configuration with each run.
- [x] Constrain formula generation to valid expression trees instead of penalizing arbitrary invalid token sequences.
- [x] Run multiple saved runs and report mean/std confidence summaries on the test split.
- [x] Align signal timestamps and forward returns explicitly, with tests covering off-by-one errors.
- [x] Fix or remove the LoRD monitor/regularizer until its parameter targeting matches the actual model modules.
- [x] Add a command that evaluates an existing formula without retraining.

## P2: Make the Tool Easier to Use

- [x] Add `.env.example` with safe placeholder values.
- [x] Add a single research command that initializes the database, refreshes data, and runs validation checks.
- [x] Add dashboard views for training history and backtest metrics.
- [x] Show data freshness and row counts in the dashboard.
- [x] Show API-rate-limit status in the dashboard.
- [x] Add automated tests for the data loader, VM operators, and backtest accounting.
- [x] Add automated tests for database queries.

## P3: Paper Trading Only

- [x] Build a paper-trading simulator with persisted virtual balances.
- [x] Add signal logging without transaction signing.
- [x] Add explicit kill switch and dry-run-only defaults.
- [x] Document live execution separately, while keeping it disabled until paper-trading results are reproducible.

## Out of Scope Until P0/P1 Are Complete

- [ ] Live wallet integration.
- [ ] Automatic Solana/Jupiter order execution.
- [ ] Claims that a mined formula is profitable or production-ready.
