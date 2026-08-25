# TODO

This list is ordered by research reliability, not by visual polish.

## P0: Make Results Trustworthy

- [ ] Add explicit time-based train/validation/test splits.
- [ ] Add walk-forward evaluation and prevent future data from entering feature construction.
- [ ] Replace the current-trending-token universe with a reproducible historical universe.
- [ ] Add data-quality checks for missing candles, zero prices, duplicate rows, and stale liquidity/FDV values.
- [ ] Produce a backtest report with cumulative return, volatility, Sharpe, maximum drawdown, turnover, trade count, win rate, and fees.
- [ ] Add baseline comparisons: buy-and-hold, liquidity filter, momentum, and random formula.

## P1: Improve the Research Loop

- [ ] Save model checkpoints and resume interrupted training.
- [ ] Add deterministic seeds and record the full configuration with each run.
- [ ] Constrain formula generation to valid expression trees instead of penalizing arbitrary invalid token sequences.
- [ ] Run multiple seeds and report confidence intervals rather than a single best formula.
- [ ] Align signal timestamps and forward returns explicitly, with tests covering off-by-one errors.
- [ ] Fix or remove the LoRD monitor/regularizer until its parameter targeting matches the actual model modules.
- [ ] Add a command that evaluates an existing formula without retraining.

## P2: Make the Tool Easier to Use

- [ ] Add `.env.example` with safe placeholder values.
- [ ] Add a single research command that initializes the database, refreshes data, and runs validation checks.
- [ ] Add dashboard views for training history and backtest metrics.
- [ ] Show data freshness, row counts, and API-rate-limit status in the dashboard.
- [ ] Add automated tests for the data loader, VM operators, backtest accounting, and database queries.

## P3: Paper Trading Only

- [ ] Build a paper-trading simulator with persisted virtual balances.
- [ ] Add signal logging without transaction signing.
- [ ] Add explicit kill switches and dry-run defaults.
- [ ] Document live execution separately, after paper-trading results are reproducible.

## Out of Scope Until P0/P1 Are Complete

- [ ] Live wallet integration.
- [ ] Automatic Solana/Jupiter order execution.
- [ ] Claims that a mined formula is profitable or production-ready.
