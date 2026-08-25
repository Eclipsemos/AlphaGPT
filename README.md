# AlphaGPT

AlphaGPT is an experimental crypto-market research toolkit. It collects Solana token OHLCV data, computes interpretable features, searches for formula-based signals, and evaluates those signals with a lightweight backtest.

It is a research prototype, not a profitable-strategy guarantee and not a production trading system. The trading executor is intentionally outside the recommended workflow.

## What It Does

```text
Birdeye API
    -> PostgreSQL
    -> feature engineering
    -> formula search on GPU
    -> in-sample backtest
    -> Streamlit dashboard
```

Main components:

- `data_pipeline/`: discovers tokens and stores OHLCV data.
- `model_core/`: computes factors, generates formula candidates, and scores them.
- `dashboard/`: displays the database snapshot, portfolio state, strategy file, and logs.
- `strategy_manager/` and `execution/`: live-trading code. Do not run these unless a separate paper-trading and risk review has been completed.

## Current Status

The end-to-end research path has been tested with PostgreSQL, Birdeye, CUDA/PyTorch, and the local Streamlit dashboard. The current implementation still has important research limitations:

- training and evaluation use the same historical sample;
- the data universe is small and selected from current trending tokens;
- the backtest reports a simplified score rather than a full performance report;
- there is no robust walk-forward or out-of-sample validation;
- generated formulas have not demonstrated positive returns in the current dataset.

Treat every generated formula as an experiment. Do not infer that a positive in-sample score is deployable.

## Requirements

- Ubuntu/Linux with Python 3.10+ (the tested environment uses Python 3.14).
- A virtual environment containing PyTorch and the packages in `requirements.txt`.
- PostgreSQL 18 or a compatible PostgreSQL server.
- A Birdeye API key for data collection.
- An NVIDIA GPU is recommended for training, but CPU execution is possible.

## Configuration

Create `.env` in the project root. Keep this file private and never commit it:

```env
BIRDEYE_API_KEY=replace-with-your-key
DB_USER=alphagpt
DB_PASSWORD=replace-with-your-db-password
DB_HOST=127.0.0.1
DB_PORT=5432
DB_NAME=crypto_quant
```

No wallet private key is required for the research workflow. Do not add `SOLANA_PRIVATE_KEY` unless you have deliberately reviewed and enabled live execution.

## Research Workflow

Activate the tested environment:

```bash
cd ~/workspace/AlphaGPT
source ~/gpu/venv/bin/activate
```

Collect or refresh market data:

```bash
python -m data_pipeline.run_pipeline
```

Run non-trading research checks in one command:

```bash
python -m research
python -m research --refresh --evaluate --walk-forward
```

Run formula mining:

```bash
python -m model_core.engine
```

Use `--seed` for reproducible sampling, `--steps` for a shorter experiment, and `--resume` to continue from `training_checkpoint.pt`:

```bash
python -m model_core.engine --seed 7 --steps 1000
python -m model_core.engine --resume --seed 7 --steps 1000
```

The command writes local experiment artifacts:

- `best_meme_strategy.json`: the best formula found in that run;
- `training_history.json`: reward history for the run.
- `evaluation_report.json`: validation/test metrics and test-set baseline comparisons.

Evaluate an existing formula without retraining:

```bash
python -m model_core.evaluate --formula best_meme_strategy.json
```

Run a simple walk-forward evaluation over unseen temporal windows:

```bash
python -m model_core.walk_forward --formula best_meme_strategy.json --windows 4
```

Run the research-core tests:

```bash
python -m unittest discover -s tests -v
```

Inspect the dashboard:

```bash
streamlit run dashboard/app.py --server.address 0.0.0.0 --server.port 8501
```

Open `http://127.0.0.1:8501` locally, or `http://<server-lan-ip>:8501` from another computer on the same network.

## Before Trusting Any Result

At minimum, a strategy needs:

1. a fixed historical universe rather than only today's trending tokens;
2. time-based train, validation, and test splits;
3. walk-forward evaluation on unseen data;
4. transaction-cost, drawdown, turnover, Sharpe, and trade-count metrics;
5. a paper-trading period with no private key and no live execution.

The repository does not provide all of these safeguards yet. Running more training steps alone does not solve that problem.

## Safety

Do not run this command as part of the research workflow:

```bash
python -m strategy_manager.runner
```

That module is intended to connect signals to Solana/Jupiter execution. The project currently has no complete paper-trading switch, so live execution must remain disabled.

## License

See [LICENSE](LICENSE).
