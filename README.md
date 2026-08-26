# AlphaGPT

AlphaGPT is a reproducible factor-mining toolkit for public Binance Spot USDT
market data. It builds immutable `1h` datasets, searches for interpretable
formula factors on GPU, selects candidates on validation data, and performs one
final historical test after selection.

This repository is research software. It has no Binance account integration,
API keys, wallet, simulated account, order submission, or live trading path.
A `promising` report is a research classification, not evidence that a formula
will remain profitable or is production-ready.

## Research Pipeline

```text
Binance public REST / data.binance.vision
    -> PostgreSQL immutable dataset snapshot
    -> causal Binance features with train-only normalization
    -> independent GPU formula-mining seeds
    -> semantic formula deduplication
    -> validation-only anchored and rolling evaluation
    -> one final untouched-test evaluation
    -> reject / research-only / promising report
```

The supported first-release contract is fixed:

- Binance Spot only; no Futures, margin, leverage, or funding data.
- USDT quote markets selected by deterministic listing-age and volume rules.
- `1h` bars covering at least one year.
- Public endpoints only; no Binance API key is read or sent.
- PostgreSQL storage with immutable snapshot IDs and per-symbol quality gates.

See [docs/BINANCE_RESEARCH.md](docs/BINANCE_RESEARCH.md) for schemas, feature
definitions, label timing, costs, and known biases. See [TODO.md](TODO.md) for
the implementation status and acceptance gates.

## Requirements

- Linux with Python 3.10 or newer. The current machine uses Python 3.14.
- PostgreSQL 14 or newer.
- An NVIDIA GPU with a working PyTorch CUDA build for practical mining speed.
- Enough PostgreSQL storage for the selected universe and history range.

CPU execution works for tests and evaluation but is not practical for a full
`5 × 1000`-step mining batch.

## Installation

```bash
git clone https://github.com/Eclipsemos/AlphaGPT.git
cd AlphaGPT
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For a CUDA environment, install the PyTorch wheel appropriate for the host
driver before installing the remaining requirements. Verify it with:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## PostgreSQL

Create a dedicated role and database. Replace the example password before
running these commands:

```bash
sudo -u postgres psql -c "CREATE ROLE alphagpt LOGIN PASSWORD 'replace-this-password';"
sudo -u postgres psql -c "CREATE DATABASE crypto_quant OWNER alphagpt;"
```

Create `.env` in the repository root:

```env
DB_USER=alphagpt
DB_PASSWORD=replace-this-password
DB_HOST=127.0.0.1
DB_PORT=5432
DB_NAME=crypto_quant
```

No Birdeye key, Binance key, Solana RPC URL, or private key is used by the
Binance research workflow.

## Build A Dataset

Activate the environment and initialize or refresh a quality-gated snapshot:

```bash
source venv/bin/activate
python -m data_pipeline.run_binance_pipeline
```

Use official monthly archives for complete months and public REST for partial
months:

```bash
python -m data_pipeline.run_binance_pipeline --source archive
```

Discover the current universe without downloading bars:

```bash
python -m data_pipeline.run_binance_pipeline --dry-run
```

Successful ingestion prints a `dataset_snapshot_id` and writes
`runs/binance_latest/dataset_report.json`. A snapshot is created only after all
symbols pass timestamp, cadence, OHLC, volume, listing-boundary, and configured
coverage checks.

List available snapshot IDs:

```bash
psql "postgresql://alphagpt@127.0.0.1/crypto_quant" -c \
  "SELECT snapshot_id, interval, start_time, end_time, created_at FROM dataset_snapshots ORDER BY created_at DESC;"
```

## Mine Factors

First run a small-universe smoke test. The symbol subset becomes part of the
formula artifact and must match later evaluation:

```bash
python -m model_core.binance_engine \
  --snapshot-id <snapshot-id> \
  --symbols BTCUSDT,ETHUSDT \
  --seed 17 \
  --steps 1 \
  --batch-size 2048 \
  --output-dir runs/binance/smoke
```

Resume the same seed and compatible mining configuration:

```bash
python -m model_core.binance_engine \
  --snapshot-id <snapshot-id> \
  --symbols BTCUSDT,ETHUSDT \
  --seed 17 \
  --steps 1000 \
  --batch-size 2048 \
  --output-dir runs/binance/smoke \
  --resume
```

Changing the snapshot, universe, feature vocabulary, seed, or mining config
causes resume/evaluation to fail instead of silently mixing experiments.

## Run A Research Batch

The default command uses five independent seeds and writes a timestamped run
under `runs/binance/`:

```bash
python -m model_core.binance_batch \
  --snapshot-id <snapshot-id> \
  --seeds 1,2,3,4,5 \
  --steps 1000 \
  --batch-size 8192 \
  --windows 4
```

To resume an interrupted batch, pass its existing directory and `--resume`:

```bash
python -m model_core.binance_batch \
  --snapshot-id <snapshot-id> \
  --seeds 1,2,3,4,5 \
  --steps 1000 \
  --batch-size 8192 \
  --windows 4 \
  --output-dir runs/binance/<run-id> \
  --resume
```

Formal candidate selection never reads test tensors. It aggregates canonical
formulas across seeds, shortlists by validation IC, ranks on sequential
validation walk-forward windows, persists the selected formula, and only then
runs one final test comparison.

Each batch contains:

- `experiment_manifest.json`: command, Git commit, packages, device, CUDA, and timing.
- `seed_<n>/`: formula, candidate pool, history, and resumable checkpoint.
- `candidate_aggregation.json`: canonical cross-seed candidate groups.
- `walk_forward_report.json`: validation-only anchored and rolling results.
- `selected_formula.json`: immutable formula and research metadata.
- `final_evaluation_report.json`: split metrics, baselines, and cost sensitivity.
- `decision_report.json`: explicit gates and research status.
- `batch_report.json`: top-level artifact index and cross-seed summary.

`reject` means validation or robustness gates failed. `research-only` means the
candidate survived pre-test selection but failed at least one final gate.
`promising` requires every configured validation, stability, baseline, cost,
drawdown, and capacity gate; it still does not authorize trading.

## Evaluate One Saved Formula

```bash
python -m model_core.evaluate_binance \
  --formula runs/binance/<run-id>/selected_formula.json \
  --snapshot-id <snapshot-id> \
  --max-positions 10 \
  --weighting equal \
  --rebalance-hours 24 \
  --taker-fee-bps 10 \
  --slippage-bps 5 \
  --minimum-quote-volume-usd 1000000 \
  --cost-scenarios 0,15,30 \
  --output runs/binance/<run-id>/manual_evaluation.json
```

Weights, turnover, costs, and returns in this report are historical statistical
constructs. AlphaGPT does not persist balances, positions, fills, or orders.

## Dashboard

```bash
streamlit run dashboard/app.py --server.address 0.0.0.0 --server.port 8501
```

Open `http://127.0.0.1:8501` on the server or
`http://<server-lan-ip>:8501` from another computer on the same trusted network.
The dashboard is read-only and shows Binance snapshots, coverage, freshness,
features, batch decisions, baselines, and cost sensitivity.

## Verification

```bash
python -m unittest discover -s tests -v
python -m compileall -q data_pipeline model_core dashboard tests
```

Run the full synthetic acceptance workflow, including short mining and final
report-schema verification:

```bash
python -m model_core.binance_acceptance
```

The acceptance fixture and deterministic tests require no PostgreSQL server,
Binance API key, wallet, network access, or private endpoint.

## Known Limitations

- The initial universe is reconstructed from currently listed symbols and has survivorship bias.
- Quote-volume capacity is a proxy, not an order-book or fill model.
- Historical costs are sensitivity assumptions, not executable quotes.
- Formula search can overfit through repeated experiments even with a held-out test split.
- A result from one dataset, universe, or market regime does not establish future performance.

## License

See [LICENSE](LICENSE).
