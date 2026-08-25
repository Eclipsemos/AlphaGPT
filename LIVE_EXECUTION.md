# Live Execution Status

Live wallet integration and automatic Solana/Jupiter order execution are intentionally out of scope until the research and paper-trading gates in `TODO.md` are complete.

The safe commands are:

```bash
python -m model_core.evaluate --formula best_meme_strategy.json
python -m model_core.walk_forward --formula best_meme_strategy.json
python -m strategy_manager.paper --formula best_meme_strategy.json
```

`strategy_manager.paper` uses historical test data, virtual cash, and a local JSON state file. It does not import a private key, create an RPC signer, or submit a transaction.

Do not run this command in research mode:

```bash
python -m strategy_manager.runner
```

Before any future live-execution work, require all of the following:

- reproducible out-of-sample and walk-forward results;
- paper-trading results over a meaningful period;
- explicit risk limits and a kill switch;
- independent review of transaction construction, slippage, custody, and key handling;
- a separate opt-in configuration that cannot be enabled by accident.
