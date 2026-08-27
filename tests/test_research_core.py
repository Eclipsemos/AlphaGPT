import unittest
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
import subprocess
import sys

import torch
import pandas as pd

from model_core.backtest import MemeBacktest
from model_core.data_loader import (
    CryptoDataLoader,
    compute_forward_returns,
    inspect_market_data,
)
from model_core.vm import StackVM
from dashboard.data_service import DashboardService
from batch_research import build_parser, parse_seeds
from data_pipeline.binance_contracts import (
    BinanceUniverseRules,
    dataset_snapshot_id,
    dataset_snapshot_payload,
)
from data_pipeline.db_manager import binance_schema_statements


class ResearchCoreTests(unittest.TestCase):
    def test_dashboard_imports_are_independent_of_launch_directory(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-c", "import app"],
            cwd=root / "dashboard",
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_dashboard_imports_without_a_configured_database(self):
        root = Path(__file__).resolve().parents[1]
        environment = {
            "DB_HOST": "127.0.0.1",
            "DB_PORT": "1",
            "DB_USER": "missing",
            "DB_PASSWORD": "missing",
            "DB_NAME": "missing",
        }
        result = subprocess.run(
            [sys.executable, "-c", "import app"],
            cwd=root / "dashboard",
            env={**__import__("os").environ, **environment},
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_repository_has_no_simulated_or_live_trading_modules(self):
        root = Path(__file__).resolve().parents[1]
        self.assertFalse((root / "execution").joinpath("trader.py").exists())
        self.assertFalse((root / "strategy_manager").joinpath("paper.py").exists())
        self.assertFalse((root / "strategy_manager").joinpath("runner.py").exists())
        self.assertFalse((root / "LIVE_EXECUTION.md").exists())

    def test_binance_dataset_contract_is_deterministic(self):
        rules = BinanceUniverseRules()
        first = dataset_snapshot_payload(
            rules,
            ["ETHUSDT", "BTCUSDT", "ETHUSDT"],
            datetime(2025, 1, 1, tzinfo=UTC),
            datetime(2026, 1, 1, tzinfo=UTC),
            source="binance-rest",
            code_version="abc123",
        )
        second = dataset_snapshot_payload(
            rules,
            ["BTCUSDT", "ETHUSDT"],
            datetime(2025, 1, 1, tzinfo=UTC),
            datetime(2026, 1, 1, tzinfo=UTC),
            source="binance-rest",
            code_version="abc123",
        )
        self.assertEqual(first, second)
        self.assertEqual(dataset_snapshot_id(first), dataset_snapshot_id(second))

    def test_binance_contract_rejects_non_spot_scope(self):
        with self.assertRaises(ValueError):
            BinanceUniverseRules(market_type="futures")
        with self.assertRaises(ValueError):
            BinanceUniverseRules(minimum_quote_volume=Decimal("0"))

    def test_binance_contract_rejects_invalid_minimum_universe(self):
        with self.assertRaises(ValueError):
            BinanceUniverseRules(minimum_symbols=0)

    def test_binance_schema_does_not_overload_solana_address(self):
        schema = "\n".join(binance_schema_statements())
        self.assertIn("market_instruments", schema)
        self.assertIn("market_bars", schema)
        self.assertIn("dataset_snapshots", schema)
        self.assertNotIn("address", schema)

    def test_batch_seed_parser_and_defaults(self):
        self.assertEqual(parse_seeds("7, 11,13"), [7, 11, 13])
        args = build_parser().parse_args([])
        self.assertEqual(args.seeds, [1, 2, 3])
        self.assertIsNone(args.output_dir)

    def test_batch_seed_parser_rejects_duplicates(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["--seeds", "1,1"])

    def test_forward_return_alignment_and_purge(self):
        opens = torch.tensor([[10.0, 11.0, 22.0, 44.0, 88.0]])
        actual = compute_forward_returns(opens)
        expected = torch.tensor([[torch.log(torch.tensor(22.0 / 11.0)), torch.log(torch.tensor(44.0 / 22.0)), torch.log(torch.tensor(88.0 / 44.0)), 0.0, 0.0]])
        self.assertTrue(torch.allclose(actual, expected))

    def test_forward_returns_expose_valid_labels(self):
        opens = torch.tensor([[10.0, 11.0, 22.0, 44.0, 88.0]])
        returns, valid = compute_forward_returns(
            opens,
            torch.tensor([[True, True, False, True, True]]),
            return_valid=True,
        )
        self.assertTrue(torch.equal(valid, torch.tensor([[False, False, False, False, False]])))
        self.assertTrue(torch.isfinite(returns).all())

    def test_backtest_compounds_simple_returns_from_log_labels(self):
        raw = {
            "liquidity": torch.full((1, 2), 1_000_000.0),
            "close": torch.ones((1, 2)),
        }
        backtest = MemeBacktest()
        backtest.trade_size = 0.0
        backtest.base_fee = 0.0
        factors = torch.full((1, 2), 10.0)
        target = torch.full((1, 2), torch.log(torch.tensor(1.1)))
        report = backtest.evaluate_report(factors, raw, target)
        self.assertAlmostEqual(report.cumulative_return, 0.21, places=5)

    def test_quality_report_detects_bad_rows(self):
        frame = pd.DataFrame({
            "time": pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-02"]),
            "address": ["A", "A", "B"],
            "open": [1.0, 1.0, 0.0], "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0], "close": [1.0, 1.0, 1.0],
            "volume": [1.0, 1.0, 1.0], "liquidity": [1.0, 1.0, 1.0],
            "fdv": [1.0, 1.0, 1.0],
        })
        report = inspect_market_data(frame)
        self.assertEqual(report.duplicate_rows, 1)
        self.assertEqual(report.nonpositive_price_rows, 1)
        self.assertTrue(report.has_fatal_errors)

    def test_temporal_split_boundaries(self):
        loader = CryptoDataLoader.__new__(CryptoDataLoader)
        split = loader._build_splits(100)
        self.assertEqual(split.train.stop, 60)
        self.assertEqual(split.validation.start, 60)
        self.assertEqual(split.validation.stop, 80)
        self.assertEqual(split.test.start, 80)

    def test_vm_rejects_invalid_stack_and_executes_valid_formula(self):
        features = torch.arange(12.0).reshape(1, 2, 6)
        vm = StackVM()
        self.assertIsNone(vm.execute([6], features))
        result = vm.execute([0, 1, 6], features)
        self.assertTrue(torch.equal(result, features[:, 0, :] + features[:, 1, :]))

    def test_backtest_report_contains_auditable_metrics(self):
        raw = {
            "liquidity": torch.full((1, 6), 1_000_000.0),
            "close": torch.ones((1, 6)),
        }
        target = torch.tensor([[0.01, -0.01, 0.02, 0.0, 0.0, 0.0]])
        factors = torch.full((1, 6), 10.0)
        report = MemeBacktest().evaluate_report(factors, raw, target)
        self.assertIn("sharpe", report.as_dict())
        self.assertGreaterEqual(report.trade_count, 1)
        self.assertTrue(torch.isfinite(torch.tensor(list(report.as_dict().values()))).all())

    def test_dashboard_market_query_uses_latest_binance_bar_per_symbol(self):
        query = DashboardService.binance_market_overview_query(25)
        self.assertIn("latest.symbol = b.symbol", query)
        self.assertIn("MAX(open_time)", query)
        self.assertIn("market_type = 'spot'", query)
        self.assertIn("LIMIT 25", query)

    def test_dashboard_market_query_clamps_limit(self):
        self.assertIn("LIMIT 1", DashboardService.binance_market_overview_query(0))
        self.assertIn("LIMIT 500", DashboardService.binance_market_overview_query(9999))


if __name__ == "__main__":
    unittest.main()
