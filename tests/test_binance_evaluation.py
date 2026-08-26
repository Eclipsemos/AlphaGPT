import math
import unittest

import torch

from model_core.binance_evaluation import (
    HOURS_PER_YEAR,
    BinanceEvaluationConfig,
    BinanceFactorEvaluator,
    baseline_reports,
    cost_sensitivity_reports,
    mean_rank_ic,
)


def raw_data(symbols, periods):
    close = torch.arange(periods, dtype=torch.float32).repeat(symbols, 1) + 100
    return {
        "close": close,
        "quote_volume": torch.full((symbols, periods), 1_000_000.0),
    }


class BinanceEvaluationTests(unittest.TestCase):
    def test_signal_ranking_maps_to_next_open_return_label(self):
        factors = torch.tensor([[3.0, 0.0, 3.0], [0.0, 3.0, 0.0]])
        valid = torch.ones_like(factors, dtype=torch.bool)
        evaluator = BinanceFactorEvaluator(
            BinanceEvaluationConfig(max_positions=1, rebalance_hours=1, taker_fee_bps=0, slippage_bps=0)
        )
        weights = evaluator.construct_weights(factors, raw_data(2, 3), valid)
        self.assertTrue(torch.equal(weights, torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])))
        target = torch.log1p(torch.tensor([[0.10, 0.00, 0.20], [0.00, 0.30, 0.00]]))
        report = evaluator.evaluate(factors, raw_data(2, 3), target, valid, valid, ["BTCUSDT", "ETHUSDT"])
        self.assertAlmostEqual(report.cumulative_return, 1.1 * 1.3 * 1.2 - 1, places=5)

    def test_compounding_charges_entry_and_terminal_exit_cost(self):
        factors = torch.ones((1, 2))
        valid = torch.ones_like(factors, dtype=torch.bool)
        target = torch.log1p(torch.full((1, 2), 0.10))
        evaluator = BinanceFactorEvaluator(
            BinanceEvaluationConfig(max_positions=1, rebalance_hours=24, taker_fee_bps=10, slippage_bps=0)
        )
        report = evaluator.evaluate(factors, raw_data(1, 2), target, valid, valid, ["BTCUSDT"])
        self.assertAlmostEqual(report.total_turnover, 2.0, places=6)
        self.assertAlmostEqual(report.fee_cost, 0.002, places=6)
        self.assertAlmostEqual(report.cumulative_return, (1.099 * 1.099) - 1, places=5)

    def test_rebalance_cadence_holds_weights_between_rank_updates(self):
        factors = torch.tensor([[3.0, 0.0, 0.0], [0.0, 3.0, 3.0]])
        valid = torch.ones_like(factors, dtype=torch.bool)
        evaluator = BinanceFactorEvaluator(
            BinanceEvaluationConfig(max_positions=1, rebalance_hours=2, taker_fee_bps=0, slippage_bps=0)
        )
        weights = evaluator.construct_weights(factors, raw_data(2, 3), valid)
        self.assertTrue(torch.equal(weights, torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])))

    def test_missing_symbol_forces_historical_exit(self):
        factors = torch.tensor([[3.0, 3.0, 3.0], [0.0, 0.0, 0.0]])
        valid = torch.ones_like(factors, dtype=torch.bool)
        valid[0, 1:] = False
        evaluator = BinanceFactorEvaluator(
            BinanceEvaluationConfig(max_positions=1, rebalance_hours=24, taker_fee_bps=0, slippage_bps=0)
        )
        weights = evaluator.construct_weights(factors, raw_data(2, 3), valid)
        self.assertTrue(torch.equal(weights[:, 0], torch.tensor([1.0, 0.0])))
        self.assertTrue(torch.equal(weights[:, 1], torch.tensor([0.0, 0.0])))

    def test_sharpe_uses_fixed_hourly_annualization(self):
        factors = torch.ones((1, 4))
        valid = torch.ones_like(factors, dtype=torch.bool)
        simple = torch.tensor([[0.01, -0.005, 0.01, -0.005]])
        evaluator = BinanceFactorEvaluator(
            BinanceEvaluationConfig(max_positions=1, rebalance_hours=24, taker_fee_bps=0, slippage_bps=0)
        )
        report = evaluator.evaluate(
            factors, raw_data(1, 4), torch.log1p(simple), valid, valid, ["BTCUSDT"]
        )
        expected = simple.flatten().mean() / simple.flatten().std(unbiased=False) * math.sqrt(HOURS_PER_YEAR)
        self.assertAlmostEqual(report.sharpe, float(expected), places=4)

    def test_cost_sensitivity_and_baselines_are_deterministic(self):
        factors = torch.tensor([[2.0] * 30, [1.0] * 30])
        valid = torch.ones_like(factors, dtype=torch.bool)
        target = torch.log1p(torch.full_like(factors, 0.001))
        config = BinanceEvaluationConfig(max_positions=1, rebalance_hours=24)
        scenarios = cost_sensitivity_reports(
            factors, raw_data(2, 30), target, valid, valid,
            ["BTCUSDT", "ETHUSDT"], config, [0, 20]
        )
        self.assertGreater(scenarios["0_bps"].cumulative_return, scenarios["20_bps"].cumulative_return)
        first = baseline_reports(
            raw_data(2, 30), target, valid, valid, ["BTCUSDT", "ETHUSDT"], config, seed=7
        )
        second = baseline_reports(
            raw_data(2, 30), target, valid, valid, ["BTCUSDT", "ETHUSDT"], config, seed=7
        )
        self.assertEqual(set(first), {
            "equal_weight_cross_section", "btcusdt_reference",
            "cross_sectional_momentum", "random_rank",
        })
        self.assertEqual(first["random_rank"].as_dict(), second["random_rank"].as_dict())

    def test_rank_ic_does_not_break_ties_arbitrarily(self):
        factors = torch.ones((3, 1))
        targets = torch.tensor([[0.1], [0.2], [0.3]])
        valid = torch.ones_like(factors, dtype=torch.bool)
        self.assertEqual(mean_rank_ic(factors, targets, valid), 0.0)


if __name__ == "__main__":
    unittest.main()
