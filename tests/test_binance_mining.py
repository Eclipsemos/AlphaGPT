import inspect
import unittest

import torch

from model_core.binance_engine import BinanceAlphaEngine, BinanceMiningConfig, parse_symbols
from model_core.binance_mining import cross_sectional_ic_score
from model_core.formula_canonical import canonical_formula, formula_complexity
from model_core.vocab import BINANCE_FORMULA_VOCAB


class BinanceMiningTests(unittest.TestCase):
    def test_cross_sectional_ic_rewards_predictive_and_rejects_constant_factors(self):
        target = torch.tensor([[0.1, -0.1], [0.2, 0.0], [0.3, 0.1]])
        valid = torch.ones_like(target, dtype=torch.bool)
        positive = cross_sectional_ic_score(target, target, valid)
        negative = cross_sectional_ic_score(-target, target, valid)
        constant = cross_sectional_ic_score(torch.ones_like(target), target, valid)
        self.assertGreater(float(positive), 0)
        self.assertLess(float(negative), 0)
        self.assertEqual(float(constant), -10.0)

    def test_canonical_formula_deduplicates_commutative_children(self):
        add = BINANCE_FORMULA_VOCAB.operator_offset
        first = canonical_formula([0, 1, add], BINANCE_FORMULA_VOCAB)
        second = canonical_formula([1, 0, add], BINANCE_FORMULA_VOCAB)
        subtract = canonical_formula([0, 1, add + 1], BINANCE_FORMULA_VOCAB)
        reversed_subtract = canonical_formula([1, 0, add + 1], BINANCE_FORMULA_VOCAB)
        self.assertEqual(first, second)
        self.assertNotEqual(subtract, reversed_subtract)
        self.assertEqual(
            formula_complexity([0, 1, add], BINANCE_FORMULA_VOCAB),
            {
                "token_count": 3,
                "operator_count": 1,
                "unique_feature_count": 2,
                "tree_depth": 2,
            },
        )

    def test_mining_config_and_symbol_parser_reject_invalid_values(self):
        with self.assertRaises(ValueError):
            BinanceMiningConfig(steps=0)
        self.assertEqual(parse_symbols("btcusdt, ETHUSDT"), ["BTCUSDT", "ETHUSDT"])
        with self.assertRaises(Exception):
            parse_symbols("BTCUSDT,BTCUSDT")

    def test_training_source_does_not_access_test_tensors(self):
        source = inspect.getsource(BinanceAlphaEngine.train)
        scorer_source = inspect.getsource(BinanceAlphaEngine._score_formula)
        self.assertNotIn("test_", source)
        self.assertNotIn('"test"', scorer_source)

    def test_checkpoint_cuda_state_is_normalized_before_restore(self):
        source = inspect.getsource(BinanceAlphaEngine.load_checkpoint)
        self.assertIn("torch.as_tensor", source)
        self.assertIn("dtype=torch.uint8", source)


if __name__ == "__main__":
    unittest.main()
