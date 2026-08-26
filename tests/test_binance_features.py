import unittest
import math

import torch

from model_core.binance_features import BINANCE_FEATURE_DEFINITIONS, BinanceFeatureEngineer
from model_core.formula_artifact import build_formula_artifact, validate_formula_artifact
from model_core.vocab import BINANCE_FORMULA_VOCAB, FORMULA_VOCAB
from model_core.vm import StackVM


def feature_input(length=40):
    time = torch.arange(length, dtype=torch.float32)
    close = (100 + time).unsqueeze(0)
    observed = torch.ones_like(close, dtype=torch.bool)
    return {
        "open": close - 0.25,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "base_volume": (1000 + 10 * time).unsqueeze(0),
        "quote_volume": (100_000 + 1000 * time).unsqueeze(0),
        "trade_count": (100 + time).unsqueeze(0),
        "taker_buy_quote_volume": (50_000 + 500 * time).unsqueeze(0),
    }, observed


class BinanceFeatureTests(unittest.TestCase):
    def test_feature_shape_warmup_and_finite_values(self):
        raw, observed = feature_input()
        result = BinanceFeatureEngineer.compute(raw, observed, fit_end=25)
        self.assertEqual(result.values.shape, (1, 11, 40))
        self.assertEqual(result.valid.shape, (1, 11, 40))
        self.assertFalse(result.valid[0, 0, 0])
        self.assertFalse(result.valid[0, 2, 13])
        self.assertTrue(result.valid[0, 2, 14])
        self.assertFalse(result.valid[0, 4, 23])
        self.assertTrue(result.valid[0, 4, 24])
        self.assertTrue(torch.isfinite(result.values).all())
        self.assertEqual(tuple(BINANCE_FEATURE_DEFINITIONS), result.normalization.feature_names)

    def test_missing_bar_invalidates_rolling_features_without_fill(self):
        raw, observed = feature_input()
        observed[:, 20] = False
        result = BinanceFeatureEngineer.compute(raw, observed, fit_end=25)
        self.assertFalse(result.valid[0, :, 20].any())
        self.assertFalse(result.valid[0, 5, 40 - 1])

    def test_future_values_do_not_change_past_features_or_normalization(self):
        raw, observed = feature_input()
        first = BinanceFeatureEngineer.compute(raw, observed, fit_end=25)
        changed = {key: value.clone() for key, value in raw.items()}
        for value in changed.values():
            value[:, 30:] *= 1000
        second = BinanceFeatureEngineer.compute(changed, observed, fit_end=25)
        self.assertTrue(torch.equal(first.values[:, :, :25], second.values[:, :, :25]))
        self.assertTrue(torch.equal(first.normalization.median, second.normalization.median))
        self.assertTrue(torch.equal(first.normalization.mad, second.normalization.mad))

    def test_all_feature_definitions_match_hand_calculation(self):
        raw, observed = feature_input()
        result = BinanceFeatureEngineer.compute(raw, observed, fit_end=40)
        reconstructed = (
            result.values * result.normalization.mad.unsqueeze(-1)
            + result.normalization.median.unsqueeze(-1)
        )
        index = 24
        close = 124.0
        previous_close = 123.0
        returns = [math.log((100 + value) / (99 + value)) for value in range(1, 25)]
        true_ranges = [2.0 / (99 + value) for value in range(11, 25)]
        expected = torch.tensor(
            [
                math.log(close / previous_close),
                2.0 / previous_close,
                sum(true_ranges) / 14,
                0.5,
                math.log(close / 100.0),
                math.sqrt(sum(value * value for value in returns) / 24),
                math.log1p(1240.0),
                math.log1p(124000.0),
                math.log1p(124000.0) - math.log1p(123000.0),
                math.log1p(124.0),
                0.0,
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(reconstructed[0, :, index], expected, atol=1e-5))

    def test_formula_artifact_rejects_cross_market_vocabulary(self):
        formula = [0]
        artifact = build_formula_artifact(
            formula,
            BINANCE_FORMULA_VOCAB,
            {"dataset_snapshot_id": "fixture"},
        )
        self.assertEqual(validate_formula_artifact(artifact, BINANCE_FORMULA_VOCAB), formula)
        with self.assertRaises(ValueError):
            validate_formula_artifact(artifact, FORMULA_VOCAB)
        with self.assertRaises(ValueError):
            validate_formula_artifact(formula, BINANCE_FORMULA_VOCAB)

    def test_vm_operator_offset_is_vocabulary_specific(self):
        self.assertNotEqual(FORMULA_VOCAB.operator_offset, BINANCE_FORMULA_VOCAB.operator_offset)
        features = torch.ones((1, BINANCE_FORMULA_VOCAB.feature_count, 3))
        vm = StackVM(BINANCE_FORMULA_VOCAB)
        formula = [0, 1, BINANCE_FORMULA_VOCAB.operator_offset]
        self.assertTrue(vm.is_valid_formula(formula))
        self.assertTrue(torch.equal(vm.execute(formula, features), torch.full((1, 3), 2.0)))


if __name__ == "__main__":
    unittest.main()
