import inspect
import unittest
from types import SimpleNamespace

import torch

from model_core.binance_engine import BinanceAlphaEngine, BinanceMiningConfig, parse_symbols
from model_core.binance_mining import cross_sectional_ic_score, cross_sectional_ic_scores
from model_core.formula_canonical import canonical_formula, formula_complexity
from model_core.vm import StackVM
from model_core.vocab import BINANCE_FORMULA_VOCAB


class BinanceMiningTests(unittest.TestCase):
    def test_batch_vm_matches_scalar_execution_and_validity(self):
        vm = StackVM(BINANCE_FORMULA_VOCAB)
        add = BINANCE_FORMULA_VOCAB.operator_offset
        formulas = torch.tensor(
            [
                [0, 1, add],
                [2, add + 5, add + 4],
                [1, 0, add + 1],
                [add, 0, 0],
            ],
            dtype=torch.long,
        )
        features = torch.randn(5, BINANCE_FORMULA_VOCAB.feature_count, 7)
        expected_validity = torch.tensor(
            [vm.is_valid_formula(formula) for formula in formulas.tolist()]
        )
        factors, actual_validity = vm.execute_batch(formulas, features, chunk_size=2)
        self.assertTrue(torch.equal(actual_validity, expected_validity))
        for index, formula in enumerate(formulas.tolist()):
            expected = vm.execute(formula, features)
            if expected is None:
                self.assertTrue(torch.equal(factors[index], torch.zeros_like(factors[index])))
            else:
                self.assertTrue(torch.allclose(factors[index], expected))

    def test_batch_vm_matches_every_operator(self):
        vm = StackVM(BINANCE_FORMULA_VOCAB)
        neg = BINANCE_FORMULA_VOCAB.operator_offset + 4
        formulas = []
        for operator_token, arity in vm.arity_map.items():
            formula = list(range(arity)) + [operator_token]
            formula.extend([neg] * (12 - len(formula)))
            formulas.append(formula)
        formula_tensor = torch.tensor(formulas, dtype=torch.long)
        features = torch.randn(4, BINANCE_FORMULA_VOCAB.feature_count, 16)
        factors, validity = vm.execute_batch(formula_tensor, features, chunk_size=5)
        self.assertTrue(bool(validity.all()))
        for index, formula in enumerate(formulas):
            self.assertTrue(torch.allclose(factors[index], vm.execute(formula, features)))

    def test_batch_validity_matches_scalar_for_random_formulas(self):
        generator = torch.Generator().manual_seed(17)
        formulas = torch.randint(
            -1,
            BINANCE_FORMULA_VOCAB.size + 1,
            (512, 12),
            generator=generator,
        )
        vm = StackVM(BINANCE_FORMULA_VOCAB)
        expected = torch.tensor(
            [vm.is_valid_formula(formula) for formula in formulas.tolist()]
        )
        self.assertTrue(torch.equal(vm.valid_formula_mask(formulas), expected))

    def test_batch_ic_matches_individual_scores(self):
        generator = torch.Generator().manual_seed(23)
        factors = torch.randn(7, 12, 9, generator=generator)
        target = torch.randn(12, 9, generator=generator)
        valid = torch.rand(12, 9, generator=generator) > 0.1
        expected = torch.stack(
            [
                cross_sectional_ic_score(
                    factor,
                    target,
                    valid,
                    minimum_cross_section=10,
                )
                for factor in factors
            ]
        )
        actual = cross_sectional_ic_scores(
            factors,
            target,
            valid,
            minimum_cross_section=10,
        )
        self.assertTrue(torch.allclose(actual, expected, atol=1e-7, rtol=1e-6))

    def test_engine_streams_score_chunks_without_changing_scores(self):
        generator = torch.Generator().manual_seed(29)
        features = torch.randn(
            12,
            BINANCE_FORMULA_VOCAB.feature_count,
            11,
            generator=generator,
        )
        target = torch.randn(12, 11, generator=generator)
        valid = torch.ones_like(target, dtype=torch.bool)
        add = BINANCE_FORMULA_VOCAB.operator_offset
        formulas = torch.tensor(
            [[0, 1, add], [2, 3, add + 1], [4, add + 4, add + 5]],
            dtype=torch.long,
        )
        engine = BinanceAlphaEngine.__new__(BinanceAlphaEngine)
        engine.config = BinanceMiningConfig(
            steps=1,
            batch_size=3,
            scoring_chunk_size=2,
            minimum_cross_section=10,
        )
        engine.vm = StackVM(BINANCE_FORMULA_VOCAB)
        engine.loader = SimpleNamespace(
            train_feat_tensor=features,
            train_target_ret=target,
            train_target_valid=valid,
        )
        actual = engine._score_formulas_batch(formulas, "train")
        expected = torch.stack(
            [
                cross_sectional_ic_score(
                    engine.vm.execute(formula, features),
                    target,
                    valid,
                    minimum_cross_section=10,
                )
                for formula in formulas.tolist()
            ]
        )
        self.assertTrue(torch.allclose(actual, expected, atol=1e-7, rtol=1e-6))

    def test_cross_sectional_ic_rewards_predictive_and_rejects_constant_factors(self):
        target = torch.tensor(
            [[0.1, -0.1], [0.2, 0.0], [0.3, 0.1], [0.0, 0.2], [-0.2, 0.3], [0.4, -0.3], [0.5, 0.4], [-0.1, 0.5], [0.6, -0.4], [0.7, 0.6]]
        )
        valid = torch.ones_like(target, dtype=torch.bool)
        positive = cross_sectional_ic_score(target, target, valid)
        negative = cross_sectional_ic_score(-target, target, valid)
        constant = cross_sectional_ic_score(torch.ones_like(target), target, valid)
        self.assertGreater(float(positive), 0)
        self.assertLess(float(negative), 0)
        self.assertEqual(float(constant), -10.0)

    def test_cross_sectional_ic_requires_configured_sample_and_does_not_saturate(self):
        factors = torch.tensor([[0.1, 0.2], [0.2, 0.1], [0.3, 0.4]])
        target = torch.tensor([[0.2, 0.1], [0.1, 0.3], [0.3, 0.2]])
        valid = torch.ones_like(factors, dtype=torch.bool)
        score = cross_sectional_ic_score(
            factors, target, valid, minimum_cross_section=3
        )
        self.assertGreaterEqual(float(score), -1.0)
        self.assertLessEqual(float(score), 1.0)
        too_small = cross_sectional_ic_score(
            factors[:2], target[:2], valid[:2], minimum_cross_section=3
        )
        self.assertEqual(float(too_small), -10.0)

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
        with self.assertRaises(ValueError):
            BinanceMiningConfig(minimum_cross_section=1)
        self.assertEqual(parse_symbols("btcusdt, ETHUSDT"), ["BTCUSDT", "ETHUSDT"])
        with self.assertRaises(Exception):
            parse_symbols("BTCUSDT,BTCUSDT")

    def test_training_source_does_not_access_test_tensors(self):
        source = inspect.getsource(BinanceAlphaEngine.train)
        scorer_source = inspect.getsource(BinanceAlphaEngine._score_formula)
        self.assertNotIn("test_", source)
        self.assertNotIn('"test"', scorer_source)
        self.assertIn("progress_callback", source)

    def test_checkpoint_cuda_state_is_normalized_before_restore(self):
        source = inspect.getsource(BinanceAlphaEngine.load_checkpoint)
        self.assertIn("torch.as_tensor", source)
        self.assertIn("dtype=torch.uint8", source)


if __name__ == "__main__":
    unittest.main()
