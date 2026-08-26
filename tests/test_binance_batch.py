import inspect
import unittest

from model_core.binance_batch import (
    DEFAULT_SEEDS,
    DecisionThresholds,
    aggregate_candidates,
    build_decision_report,
    parse_seeds,
    rank_walk_forward_candidates,
    walk_forward_windows,
)
from model_core.vocab import BINANCE_FORMULA_VOCAB


def candidate(canonical, validation, sharpe, score, seed_support=1):
    summary_metric = lambda value: {
        "count": 2,
        "mean": value,
        "std": 0.0,
        "ci95_low": value,
        "ci95_high": value,
        "median": value,
        "minimum": value,
        "maximum": value,
    }
    return {
        "canonical_formula": canonical,
        "formula": [0],
        "formula_length": 1,
        "seed_support": seed_support,
        "validation_score": {"mean": validation},
        "walk_forward": {
            "rolling": {
                "summary": {
                    "valid_window_count": 2,
                    "failed_window_count": 0,
                    "positive_sharpe_fraction": float(sharpe > 0),
                    "sharpe": summary_metric(sharpe),
                    "score": summary_metric(score),
                    "max_drawdown": summary_metric(-0.1),
                }
            }
        },
    }


def final_report(test_return=0.2, test_sharpe=1.0):
    metrics = {
        "score": 1.0,
        "cumulative_return": test_return,
        "sharpe": test_sharpe,
        "max_drawdown": -0.1,
        "maximum_volume_participation": 0.001,
    }
    baseline = {"score": 0.5}
    return {
        "splits": {"test": metrics},
        "test_baselines": {
            "equal_weight_cross_section": baseline,
            "btcusdt_reference": baseline,
            "cross_sectional_momentum": baseline,
            "random_rank": baseline,
        },
        "test_cost_sensitivity": {
            "0_bps": {"cumulative_return": 0.2},
            "30_bps": {"cumulative_return": 0.1},
        },
    }


class BinanceBatchTests(unittest.TestCase):
    def test_default_batch_uses_five_independent_seeds(self):
        self.assertEqual(DEFAULT_SEEDS, (1, 2, 3, 4, 5))
        self.assertEqual(parse_seeds("1, 2,3"), [1, 2, 3])
        with self.assertRaises(Exception):
            parse_seeds("1,1")

    def test_semantic_aggregation_merges_commutative_formulas(self):
        add = BINANCE_FORMULA_VOCAB.operator_offset
        result = aggregate_candidates(
            [
                {
                    "seed": 1,
                    "candidates": [
                        {"formula": [0, 1, add], "train_score": 1, "validation_score": 0.2, "first_seen_step": 2}
                    ],
                },
                {
                    "seed": 2,
                    "candidates": [
                        {"formula": [1, 0, add], "train_score": 2, "validation_score": 0.4, "first_seen_step": 3}
                    ],
                },
            ]
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["seed_support"], 2)
        self.assertAlmostEqual(result[0]["validation_score"]["mean"], 0.3)

    def test_walk_forward_has_disjoint_rolling_and_expanding_anchored_windows(self):
        windows = walk_forward_windows(10, 3)
        self.assertEqual(windows["rolling"], [(0, 3), (3, 6), (6, 10)])
        self.assertEqual(windows["anchored"], [(0, 3), (0, 6), (0, 10)])

    def test_ranking_uses_validation_walk_forward_and_not_test_fields(self):
        weak = candidate("weak", validation=0.5, sharpe=-1, score=-1)
        stable = candidate("stable", validation=0.1, sharpe=1, score=1)
        self.assertEqual(rank_walk_forward_candidates([weak, stable])[0]["canonical_formula"], "stable")
        source = inspect.getsource(rank_walk_forward_candidates)
        self.assertNotIn("test", source)

    def test_positive_test_return_alone_cannot_be_promising(self):
        selected = candidate("single-seed", validation=-0.1, sharpe=-1, score=-1, seed_support=1)
        decision = build_decision_report(selected, final_report(test_return=1.0, test_sharpe=5.0), DecisionThresholds())
        self.assertEqual(decision["status"], "reject")
        self.assertFalse(decision["test_was_used_for_candidate_selection"])
        self.assertIn("validation_ic", decision["failed_gates"]["pretest"])

    def test_candidate_that_passes_pretest_but_fails_final_is_research_only(self):
        selected = candidate("stable", validation=0.1, sharpe=1, score=1, seed_support=2)
        decision = build_decision_report(selected, final_report(test_return=-0.1, test_sharpe=-1), DecisionThresholds())
        self.assertEqual(decision["status"], "research-only")


if __name__ == "__main__":
    unittest.main()
