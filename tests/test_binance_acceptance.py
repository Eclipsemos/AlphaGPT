import json
import tempfile
import unittest
from pathlib import Path

from model_core.binance_acceptance import build_fixture_loader, verify_acceptance_artifacts
from model_core.binance_batch import BATCH_REPORT_VERSION, DECISION_REPORT_VERSION
from model_core.formula_artifact import FORMULA_ARTIFACT_VERSION


class BinanceAcceptanceTests(unittest.TestCase):
    def test_fixture_loader_is_deterministic_and_split(self):
        first = build_fixture_loader(120)
        second = build_fixture_loader(120)
        self.assertEqual(first.snapshot_id, second.snapshot_id)
        self.assertEqual(first.symbols, ["BTCUSDT", "ETHUSDT", "BNBUSDT"])
        self.assertEqual(first.feat_tensor.shape, (3, 11, 120))
        self.assertEqual(first.train_feat_tensor.shape[-1], 72)
        self.assertEqual(first.validation_feat_tensor.shape[-1], 24)
        self.assertEqual(first.test_feat_tensor.shape[-1], 24)

    def test_schema_verifier_rejects_missing_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                verify_acceptance_artifacts(directory)

    def test_schema_verifier_accepts_complete_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payloads = {
                "experiment_manifest.json": {"status": "complete"},
                "batch_report.json": {"report_version": BATCH_REPORT_VERSION},
                "candidate_aggregation.json": {},
                "walk_forward_report.json": {"test_was_accessed": False},
                "selected_formula.json": {
                    "artifact_version": FORMULA_ARTIFACT_VERSION,
                    "complexity": {
                        "token_count": 1,
                        "operator_count": 0,
                        "unique_feature_count": 1,
                        "tree_depth": 1,
                    },
                },
                "final_evaluation_report.json": {
                    "report_version": "binance-factor-evaluation-v1",
                    "splits": {"train": {}, "validation": {}, "test": {}},
                    "test_baselines": {
                        "equal_weight_cross_section": {},
                        "btcusdt_reference": {},
                        "cross_sectional_momentum": {},
                        "random_rank": {},
                    },
                    "test_regimes": {
                        "trend_up": {},
                        "trend_down": {},
                        "drawdown": {},
                        "high_volatility": {},
                        "low_volatility": {},
                    },
                    "test_robustness": {
                        "fee_bps": {},
                        "slippage_bps": {},
                        "rebalance_hours": {},
                        "max_positions": {},
                        "weighting": {},
                        "minimum_quote_volume_usd": {},
                    },
                },
                "decision_report.json": {
                    "report_version": DECISION_REPORT_VERSION,
                    "status": "reject",
                    "test_was_used_for_candidate_selection": False,
                },
            }
            for filename, payload in payloads.items():
                (root / filename).write_text(json.dumps(payload))
            result = verify_acceptance_artifacts(root)
            self.assertEqual(result["status"], "passed")


if __name__ == "__main__":
    unittest.main()
