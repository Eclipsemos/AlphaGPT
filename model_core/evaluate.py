"""Evaluate an existing formula without running the mining loop."""

import argparse
import json

from .backtest import MemeBacktest
from .data_loader import CryptoDataLoader
from .vm import StackVM


def load_formula(path):
    with open(path) as handle:
        value = json.load(handle)
    formula = value.get("formula") if isinstance(value, dict) else value
    if not isinstance(formula, list) or not formula:
        raise ValueError(f"{path} does not contain a non-empty formula token list")
    return [int(token) for token in formula]


def run(formula_path="best_meme_strategy.json", output_path="evaluation_report.json"):
    loader = CryptoDataLoader()
    loader.load_data()
    formula = load_formula(formula_path)
    vm = StackVM()
    backtest = MemeBacktest()

    reports = {}
    for name, features, raw_data, target in (
        ("train", loader.train_feat_tensor, loader.train_raw_data_cache, loader.train_target_ret),
        ("validation", loader.validation_feat_tensor, loader.validation_raw_data_cache, loader.validation_target_ret),
        ("test", loader.test_feat_tensor, loader.test_raw_data_cache, loader.test_target_ret),
    ):
        factors = vm.execute(formula, features)
        if factors is None:
            raise ValueError(f"Formula is invalid for the {name} feature tensor")
        reports[name] = backtest.evaluate_report(factors, raw_data, target).as_dict()

    reports["test_baselines"] = {
        name: report.as_dict()
        for name, report in backtest.baseline_reports(
            loader.test_raw_data_cache,
            loader.test_target_ret,
        ).items()
    }
    reports["formula"] = formula
    reports["data_quality"] = loader.quality_report.as_dict()
    with open(output_path, "w") as handle:
        json.dump(reports, handle, indent=2)
    print(json.dumps(reports, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved AlphaGPT formula.")
    parser.add_argument("--formula", default="best_meme_strategy.json")
    parser.add_argument("--output", default="evaluation_report.json")
    args = parser.parse_args()
    run(args.formula, args.output)


if __name__ == "__main__":
    main()
