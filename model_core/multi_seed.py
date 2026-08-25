"""Summarize repeated saved-formula evaluations across random seeds.

This command evaluates formula files produced by independent runs. It does
not start training or trading; callers can pass a comma-separated list of
JSON files and receive mean/std confidence summaries on unseen data.
"""

import argparse
import json
import math

from .data_loader import CryptoDataLoader
from .evaluate import load_formula
from .backtest import MemeBacktest
from .vm import StackVM


def run(formula_paths, output_path="multi_seed_report.json"):
    loader = CryptoDataLoader()
    loader.load_data()
    vm = StackVM()
    backtest = MemeBacktest()
    rows = []
    for path in formula_paths:
        formula = load_formula(path)
        factors = vm.execute(formula, loader.test_feat_tensor)
        if factors is None:
            raise ValueError(f"Invalid formula in {path}")
        report = backtest.evaluate_report(factors, loader.test_raw_data_cache, loader.test_target_ret)
        rows.append({"formula_path": path, **report.as_dict()})
    summary = {}
    for key in rows[0]:
        if key == "formula_path":
            continue
        values = [float(row[key]) for row in rows]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
        std = math.sqrt(variance)
        margin = 1.96 * std / math.sqrt(len(values)) if values else 0.0
        summary[key] = {
            "mean": mean,
            "std": std,
            "ci95_low": mean - margin,
            "ci95_high": mean + margin,
            "count": len(values),
        }
    result = {"runs": rows, "summary": summary, "data_quality": loader.quality_report.as_dict()}
    with open(output_path, "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description="Summarize saved formula runs on the test split.")
    parser.add_argument("--formulas", required=True, help="Comma-separated JSON formula files")
    parser.add_argument("--output", default="multi_seed_report.json")
    args = parser.parse_args()
    run([path.strip() for path in args.formulas.split(",") if path.strip()], args.output)


if __name__ == "__main__":
    main()
