"""Walk-forward evaluation for a saved formula on unseen temporal windows."""

import argparse
import json

from .backtest import MemeBacktest
from .data_loader import CryptoDataLoader
from .vm import StackVM


def evaluate(formula, loader, windows):
    vm = StackVM()
    backtest = MemeBacktest()
    reports = []
    total = loader.feat_tensor.shape[-1]
    for index, (train_end, test_end) in enumerate(windows):
        if not 0 < train_end < test_end <= total:
            raise ValueError(f"Invalid walk-forward window: {train_end}, {test_end}, total={total}")
        test_slice = slice(train_end, test_end)
        raw = loader._slice_raw(loader.raw_data_cache, test_slice)
        factors = vm.execute(formula, loader.feat_tensor[:, :, test_slice])
        if factors is None:
            raise ValueError("Formula is invalid for a walk-forward window")
        target = loader.target_ret[:, test_slice]
        report = backtest.evaluate_report(factors, raw, target)
        reports.append({
            "window": index,
            "train_end": train_end,
            "test_start": train_end,
            "test_end": test_end,
            **report.as_dict(),
        })
    return reports


def run(formula_path="best_meme_strategy.json", output_path="walk_forward_report.json", window_count=4):
    if window_count < 1:
        raise ValueError("window_count must be positive")
    with open(formula_path) as handle:
        value = json.load(handle)
    formula = value.get("formula") if isinstance(value, dict) else value
    formula = [int(token) for token in formula]
    loader = CryptoDataLoader()
    loader.load_data()
    total = loader.feat_tensor.shape[-1]
    test_start = int(total * 0.5)
    remaining = total - test_start
    step = max(1, remaining // window_count)
    windows = [(test_start + i * step, min(total, test_start + (i + 1) * step)) for i in range(window_count)]
    windows = [(start, end) for start, end in windows if end > start]
    report = {"formula": formula, "windows": evaluate(formula, loader, windows), "data_quality": loader.quality_report.as_dict()}
    with open(output_path, "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Evaluate a formula across walk-forward windows.")
    parser.add_argument("--formula", default="best_meme_strategy.json")
    parser.add_argument("--output", default="walk_forward_report.json")
    parser.add_argument("--windows", type=int, default=4)
    args = parser.parse_args()
    run(args.formula, args.output, args.windows)


if __name__ == "__main__":
    main()
