"""Unified non-trading research workflow."""

import argparse
import asyncio
import json

from data_pipeline.data_manager import DataManager
from model_core.data_loader import CryptoDataLoader


async def refresh_data():
    manager = DataManager()
    try:
        await manager.initialize()
        await manager.pipeline_sync_daily()
    finally:
        await manager.close()


def main():
    parser = argparse.ArgumentParser(description="Run AlphaGPT research checks without live trading.")
    parser.add_argument("--refresh", action="store_true", help="Fetch and store a fresh Birdeye snapshot")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate best_meme_strategy.json")
    parser.add_argument("--walk-forward", action="store_true", help="Run walk-forward evaluation")
    parser.add_argument("--tokens", type=int, default=500, help="Maximum tokens to load for validation")
    args = parser.parse_args()

    if args.refresh:
        asyncio.run(refresh_data())

    loader = CryptoDataLoader()
    loader.load_data(limit_tokens=args.tokens)
    print("Data quality:", json.dumps(loader.quality_report.as_dict(), indent=2))
    print("Temporal splits:", {
        "train": loader.train_feat_tensor.shape[-1],
        "validation": loader.validation_feat_tensor.shape[-1],
        "test": loader.test_feat_tensor.shape[-1],
    })

    if args.evaluate:
        from model_core.evaluate import run as evaluate_run
        evaluate_run()
    if args.walk_forward:
        from model_core.walk_forward import run as walk_forward_run
        walk_forward_run()

    if not args.evaluate and not args.walk_forward:
        print("Research checks complete. No training or trading was started.")


if __name__ == "__main__":
    main()
