"""Historical paper-trading simulator with no wallet or transaction signing."""

import argparse
import json
import os

import torch

from model_core.data_loader import CryptoDataLoader
from model_core.evaluate import load_formula
from model_core.vm import StackVM


class PaperPortfolio:
    def __init__(self, cash=10_000.0, state_path="paper_portfolio.json"):
        self.cash = float(cash)
        self.state_path = state_path
        self.positions = {}
        self.trades = []

    def buy(self, address, price, notional, timestamp):
        if notional <= 0 or notional > self.cash:
            return False
        amount = notional / max(price, 1e-12)
        self.cash -= notional
        position = self.positions.setdefault(address, {"amount": 0.0, "cost": 0.0})
        position["amount"] += amount
        position["cost"] += notional
        self.trades.append({"time": str(timestamp), "side": "BUY", "address": address, "price": price, "notional": notional})
        return True

    def sell(self, address, price, timestamp):
        position = self.positions.pop(address, None)
        if not position:
            return False
        notional = position["amount"] * price
        self.cash += notional
        self.trades.append({"time": str(timestamp), "side": "SELL", "address": address, "price": price, "notional": notional})
        return True

    def save(self):
        with open(self.state_path, "w") as handle:
            json.dump({"cash": self.cash, "positions": self.positions, "trades": self.trades}, handle, indent=2)


def run(formula_path="best_meme_strategy.json", state_path="paper_portfolio.json", initial_cash=10_000.0):
    loader = CryptoDataLoader()
    loader.load_data()
    formula = load_formula(formula_path)
    factors = StackVM().execute(formula, loader.test_feat_tensor)
    if factors is None:
        raise ValueError("Formula is invalid for the test split")
    signals = torch.sigmoid(factors) > 0.85
    portfolio = PaperPortfolio(initial_cash, state_path)
    addresses = loader.addresses
    prices = loader.test_raw_data_cache["open"].detach().cpu()
    for step in range(signals.shape[1]):
        if os.path.exists("STOP_SIGNAL"):
            print("STOP_SIGNAL found; ending paper simulation early.")
            break
        timestamp = loader.times[loader.splits.test.start + step]
        for index, address in enumerate(addresses):
            price = float(prices[index, step])
            if signals[index, step] and address not in portfolio.positions:
                portfolio.buy(address, price, min(100.0, portfolio.cash), timestamp)
            elif not signals[index, step] and address in portfolio.positions:
                portfolio.sell(address, price, timestamp)
    portfolio.save()
    marked_value = portfolio.cash
    final_prices = loader.test_raw_data_cache["close"][:, -1].detach().cpu()
    for index, address in enumerate(addresses):
        position = portfolio.positions.get(address)
        if position:
            marked_value += position["amount"] * float(final_prices[index])
    result = {"initial_cash": initial_cash, "cash": portfolio.cash, "marked_value": marked_value, "trade_count": len(portfolio.trades), "state_path": state_path}
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description="Run a historical paper simulation; no live trading is possible.")
    parser.add_argument("--formula", default="best_meme_strategy.json")
    parser.add_argument("--state", default="paper_portfolio.json")
    parser.add_argument("--cash", type=float, default=10_000.0)
    args = parser.parse_args()
    run(args.formula, args.state, args.cash)


if __name__ == "__main__":
    main()
