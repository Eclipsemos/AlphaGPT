"""GPU formula miner for immutable Binance Spot research snapshots."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical
from tqdm import tqdm

from .alphagpt import AlphaGPT, NewtonSchulzLowRankDecay
from .binance_data_loader import BinanceDataLoader
from .binance_mining import cross_sectional_ic_score
from .config import ModelConfig
from .formula_artifact import build_formula_artifact
from .formula_canonical import canonical_formula
from .vm import StackVM
from .vocab import BINANCE_FORMULA_VOCAB


@dataclass(frozen=True)
class BinanceMiningConfig:
    steps: int = 1000
    batch_size: int = 8192
    max_formula_length: int = 12
    validation_candidates_per_step: int = 8
    checkpoint_interval: int = 100
    learning_rate: float = 1e-3

    def __post_init__(self) -> None:
        if min(
            self.steps,
            self.batch_size,
            self.max_formula_length,
            self.validation_candidates_per_step,
            self.checkpoint_interval,
        ) <= 0:
            raise ValueError("Binance mining limits must be positive")
        if self.max_formula_length > ModelConfig.MAX_FORMULA_LEN:
            raise ValueError(
                f"max_formula_length cannot exceed model limit {ModelConfig.MAX_FORMULA_LEN}"
            )
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")


class BinanceAlphaEngine:
    def __init__(
        self,
        snapshot_id: str,
        *,
        symbols: list[str] | None = None,
        seed: int = 0,
        output_dir: str | Path = ".",
        config: BinanceMiningConfig | None = None,
        use_lord_regularization: bool = True,
    ):
        self.seed = int(seed)
        self.config = config or BinanceMiningConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.output_dir / "binance_training_checkpoint.pt"
        self._set_seed(self.seed)
        self.loader = BinanceDataLoader(snapshot_id, symbols=symbols)
        self.loader.load_data()
        if len(self.loader.symbols) < 2:
            raise ValueError("Binance cross-sectional mining requires at least two symbols")
        self.model = AlphaGPT(vocab=BINANCE_FORMULA_VOCAB).to(ModelConfig.DEVICE)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.learning_rate
        )
        self.use_lord_regularization = use_lord_regularization
        self.lord_optimizer = (
            NewtonSchulzLowRankDecay(
                self.model.named_parameters(),
                decay_rate=1e-3,
                num_iterations=5,
                target_keywords=["in_proj_weight", "out_proj.weight", "attention"],
            )
            if use_lord_regularization
            else None
        )
        self.vm = StackVM(BINANCE_FORMULA_VOCAB)
        self.start_step = 0
        self.candidates: dict[str, dict] = {}
        self.history: dict[str, object] = {
            "research_metadata": self.loader.research_metadata,
            "mining_config": asdict(self.config),
            "seed": self.seed,
            "steps": [],
        }

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _checkpoint_payload(self, next_step: int) -> dict:
        return {
            "next_step": next_step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "candidates": self.candidates,
            "history": self.history,
            "seed": self.seed,
            "snapshot_id": self.loader.snapshot_id,
            "symbols": self.loader.symbols,
            "formula_vocab_version": BINANCE_FORMULA_VOCAB.version,
            "research_metadata": self.loader.research_metadata,
            "mining_config": asdict(self.config),
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

    def save_checkpoint(self, next_step: int) -> None:
        torch.save(self._checkpoint_payload(next_step), self.checkpoint_path)

    def load_checkpoint(self) -> None:
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=ModelConfig.DEVICE,
            weights_only=False,
        )
        expected = {
            "snapshot_id": self.loader.snapshot_id,
            "symbols": self.loader.symbols,
            "formula_vocab_version": BINANCE_FORMULA_VOCAB.version,
            "research_metadata": self.loader.research_metadata,
        }
        for key, value in expected.items():
            if checkpoint.get(key) != value:
                raise ValueError(f"Checkpoint is incompatible with current {key}")
        saved_config = dict(checkpoint.get("mining_config", {}))
        current_config = asdict(self.config)
        for key in (
            "batch_size",
            "max_formula_length",
            "validation_candidates_per_step",
            "learning_rate",
        ):
            if saved_config.get(key) != current_config[key]:
                raise ValueError(f"Checkpoint is incompatible with mining config {key}")
        if self.config.steps < int(checkpoint["next_step"]):
            raise ValueError("Configured steps precede the checkpoint's completed step")
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.candidates = checkpoint["candidates"]
        self.history = checkpoint["history"]
        self.history["mining_config"] = current_config
        self.start_step = int(checkpoint["next_step"])
        random.setstate(checkpoint["python_random_state"])
        np.random.set_state(checkpoint["numpy_random_state"])
        torch.set_rng_state(checkpoint["torch_random_state"].cpu())
        if torch.cuda.is_available() and checkpoint.get("cuda_random_state") is not None:
            cuda_states = [
                torch.as_tensor(state, dtype=torch.uint8, device="cpu").detach().cpu()
                for state in checkpoint["cuda_random_state"]
            ]
            torch.cuda.set_rng_state_all(cuda_states)

    def _sample(self) -> tuple[torch.Tensor, list[torch.Tensor]]:
        batch_size = self.config.batch_size
        inputs = torch.zeros((batch_size, 1), dtype=torch.long, device=ModelConfig.DEVICE)
        log_probabilities: list[torch.Tensor] = []
        tokens: list[torch.Tensor] = []
        for _ in range(self.config.max_formula_length):
            logits, _, _ = self.model(inputs)
            if not torch.isfinite(logits).all():
                raise RuntimeError("Binance miner produced non-finite logits")
            distribution = Categorical(logits=logits)
            action = distribution.sample()
            log_probabilities.append(distribution.log_prob(action))
            tokens.append(action)
            inputs = torch.cat([inputs, action.unsqueeze(1)], dim=1)
        return torch.stack(tokens, dim=1), log_probabilities

    def _score_formula(self, formula: list[int], split: str) -> float:
        features = getattr(self.loader, f"{split}_feat_tensor")
        target = getattr(self.loader, f"{split}_target_ret")
        valid = getattr(self.loader, f"{split}_target_valid")
        factors = self.vm.execute(formula, features)
        if factors is None or not torch.isfinite(factors).all():
            return -10.0
        score = cross_sectional_ic_score(factors, target, valid)
        return float(score.item()) if torch.isfinite(score) else -10.0

    def _record_candidates(
        self,
        step: int,
        formulas: list[tuple[float, list[int]]],
    ) -> None:
        formulas.sort(key=lambda item: (-item[0], item[1]))
        checked = 0
        for train_score, formula in formulas:
            try:
                canonical = canonical_formula(formula, BINANCE_FORMULA_VOCAB)
            except ValueError:
                continue
            if canonical in self.candidates:
                continue
            validation_score = self._score_formula(formula, "validation")
            self.candidates[canonical] = {
                "formula": formula,
                "canonical_formula": canonical,
                "train_score": train_score,
                "validation_score": validation_score,
                "first_seen_step": step,
                "seed": self.seed,
            }
            checked += 1
            if checked >= self.config.validation_candidates_per_step:
                break

    def train(self, *, resume: bool = False) -> dict:
        if resume:
            self.load_checkpoint()
        progress = tqdm(range(self.start_step, self.config.steps), desc="Binance factor mining")
        for step in progress:
            sequences, log_probabilities = self._sample()
            rewards = torch.full(
                (self.config.batch_size,), -5.0, device=ModelConfig.DEVICE
            )
            valid_formulas: list[tuple[float, list[int]]] = []
            for index in range(self.config.batch_size):
                formula = sequences[index].tolist()
                if not self.vm.is_valid_formula(formula):
                    continue
                score = self._score_formula(formula, "train")
                rewards[index] = score
                valid_formulas.append((score, formula))
            self._record_candidates(step, valid_formulas)
            advantage = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            loss = sum(-value * advantage for value in log_probabilities).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Binance miner loss became non-finite")
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            if self.lord_optimizer is not None:
                self.lord_optimizer.step()
            best_validation = max(
                (item["validation_score"] for item in self.candidates.values()),
                default=-10.0,
            )
            self.history["steps"].append(
                {
                    "step": step,
                    "average_reward": float(rewards.mean().item()),
                    "valid_formula_count": len(valid_formulas),
                    "unique_candidate_count": len(self.candidates),
                    "best_validation_score": best_validation,
                }
            )
            progress.set_postfix(
                avg=f"{rewards.mean().item():.3f}",
                candidates=len(self.candidates),
                validation=f"{best_validation:.3f}",
            )
            if (step + 1) % self.config.checkpoint_interval == 0:
                self.save_checkpoint(step + 1)
        if not self.candidates:
            raise RuntimeError("Binance mining found no valid candidate formula")
        self.save_checkpoint(self.config.steps)
        return self.write_artifacts()

    def write_artifacts(self) -> dict:
        ranked = sorted(
            self.candidates.values(),
            key=lambda item: (
                -item["validation_score"],
                -item["train_score"],
                item["canonical_formula"],
            ),
        )
        best = ranked[0]
        artifact = build_formula_artifact(
            best["formula"],
            BINANCE_FORMULA_VOCAB,
            self.loader.research_metadata,
        )
        artifact["canonical_formula"] = best["canonical_formula"]
        artifact["selection"] = {
            "criterion": "validation_ic_score",
            "train_score": best["train_score"],
            "validation_score": best["validation_score"],
            "test_was_accessed": False,
        }
        artifact["discovery"] = {
            "engine": "binance-alpha-engine-v1",
            "seed": self.seed,
            "mining_config": asdict(self.config),
            "canonical_formula": best["canonical_formula"],
        }
        formula_path = self.output_dir / "best_formula.json"
        candidates_path = self.output_dir / "candidates.json"
        history_path = self.output_dir / "training_history.json"
        metadata_path = self.output_dir / "run_metadata.json"
        formula_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
        candidates_path.write_text(
            json.dumps(
                {
                    "research_metadata": self.loader.research_metadata,
                    "seed": self.seed,
                    "mining_config": asdict(self.config),
                    "selection_criterion": "validation_ic_score",
                    "test_was_accessed": False,
                    "candidates": ranked,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        history_path.write_text(json.dumps(self.history, indent=2, sort_keys=True) + "\n")
        metadata = {
            "market": "binance-spot",
            "seed": self.seed,
            "research_metadata": self.loader.research_metadata,
            "mining_config": asdict(self.config),
            "formula_vocab_version": BINANCE_FORMULA_VOCAB.version,
            "test_was_accessed": False,
            "artifacts": {
                "formula": str(formula_path),
                "candidates": str(candidates_path),
                "history": str(history_path),
                "checkpoint": str(self.checkpoint_path),
            },
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        return {"best": best, "artifact": artifact, "metadata": metadata}


def parse_symbols(value: str) -> list[str]:
    symbols = [item.strip().upper() for item in value.split(",") if item.strip()]
    if len(set(symbols)) != len(symbols):
        raise argparse.ArgumentTypeError("symbols must be unique")
    return symbols


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mine Binance Spot factors (research only)")
    parser.add_argument("--market", choices=("binance-spot",), default="binance-spot")
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--interval", choices=("1h",), default="1h")
    parser.add_argument("--symbols", type=parse_symbols, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-lord", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    engine = BinanceAlphaEngine(
        args.snapshot_id,
        symbols=args.symbols,
        seed=args.seed,
        output_dir=args.output_dir,
        config=BinanceMiningConfig(steps=args.steps, batch_size=args.batch_size),
        use_lord_regularization=not args.no_lord,
    )
    result = engine.train(resume=args.resume)
    print(json.dumps(result["best"], indent=2))


if __name__ == "__main__":
    main()
