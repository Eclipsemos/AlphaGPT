"""GPU formula miner for immutable Binance Spot research snapshots."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.distributions import Categorical
from tqdm import tqdm

from .alphagpt import AlphaGPT, NewtonSchulzLowRankDecay
from .binance_data_loader import BinanceDataLoader
from .binance_mining import cross_sectional_ic_scores
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
    scoring_chunk_size: int = 64
    candidate_pool_size: int = 256
    profiling_interval: int = 25
    learning_rate: float = 1e-3
    minimum_cross_section: int = 10

    def __post_init__(self) -> None:
        if min(
            self.steps,
            self.batch_size,
            self.max_formula_length,
            self.validation_candidates_per_step,
            self.checkpoint_interval,
            self.scoring_chunk_size,
            self.candidate_pool_size,
            self.profiling_interval,
        ) <= 0:
            raise ValueError("Binance mining limits must be positive")
        if self.max_formula_length > ModelConfig.MAX_FORMULA_LEN:
            raise ValueError(
                f"max_formula_length cannot exceed model limit {ModelConfig.MAX_FORMULA_LEN}"
            )
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.minimum_cross_section < 2:
            raise ValueError("minimum_cross_section must be at least 2")


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
        loader: Any | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.seed = int(seed)
        self.config = config or BinanceMiningConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.output_dir / "binance_training_checkpoint.pt"
        self._set_seed(self.seed)
        self.loader = loader or BinanceDataLoader(snapshot_id, symbols=symbols)
        if getattr(self.loader, "feat_tensor", None) is None:
            self.loader.load_data()
        if self.loader.snapshot_id != snapshot_id:
            raise ValueError("Injected Binance loader does not match the requested snapshot")
        if symbols is not None and list(self.loader.symbols) != [item.upper() for item in symbols]:
            raise ValueError("Injected Binance loader does not match the requested symbols")
        if len(self.loader.symbols) < self.config.minimum_cross_section:
            raise ValueError(
                "Binance cross-sectional mining requires at least "
                f"{self.config.minimum_cross_section} symbols"
            )
        self.model = AlphaGPT(vocab=BINANCE_FORMULA_VOCAB).to(ModelConfig.DEVICE)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.learning_rate
        )
        self.use_lord_regularization = use_lord_regularization
        self.progress_callback = progress_callback
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
            "minimum_cross_section",
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

    @torch.no_grad()
    def _score_formulas_batch(self, formulas: torch.Tensor, split: str) -> torch.Tensor:
        features = getattr(self.loader, f"{split}_feat_tensor")
        target = getattr(self.loader, f"{split}_target_ret")
        valid_labels = getattr(self.loader, f"{split}_target_valid")
        if formulas.shape[0] == 0:
            return features.new_empty((0,))
        score_chunks: list[torch.Tensor] = []
        chunk_size = self.config.scoring_chunk_size
        for start in range(0, formulas.shape[0], chunk_size):
            formula_chunk = formulas[start : start + chunk_size]
            factors, valid_formulas = self.vm.execute_batch(
                formula_chunk,
                features,
                chunk_size=chunk_size,
            )
            scores = cross_sectional_ic_scores(
                factors,
                target,
                valid_labels,
                minimum_cross_section=self.config.minimum_cross_section,
            )
            score_chunks.append(
                torch.where(valid_formulas, scores, torch.full_like(scores, -10.0))
            )
        return torch.cat(score_chunks)

    def _score_formula(self, formula: list[int], split: str) -> float:
        values = self._score_formulas_batch(
            torch.tensor([formula], dtype=torch.long, device=ModelConfig.DEVICE), split
        )
        return float(values[0].item())

    def _record_candidates(
        self,
        step: int,
        formulas: torch.Tensor,
        train_scores: torch.Tensor,
    ) -> None:
        if train_scores.numel() == 0:
            return
        pool_size = min(self.config.candidate_pool_size, train_scores.shape[0])
        pool_indices = torch.topk(train_scores, pool_size, sorted=True).indices
        pool_scores = train_scores[pool_indices].detach().cpu().tolist()
        pool_formulas = formulas[pool_indices].detach().cpu().tolist()
        ranked = sorted(
            zip(pool_scores, pool_formulas),
            key=lambda item: (-item[0], item[1]),
        )
        pending: list[tuple[str, float, list[int]]] = []
        for train_score, formula in ranked:
            try:
                canonical = canonical_formula(formula, BINANCE_FORMULA_VOCAB)
            except ValueError:
                continue
            if canonical in self.candidates:
                continue
            pending.append((canonical, train_score, formula))
            if len(pending) >= self.config.validation_candidates_per_step:
                break
        if not pending:
            return
        scores = self._score_formulas_batch(
            torch.tensor([item[2] for item in pending], dtype=torch.long, device=ModelConfig.DEVICE),
            "validation",
        ).detach().cpu().tolist()
        for (canonical, train_score, formula), validation_score in zip(pending, scores):
            self.candidates[canonical] = {
                "formula": formula,
                "canonical_formula": canonical,
                "train_score": train_score,
                "validation_score": validation_score,
                "first_seen_step": step,
                "seed": self.seed,
            }

    def train(self, *, resume: bool = False) -> dict:
        if resume:
            self.load_checkpoint()
        progress = tqdm(range(self.start_step, self.config.steps), desc="Binance factor mining")
        for step in progress:
            profile_step = (
                step == self.start_step
                or (step + 1) % self.config.profiling_interval == 0
            )
            timings_ms: dict[str, float] = {}
            if profile_step and ModelConfig.DEVICE.type == "cuda":
                torch.cuda.synchronize(ModelConfig.DEVICE)
            stage_started = time.perf_counter()

            def finish_stage(name: str) -> None:
                nonlocal stage_started
                if not profile_step:
                    return
                if ModelConfig.DEVICE.type == "cuda":
                    torch.cuda.synchronize(ModelConfig.DEVICE)
                now = time.perf_counter()
                timings_ms[name] = (now - stage_started) * 1000.0
                stage_started = now

            sequences, log_probabilities = self._sample()
            finish_stage("sampling")
            rewards = torch.full((self.config.batch_size,), -5.0, device=ModelConfig.DEVICE)
            valid_mask = self.vm.valid_formula_mask(sequences)
            valid_sequences = sequences[valid_mask]
            finish_stage("validity")
            train_scores = self._score_formulas_batch(valid_sequences, "train")
            rewards[valid_mask] = train_scores
            finish_stage("vm_and_ic")
            self._record_candidates(step, valid_sequences, train_scores)
            finish_stage("candidate_validation")
            advantage = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            loss = sum(-value * advantage for value in log_probabilities).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Binance miner loss became non-finite")
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            finish_stage("backward_and_optimizer")
            if self.lord_optimizer is not None:
                self.lord_optimizer.step()
            finish_stage("lord")
            valid_formula_count = int(valid_mask.sum().item())
            best_validation = max(
                (item["validation_score"] for item in self.candidates.values()),
                default=-10.0,
            )
            step_history = {
                "step": step,
                "average_reward": float(rewards.mean().item()),
                "valid_formula_count": valid_formula_count,
                "unique_candidate_count": len(self.candidates),
                "best_validation_score": best_validation,
            }
            if profile_step:
                step_history["timings_ms"] = timings_ms
            self.history["steps"].append(step_history)
            if self.progress_callback is not None:
                self.progress_callback(
                    {
                        "phase": "mining",
                        "step": step + 1,
                        "steps": self.config.steps,
                        "average_reward": float(rewards.mean().item()),
                        "valid_formula_count": valid_formula_count,
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
    parser.add_argument("--minimum-cross-section", type=int, default=10)
    parser.add_argument("--scoring-chunk-size", type=int, default=64)
    parser.add_argument("--profiling-interval", type=int, default=25)
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
        config=BinanceMiningConfig(
            steps=args.steps,
            batch_size=args.batch_size,
            minimum_cross_section=args.minimum_cross_section,
            scoring_chunk_size=args.scoring_chunk_size,
            profiling_interval=args.profiling_interval,
        ),
        use_lord_regularization=not args.no_lord,
    )
    result = engine.train(resume=args.resume)
    print(json.dumps(result["best"], indent=2))


if __name__ == "__main__":
    main()
