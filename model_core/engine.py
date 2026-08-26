import torch
from torch.distributions import Categorical
from tqdm import tqdm
import json
import random
import numpy as np
from pathlib import Path

from .config import ModelConfig
from .data_loader import CryptoDataLoader
from .alphagpt import AlphaGPT, NewtonSchulzLowRankDecay, StableRankMonitor
from .vm import StackVM
from .backtest import MemeBacktest

class AlphaEngine:
    def __init__(self, use_lord_regularization=True, lord_decay_rate=1e-3, lord_num_iterations=5, seed=0, output_dir="."):
        """
        Initialize AlphaGPT training engine.
        
        Args:
            use_lord_regularization: Enable Low-Rank Decay (LoRD) regularization
            lord_decay_rate: Strength of LoRD regularization
            lord_num_iterations: Number of Newton-Schulz iterations per step
        """
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.output_dir / ModelConfig.CHECKPOINT_PATH
        self._set_seed(seed)
        self.loader = CryptoDataLoader()
        self.loader.load_data()
        
        self.model = AlphaGPT().to(ModelConfig.DEVICE)
        
        # Standard optimizer
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        
        # Low-Rank Decay regularizer
        self.use_lord = use_lord_regularization
        if self.use_lord:
            self.lord_opt = NewtonSchulzLowRankDecay(
                self.model.named_parameters(),
                decay_rate=lord_decay_rate,
                num_iterations=lord_num_iterations,
                target_keywords=["in_proj_weight", "out_proj.weight", "attention"]
            )
            self.rank_monitor = StableRankMonitor(
                self.model,
                target_keywords=["in_proj_weight", "out_proj.weight", "attention"]
            )
        else:
            self.lord_opt = None
            self.rank_monitor = None
        
        self.vm = StackVM()
        self.bt = MemeBacktest()
        
        self.best_score = -float('inf')
        self.best_formula = None
        self.training_history = {
            'step': [],
            'avg_reward': [],
            'best_score': [],
            'stable_rank': [],
            'validation_score': [],
            'validation_return': [],
            'test_score': [],
            'test_return': [],
            'config': {
                'seed': seed,
                'train_steps': ModelConfig.TRAIN_STEPS,
                'batch_size': ModelConfig.BATCH_SIZE,
                'train_ratio': ModelConfig.TRAIN_RATIO,
                'validation_ratio': ModelConfig.VALIDATION_RATIO,
                'test_ratio': ModelConfig.TEST_RATIO,
                'device': str(ModelConfig.DEVICE),
            },
        }
        self.start_step = 0

    @staticmethod
    def _set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _save_checkpoint(self, next_step):
        torch.save({
            'next_step': next_step,
            'model': self.model.state_dict(),
            'optimizer': self.opt.state_dict(),
            'best_score': self.best_score,
            'best_formula': self.best_formula,
            'training_history': self.training_history,
            'seed': self.seed,
        }, self.checkpoint_path)

    def _load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path, map_location=ModelConfig.DEVICE, weights_only=False)
        self.model.load_state_dict(checkpoint['model'])
        self.opt.load_state_dict(checkpoint['optimizer'])
        self.best_score = checkpoint['best_score']
        self.best_formula = checkpoint['best_formula']
        self.training_history = checkpoint['training_history']
        self.start_step = int(checkpoint['next_step'])
        self._set_seed(int(checkpoint.get('seed', self.seed)))
        print(f"Resuming from checkpoint at step {self.start_step}.")

    def evaluate_formula(self, formula, feat_tensor, raw_data, target_ret, valid_mask):
        if formula is None:
            return float('-inf'), 0.0
        result = self.vm.execute(formula, feat_tensor)
        if result is None:
            return float('-inf'), 0.0
        report = self.bt.evaluate_report(result, raw_data, target_ret, valid_mask)
        return report.score, report.cumulative_return

    def train(self, resume=False):
        if resume:
            self._load_checkpoint()
        print("🚀 Starting Meme Alpha Mining with LoRD Regularization..." if self.use_lord else "🚀 Starting Meme Alpha Mining...")
        if self.use_lord:
            print(f"   LoRD Regularization enabled")
            print(f"   Target keywords: ['in_proj_weight', 'out_proj.weight', 'attention']")
        
        pbar = tqdm(range(self.start_step, ModelConfig.TRAIN_STEPS))
        
        for step in pbar:
            bs = ModelConfig.BATCH_SIZE
            inp = torch.zeros((bs, 1), dtype=torch.long, device=ModelConfig.DEVICE)
            
            log_probs = []
            tokens_list = []
            
            for _ in range(ModelConfig.MAX_FORMULA_LEN):
                logits, _, _ = self.model(inp)
                if not torch.isfinite(logits).all():
                    raise RuntimeError("Model produced non-finite logits")
                dist = Categorical(logits=logits)
                action = dist.sample()
                
                log_probs.append(dist.log_prob(action))
                tokens_list.append(action)
                inp = torch.cat([inp, action.unsqueeze(1)], dim=1)
            
            seqs = torch.stack(tokens_list, dim=1)
            
            rewards = torch.zeros(bs, device=ModelConfig.DEVICE)
            
            for i in range(bs):
                formula = seqs[i].tolist()

                if not self.vm.is_valid_formula(formula):
                    rewards[i] = -5.0
                    continue

                res = self.vm.execute(formula, self.loader.train_feat_tensor)
                
                if res is None:
                    rewards[i] = -5.0
                    continue
                
                if res.std() < 1e-4:
                    rewards[i] = -2.0
                    continue
                
                score, ret_val = self.bt.evaluate(
                    res,
                    self.loader.train_raw_data_cache,
                    self.loader.train_target_ret,
                    self.loader.train_target_valid,
                )
                rewards[i] = score if torch.isfinite(score) else -10.0
                
                if score.item() > self.best_score:
                    self.best_score = score.item()
                    self.best_formula = formula
                    tqdm.write(f"[!] New King: Score {score:.2f} | Ret {ret_val:.2%} | Formula {formula}")
            
            # Normalize rewards
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            
            loss = 0
            for t in range(len(log_probs)):
                loss += -log_probs[t] * adv
            
            loss = loss.mean()

            if not torch.isfinite(loss):
                tqdm.write("[!] Non-finite loss; skipping optimizer step")
                continue
            
            # Gradient step
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.opt.step()
            
            # Apply Low-Rank Decay regularization
            if self.use_lord:
                self.lord_opt.step()
            
            # Logging
            avg_reward = rewards.mean().item()
            postfix_dict = {'AvgRew': f"{avg_reward:.3f}", 'BestScore': f"{self.best_score:.3f}"}
            
            if self.use_lord and step % 100 == 0:
                stable_rank = self.rank_monitor.compute()
                postfix_dict['Rank'] = f"{stable_rank:.2f}"
                self.training_history['stable_rank'].append(stable_rank)
            
            self.training_history['step'].append(step)
            self.training_history['avg_reward'].append(avg_reward)
            self.training_history['best_score'].append(self.best_score)

            validation_score, validation_return = self.evaluate_formula(
                self.best_formula,
                self.loader.validation_feat_tensor,
                self.loader.validation_raw_data_cache,
                self.loader.validation_target_ret,
                self.loader.validation_target_valid,
            )
            test_score, test_return = self.evaluate_formula(
                self.best_formula,
                self.loader.test_feat_tensor,
                self.loader.test_raw_data_cache,
                self.loader.test_target_ret,
                self.loader.test_target_valid,
            )
            self.training_history['validation_score'].append(validation_score)
            self.training_history['validation_return'].append(validation_return)
            self.training_history['test_score'].append(test_score)
            self.training_history['test_return'].append(test_return)
            
            pbar.set_postfix(postfix_dict)
            if (step + 1) % ModelConfig.CHECKPOINT_INTERVAL == 0:
                self._save_checkpoint(step + 1)

        # Save best formula
        if self.best_formula is None:
            raise RuntimeError("No valid formula was found; increase --steps or --batch-size")
        self._save_checkpoint(ModelConfig.TRAIN_STEPS)
        with open(self.output_dir / "best_meme_strategy.json", "w") as f:
            json.dump(self.best_formula, f)
        
        # Save training history
        import json as js
        with open(self.output_dir / "training_history.json", "w") as f:
            js.dump(self.training_history, f)

        validation_result = self.vm.execute(self.best_formula, self.loader.validation_feat_tensor)
        test_result = self.vm.execute(self.best_formula, self.loader.test_feat_tensor)
        evaluation_report = {
            "validation": self.bt.evaluate_report(
                validation_result,
                self.loader.validation_raw_data_cache,
                self.loader.validation_target_ret,
                self.loader.validation_target_valid,
            ).as_dict(),
            "test": self.bt.evaluate_report(
                test_result,
                self.loader.test_raw_data_cache,
                self.loader.test_target_ret,
                self.loader.test_target_valid,
            ).as_dict(),
            "test_baselines": {
                name: report.as_dict()
                for name, report in self.bt.baseline_reports(
                    self.loader.test_raw_data_cache,
                    self.loader.test_target_ret,
                    valid_mask=self.loader.test_target_valid,
                ).items()
            },
        }
        with open(self.output_dir / "evaluation_report.json", "w") as f:
            js.dump(evaluation_report, f, indent=2)
        
        print(f"\n✓ Training completed!")
        print(f"  Best score: {self.best_score:.4f}")
        print(f"  Best formula: {self.best_formula}")
        validation_score, validation_return = self.evaluate_formula(
            self.best_formula,
            self.loader.validation_feat_tensor,
            self.loader.validation_raw_data_cache,
            self.loader.validation_target_ret,
            self.loader.validation_target_valid,
        )
        test_score, test_return = self.evaluate_formula(
            self.best_formula,
            self.loader.test_feat_tensor,
            self.loader.test_raw_data_cache,
            self.loader.test_target_ret,
            self.loader.test_target_valid,
        )
        print(f"  Validation score/return: {validation_score:.4f} / {validation_return:.2%}")
        print(f"  Test score/return: {test_score:.4f} / {test_return:.2%}")
        for name, report in evaluation_report["test_baselines"].items():
            print(f"  Test baseline {name}: score={report['score']:.4f}, return={report['cumulative_return']:.2%}, drawdown={report['max_drawdown']:.2%}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mine formula-based market signals.")
    parser.add_argument("--resume", action="store_true", help="Resume from training_checkpoint.pt")
    parser.add_argument("--steps", type=int, help="Override the configured total training steps")
    parser.add_argument("--batch-size", type=int, help="Override the formula batch size")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible runs")
    parser.add_argument("--output-dir", default=".", help="Directory for formula, reports, and checkpoint")
    args = parser.parse_args()
    if args.steps is not None:
        if args.steps <= 0:
            parser.error("--steps must be positive")
        ModelConfig.TRAIN_STEPS = args.steps
    if args.batch_size is not None:
        if args.batch_size <= 0:
            parser.error("--batch-size must be positive")
        ModelConfig.BATCH_SIZE = args.batch_size
    eng = AlphaEngine(use_lord_regularization=True, seed=args.seed, output_dir=args.output_dir)
    eng.train(resume=args.resume)
