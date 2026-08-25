import torch
from torch.distributions import Categorical
from tqdm import tqdm
import json

from .config import ModelConfig
from .data_loader import CryptoDataLoader
from .alphagpt import AlphaGPT, NewtonSchulzLowRankDecay, StableRankMonitor
from .vm import StackVM
from .backtest import MemeBacktest

class AlphaEngine:
    def __init__(self, use_lord_regularization=True, lord_decay_rate=1e-3, lord_num_iterations=5):
        """
        Initialize AlphaGPT training engine.
        
        Args:
            use_lord_regularization: Enable Low-Rank Decay (LoRD) regularization
            lord_decay_rate: Strength of LoRD regularization
            lord_num_iterations: Number of Newton-Schulz iterations per step
        """
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
                target_keywords=["q_proj", "k_proj", "attention", "qk_norm"]
            )
            self.rank_monitor = StableRankMonitor(
                self.model,
                target_keywords=["q_proj", "k_proj"]
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
        }

    def evaluate_formula(self, formula, feat_tensor, raw_data, target_ret):
        if formula is None:
            return float('-inf'), 0.0
        result = self.vm.execute(formula, feat_tensor)
        if result is None:
            return float('-inf'), 0.0
        report = self.bt.evaluate_report(result, raw_data, target_ret)
        return report.score, report.cumulative_return

    def train(self):
        print("🚀 Starting Meme Alpha Mining with LoRD Regularization..." if self.use_lord else "🚀 Starting Meme Alpha Mining...")
        if self.use_lord:
            print(f"   LoRD Regularization enabled")
            print(f"   Target keywords: ['q_proj', 'k_proj', 'attention', 'qk_norm']")
        
        pbar = tqdm(range(ModelConfig.TRAIN_STEPS))
        
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
            )
            test_score, test_return = self.evaluate_formula(
                self.best_formula,
                self.loader.test_feat_tensor,
                self.loader.test_raw_data_cache,
                self.loader.test_target_ret,
            )
            self.training_history['validation_score'].append(validation_score)
            self.training_history['validation_return'].append(validation_return)
            self.training_history['test_score'].append(test_score)
            self.training_history['test_return'].append(test_return)
            
            pbar.set_postfix(postfix_dict)

        # Save best formula
        with open("best_meme_strategy.json", "w") as f:
            json.dump(self.best_formula, f)
        
        # Save training history
        import json as js
        with open("training_history.json", "w") as f:
            js.dump(self.training_history, f)

        validation_result = self.vm.execute(self.best_formula, self.loader.validation_feat_tensor)
        test_result = self.vm.execute(self.best_formula, self.loader.test_feat_tensor)
        evaluation_report = {
            "validation": self.bt.evaluate_report(
                validation_result,
                self.loader.validation_raw_data_cache,
                self.loader.validation_target_ret,
            ).as_dict(),
            "test": self.bt.evaluate_report(
                test_result,
                self.loader.test_raw_data_cache,
                self.loader.test_target_ret,
            ).as_dict(),
            "test_baselines": {
                name: report.as_dict()
                for name, report in self.bt.baseline_reports(
                    self.loader.test_raw_data_cache,
                    self.loader.test_target_ret,
                ).items()
            },
        }
        with open("evaluation_report.json", "w") as f:
            js.dump(evaluation_report, f, indent=2)
        
        print(f"\n✓ Training completed!")
        print(f"  Best score: {self.best_score:.4f}")
        print(f"  Best formula: {self.best_formula}")
        validation_score, validation_return = self.evaluate_formula(
            self.best_formula,
            self.loader.validation_feat_tensor,
            self.loader.validation_raw_data_cache,
            self.loader.validation_target_ret,
        )
        test_score, test_return = self.evaluate_formula(
            self.best_formula,
            self.loader.test_feat_tensor,
            self.loader.test_raw_data_cache,
            self.loader.test_target_ret,
        )
        print(f"  Validation score/return: {validation_score:.4f} / {validation_return:.2%}")
        print(f"  Test score/return: {test_score:.4f} / {test_return:.2%}")
        for name, report in evaluation_report["test_baselines"].items():
            print(f"  Test baseline {name}: score={report['score']:.4f}, return={report['cumulative_return']:.2%}, drawdown={report['max_drawdown']:.2%}")


if __name__ == "__main__":
    eng = AlphaEngine(use_lord_regularization=True)
    eng.train()
