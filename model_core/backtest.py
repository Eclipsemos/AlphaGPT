from dataclasses import asdict, dataclass
import math

import torch


@dataclass(frozen=True)
class BacktestReport:
    """Aggregate metrics for a signal across the loaded token universe."""

    score: float
    cumulative_return: float
    volatility: float
    sharpe: float
    max_drawdown: float
    turnover: float
    trade_count: float
    win_rate: float
    fees: float
    active_fraction: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


class MemeBacktest:
    def __init__(self):
        self.trade_size = 1000.0
        self.min_liq = 500000.0
        self.base_fee = 0.0060

    def _simulate(self, position: torch.Tensor, raw_data: dict[str, torch.Tensor], target_ret: torch.Tensor):
        liquidity = raw_data["liquidity"]
        impact_slippage = self.trade_size / (liquidity + 1e-9)
        impact_slippage = torch.clamp(impact_slippage, 0.0, 0.05)
        total_slippage_one_way = self.base_fee + impact_slippage
        previous = torch.roll(position, 1, dims=1)
        previous[:, 0] = 0
        turnover = torch.abs(position - previous)
        fees = turnover * total_slippage_one_way
        gross_pnl = position * target_ret
        net_pnl = gross_pnl - fees
        return net_pnl, fees, turnover

    def _report(self, position: torch.Tensor, raw_data: dict[str, torch.Tensor], target_ret: torch.Tensor) -> BacktestReport:
        net_pnl, fees, turnover = self._simulate(position, raw_data, target_ret)
        cumulative = net_pnl.sum(dim=1)
        mean_bar_return = net_pnl.mean(dim=1)
        volatility = net_pnl.std(dim=1, unbiased=False)
        sharpe_values = torch.where(
            volatility > 1e-9,
            mean_bar_return / volatility * math.sqrt(max(1, net_pnl.shape[1])),
            torch.zeros_like(volatility),
        )
        equity = torch.cat([torch.zeros((net_pnl.shape[0], 1), device=net_pnl.device), net_pnl.cumsum(dim=1)], dim=1)
        drawdown = equity - torch.cummax(equity, dim=1).values
        max_drawdown = drawdown.min(dim=1).values
        active_bars = position.sum(dim=1)
        trades = turnover.sum(dim=1)
        wins = (net_pnl > 0).float().sum(dim=1)
        active = (position > 0).float().sum(dim=1)
        win_rate = torch.where(active > 0, wins / active, torch.zeros_like(active))
        big_drawdowns = (net_pnl < -0.05).float().sum(dim=1)
        score_values = cumulative - big_drawdowns * 2.0
        score_values = torch.where(active_bars < 5, torch.full_like(score_values, -10.0), score_values)

        def median(values):
            return float(torch.median(values).item())

        return BacktestReport(
            score=median(score_values),
            cumulative_return=float(cumulative.mean().item()),
            volatility=float(volatility.mean().item()),
            sharpe=median(sharpe_values),
            max_drawdown=median(max_drawdown),
            turnover=float(turnover.sum(dim=1).mean().item()),
            trade_count=float(trades.mean().item()),
            win_rate=float(win_rate.mean().item()),
            fees=float(fees.sum(dim=1).mean().item()),
            active_fraction=float((active_bars / max(1, position.shape[1])).mean().item()),
        )

    def positions_from_factors(self, factors: torch.Tensor, raw_data: dict[str, torch.Tensor]) -> torch.Tensor:
        signal = torch.sigmoid(factors)
        is_safe = (raw_data["liquidity"] > self.min_liq).float()
        return (signal > 0.85).float() * is_safe

    def evaluate_report(self, factors: torch.Tensor, raw_data: dict[str, torch.Tensor], target_ret: torch.Tensor) -> BacktestReport:
        position = self.positions_from_factors(factors, raw_data)
        return self._report(position, raw_data, target_ret)

    def evaluate(self, factors: torch.Tensor, raw_data: dict[str, torch.Tensor], target_ret: torch.Tensor):
        report = self.evaluate_report(factors, raw_data, target_ret)
        return torch.tensor(report.score, device=target_ret.device), report.cumulative_return

    def baseline_reports(self, raw_data: dict[str, torch.Tensor], target_ret: torch.Tensor, seed: int = 0):
        liquidity = raw_data["liquidity"]
        safe = (liquidity > self.min_liq).float()
        close = raw_data["close"]
        previous_close = torch.roll(close, 1, dims=1)
        previous_close[:, 0] = close[:, 0]
        generator = torch.Generator(device="cpu").manual_seed(seed)
        random_signal = torch.rand(liquidity.shape, generator=generator).to(liquidity.device)
        positions = {
            "buy_and_hold": torch.ones_like(liquidity),
            "liquidity_filter": safe,
            "momentum": (close > previous_close).float() * safe,
            "random": (random_signal > 0.85).float() * safe,
        }
        return {name: self._report(position, raw_data, target_ret) for name, position in positions.items()}
