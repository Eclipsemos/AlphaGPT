from dataclasses import asdict, dataclass
import math

import torch


@dataclass(frozen=True)
class BacktestReport:
    """Aggregate equal-weight portfolio metrics across the loaded universe.

    ``target_ret`` contains log-return labels for model training. Reports use
    ``expm1(target_ret)`` and compound the resulting net simple returns.
    """

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

    def _simulate(
        self,
        position: torch.Tensor,
        raw_data: dict[str, torch.Tensor],
        target_ret: torch.Tensor,
        valid_mask: torch.Tensor,
    ):
        position = position * valid_mask.float()
        liquidity = raw_data["liquidity"]
        impact_slippage = self.trade_size / (liquidity + 1e-9)
        impact_slippage = torch.clamp(impact_slippage, 0.0, 0.05)
        total_slippage_one_way = self.base_fee + impact_slippage
        previous = torch.roll(position, 1, dims=1)
        previous[:, 0] = 0
        turnover = torch.abs(position - previous)
        fees = turnover * total_slippage_one_way
        # Labels are log returns for stable model training; portfolio accounting
        # must use simple returns before compounding and subtracting fees.
        gross_pnl = position * torch.expm1(target_ret)
        gross_pnl = torch.where(valid_mask, gross_pnl, torch.zeros_like(gross_pnl))
        fees = torch.where(valid_mask, fees, torch.zeros_like(fees))
        turnover = torch.where(valid_mask, turnover, torch.zeros_like(turnover))
        net_pnl = gross_pnl - fees
        return net_pnl, fees, turnover

    def _report(
        self,
        position: torch.Tensor,
        raw_data: dict[str, torch.Tensor],
        target_ret: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> BacktestReport:
        net_pnl, fees, turnover = self._simulate(position, raw_data, target_ret, valid_mask)

        # Treat the token universe as an equal-weight portfolio. Missing or
        # purged labels are excluded from that timestamp's denominator.
        valid_count = valid_mask.float().sum(dim=0)
        valid_time = valid_count > 0
        portfolio_returns = torch.where(
            valid_time,
            net_pnl.sum(dim=0) / valid_count.clamp_min(1.0),
            torch.zeros_like(valid_count),
        )
        returns = portfolio_returns[valid_time]
        if returns.numel() == 0:
            raise ValueError("No valid return labels available for backtest")
        cumulative = torch.prod(1.0 + returns).sub(1.0)
        volatility = returns.std(unbiased=False)
        sharpe = torch.where(
            volatility > 1e-9,
            returns.mean() / volatility * math.sqrt(max(1, returns.numel())),
            torch.zeros_like(volatility),
        )
        equity = torch.cat([torch.ones((1,), device=returns.device), torch.cumprod(1.0 + returns, dim=0)])
        drawdown = equity / torch.cummax(equity, dim=0).values - 1.0
        max_drawdown = drawdown.min()
        effective_position = position * valid_mask.float()
        active_bars = effective_position.sum()
        trades = turnover.sum()
        wins = (net_pnl > 0).float().sum()
        active = effective_position.sum()
        win_rate = wins / active.clamp_min(1.0)
        big_drawdowns = (returns < -0.05).float().sum()
        score = cumulative - big_drawdowns * 2.0
        if active_bars < 5:
            score = torch.tensor(-10.0, device=returns.device)

        return BacktestReport(
            score=float(score.item()),
            cumulative_return=float(cumulative.item()),
            volatility=float(volatility.item()),
            sharpe=float(sharpe.item()),
            max_drawdown=float(max_drawdown.item()),
            turnover=float(trades.item() / max(1, position.shape[0])),
            trade_count=float(trades.item() / max(1, position.shape[0])),
            win_rate=float(win_rate.item()),
            fees=float(fees.sum().item() / max(1, position.shape[0])),
            active_fraction=float(active_bars.item() / valid_mask.float().sum().clamp_min(1.0).item()),
        )

    def positions_from_factors(self, factors: torch.Tensor, raw_data: dict[str, torch.Tensor]) -> torch.Tensor:
        signal = torch.sigmoid(factors)
        is_safe = (raw_data["liquidity"] > self.min_liq).float()
        return (signal > 0.85).float() * is_safe

    def evaluate_report(
        self,
        factors: torch.Tensor,
        raw_data: dict[str, torch.Tensor],
        target_ret: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> BacktestReport:
        position = self.positions_from_factors(factors, raw_data)
        valid_mask = torch.ones_like(target_ret, dtype=torch.bool) if valid_mask is None else valid_mask
        return self._report(position, raw_data, target_ret, valid_mask)

    def evaluate(self, factors: torch.Tensor, raw_data: dict[str, torch.Tensor], target_ret: torch.Tensor, valid_mask: torch.Tensor | None = None):
        report = self.evaluate_report(factors, raw_data, target_ret, valid_mask)
        return torch.tensor(report.score, device=target_ret.device), report.cumulative_return

    def baseline_reports(
        self,
        raw_data: dict[str, torch.Tensor],
        target_ret: torch.Tensor,
        seed: int = 0,
        valid_mask: torch.Tensor | None = None,
    ):
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
        valid_mask = torch.ones_like(target_ret, dtype=torch.bool) if valid_mask is None else valid_mask
        return {name: self._report(position, raw_data, target_ret, valid_mask) for name, position in positions.items()}
