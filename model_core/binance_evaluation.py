"""Historical cross-sectional factor evaluation for Binance Spot research."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any, Sequence

import torch


HOURS_PER_YEAR = 365 * 24


@dataclass(frozen=True)
class BinanceEvaluationConfig:
    interval: str = "1h"
    max_positions: int = 10
    weighting: str = "equal"
    rebalance_hours: int = 24
    risk_lookback_hours: int = 24
    taker_fee_bps: float = 10.0
    slippage_bps: float = 5.0
    portfolio_notional_usd: float = 100_000.0

    def __post_init__(self) -> None:
        if self.interval != "1h":
            raise ValueError("Binance evaluation currently supports 1h only")
        if self.max_positions <= 0 or self.rebalance_hours <= 0:
            raise ValueError("max_positions and rebalance_hours must be positive")
        if self.risk_lookback_hours <= 1:
            raise ValueError("risk_lookback_hours must exceed one")
        if self.weighting not in {"equal", "risk"}:
            raise ValueError("weighting must be equal or risk")
        if self.taker_fee_bps < 0 or self.slippage_bps < 0:
            raise ValueError("research costs cannot be negative")
        if self.portfolio_notional_usd <= 0:
            raise ValueError("portfolio_notional_usd must be positive")

    @property
    def fee_rate(self) -> float:
        return self.taker_fee_bps / 10_000.0

    @property
    def slippage_rate(self) -> float:
        return self.slippage_bps / 10_000.0


@dataclass(frozen=True)
class BinanceFactorReport:
    score: float
    cumulative_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe: float
    max_drawdown: float
    total_turnover: float
    annualized_turnover: float
    rebalance_leg_count: int
    fee_cost: float
    slippage_cost: float
    total_cost: float
    gross_return_sum: float
    net_return_sum: float
    win_rate: float
    average_exposure: float
    average_active_positions: float
    maximum_volume_participation: float
    mean_rank_ic: float
    evaluated_hours: int
    skipped_return_hours: int
    config: dict[str, Any]
    per_symbol: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["per_symbol"] = list(self.per_symbol)
        return value


class BinanceFactorEvaluator:
    def __init__(self, config: BinanceEvaluationConfig | None = None):
        self.config = config or BinanceEvaluationConfig()

    def evaluate(
        self,
        factors: torch.Tensor,
        raw_data: dict[str, torch.Tensor],
        target_log_returns: torch.Tensor,
        return_valid: torch.Tensor,
        signal_valid: torch.Tensor,
        symbols: Sequence[str],
    ) -> BinanceFactorReport:
        self._validate_inputs(
            factors, raw_data, target_log_returns, return_valid, signal_valid, symbols
        )
        weights = self.construct_weights(factors, raw_data, signal_valid)
        previous = torch.roll(weights, 1, dims=1)
        previous[:, 0] = 0
        turnover_by_symbol = torch.abs(weights - previous)
        # Close research exposure at the evaluation boundary.
        turnover_by_symbol[:, -1] += torch.abs(weights[:, -1])
        fee_cost_by_symbol = turnover_by_symbol * self.config.fee_rate
        slippage_cost_by_symbol = turnover_by_symbol * self.config.slippage_rate
        gross_by_symbol = weights * torch.expm1(target_log_returns)
        gross_by_symbol = torch.where(return_valid, gross_by_symbol, torch.zeros_like(gross_by_symbol))

        held = weights.abs() > 0
        valid_return_hours = ((~held) | return_valid).all(dim=0) & return_valid.any(dim=0)
        activity_hours = valid_return_hours | (turnover_by_symbol.sum(dim=0) > 0)
        gross_returns = gross_by_symbol.sum(dim=0)
        fees = fee_cost_by_symbol.sum(dim=0)
        slippage = slippage_cost_by_symbol.sum(dim=0)
        net_returns = gross_returns - fees - slippage
        returns = net_returns[activity_hours]
        if returns.numel() == 0:
            raise ValueError("No evaluable Binance factor hours")
        if torch.any(returns <= -1):
            raise ValueError("Research costs produced a return at or below -100%")

        cumulative = torch.prod(1 + returns) - 1
        annualized_return = torch.pow(1 + cumulative, HOURS_PER_YEAR / returns.numel()) - 1
        hourly_volatility = returns.std(unbiased=False)
        annualized_volatility = hourly_volatility * math.sqrt(HOURS_PER_YEAR)
        sharpe = (
            returns.mean() / hourly_volatility * math.sqrt(HOURS_PER_YEAR)
            if hourly_volatility > 1e-12
            else torch.zeros((), device=returns.device)
        )
        equity = torch.cat(
            [torch.ones(1, device=returns.device), torch.cumprod(1 + returns, dim=0)]
        )
        drawdown = equity / torch.cummax(equity, dim=0).values - 1
        max_drawdown = drawdown.min()
        total_turnover = turnover_by_symbol.sum()
        years = max(returns.numel() / HOURS_PER_YEAR, 1 / HOURS_PER_YEAR)
        annualized_turnover = total_turnover / years
        exposure = weights.abs().sum(dim=0)
        active_positions = (weights.abs() > 0).sum(dim=0).float()
        quote_volume = raw_data["quote_volume"]
        trade_notional = turnover_by_symbol * self.config.portfolio_notional_usd
        participation = torch.where(
            quote_volume > 0,
            trade_notional / quote_volume,
            torch.zeros_like(trade_notional),
        )
        rank_ic = mean_rank_ic(factors, target_log_returns, return_valid & signal_valid)
        score = sharpe - 0.5 * torch.abs(max_drawdown)
        per_symbol = tuple(
            {
                "symbol": str(symbol),
                "gross_contribution": float(gross_by_symbol[index].sum().item()),
                "fee_cost": float(fee_cost_by_symbol[index].sum().item()),
                "slippage_cost": float(slippage_cost_by_symbol[index].sum().item()),
                "net_contribution": float(
                    (
                        gross_by_symbol[index]
                        - fee_cost_by_symbol[index]
                        - slippage_cost_by_symbol[index]
                    ).sum().item()
                ),
                "turnover": float(turnover_by_symbol[index].sum().item()),
                "average_weight": float(weights[index].mean().item()),
            }
            for index, symbol in enumerate(symbols)
        )
        return BinanceFactorReport(
            score=float(score.item()),
            cumulative_return=float(cumulative.item()),
            annualized_return=float(annualized_return.item()),
            annualized_volatility=float(annualized_volatility.item()),
            sharpe=float(sharpe.item()),
            max_drawdown=float(max_drawdown.item()),
            total_turnover=float(total_turnover.item()),
            annualized_turnover=float(annualized_turnover.item()),
            rebalance_leg_count=int((turnover_by_symbol > 1e-12).sum().item()),
            fee_cost=float(fees.sum().item()),
            slippage_cost=float(slippage.sum().item()),
            total_cost=float((fees + slippage).sum().item()),
            gross_return_sum=float(gross_returns[valid_return_hours].sum().item()),
            net_return_sum=float(returns.sum().item()),
            win_rate=float((returns > 0).float().mean().item()),
            average_exposure=float(exposure.mean().item()),
            average_active_positions=float(active_positions.mean().item()),
            maximum_volume_participation=float(participation.max().item()),
            mean_rank_ic=rank_ic,
            evaluated_hours=int(activity_hours.sum().item()),
            skipped_return_hours=int((~valid_return_hours).sum().item()),
            config=asdict(self.config),
            per_symbol=per_symbol,
        )

    def construct_weights(
        self,
        factors: torch.Tensor,
        raw_data: dict[str, torch.Tensor],
        signal_valid: torch.Tensor,
    ) -> torch.Tensor:
        symbol_count, time_count = factors.shape
        weights = torch.zeros_like(factors)
        risk = causal_realized_volatility(
            raw_data["close"], signal_valid, self.config.risk_lookback_hours
        )
        previous = torch.zeros(symbol_count, dtype=factors.dtype, device=factors.device)
        for index in range(time_count):
            available = signal_valid[:, index] & torch.isfinite(factors[:, index])
            if index % self.config.rebalance_hours == 0:
                order = torch.argsort(factors[:, index], descending=True, stable=True)
                selected = order[available[order]][: self.config.max_positions]
                desired = torch.zeros_like(previous)
                if selected.numel() > 0:
                    if self.config.weighting == "risk":
                        inverse_risk = torch.where(
                            torch.isfinite(risk[selected, index]),
                            1 / torch.clamp(risk[selected, index], min=1e-6),
                            torch.zeros_like(risk[selected, index]),
                        )
                        if inverse_risk.sum() > 0:
                            desired[selected] = inverse_risk / inverse_risk.sum()
                        else:
                            desired[selected] = 1 / selected.numel()
                    else:
                        desired[selected] = 1 / selected.numel()
            else:
                desired = previous.clone()
                desired[~available] = 0
            weights[:, index] = desired
            previous = desired
        return weights

    @staticmethod
    def _validate_inputs(
        factors: torch.Tensor,
        raw_data: dict[str, torch.Tensor],
        target_log_returns: torch.Tensor,
        return_valid: torch.Tensor,
        signal_valid: torch.Tensor,
        symbols: Sequence[str],
    ) -> None:
        shape = factors.shape
        if factors.ndim != 2 or target_log_returns.shape != shape:
            raise ValueError("Binance factors and returns must share [symbols, time] shape")
        if return_valid.shape != shape or signal_valid.shape != shape:
            raise ValueError("Binance validity masks must match factor shape")
        if len(symbols) != shape[0]:
            raise ValueError("Symbol count does not match factor rows")
        for key in ("close", "quote_volume"):
            if key not in raw_data or raw_data[key].shape != shape:
                raise ValueError(f"Binance evaluation requires {key} with factor shape")


def causal_realized_volatility(
    close: torch.Tensor,
    valid: torch.Tensor,
    window: int,
) -> torch.Tensor:
    previous = torch.roll(close, 1, dims=1)
    previous[:, 0] = 0
    previous_valid = torch.roll(valid, 1, dims=1)
    previous_valid[:, 0] = False
    return_valid = valid & previous_valid & (close > 0) & (previous > 0)
    returns = torch.where(
        return_valid,
        torch.log(close / torch.clamp(previous, min=1e-12)),
        torch.zeros_like(close),
    )
    squared = torch.nn.functional.pad(returns.square(), (window - 1, 0))
    counts = torch.nn.functional.pad(return_valid.to(torch.int16), (window - 1, 0))
    variance = squared.unfold(1, window, 1).mean(dim=-1)
    complete = counts.unfold(1, window, 1).sum(dim=-1) == window
    volatility = torch.sqrt(torch.clamp(variance, min=1e-12))
    return torch.where(complete, volatility, torch.full_like(volatility, float("inf")))


def mean_rank_ic(
    factors: torch.Tensor,
    target_log_returns: torch.Tensor,
    valid: torch.Tensor,
) -> float:
    values: list[torch.Tensor] = []
    for index in range(factors.shape[1]):
        mask = valid[:, index] & torch.isfinite(factors[:, index])
        if int(mask.sum()) < 2:
            continue
        factor = factors[mask, index]
        target = target_log_returns[mask, index]
        factor_rank = average_ranks(factor)
        target_rank = average_ranks(target)
        factor_rank -= factor_rank.mean()
        target_rank -= target_rank.mean()
        denominator = torch.sqrt(factor_rank.square().sum() * target_rank.square().sum())
        if denominator > 0:
            values.append((factor_rank * target_rank).sum() / denominator)
    return float(torch.stack(values).mean().item()) if values else 0.0


def average_ranks(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    ranks = torch.empty_like(values, dtype=torch.float32)
    start = 0
    while start < values.numel():
        end = start + 1
        while end < values.numel() and bool(sorted_values[end] == sorted_values[start]):
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def baseline_reports(
    raw_data: dict[str, torch.Tensor],
    target_log_returns: torch.Tensor,
    return_valid: torch.Tensor,
    signal_valid: torch.Tensor,
    symbols: Sequence[str],
    config: BinanceEvaluationConfig,
    *,
    seed: int = 0,
) -> dict[str, BinanceFactorReport]:
    shape = target_log_returns.shape
    equal_config = replace(config, max_positions=len(symbols), weighting="equal")
    equal = BinanceFactorEvaluator(equal_config).evaluate(
        torch.zeros(shape, device=target_log_returns.device),
        raw_data,
        target_log_returns,
        return_valid,
        signal_valid,
        symbols,
    )
    if "BTCUSDT" not in symbols:
        raise ValueError("BTCUSDT is required for the Binance reference baseline")
    btc_factor = torch.full(shape, float("-inf"), device=target_log_returns.device)
    btc_factor[symbols.index("BTCUSDT")] = 0
    btc = BinanceFactorEvaluator(replace(config, max_positions=1, weighting="equal")).evaluate(
        btc_factor,
        raw_data,
        target_log_returns,
        return_valid,
        signal_valid,
        symbols,
    )
    close = raw_data["close"]
    lagged = torch.roll(close, 24, dims=1)
    lagged[:, :24] = 0
    momentum = torch.where(
        (close > 0) & (lagged > 0),
        torch.log(close / torch.clamp(lagged, min=1e-12)),
        torch.zeros_like(close),
    )
    momentum_valid = signal_valid.clone()
    momentum_valid[:, :24] = False
    momentum_report = BinanceFactorEvaluator(config).evaluate(
        momentum,
        raw_data,
        target_log_returns,
        return_valid,
        momentum_valid,
        symbols,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    random_factor = torch.rand(shape, generator=generator).to(target_log_returns.device)
    random_report = BinanceFactorEvaluator(config).evaluate(
        random_factor,
        raw_data,
        target_log_returns,
        return_valid,
        signal_valid,
        symbols,
    )
    return {
        "equal_weight_cross_section": equal,
        "btcusdt_reference": btc,
        "cross_sectional_momentum": momentum_report,
        "random_rank": random_report,
    }


def cost_sensitivity_reports(
    factors: torch.Tensor,
    raw_data: dict[str, torch.Tensor],
    target_log_returns: torch.Tensor,
    return_valid: torch.Tensor,
    signal_valid: torch.Tensor,
    symbols: Sequence[str],
    config: BinanceEvaluationConfig,
    total_cost_bps: Sequence[float],
) -> dict[str, BinanceFactorReport]:
    reports: dict[str, BinanceFactorReport] = {}
    fee_share = (
        config.taker_fee_bps / (config.taker_fee_bps + config.slippage_bps)
        if config.taker_fee_bps + config.slippage_bps > 0
        else 0.5
    )
    for total in total_cost_bps:
        if total < 0:
            raise ValueError("cost sensitivity values cannot be negative")
        scenario = replace(
            config,
            taker_fee_bps=float(total) * fee_share,
            slippage_bps=float(total) * (1 - fee_share),
        )
        reports[f"{float(total):g}_bps"] = BinanceFactorEvaluator(scenario).evaluate(
            factors,
            raw_data,
            target_log_returns,
            return_valid,
            signal_valid,
            symbols,
        )
    return reports
