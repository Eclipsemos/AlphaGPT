"""Causal Binance Spot feature vocabulary and train-only normalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .vocab import BINANCE_FEATURE_NAMES


BINANCE_FEATURE_CODE_VERSION = "binance-features-v1"
BINANCE_FEATURE_WARMUPS = (1, 0, 14, 0, 24, 24, 0, 0, 1, 0, 0)


@dataclass(frozen=True)
class BinanceNormalizationState:
    feature_names: tuple[str, ...]
    fit_end: int
    median: torch.Tensor
    mad: torch.Tensor

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": "per-symbol-median-mad",
            "fit_start": 0,
            "fit_end": self.fit_end,
            "feature_names": list(self.feature_names),
            "median": self.median.detach().cpu().tolist(),
            "mad": self.mad.detach().cpu().tolist(),
        }


@dataclass(frozen=True)
class BinanceFeatureSet:
    values: torch.Tensor
    valid: torch.Tensor
    normalization: BinanceNormalizationState


class BinanceFeatureEngineer:
    @staticmethod
    def compute(
        raw: dict[str, torch.Tensor],
        observed: torch.Tensor,
        *,
        fit_end: int,
    ) -> BinanceFeatureSet:
        required = {
            "open",
            "high",
            "low",
            "close",
            "base_volume",
            "quote_volume",
            "trade_count",
            "taker_buy_quote_volume",
        }
        missing = required.difference(raw)
        if missing:
            raise ValueError(f"Binance feature input is missing: {sorted(missing)}")
        shape = raw["close"].shape
        if observed.shape != shape or any(raw[name].shape != shape for name in required):
            raise ValueError("All Binance feature inputs must share [symbols, time] shape")
        if not 1 <= fit_end <= shape[1]:
            raise ValueError("fit_end must fall inside the time axis")

        close = raw["close"]
        high = raw["high"]
        low = raw["low"]
        base_volume = raw["base_volume"]
        quote_volume = raw["quote_volume"]
        trade_count = raw["trade_count"]
        taker_buy_quote = raw["taker_buy_quote_volume"]

        previous_close = lag(close, 1)
        previous_quote_volume = lag(quote_volume, 1)
        ret_1h = safe_log_ratio(close, previous_close)
        value_range = (high - low) / torch.clamp(previous_close, min=1e-12)
        true_range = torch.maximum(
            high - low,
            torch.maximum(torch.abs(high - previous_close), torch.abs(low - previous_close)),
        ) / torch.clamp(previous_close, min=1e-12)
        atr_14 = rolling_mean(true_range, 14)
        close_position = (close - low) / torch.clamp(high - low, min=1e-12)
        momentum_24h = safe_log_ratio(close, lag(close, 24))
        realized_vol_24h = torch.sqrt(torch.clamp(rolling_mean(ret_1h.square(), 24), min=0))
        log_base_volume = torch.log1p(torch.clamp(base_volume, min=0))
        log_quote_volume = torch.log1p(torch.clamp(quote_volume, min=0))
        quote_volume_change = torch.log1p(torch.clamp(quote_volume, min=0)) - torch.log1p(
            torch.clamp(previous_quote_volume, min=0)
        )
        log_trade_count = torch.log1p(torch.clamp(trade_count, min=0))
        taker_buy_imbalance = 2 * taker_buy_quote / torch.clamp(quote_volume, min=1e-12) - 1

        observed = observed.bool()
        valid_1 = observed & lag(observed, 1)
        valid_14 = rolling_all(valid_1, 14)
        valid_24 = rolling_all(valid_1, 24)
        current_valid = observed
        range_valid = valid_1 & (previous_close > 0) & (high >= low)
        close_position_valid = current_valid & (high > low)
        volume_valid = current_valid & (base_volume >= 0) & (quote_volume >= 0)
        taker_valid = volume_valid & (quote_volume > 0) & (taker_buy_quote >= 0) & (
            taker_buy_quote <= quote_volume
        )
        raw_features = torch.stack(
            [
                ret_1h,
                value_range,
                atr_14,
                close_position,
                momentum_24h,
                realized_vol_24h,
                log_base_volume,
                log_quote_volume,
                quote_volume_change,
                log_trade_count,
                taker_buy_imbalance,
            ],
            dim=1,
        )
        valid = torch.stack(
            [
                valid_1 & (close > 0) & (previous_close > 0),
                range_valid,
                valid_14 & range_valid,
                close_position_valid,
                rolling_all(observed, 25) & (close > 0) & (lag(close, 24) > 0),
                valid_24,
                volume_valid,
                volume_valid,
                valid_1 & (quote_volume >= 0) & (previous_quote_volume >= 0),
                current_valid & (trade_count >= 0),
                taker_valid,
            ],
            dim=1,
        )
        finite = torch.isfinite(raw_features)
        valid &= finite
        normalized, state = fit_robust_normalization(raw_features, valid, fit_end)
        return BinanceFeatureSet(values=normalized, valid=valid, normalization=state)


def lag(value: torch.Tensor, periods: int) -> torch.Tensor:
    if periods == 0:
        return value
    pad = torch.zeros((*value.shape[:-1], periods), dtype=value.dtype, device=value.device)
    return torch.cat([pad, value[..., :-periods]], dim=-1)


def safe_log_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    valid = (numerator > 0) & (denominator > 0)
    ratio = torch.where(valid, numerator / torch.clamp(denominator, min=1e-12), torch.ones_like(numerator))
    return torch.where(valid, torch.log(ratio), torch.zeros_like(numerator))


def rolling_mean(value: torch.Tensor, window: int) -> torch.Tensor:
    padded = torch.nn.functional.pad(value, (window - 1, 0))
    return padded.unfold(-1, window, 1).mean(dim=-1)


def rolling_all(mask: torch.Tensor, window: int) -> torch.Tensor:
    padded = torch.nn.functional.pad(mask.to(torch.int16), (window - 1, 0))
    return padded.unfold(-1, window, 1).sum(dim=-1) == window


def fit_robust_normalization(
    values: torch.Tensor,
    valid: torch.Tensor,
    fit_end: int,
) -> tuple[torch.Tensor, BinanceNormalizationState]:
    fit_values = values[..., :fit_end]
    fit_valid = valid[..., :fit_end]
    masked = torch.where(fit_valid, fit_values, torch.full_like(fit_values, float("nan")))
    median = torch.nanmedian(masked, dim=-1, keepdim=True).values
    absolute = torch.abs(masked - median)
    mad = torch.nanmedian(absolute, dim=-1, keepdim=True).values
    median = torch.nan_to_num(median, nan=0.0)
    mad = torch.nan_to_num(mad, nan=1.0).clamp_min(1e-6)
    normalized = torch.clamp((values - median) / mad, -5.0, 5.0)
    normalized = torch.where(valid, normalized, torch.zeros_like(normalized))
    state = BinanceNormalizationState(
        feature_names=BINANCE_FEATURE_NAMES,
        fit_end=fit_end,
        median=median.squeeze(-1),
        mad=mad.squeeze(-1),
    )
    return normalized, state
