"""Vectorized train/validation scoring for Binance factor discovery."""

from __future__ import annotations

import torch


def cross_sectional_ic_score(
    factors: torch.Tensor,
    target_log_returns: torch.Tensor,
    valid: torch.Tensor,
    *,
    minimum_cross_section: int = 10,
) -> torch.Tensor:
    if factors.shape != target_log_returns.shape or valid.shape != factors.shape:
        raise ValueError("Mining factors, targets, and masks must share shape")
    if minimum_cross_section < 2:
        raise ValueError("minimum_cross_section must be at least 2")
    mask = valid & torch.isfinite(factors) & torch.isfinite(target_log_returns)
    count = mask.sum(dim=0)
    enough = count >= minimum_cross_section
    safe_count = count.clamp_min(1).to(factors.dtype)
    x = torch.where(mask, factors, torch.zeros_like(factors))
    y = torch.where(mask, target_log_returns, torch.zeros_like(target_log_returns))
    x_mean = x.sum(dim=0) / safe_count
    y_mean = y.sum(dim=0) / safe_count
    x_centered = torch.where(mask, x - x_mean, torch.zeros_like(x))
    y_centered = torch.where(mask, y - y_mean, torch.zeros_like(y))
    covariance = (x_centered * y_centered).sum(dim=0)
    denominator = torch.sqrt(
        x_centered.square().sum(dim=0) * y_centered.square().sum(dim=0)
    )
    usable = enough & (denominator > 1e-12)
    if not bool(usable.any()):
        return torch.tensor(-10.0, dtype=factors.dtype, device=factors.device)
    ic = torch.where(usable, covariance / denominator.clamp_min(1e-12), torch.zeros_like(denominator))
    values = ic[usable]
    # IC is bounded and interpretable. Do not amplify a tiny cross-section into
    # an artificial score of 10 via an unstable IC information ratio.
    return values.mean()
