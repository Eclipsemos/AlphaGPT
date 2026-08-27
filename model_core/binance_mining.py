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
    return cross_sectional_ic_scores(
        factors.unsqueeze(0),
        target_log_returns,
        valid,
        minimum_cross_section=minimum_cross_section,
    )[0]


def cross_sectional_ic_scores(
    factors: torch.Tensor,
    target_log_returns: torch.Tensor,
    valid: torch.Tensor,
    *,
    minimum_cross_section: int = 10,
) -> torch.Tensor:
    """Score a batch of factors without synchronizing once per formula."""
    if factors.ndim != 3:
        raise ValueError("Mining factor batch must have shape [formula, symbol, time]")
    if target_log_returns.shape != factors.shape[1:] or valid.shape != factors.shape[1:]:
        raise ValueError("Mining targets and masks must match factor symbol/time shape")
    if minimum_cross_section < 2:
        raise ValueError("minimum_cross_section must be at least 2")
    expanded_target = target_log_returns.unsqueeze(0)
    mask = valid.unsqueeze(0) & torch.isfinite(factors) & torch.isfinite(expanded_target)
    count = mask.sum(dim=1)
    enough = count >= minimum_cross_section
    safe_count = count.clamp_min(1).to(factors.dtype)
    x = torch.where(mask, factors, torch.zeros_like(factors))
    y = torch.where(mask, expanded_target, torch.zeros_like(factors))
    x_mean = x.sum(dim=1, keepdim=True) / safe_count.unsqueeze(1)
    y_mean = y.sum(dim=1, keepdim=True) / safe_count.unsqueeze(1)
    x_centered = torch.where(mask, x - x_mean, torch.zeros_like(x))
    y_centered = torch.where(mask, y - y_mean, torch.zeros_like(y))
    covariance = (x_centered * y_centered).sum(dim=1)
    denominator = torch.sqrt(
        x_centered.square().sum(dim=1) * y_centered.square().sum(dim=1)
    )
    usable = enough & (denominator > 1e-12)
    ic = torch.where(usable, covariance / denominator.clamp_min(1e-12), torch.zeros_like(denominator))
    # IC is bounded and interpretable. Do not amplify a tiny cross-section into
    # an artificial score of 10 via an unstable IC information ratio.
    usable_count = usable.sum(dim=1)
    scores = ic.sum(dim=1) / usable_count.clamp_min(1).to(ic.dtype)
    return torch.where(
        usable_count > 0,
        scores,
        torch.full_like(scores, -10.0),
    )
