"""Versioned saved-formula contract used across research markets."""

from __future__ import annotations

from typing import Any, Sequence

from .vocab import FormulaVocab


FORMULA_ARTIFACT_VERSION = "alphagpt-formula-v1"


def build_formula_artifact(
    formula: Sequence[int],
    vocab: FormulaVocab,
    research_metadata: dict[str, Any],
) -> dict[str, Any]:
    tokens = [int(token) for token in formula]
    if not tokens:
        raise ValueError("Formula artifact requires at least one token")
    if any(token < 0 or token >= vocab.size for token in tokens):
        raise ValueError(f"Formula contains a token outside {vocab.version}")
    return {
        "artifact_version": FORMULA_ARTIFACT_VERSION,
        "formula": tokens,
        "formula_vocab_version": vocab.version,
        "market": vocab.market,
        "token_names": list(vocab.token_names),
        "research_metadata": research_metadata,
    }


def validate_formula_artifact(value: Any, vocab: FormulaVocab) -> list[int]:
    if not isinstance(value, dict) or value.get("artifact_version") != FORMULA_ARTIFACT_VERSION:
        raise ValueError(f"A versioned {vocab.market} formula artifact is required")
    if value.get("formula_vocab_version") != vocab.version:
        raise ValueError(
            f"Formula vocabulary mismatch: expected {vocab.version}, "
            f"got {value.get('formula_vocab_version')!r}"
        )
    if value.get("market") != vocab.market:
        raise ValueError(f"Formula market mismatch: expected {vocab.market}")
    if value.get("token_names") != list(vocab.token_names):
        raise ValueError("Formula token names do not match the requested vocabulary")
    formula = value.get("formula")
    if not isinstance(formula, list) or not formula:
        raise ValueError("Formula artifact contains no token list")
    tokens = [int(token) for token in formula]
    if any(token < 0 or token >= vocab.size for token in tokens):
        raise ValueError(f"Formula contains a token outside {vocab.version}")
    return tokens
