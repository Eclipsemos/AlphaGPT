"""Canonical RPN formula representation for cross-run deduplication."""

from __future__ import annotations

from typing import Sequence

from .ops import OPS_CONFIG
from .vocab import FormulaVocab


COMMUTATIVE_OPERATORS = {"ADD", "MUL"}


def canonical_formula(formula: Sequence[int], vocab: FormulaVocab) -> str:
    stack: list[str] = []
    operators = {
        vocab.operator_offset + index: (name, arity)
        for index, (name, _, arity) in enumerate(OPS_CONFIG)
    }
    for raw_token in formula:
        token = int(raw_token)
        if 0 <= token < vocab.feature_count:
            stack.append(vocab.feature_names[token])
            continue
        operator = operators.get(token)
        if operator is None:
            raise ValueError(f"Unknown token {token} for {vocab.version}")
        name, arity = operator
        if len(stack) < arity:
            raise ValueError("Formula is not a valid stack expression")
        arguments = stack[-arity:]
        del stack[-arity:]
        if name in COMMUTATIVE_OPERATORS:
            arguments.sort()
        stack.append(f"{name}({','.join(arguments)})")
    if len(stack) != 1:
        raise ValueError("Formula is not a single stack expression")
    return stack[0]
