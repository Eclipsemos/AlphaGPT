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


def formula_complexity(formula: Sequence[int], vocab: FormulaVocab) -> dict[str, int]:
    stack: list[int] = []
    operators = {
        vocab.operator_offset + index: arity
        for index, (_, _, arity) in enumerate(OPS_CONFIG)
    }
    operator_count = 0
    feature_tokens: set[int] = set()
    maximum_depth = 0
    for raw_token in formula:
        token = int(raw_token)
        if 0 <= token < vocab.feature_count:
            stack.append(1)
            feature_tokens.add(token)
            maximum_depth = max(maximum_depth, 1)
            continue
        arity = operators.get(token)
        if arity is None or len(stack) < arity:
            raise ValueError("Formula is not a valid stack expression")
        arguments = stack[-arity:]
        del stack[-arity:]
        depth = max(arguments) + 1
        stack.append(depth)
        maximum_depth = max(maximum_depth, depth)
        operator_count += 1
    if len(stack) != 1:
        raise ValueError("Formula is not a single stack expression")
    return {
        "token_count": len(formula),
        "operator_count": operator_count,
        "unique_feature_count": len(feature_tokens),
        "tree_depth": maximum_depth,
    }
