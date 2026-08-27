import torch
from .ops import OPS_CONFIG
from .vocab import FORMULA_VOCAB, FormulaVocab

class StackVM:
    def __init__(self, vocab: FormulaVocab = FORMULA_VOCAB):
        self.vocab = vocab
        self.feat_offset = vocab.operator_offset
        self.op_map = {i + self.feat_offset: cfg[1] for i, cfg in enumerate(OPS_CONFIG)}
        self.arity_map = {i + self.feat_offset: cfg[2] for i, cfg in enumerate(OPS_CONFIG)}

    def execute(self, formula_tokens, feat_tensor):
        stack = []
        try:
            for token in formula_tokens:
                token = int(token)
                if token < self.feat_offset:
                    if token >= feat_tensor.shape[1]:
                        return None
                    stack.append(feat_tensor[:, token, :])
                elif token in self.op_map:
                    arity = self.arity_map[token]
                    if len(stack) < arity: return None
                    args = []
                    for _ in range(arity):
                        args.append(stack.pop())
                    args.reverse()
                    func = self.op_map[token]
                    res = func(*args)
                    if torch.isnan(res).any() or torch.isinf(res).any():
                        res = torch.nan_to_num(res, nan=0.0, posinf=1.0, neginf=-1.0)
                    stack.append(res)
                else:
                    return None
            if len(stack) == 1:
                return stack[0]
            else:
                return None
        except Exception:
            return None

    def is_valid_formula(self, formula_tokens) -> bool:
        """Return whether tokens form a single valid stack expression."""
        depth = 0
        for raw_token in formula_tokens:
            token = int(raw_token)
            if token < 0 or token >= self.vocab.size:
                return False
            if token < self.feat_offset:
                depth += 1
                continue
            arity = self.arity_map.get(token)
            if arity is None or depth < arity:
                return False
            depth = depth - arity + 1
        return depth == 1

    def valid_formula_mask(self, formulas: torch.Tensor) -> torch.Tensor:
        """Validate a [formula, token] tensor without Python per-formula loops."""
        if formulas.ndim != 2:
            raise ValueError("formulas must have shape [formula, token]")
        depth = torch.zeros(formulas.shape[0], dtype=torch.int16, device=formulas.device)
        valid = torch.ones(formulas.shape[0], dtype=torch.bool, device=formulas.device)
        for position in range(formulas.shape[1]):
            token = formulas[:, position]
            in_range = (token >= 0) & (token < self.vocab.size)
            feature = in_range & (token < self.feat_offset)
            operator = in_range & (token >= self.feat_offset)
            arity = torch.zeros_like(depth)
            for operator_token, operator_arity in self.arity_map.items():
                arity = torch.where(
                    token == operator_token,
                    torch.as_tensor(operator_arity, dtype=depth.dtype, device=depth.device),
                    arity,
                )
            valid &= in_range & (feature | (operator & (depth >= arity)))
            depth = torch.where(feature, depth + 1, depth - arity + 1)
        return valid & (depth == 1)

    def execute_batch(
        self,
        formulas: torch.Tensor,
        feat_tensor: torch.Tensor,
        *,
        chunk_size: int = 128,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute many postfix formulas in bounded GPU-memory chunks.

        Returns factors with shape [formula, symbol, time] and a validity mask.
        Scoring callers can discard invalid formulas without synchronizing one
        Python loop per sampled sequence.
        """
        if formulas.ndim != 2:
            raise ValueError("formulas must have shape [formula, token]")
        if feat_tensor.ndim != 3:
            raise ValueError("feat_tensor must have shape [symbol, feature, time]")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        formula_count, token_count = formulas.shape
        symbol_count, feature_count, time_count = feat_tensor.shape
        if feature_count < self.feat_offset:
            raise ValueError("feature tensor does not contain the VM feature vocabulary")

        if formulas.device != feat_tensor.device:
            raise ValueError("formulas and feat_tensor must be on the same device")
        valid_formulas = self.valid_formula_mask(formulas)
        if formula_count == 0:
            return feat_tensor.new_empty((0, symbol_count, time_count)), valid_formulas

        outputs: list[torch.Tensor] = []
        for start in range(0, formula_count, chunk_size):
            chunk_tokens = formulas[start : start + chunk_size]
            chunk_valid = valid_formulas[start : start + chunk_size]
            chunk_output = feat_tensor.new_zeros(
                (chunk_tokens.shape[0], symbol_count, time_count)
            )
            valid_rows = torch.nonzero(chunk_valid, as_tuple=False).flatten()
            if valid_rows.numel() == 0:
                outputs.append(chunk_output)
                continue

            tokens = chunk_tokens[valid_rows]
            batch = tokens.shape[0]
            depth = torch.zeros(batch, dtype=torch.long, device=tokens.device)
            stack = torch.zeros(
                (batch, token_count, symbol_count, time_count),
                dtype=feat_tensor.dtype,
                device=feat_tensor.device,
            )
            for position in range(token_count):
                token = tokens[:, position]
                feature_rows = torch.nonzero(token < self.feat_offset, as_tuple=False).flatten()
                if feature_rows.numel() > 0:
                    feature_index = token[feature_rows]
                    feature_values = feat_tensor[:, feature_index, :].permute(1, 0, 2)
                    stack[feature_rows, depth[feature_rows]] = feature_values
                    depth[feature_rows] += 1
                for operator_token, operator_arity in self.arity_map.items():
                    operator_rows = torch.nonzero(
                        token == operator_token, as_tuple=False
                    ).flatten()
                    if operator_rows.numel() == 0:
                        continue
                    base_depth = depth[operator_rows] - operator_arity
                    args = [
                        stack[operator_rows, base_depth + offset]
                        for offset in range(operator_arity)
                    ]
                    flat_args = [arg.reshape(-1, time_count) for arg in args]
                    result = self.op_map[operator_token](*flat_args).reshape(
                        operator_rows.shape[0], symbol_count, time_count
                    )
                    result = torch.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)
                    stack[operator_rows, base_depth] = result
                    depth[operator_rows] = base_depth + 1
            rows = torch.arange(batch, device=tokens.device)
            chunk_output[valid_rows] = stack[rows, depth - 1]
            outputs.append(chunk_output)
        return torch.cat(outputs, dim=0), valid_formulas
