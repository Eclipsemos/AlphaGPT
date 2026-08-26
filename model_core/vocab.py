from dataclasses import dataclass

from .ops import OPS_CONFIG


FEATURE_NAMES = (
    "RET",
    "LIQ_SCORE",
    "PRESSURE",
    "FOMO",
    "DEV",
    "LOG_VOL",
)


@dataclass(frozen=True)
class FormulaVocab:
    feature_names: tuple[str, ...]
    operator_names: tuple[str, ...]
    version: str = "solana-formula-v1"
    market: str = "solana"

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)

    @property
    def operator_offset(self) -> int:
        return self.feature_count

    @property
    def token_names(self) -> tuple[str, ...]:
        return self.feature_names + self.operator_names

    @property
    def size(self) -> int:
        return len(self.token_names)


FORMULA_VOCAB = FormulaVocab(
    feature_names=FEATURE_NAMES,
    operator_names=tuple(cfg[0] for cfg in OPS_CONFIG),
)


BINANCE_FEATURE_NAMES = (
    "RET_1H",
    "RANGE",
    "ATR_14",
    "CLOSE_POSITION",
    "MOMENTUM_24H",
    "REALIZED_VOL_24H",
    "LOG_BASE_VOLUME",
    "LOG_QUOTE_VOLUME",
    "QUOTE_VOLUME_CHANGE",
    "LOG_TRADE_COUNT",
    "TAKER_BUY_IMBALANCE",
)


BINANCE_FORMULA_VOCAB = FormulaVocab(
    feature_names=BINANCE_FEATURE_NAMES,
    operator_names=tuple(cfg[0] for cfg in OPS_CONFIG),
    version="binance-formula-v1",
    market="binance-spot",
)
