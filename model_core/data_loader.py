from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch
import sqlalchemy

from .config import ModelConfig
from .factors import FeatureEngineer


@dataclass(frozen=True)
class DataQualityReport:
    """Quality checks performed before tensors are built."""

    rows: int
    addresses: int
    null_rows: int
    duplicate_rows: int
    nonpositive_price_rows: int
    nonpositive_liquidity_rows: int
    nonpositive_fdv_rows: int
    timestamp_gaps: int

    @property
    def has_fatal_errors(self) -> bool:
        return self.null_rows > 0 or self.duplicate_rows > 0

    def as_dict(self) -> dict[str, int]:
        return {
            "rows": self.rows,
            "addresses": self.addresses,
            "null_rows": self.null_rows,
            "duplicate_rows": self.duplicate_rows,
            "nonpositive_price_rows": self.nonpositive_price_rows,
            "nonpositive_liquidity_rows": self.nonpositive_liquidity_rows,
            "nonpositive_fdv_rows": self.nonpositive_fdv_rows,
            "timestamp_gaps": self.timestamp_gaps,
        }


@dataclass(frozen=True)
class TimeSplit:
    """Half-open temporal ranges into the loaded time axis."""

    train: slice
    validation: slice
    test: slice


def _count_timestamp_gaps(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    gaps = 0
    for _, group in frame.sort_values("time").groupby("address"):
        deltas = group["time"].drop_duplicates().diff().dropna()
        if deltas.empty:
            continue
        cadence = deltas.median()
        gaps += int((deltas > cadence * 1.5).sum())
    return gaps


def inspect_market_data(frame: pd.DataFrame) -> DataQualityReport:
    required = {"time", "address", "open", "high", "low", "close", "volume", "liquidity", "fdv"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"OHLCV data is missing columns: {sorted(missing)}")

    price_columns = ["open", "high", "low", "close"]
    return DataQualityReport(
        rows=len(frame),
        addresses=int(frame["address"].nunique()),
        null_rows=int(frame[list(required)].isna().any(axis=1).sum()),
        duplicate_rows=int(frame.duplicated(["time", "address"]).sum()),
        nonpositive_price_rows=int((frame[price_columns] <= 0).any(axis=1).sum()),
        nonpositive_liquidity_rows=int((frame["liquidity"] <= 0).sum()),
        nonpositive_fdv_rows=int((frame["fdv"] <= 0).sum()),
        timestamp_gaps=_count_timestamp_gaps(frame),
    )


def compute_forward_returns(
    open_prices: torch.Tensor,
    observed_mask: torch.Tensor | None = None,
    *,
    return_valid: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Return from the next bar's open to the following bar's open.

    A signal observed at bar ``t`` is assumed to enter at ``t+1`` and hold
    through ``t+2``. The final two labels are purged because they cross the
    available data boundary. With ``observed_mask``, labels are also invalid
    when the signal, entry, or exit candle was filled rather than observed.
    """
    next_open = torch.roll(open_prices, -1, dims=1)
    exit_open = torch.roll(open_prices, -2, dims=1)
    valid = (open_prices > 0) & (next_open > 0) & (exit_open > 0)
    if observed_mask is not None:
        if observed_mask.shape != open_prices.shape:
            raise ValueError("observed_mask must have the same shape as open_prices")
        valid &= observed_mask
        valid &= torch.roll(observed_mask, -1, dims=1)
        valid &= torch.roll(observed_mask, -2, dims=1)
    valid[:, -2:] = False
    ratio = torch.where(valid, exit_open / (next_open + 1e-9), torch.ones_like(next_open))
    forward_return = torch.where(valid, torch.log(ratio), torch.zeros_like(ratio))
    forward_return = torch.nan_to_num(forward_return, nan=0.0, posinf=0.0, neginf=0.0)
    forward_return[:, -2:] = 0.0
    return (forward_return, valid) if return_valid else forward_return


class CryptoDataLoader:
    def __init__(self):
        self.engine = sqlalchemy.create_engine(ModelConfig.DB_URL)
        self.feat_tensor = None
        self.raw_data_cache = None
        self.target_ret = None
        self.target_valid = None
        self.addresses: list[str] = []
        self.times: list[Any] = []
        self.splits: TimeSplit | None = None
        self.quality_report: DataQualityReport | None = None
        self.train_feat_tensor = None
        self.validation_feat_tensor = None
        self.test_feat_tensor = None
        self.train_raw_data_cache = None
        self.validation_raw_data_cache = None
        self.test_raw_data_cache = None
        self.train_target_ret = None
        self.validation_target_ret = None
        self.test_target_ret = None
        self.train_target_valid = None
        self.validation_target_valid = None
        self.test_target_valid = None

    @staticmethod
    def _slice_raw(raw_data: dict[str, torch.Tensor], window: slice) -> dict[str, torch.Tensor]:
        return {key: value[:, window] for key, value in raw_data.items()}

    def _build_splits(self, length: int) -> TimeSplit:
        if length < 10:
            raise ValueError(f"At least 10 time points are required for temporal splits; got {length}")
        train_end = max(1, int(length * ModelConfig.TRAIN_RATIO))
        validation_end = max(train_end + 1, int(length * (ModelConfig.TRAIN_RATIO + ModelConfig.VALIDATION_RATIO)))
        validation_end = min(validation_end, length - 1)
        if not 0 < train_end < validation_end < length:
            raise ValueError(f"Invalid split ratios for {length} time points")
        return TimeSplit(slice(0, train_end), slice(train_end, validation_end), slice(validation_end, length))

    def _assign_split_views(self) -> None:
        assert self.splits is not None
        self.train_feat_tensor = self.feat_tensor[:, :, self.splits.train]
        self.validation_feat_tensor = self.feat_tensor[:, :, self.splits.validation]
        self.test_feat_tensor = self.feat_tensor[:, :, self.splits.test]
        self.train_raw_data_cache = self._slice_raw(self.raw_data_cache, self.splits.train)
        self.validation_raw_data_cache = self._slice_raw(self.raw_data_cache, self.splits.validation)
        self.test_raw_data_cache = self._slice_raw(self.raw_data_cache, self.splits.test)
        self.train_target_ret = self.target_ret[:, self.splits.train].clone()
        self.validation_target_ret = self.target_ret[:, self.splits.validation].clone()
        self.test_target_ret = self.target_ret[:, self.splits.test].clone()
        self.train_target_valid = self.target_valid[:, self.splits.train].clone()
        self.validation_target_valid = self.target_valid[:, self.splits.validation].clone()
        self.test_target_valid = self.target_valid[:, self.splits.test].clone()
        # The target uses the next two bars; purge labels that cross a split boundary.
        self.train_target_ret[:, -2:] = 0.0
        self.validation_target_ret[:, -2:] = 0.0
        self.test_target_ret[:, -2:] = 0.0
        self.train_target_valid[:, -2:] = False
        self.validation_target_valid[:, -2:] = False
        self.test_target_valid[:, -2:] = False

    def load_data(self, limit_tokens=500):
        print("Loading data from SQL...")
        top_query = f"SELECT address FROM tokens ORDER BY address LIMIT {int(limit_tokens)}"
        self.addresses = pd.read_sql(top_query, self.engine)["address"].tolist()
        if not self.addresses:
            raise ValueError("No tokens found.")

        addr_str = "'" + "','".join(self.addresses) + "'"
        data_query = f"""
        SELECT time, address, open, high, low, close, volume, liquidity, fdv
        FROM ohlcv
        WHERE address IN ({addr_str})
        ORDER BY time ASC
        """
        frame = pd.read_sql(data_query, self.engine)
        self.quality_report = inspect_market_data(frame)
        if self.quality_report.has_fatal_errors:
            raise ValueError(f"Fatal market-data quality errors: {self.quality_report.as_dict()}")
        if self.quality_report.nonpositive_price_rows:
            print(f"Warning: {self.quality_report.nonpositive_price_rows} rows have non-positive prices; values will be forward-filled.")
        if self.quality_report.timestamp_gaps:
            print(f"Warning: detected {self.quality_report.timestamp_gaps} timestamp gaps.")

        self.times = sorted(frame["time"].drop_duplicates().tolist())

        observed_frame = frame.pivot(index="time", columns="address", values="open")
        observed_frame = observed_frame.reindex(index=self.times, columns=self.addresses)
        observed_mask = torch.tensor(
            observed_frame.notna().values.T,
            dtype=torch.bool,
            device=ModelConfig.DEVICE,
        )

        def to_tensor(col):
            pivot = frame.pivot(index="time", columns="address", values=col)
            pivot = pivot.reindex(index=self.times, columns=self.addresses)
            if col in {"open", "high", "low", "close"}:
                pivot = pivot.mask(pivot <= 0)
            pivot = pivot.ffill().bfill().fillna(0.0)
            return torch.tensor(pivot.values.T, dtype=torch.float32, device=ModelConfig.DEVICE)

        self.raw_data_cache = {key: to_tensor(key) for key in ("open", "high", "low", "close", "volume", "liquidity", "fdv")}
        train_end = int(self.raw_data_cache["open"].shape[1] * ModelConfig.TRAIN_RATIO)
        self.feat_tensor = FeatureEngineer.compute_features(self.raw_data_cache, fit_end=train_end)
        self.target_ret, self.target_valid = compute_forward_returns(
            self.raw_data_cache["open"], observed_mask, return_valid=True
        )
        self.splits = self._build_splits(self.feat_tensor.shape[-1])
        self._assign_split_views()
        print(
            f"Data Ready. Shape: {self.feat_tensor.shape}; "
            f"splits train/validation/test = "
            f"{self.train_feat_tensor.shape[-1]}/"
            f"{self.validation_feat_tensor.shape[-1]}/"
            f"{self.test_feat_tensor.shape[-1]}"
        )
