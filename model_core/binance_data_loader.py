"""Load an immutable Binance dataset snapshot for factor mining."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import pandas as pd
import sqlalchemy
import torch

from .binance_features import (
    BINANCE_FEATURE_CODE_VERSION,
    BINANCE_FEATURE_WARMUPS,
    BinanceFeatureEngineer,
)
from .config import ModelConfig
from .data_loader import TimeSplit, compute_forward_returns
from .vocab import BINANCE_FORMULA_VOCAB


class BinanceDataLoader:
    def __init__(self, snapshot_id: str | None = None):
        self.engine = sqlalchemy.create_engine(ModelConfig.DB_URL)
        self.snapshot_id = snapshot_id
        self.snapshot: dict[str, Any] | None = None
        self.symbols: list[str] = []
        self.times: list[datetime] = []
        self.raw_data_cache: dict[str, torch.Tensor] | None = None
        self.observed_mask: torch.Tensor | None = None
        self.feature_valid: torch.Tensor | None = None
        self.feat_tensor: torch.Tensor | None = None
        self.feature_normalization: dict[str, Any] | None = None
        self.target_ret: torch.Tensor | None = None
        self.target_valid: torch.Tensor | None = None
        self.splits: TimeSplit | None = None
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
            raise ValueError(f"At least 10 Binance time points are required; got {length}")
        train_end = max(1, int(length * ModelConfig.TRAIN_RATIO))
        validation_end = max(
            train_end + 1,
            int(length * (ModelConfig.TRAIN_RATIO + ModelConfig.VALIDATION_RATIO)),
        )
        validation_end = min(validation_end, length - 1)
        if not 0 < train_end < validation_end < length:
            raise ValueError(f"Invalid Binance split ratios for {length} time points")
        return TimeSplit(slice(0, train_end), slice(train_end, validation_end), slice(validation_end, length))

    def _assign_split_views(self) -> None:
        assert self.splits is not None
        assert self.feat_tensor is not None
        assert self.raw_data_cache is not None
        assert self.target_ret is not None
        assert self.target_valid is not None
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
        self.train_target_ret[:, -2:] = 0
        self.validation_target_ret[:, -2:] = 0
        self.test_target_ret[:, -2:] = 0
        self.train_target_valid[:, -2:] = False
        self.validation_target_valid[:, -2:] = False
        self.test_target_valid[:, -2:] = False

    def _load_snapshot(self) -> tuple[dict[str, Any], list[str]]:
        if self.snapshot_id:
            frame = pd.read_sql(
                sqlalchemy.text(
                    "SELECT snapshot_id, payload FROM dataset_snapshots WHERE snapshot_id = :snapshot_id"
                ),
                self.engine,
                params={"snapshot_id": self.snapshot_id},
            )
        else:
            frame = pd.read_sql(
                sqlalchemy.text("""
                SELECT snapshot_id, payload FROM dataset_snapshots
                WHERE venue = 'binance' AND market_type = 'spot'
                ORDER BY created_at DESC LIMIT 1
                """),
                self.engine,
            )
        if frame.empty:
            raise ValueError("No Binance dataset snapshot found; build one with data_pipeline.run_binance_pipeline")
        resolved_snapshot_id = str(frame.iloc[0]["snapshot_id"])
        payload = frame.iloc[0]["payload"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        if payload.get("feature_schema_version") != "binance-features-v1":
            raise ValueError("Unsupported or missing Binance feature schema version")
        symbols = [str(value).upper() for value in payload.get("symbols", [])]
        if not symbols:
            raise ValueError("Binance snapshot contains no symbols")
        coverage = pd.read_sql(
            sqlalchemy.text(
                """
                SELECT symbol, source_metadata
                FROM dataset_snapshot_coverage
                WHERE snapshot_id = :snapshot_id
                """
            ),
            self.engine,
            params={"snapshot_id": resolved_snapshot_id},
        )
        accepted_symbols = set()
        for _, row in coverage.iterrows():
            metadata = row["source_metadata"]
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            if metadata.get("quality", {}).get("accepted") is True:
                accepted_symbols.add(str(row["symbol"]).upper())
        if accepted_symbols != set(symbols):
            raise ValueError(
                f"Binance snapshot {resolved_snapshot_id} is missing accepted B2 quality coverage"
            )
        self.snapshot_id = resolved_snapshot_id
        return payload, symbols

    def load_data(self) -> None:
        payload, symbols = self._load_snapshot()
        self.snapshot = payload
        start_time = pd.Timestamp(payload["start_time"]).tz_convert("UTC")
        end_time = pd.Timestamp(payload["end_time"]).tz_convert("UTC")
        if payload.get("rules", {}).get("interval") != "1h":
            raise ValueError("BinanceDataLoader only supports 1h snapshots")
        symbol_params = {f"symbol_{index}": symbol for index, symbol in enumerate(symbols)}
        placeholders = ", ".join(f":symbol_{index}" for index in range(len(symbols)))
        frame = pd.read_sql(
            sqlalchemy.text(f"""
            SELECT symbol, interval, open_time, close_time, open, high, low, close,
                   base_volume, quote_volume, trade_count,
                   taker_buy_base_volume, taker_buy_quote_volume
            FROM market_bars
            WHERE venue = 'binance' AND market_type = 'spot'
              AND symbol IN ({placeholders}) AND interval = '1h'
              AND open_time >= :start_time AND open_time < :end_time
            ORDER BY open_time, symbol
            """),
            self.engine,
            params={**symbol_params, "start_time": start_time.to_pydatetime(), "end_time": end_time.to_pydatetime()},
        )
        if frame.empty:
            raise ValueError(f"Binance snapshot {self.snapshot_id} has no stored bars")
        frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True)
        self.times = pd.date_range(
            start=start_time,
            end=end_time,
            freq="1h",
            inclusive="left",
        ).tolist()
        observed = frame.pivot(index="open_time", columns="symbol", values="open").reindex(
            index=self.times, columns=symbols
        )
        self.observed_mask = torch.tensor(observed.notna().to_numpy().T, dtype=torch.bool, device=ModelConfig.DEVICE)

        raw_columns = {
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "base_volume": "base_volume",
            "quote_volume": "quote_volume",
            "trade_count": "trade_count",
            "taker_buy_quote_volume": "taker_buy_quote_volume",
        }
        raw: dict[str, torch.Tensor] = {}
        for key, column in raw_columns.items():
            pivot = frame.pivot(index="open_time", columns="symbol", values=column).reindex(
                index=self.times, columns=symbols
            )
            raw[key] = torch.tensor(
                pivot.fillna(0.0).to_numpy(dtype="float32").T,
                dtype=torch.float32,
                device=ModelConfig.DEVICE,
            )
        self.symbols = symbols
        self.raw_data_cache = raw
        fit_end = max(1, int(len(self.times) * ModelConfig.TRAIN_RATIO))
        feature_set = BinanceFeatureEngineer.compute(raw, self.observed_mask, fit_end=fit_end)
        self.feat_tensor = feature_set.values
        self.feature_valid = feature_set.valid
        self.feature_normalization = feature_set.normalization.as_dict()
        self.target_ret, self.target_valid = compute_forward_returns(
            raw["open"], self.observed_mask, return_valid=True
        )
        self.target_valid &= self.feature_valid.all(dim=1)
        self.splits = self._build_splits(self.feat_tensor.shape[-1])
        self._assign_split_views()

    @property
    def quality_report(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "symbols": len(self.symbols),
            "time_points": len(self.times),
            "feature_schema_version": BINANCE_FEATURE_CODE_VERSION,
            "feature_names": list(self.feature_normalization["feature_names"])
            if self.feature_normalization
            else [],
        }

    @property
    def research_metadata(self) -> dict[str, Any]:
        if self.snapshot is None or self.feature_normalization is None:
            raise RuntimeError("Binance data must be loaded before metadata is available")
        return {
            "market": "binance-spot",
            "dataset_snapshot_id": self.snapshot_id,
            "dataset_schema_version": self.snapshot["schema_version"],
            "feature_schema_version": BINANCE_FEATURE_CODE_VERSION,
            "formula_vocab_version": BINANCE_FORMULA_VOCAB.version,
            "feature_names": list(BINANCE_FORMULA_VOCAB.feature_names),
            "feature_warmups": dict(zip(BINANCE_FORMULA_VOCAB.feature_names, BINANCE_FEATURE_WARMUPS)),
            "normalization": self.feature_normalization,
        }
