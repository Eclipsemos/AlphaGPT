"""Read-only Binance factor-research dashboard service.

This module deliberately has no wallet, RPC, account, portfolio, paper-trading,
or order API integration. Database access is limited to public market-data
snapshots and their quality metadata.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import sqlalchemy
from dotenv import load_dotenv

load_dotenv()


class DashboardService:
    def __init__(self, *, project_root: str | Path | None = None):
        db_user = os.getenv("DB_USER", "postgres")
        db_pass = os.getenv("DB_PASSWORD", "password")
        db_host = os.getenv("DB_HOST", "localhost")
        db_port = os.getenv("DB_PORT", "5432")
        db_name = os.getenv("DB_NAME", "crypto_quant")
        self.engine = sqlalchemy.create_engine(
            f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
        )
        self.project_root = Path(project_root or Path(__file__).resolve().parents[1])

    @staticmethod
    def _load_json_file(path: str | Path, default: Any) -> Any:
        try:
            with Path(path).open() as handle:
                return json.load(handle)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return default

    def list_snapshots(self, limit: int = 50) -> pd.DataFrame:
        limit = max(1, min(int(limit), 500))
        query = sqlalchemy.text(
            """
            SELECT snapshot_id, schema_version, feature_schema_version,
                   venue, market_type, interval, start_time, end_time,
                   source, code_version, created_at, payload
            FROM dataset_snapshots
            WHERE venue = 'binance' AND market_type = 'spot'
            ORDER BY created_at DESC
            LIMIT :limit
            """
        )
        try:
            frame = pd.read_sql(query, self.engine, params={"limit": limit})
        except Exception:
            return pd.DataFrame()
        if frame.empty:
            return frame
        frame["symbol_count"] = frame["payload"].apply(
            lambda payload: len(payload.get("symbols", [])) if isinstance(payload, dict) else 0
        )
        return frame.drop(columns=["payload"])

    def snapshot_payload(self, snapshot_id: str) -> dict[str, Any]:
        query = sqlalchemy.text(
            "SELECT payload FROM dataset_snapshots WHERE snapshot_id = :snapshot_id"
        )
        try:
            frame = pd.read_sql(query, self.engine, params={"snapshot_id": snapshot_id})
        except Exception:
            return {}
        if frame.empty:
            return {}
        payload = frame.iloc[0]["payload"]
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                return {}
        return payload if isinstance(payload, dict) else {}

    def get_snapshot_coverage(self, snapshot_id: str) -> pd.DataFrame:
        query = sqlalchemy.text(
            """
            SELECT symbol, requested_start_time, requested_end_time,
                   response_start_time, response_end_time, bar_count,
                   expected_bar_count, archive_checksum, source_metadata,
                   retrieved_at
            FROM dataset_snapshot_coverage
            WHERE snapshot_id = :snapshot_id
            ORDER BY symbol
            """
        )
        try:
            return pd.read_sql(query, self.engine, params={"snapshot_id": snapshot_id})
        except Exception:
            return pd.DataFrame()

    @staticmethod
    def binance_market_overview_query(limit: int = 50) -> str:
        """Return a bounded query for the latest completed bar per Binance symbol."""
        limit = max(1, min(int(limit), 500))
        return f"""
        SELECT b.symbol, b.open_time, b.open, b.high, b.low, b.close,
               b.base_volume, b.quote_volume, b.trade_count,
               b.taker_buy_quote_volume
        FROM market_bars AS b
        JOIN (
            SELECT symbol, MAX(open_time) AS latest_open_time
            FROM market_bars
            WHERE venue = 'binance' AND market_type = 'spot' AND interval = '1h'
            GROUP BY symbol
        ) AS latest
          ON latest.symbol = b.symbol AND latest.latest_open_time = b.open_time
        WHERE b.venue = 'binance' AND b.market_type = 'spot' AND b.interval = '1h'
        ORDER BY b.quote_volume DESC
        LIMIT {limit}
        """

    def get_market_overview(self, limit: int = 50) -> pd.DataFrame:
        try:
            return pd.read_sql(self.binance_market_overview_query(limit), self.engine)
        except Exception:
            return pd.DataFrame()

    def get_data_status(self, snapshot_id: str | None = None) -> dict[str, Any]:
        query = """
        SELECT
            (SELECT COUNT(*) FROM market_instruments
             WHERE venue = 'binance' AND market_type = 'spot') AS instrument_count,
            (SELECT COUNT(*) FROM market_bars
             WHERE venue = 'binance' AND market_type = 'spot' AND interval = '1h') AS bar_count,
            (SELECT MAX(open_time) FROM market_bars
             WHERE venue = 'binance' AND market_type = 'spot' AND interval = '1h') AS latest_bar,
            (SELECT COUNT(*) FROM dataset_snapshots
             WHERE venue = 'binance' AND market_type = 'spot') AS snapshot_count,
            (SELECT MAX(created_at) FROM dataset_snapshots
             WHERE venue = 'binance' AND market_type = 'spot') AS latest_snapshot
        """
        try:
            with self.engine.connect() as connection:
                row = connection.exec_driver_sql(query).mappings().one()
            result = dict(row)
        except Exception:
            result = {
                "instrument_count": 0,
                "bar_count": 0,
                "latest_bar": None,
                "snapshot_count": 0,
                "latest_snapshot": None,
            }
        if snapshot_id:
            payload = self.snapshot_payload(snapshot_id)
            result["selected_snapshot_id"] = snapshot_id
            result["selected_symbol_count"] = len(payload.get("symbols", []))
            result["selected_start_time"] = payload.get("start_time")
            result["selected_end_time"] = payload.get("end_time")
        return result

    def latest_research_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        root = self.project_root / "runs" / "binance"
        if not root.exists():
            return []
        rows = []
        paths = sorted(
            root.glob("*/decision_report.json"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        for path in paths[:limit]:
            report = self._load_json_file(path, {})
            batch = self._load_json_file(path.parent / "batch_report.json", {})
            rows.append(
                {
                    "run": path.parent.name,
                    "path": str(path.parent),
                    "status": report.get("status", "unknown"),
                    "canonical_formula": report.get("canonical_formula", ""),
                    "snapshot_id": batch.get("research_metadata", {}).get("dataset_snapshot_id", ""),
                    "seeds": batch.get("seeds", []),
                }
            )
        return rows

    def load_run_artifacts(self, run_dir: str | Path) -> dict[str, Any]:
        directory = Path(run_dir)
        if not directory.is_absolute():
            directory = self.project_root / directory
        return {
            "manifest": self._load_json_file(directory / "experiment_manifest.json", {}),
            "batch": self._load_json_file(directory / "batch_report.json", {}),
            "decision": self._load_json_file(directory / "decision_report.json", {}),
            "evaluation": self._load_json_file(directory / "final_evaluation_report.json", {}),
            "walk_forward": self._load_json_file(directory / "walk_forward_report.json", {}),
        }
