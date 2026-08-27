"""Persistent process management for dashboard-launched research batches."""

from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4


ACTIVE_STATUSES = {"queued", "running", "stopping"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class MiningJobConfig:
    snapshot_id: str
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5)
    steps: int = 1000
    batch_size: int = 8192
    windows: int = 4
    shortlist_size: int = 25
    max_positions: int = 10
    weighting: str = "equal"
    rebalance_hours: int = 24
    risk_lookback_hours: int = 24
    taker_fee_bps: float = 10.0
    slippage_bps: float = 5.0
    portfolio_notional_usd: float = 100_000.0
    minimum_quote_volume_usd: float = 0.0
    minimum_cross_section: int = 10
    use_lord_regularization: bool = True

    def __post_init__(self) -> None:
        if not self.snapshot_id or any(
            character not in "0123456789abcdef" for character in self.snapshot_id
        ):
            raise ValueError("snapshot_id must be a lowercase hexadecimal identifier")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and unique")
        if len(self.seeds) > 20:
            raise ValueError("at most 20 seeds are allowed")
        if min(self.steps, self.batch_size, self.windows, self.shortlist_size) <= 0:
            raise ValueError("steps, batch size, windows, and shortlist size must be positive")
        if self.minimum_cross_section < 10:
            raise ValueError("dashboard mining requires at least 10 cross-sectional symbols")
        if self.steps > 100_000 or self.batch_size > 65_536:
            raise ValueError("requested mining workload exceeds the dashboard limit")
        if self.weighting not in {"equal", "risk"}:
            raise ValueError("weighting must be equal or risk")
        if min(self.max_positions, self.rebalance_hours, self.risk_lookback_hours) <= 0:
            raise ValueError("evaluation limits must be positive")
        if min(
            self.taker_fee_bps,
            self.slippage_bps,
            self.portfolio_notional_usd,
            self.minimum_quote_volume_usd,
        ) < 0:
            raise ValueError("cost and liquidity assumptions cannot be negative")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MiningJobConfig":
        payload = dict(value)
        payload["seeds"] = tuple(int(seed) for seed in payload["seeds"])
        return cls(**payload)


class MiningJobManager:
    """Launch and monitor one detached research workload at a time."""

    def __init__(
        self,
        *,
        project_root: str | Path | None = None,
        python_executable: str | Path | None = None,
    ) -> None:
        self.project_root = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
        self.python_executable = str(python_executable or sys.executable)
        self.job_root = self.project_root / "runs" / "dashboard_jobs"
        self.output_root = self.project_root / "runs" / "binance"
        self.job_root.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.job_root / ".lock"

    @contextmanager
    def _lock(self) -> Iterator[None]:
        with self.lock_path.open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _state_path(self, job_id: str) -> Path:
        if not job_id or any(
            not (character.isalnum() or character in "-_") for character in job_id
        ):
            raise ValueError("invalid job identifier")
        return self.job_root / f"{job_id}.json"

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _process_start_ticks(pid: int) -> str | None:
        try:
            # Field 22 follows the parenthesized comm value, which may contain spaces.
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            return fields[19]
        except (FileNotFoundError, IndexError, OSError):
            return None

    @staticmethod
    def _process_command(pid: int) -> str:
        try:
            return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (FileNotFoundError, OSError, UnicodeDecodeError):
            return ""

    def _is_owned_process(self, state: dict[str, Any]) -> bool:
        pid = state.get("pid")
        if not isinstance(pid, int) or pid <= 0:
            return False
        expected_ticks = state.get("process_start_ticks")
        if not expected_ticks or self._process_start_ticks(pid) != expected_ticks:
            return False
        command = self._process_command(pid)
        return "dashboard.mining_job_runner" in command and state.get("job_id", "") in command

    def _load_locked(self, job_id: str) -> dict[str, Any]:
        state = self._read_json(self._state_path(job_id))
        if not state:
            raise KeyError(f"unknown mining job: {job_id}")
        return state

    def update_job(self, job_id: str, **changes: Any) -> dict[str, Any]:
        with self._lock():
            state = self._load_locked(job_id)
            state.update(changes)
            self._write_json(self._state_path(job_id), state)
            return state

    def update_progress(self, job_id: str, event: dict[str, Any]) -> dict[str, Any]:
        with self._lock():
            state = self._load_locked(job_id)
            state["status"] = "running"
            state["updated_at"] = utc_now()
            state["progress"] = event
            self._write_json(self._state_path(job_id), state)
            return state

    def _reconcile_locked(self, state: dict[str, Any]) -> dict[str, Any]:
        if state.get("status") not in ACTIVE_STATUSES:
            return state
        if self._is_owned_process(state):
            return state

        output_dir = Path(state["output_dir"])
        batch = self._read_json(output_dir / "batch_report.json")
        manifest = self._read_json(output_dir / "experiment_manifest.json")
        if batch.get("status") == "complete" or manifest.get("status") == "complete":
            state["status"] = "complete"
        elif state.get("stop_requested_at"):
            state["status"] = "stopped"
        else:
            state["status"] = "failed"
            if not state.get("error"):
                state["error"] = manifest.get("error") or "Mining process exited unexpectedly"
        if not state.get("finished_at"):
            state["finished_at"] = utc_now()
        state["updated_at"] = utc_now()
        self._write_json(self._state_path(state["job_id"]), state)
        return state

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self._lock():
            return self._reconcile_locked(self._load_locked(job_id))

    def list_jobs(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock():
            paths = sorted(
                self.job_root.glob("*.json"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            states = [self._reconcile_locked(self._read_json(path)) for path in paths[:limit]]
        return [state for state in states if state]

    def _ensure_no_active_job_locked(self, *, excluding: str | None = None) -> None:
        for path in self.job_root.glob("*.json"):
            state = self._read_json(path)
            if not state or state.get("job_id") == excluding:
                continue
            state = self._reconcile_locked(state)
            if state.get("status") in ACTIVE_STATUSES:
                raise RuntimeError(f"mining job {state['job_id']} is already active")

    def _spawn_locked(self, state: dict[str, Any]) -> dict[str, Any]:
        log_path = Path(state["log_path"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            self.python_executable,
            "-u",
            "-m",
            "dashboard.mining_job_runner",
            "--job-id",
            state["job_id"],
        ]
        environment = dict(os.environ)
        environment["PYTHONUNBUFFERED"] = "1"
        with log_path.open("ab", buffering=0) as log_handle:
            process = subprocess.Popen(
                command,
                cwd=self.project_root,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        state.update(
            {
                "status": "queued",
                "pid": process.pid,
                "process_start_ticks": self._process_start_ticks(process.pid),
                "updated_at": utc_now(),
            }
        )
        self._write_json(self._state_path(state["job_id"]), state)
        return state

    def start_job(self, config: MiningJobConfig) -> dict[str, Any]:
        with self._lock():
            self._ensure_no_active_job_locked()
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            job_id = f"{stamp}-web-{uuid4().hex[:6]}"
            output_dir = self.output_root / job_id
            state = {
                "job_version": 1,
                "job_id": job_id,
                "status": "queued",
                "created_at": utc_now(),
                "updated_at": utc_now(),
                "output_dir": str(output_dir),
                "log_path": str(self.job_root / f"{job_id}.log"),
                "config": asdict(config),
                "resume_requested": False,
                "attempt": 1,
                "progress": {"phase": "queued", "message": "Waiting for worker startup"},
            }
            self._write_json(self._state_path(job_id), state)
            return self._spawn_locked(state)

    def resume_job(self, job_id: str) -> dict[str, Any]:
        with self._lock():
            state = self._reconcile_locked(self._load_locked(job_id))
            if state.get("status") not in {"failed", "stopped"}:
                raise RuntimeError("only failed or stopped jobs can be resumed")
            self._ensure_no_active_job_locked(excluding=job_id)
            state.update(
                {
                    "status": "queued",
                    "resume_requested": True,
                    "stop_requested_at": None,
                    "finished_at": None,
                    "error": None,
                    "attempt": int(state.get("attempt", 1)) + 1,
                    "progress": {"phase": "queued", "message": "Waiting to resume"},
                }
            )
            with Path(state["log_path"]).open("a") as handle:
                handle.write(f"\n--- Resume attempt {state['attempt']} at {utc_now()} ---\n")
            return self._spawn_locked(state)

    def stop_job(self, job_id: str) -> dict[str, Any]:
        with self._lock():
            state = self._reconcile_locked(self._load_locked(job_id))
            if state.get("status") not in {"queued", "running"}:
                raise RuntimeError("job is not running")
            if not self._is_owned_process(state):
                raise RuntimeError("job process identity could not be verified")
            state.update(
                {
                    "status": "stopping",
                    "stop_requested_at": utc_now(),
                    "updated_at": utc_now(),
                }
            )
            self._write_json(self._state_path(job_id), state)
            os.killpg(int(state["pid"]), signal.SIGTERM)
            return state

    def log_tail(self, job_id: str, max_bytes: int = 24_000) -> str:
        state = self.get_job(job_id)
        path = Path(state["log_path"])
        try:
            with path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - max_bytes))
                return handle.read().decode(errors="replace").replace("\r", "\n")[-max_bytes:]
        except OSError:
            return ""
