"""Temporary-run cleanup service for agent storage."""

from __future__ import annotations

from pathlib import Path
import json
import shutil
import time
from typing import Any, Optional


class StorageCleanupService:
    """Applies cleanup policies to temporary runs only."""

    def __init__(
        self,
        *,
        agents_root: Path,
        cleanup_enabled: bool,
        temp_ttl_days: int,
        temp_keep_last_runs: int,
        temp_max_runs: int,
        max_disk_mb: int,
        event_logger: Optional[Any] = None,
    ) -> None:
        self.agents_root = agents_root.expanduser().resolve()
        self.agents_root.mkdir(parents=True, exist_ok=True)

        self.cleanup_enabled = bool(cleanup_enabled)
        self.temp_ttl_days = max(0, int(temp_ttl_days or 0))
        self.temp_keep_last_runs = max(0, int(temp_keep_last_runs or 0))
        self.temp_max_runs = max(1, int(temp_max_runs or 1))
        self.max_disk_bytes = max(0, int(max_disk_mb or 0)) * 1024 * 1024
        self.event_logger = event_logger

    def cleanup_temp_runs(self, *, exclude_agent_id: Optional[str] = None) -> dict[str, int]:
        stats = {
            "agents_scanned": 0,
            "runs_deleted": 0,
            "bytes_deleted": 0,
        }
        if not self.cleanup_enabled:
            return stats

        self._emit_cleanup_start(exclude_agent_id=exclude_agent_id)

        now = time.time()
        ttl_seconds = self.temp_ttl_days * 86400
        temp_run_paths: list[Path] = []

        for agent_root in self._iter_agent_roots():
            if exclude_agent_id and agent_root.name == exclude_agent_id:
                continue

            meta = self._read_json(agent_root / "meta" / "agent.json")
            mode = self._normalize_mode(meta.get("persistence_mode", "temp"))
            if mode != "temp":
                continue

            stats["agents_scanned"] += 1
            runs_root = agent_root / "runs"
            if not runs_root.exists():
                continue

            run_dirs = [path for path in runs_root.iterdir() if path.is_dir()]
            run_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
            protected = set(run_dirs[: self.temp_keep_last_runs])

            # Max-runs pruning (per temp agent).
            for path in run_dirs[self.temp_max_runs :]:
                if path in protected or self._is_active_run(path):
                    continue
                self._safe_delete_dir(path, reason="max_runs", stats=stats)

            # TTL pruning.
            if ttl_seconds > 0:
                for path in run_dirs:
                    if path in protected or self._is_active_run(path) or not path.exists():
                        continue
                    age_seconds = now - path.stat().st_mtime
                    if age_seconds > ttl_seconds:
                        self._safe_delete_dir(path, reason="ttl", stats=stats)

            # Collect remaining temp runs for global disk-budget pruning.
            for path in run_dirs:
                if path.exists() and not self._is_active_run(path):
                    temp_run_paths.append(path)

        # Global disk budget pruning (temp runs only).
        if self.max_disk_bytes > 0:
            total_bytes = sum(self._dir_size_bytes(path) for path in temp_run_paths if path.exists())
            if total_bytes > self.max_disk_bytes:
                temp_run_paths.sort(key=lambda path: path.stat().st_mtime)
                for path in temp_run_paths:
                    if total_bytes <= self.max_disk_bytes:
                        break
                    if not path.exists() or self._is_active_run(path):
                        continue

                    parent_runs = path.parent
                    siblings = [p for p in parent_runs.iterdir() if p.is_dir() and not self._is_active_run(p)]
                    siblings.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    keep_set = set(siblings[: self.temp_keep_last_runs])
                    if path in keep_set:
                        continue

                    removed = self._safe_delete_dir(path, reason="disk_budget", stats=stats)
                    total_bytes = max(0, total_bytes - removed)

        self._emit_cleanup_complete(stats=stats)
        return stats

    def _iter_agent_roots(self) -> list[Path]:
        if not self.agents_root.exists() or not self.agents_root.is_dir():
            return []
        found: list[Path] = []
        for candidate in self.agents_root.iterdir():
            if not candidate.is_dir():
                continue
            if (candidate / "meta" / "agent.json").exists() or (candidate / "runs").exists():
                found.append(candidate)
        return found

    @staticmethod
    def _normalize_mode(value: Any) -> str:
        mode = str(value or "temp").strip().lower()
        if mode not in {"temp", "persistent"}:
            return "temp"
        return mode

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
        return {}

    @staticmethod
    def _is_active_run(run_dir: Path) -> bool:
        return (run_dir / ".active").exists()

    @staticmethod
    def _dir_size_bytes(path: Path) -> int:
        total = 0
        try:
            for child in path.rglob("*"):
                if child.is_file():
                    total += child.stat().st_size
        except Exception:
            return total
        return total

    def _safe_delete_dir(self, path: Path, *, reason: str, stats: dict[str, int]) -> int:
        size = self._dir_size_bytes(path)
        try:
            shutil.rmtree(path, ignore_errors=True)
        except Exception:
            size = 0
        if not path.exists():
            stats["runs_deleted"] += 1
            stats["bytes_deleted"] += max(0, size)
            self._emit_cleanup_deletion(path=path, reason=reason, bytes_removed=max(0, size))
            return max(0, size)
        self._emit_cleanup_deletion(path=path, reason=f"{reason}_failed", bytes_removed=0)
        return 0

    def _emit_cleanup_start(self, *, exclude_agent_id: Optional[str]) -> None:
        if not self.event_logger:
            return
        try:
            self.event_logger.cleanup_start(
                root=str(self.agents_root),
                exclude_agent_id=exclude_agent_id,
                ttl_days=self.temp_ttl_days,
                keep_last=self.temp_keep_last_runs,
                max_runs=self.temp_max_runs,
                max_disk_bytes=self.max_disk_bytes,
            )
        except Exception:
            pass

    def _emit_cleanup_deletion(self, *, path: Path, reason: str, bytes_removed: int) -> None:
        if not self.event_logger:
            return
        try:
            self.event_logger.cleanup_deletion(
                path=str(path),
                reason=reason,
                bytes_removed=bytes_removed,
            )
        except Exception:
            pass

    def _emit_cleanup_complete(self, *, stats: dict[str, int]) -> None:
        if not self.event_logger:
            return
        try:
            self.event_logger.cleanup_complete(
                root=str(self.agents_root),
                agents_scanned=stats.get("agents_scanned", 0),
                runs_deleted=stats.get("runs_deleted", 0),
                bytes_deleted=stats.get("bytes_deleted", 0),
            )
        except Exception:
            pass
