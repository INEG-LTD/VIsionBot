"""Per-agent workspace and temporary run cleanup management."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time
import uuid
from typing import Any, Optional

from core.storage_cleanup import StorageCleanupService


@dataclass
class AgentWorkspace:
    """Resolved filesystem layout for one agent instance."""

    agent_id: str
    persistence_mode: str
    base_dir: Path
    agent_root: Path
    workspace_root: Path
    written_data_dir: Path
    browser_profile_dir: Path
    browser_downloads_dir: Path
    screenshots_dir: Path
    stream_screenshots_dir: Path
    runs_root: Path
    meta_dir: Path
    current_run_root: Optional[Path] = None

    @property
    def sandbox_audit_path(self) -> Optional[Path]:
        if not self.current_run_root:
            return None
        return self.current_run_root / "audit" / "sandbox_events.jsonl"

    @property
    def run_event_log_path(self) -> Optional[Path]:
        if not self.current_run_root:
            return None
        return self.current_run_root / "logs" / "events.jsonl"


class AgentWorkspaceManager:
    """Creates, tracks, and cleans agent workspace directories."""

    def __init__(
        self,
        *,
        base_dir: str,
        cleanup_enabled: bool,
        temp_ttl_days: int,
        temp_keep_last_runs: int,
        temp_max_runs: int,
        max_disk_mb: int,
        event_logger: Optional[Any] = None,
    ) -> None:
        resolved_base_dir = Path(base_dir).expanduser().resolve()
        resolved_base_dir.mkdir(parents=True, exist_ok=True)
        self.agents_root = self._resolve_agents_root(resolved_base_dir)
        self.agents_root.mkdir(parents=True, exist_ok=True)
        self.base_dir = self.agents_root
        self.cleanup_enabled = bool(cleanup_enabled)
        self.temp_ttl_days = max(0, int(temp_ttl_days or 0))
        self.temp_keep_last_runs = max(0, int(temp_keep_last_runs or 0))
        self.temp_max_runs = max(1, int(temp_max_runs or 1))
        self.event_logger = event_logger
        self.cleanup_service = StorageCleanupService(
            agents_root=self.agents_root,
            cleanup_enabled=self.cleanup_enabled,
            temp_ttl_days=self.temp_ttl_days,
            temp_keep_last_runs=self.temp_keep_last_runs,
            temp_max_runs=self.temp_max_runs,
            max_disk_mb=max_disk_mb,
            event_logger=self.event_logger,
        )

    def create_agent(
        self,
        *,
        persistence_mode: str = "temp",
        agent_id: Optional[str] = None,
    ) -> AgentWorkspace:
        resolved_mode = self._normalize_mode(persistence_mode)
        resolved_agent_id = (agent_id or f"agent_{uuid.uuid4().hex[:10]}").strip()
        agent_root = self.agents_root / resolved_agent_id
        workspace = AgentWorkspace(
            agent_id=resolved_agent_id,
            persistence_mode=resolved_mode,
            base_dir=self.agents_root,
            agent_root=agent_root,
            workspace_root=agent_root / "workspace",
            written_data_dir=agent_root / "data" / "written",
            browser_profile_dir=agent_root / "browser" / "profile",
            browser_downloads_dir=agent_root / "browser" / "downloads",
            screenshots_dir=agent_root / "artifacts" / "screenshots",
            stream_screenshots_dir=agent_root / "artifacts" / "stream",
            runs_root=agent_root / "runs",
            meta_dir=agent_root / "meta",
        )
        for path in (
            workspace.workspace_root,
            workspace.written_data_dir,
            workspace.browser_profile_dir,
            workspace.browser_downloads_dir,
            workspace.screenshots_dir,
            workspace.stream_screenshots_dir,
            workspace.runs_root,
            workspace.meta_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

        self._write_agent_meta(workspace, created=True)
        return workspace

    def start_run(self, workspace: AgentWorkspace, *, mission: str) -> str:
        run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        run_root = workspace.runs_root / run_id
        (run_root / "logs").mkdir(parents=True, exist_ok=True)
        (run_root / "audit").mkdir(parents=True, exist_ok=True)
        (run_root / ".active").write_text("1", encoding="utf-8")

        workspace.current_run_root = run_root
        self._write_json(
            run_root / "summary.json",
            {
                "run_id": run_id,
                "agent_id": workspace.agent_id,
                "persistence_mode": workspace.persistence_mode,
                "mission": mission,
                "started_at": time.time(),
                "status": "running",
            },
        )
        self._write_agent_meta(workspace, created=False)
        return run_id

    def finish_run(
        self,
        workspace: AgentWorkspace,
        *,
        success: bool,
        reasoning: str,
        total_actions: int,
        total_iterations: int,
        final_url: str,
        duration_s: float,
        event_count: int = 0,
    ) -> None:
        run_root = workspace.current_run_root
        if not run_root:
            return

        summary_path = run_root / "summary.json"
        payload = self._read_json(summary_path)
        payload.update(
            {
                "ended_at": time.time(),
                "status": "success" if success else "failed",
                "success": bool(success),
                "reasoning": reasoning,
                "total_actions": int(total_actions or 0),
                "total_iterations": int(total_iterations or 0),
                "final_url": final_url or "",
                "duration_s": float(duration_s or 0.0),
                "event_count": int(event_count or 0),
            }
        )
        self._write_json(summary_path, payload)
        try:
            (run_root / ".active").unlink(missing_ok=True)
        except Exception:
            pass
        self._write_agent_meta(workspace, created=False)

    def cleanup_temp_runs(self, *, exclude_agent_id: Optional[str] = None) -> None:
        self.cleanup_service.cleanup_temp_runs(exclude_agent_id=exclude_agent_id)

    @staticmethod
    def _resolve_agents_root(base_dir: Path) -> Path:
        """Normalize configured base_dir to an agents root."""
        if base_dir.name.lower() == "agents":
            return base_dir
        return base_dir / "agents"

    @staticmethod
    def _normalize_mode(value: Any) -> str:
        mode = str(value or "temp").strip().lower()
        if mode not in {"temp", "persistent"}:
            return "temp"
        return mode

    def _write_agent_meta(self, workspace: AgentWorkspace, *, created: bool) -> None:
        meta_path = workspace.meta_dir / "agent.json"
        payload = self._read_json(meta_path)
        if created:
            payload.setdefault("created_at", time.time())
        payload.update(
            {
                "agent_id": workspace.agent_id,
                "persistence_mode": workspace.persistence_mode,
                "last_seen_at": time.time(),
            }
        )
        self._write_json(meta_path, payload)

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
        return {}

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
