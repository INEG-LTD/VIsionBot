"""Per-agent workspace management."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time
import uuid
from typing import Any, Optional, List, Dict


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

    @property
    def run_checkpoint_path(self) -> Optional[Path]:
        """Path for mission-resume checkpoint of the currently attached run."""
        if not self.current_run_root:
            return None
        return self.current_run_root / "state" / "checkpoint.json"


class AgentWorkspaceManager:
    """Creates and tracks agent workspace directories."""

    def __init__(
        self,
        *,
        base_dir: str,
        event_logger: Optional[Any] = None,
    ) -> None:
        resolved_base_dir = Path(base_dir).expanduser().resolve()
        resolved_base_dir.mkdir(parents=True, exist_ok=True)
        self.agents_root = self._resolve_agents_root(resolved_base_dir)
        self.agents_root.mkdir(parents=True, exist_ok=True)
        self.base_dir = self.agents_root
        self.event_logger = event_logger

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
        (run_root / "state").mkdir(parents=True, exist_ok=True)
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

    def attach_existing_run(self, workspace: AgentWorkspace, *, run_id: str) -> Optional[Path]:
        """Attach a workspace to an existing run directory by run_id."""
        run_id_clean = str(run_id or "").strip()
        if not run_id_clean:
            return None
        run_root = workspace.runs_root / run_id_clean
        if not run_root.is_dir():
            return None
        workspace.current_run_root = run_root
        return run_root

    def list_agent_ids(self) -> List[str]:
        """Return known agent ids under the storage root."""
        try:
            ids = [path.name for path in self.agents_root.iterdir() if path.is_dir()]
        except Exception:
            return []
        ids.sort()
        return ids

    def list_runs(self, workspace: AgentWorkspace) -> List[Dict[str, Any]]:
        """
        Return run metadata sorted newest-first.

        Each item includes:
        - run_id
        - started_at
        - status
        - mission
        - active (bool)
        - has_checkpoint (bool)
        """
        runs: List[Dict[str, Any]] = []
        try:
            run_dirs = [path for path in workspace.runs_root.iterdir() if path.is_dir()]
        except Exception:
            return runs

        for run_root in run_dirs:
            run_id = run_root.name
            summary = self._read_json(run_root / "summary.json")
            started_at = float(summary.get("started_at", 0.0) or 0.0)
            active = bool((run_root / ".active").exists())
            checkpoint_path = run_root / "state" / "checkpoint.json"
            has_checkpoint = bool(checkpoint_path.exists())
            runs.append(
                {
                    "run_id": run_id,
                    "started_at": started_at,
                    "status": str(summary.get("status", "") or ""),
                    "mission": str(summary.get("mission", "") or ""),
                    "active": active,
                    "has_checkpoint": has_checkpoint,
                }
            )

        runs.sort(
            key=lambda item: (
                float(item.get("started_at", 0.0) or 0.0),
                str(item.get("run_id", "")),
            ),
            reverse=True,
        )
        return runs

    def resolve_resume_run_id(
        self,
        workspace: AgentWorkspace,
        *,
        requested_run_id: Optional[str] = None,
        prefer_active: bool = True,
    ) -> Optional[str]:
        """
        Resolve which run should be used for resume loading.

        Preference order:
        1) requested run id (if it exists)
        2) newest active run with checkpoint (when prefer_active=True)
        3) newest run with checkpoint
        4) newest active run
        5) newest run
        """
        requested = str(requested_run_id or "").strip()
        if requested:
            run_root = workspace.runs_root / requested
            if run_root.is_dir():
                return requested

        runs = self.list_runs(workspace)
        if not runs:
            return None

        if prefer_active:
            for item in runs:
                if item.get("active") and item.get("has_checkpoint"):
                    return str(item.get("run_id", "")).strip() or None

        for item in runs:
            if item.get("has_checkpoint"):
                return str(item.get("run_id", "")).strip() or None

        if prefer_active:
            for item in runs:
                if item.get("active"):
                    return str(item.get("run_id", "")).strip() or None

        return str(runs[0].get("run_id", "")).strip() or None

    def write_run_checkpoint(self, workspace: AgentWorkspace, payload: Dict[str, Any]) -> bool:
        """Write checkpoint JSON for the currently attached run."""
        path = workspace.run_checkpoint_path
        if path is None:
            return False
        try:
            self._write_json(path, payload)
            return True
        except Exception:
            return False

    def read_run_checkpoint(
        self,
        workspace: AgentWorkspace,
        *,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Read checkpoint JSON for a run (current run when run_id is omitted)."""
        if run_id:
            run_root = workspace.runs_root / str(run_id).strip()
            if not run_root.is_dir():
                return {}
            path = run_root / "state" / "checkpoint.json"
            return self._read_json(path)

        path = workspace.run_checkpoint_path
        if path is None:
            return {}
        return self._read_json(path)

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
