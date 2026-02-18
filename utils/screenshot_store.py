"""Hybrid screenshot storage: bounded memory cache plus optional disk persistence."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import hashlib
import threading
import time
import uuid
from typing import Optional, List


@dataclass
class ScreenshotMeta:
    screenshot_id: str
    timestamp: float
    iteration: int
    url: str
    title: str
    sha256: str
    byte_size: int
    in_memory: bool
    path: Optional[str] = None


class ScreenshotStore:
    """Stores screenshots in a bounded in-memory cache and optional disk archive."""

    def __init__(
        self,
        *,
        max_in_memory_items: int = 40,
        max_in_memory_mb: int = 120,
        persist_to_disk: bool = True,
        disk_dir: str = "agent_stream_screenshots",
        max_disk_files: int = 2000,
    ) -> None:
        self.max_in_memory_items = max(1, int(max_in_memory_items))
        self.max_in_memory_bytes = max(1, int(max_in_memory_mb)) * 1024 * 1024
        self.persist_to_disk = bool(persist_to_disk)
        self.disk_dir = Path(disk_dir)
        self.max_disk_files = max(0, int(max_disk_files))

        self._lock = threading.Lock()
        self._metas: "OrderedDict[str, ScreenshotMeta]" = OrderedDict()
        self._memory_bytes: "OrderedDict[str, bytes]" = OrderedDict()
        self._memory_total_bytes: int = 0

        if self.persist_to_disk:
            self.disk_dir.mkdir(parents=True, exist_ok=True)

    def put(
        self,
        screenshot_bytes: bytes,
        *,
        iteration: int,
        url: str = "",
        title: str = "",
    ) -> ScreenshotMeta:
        screenshot_id = f"ss_{uuid.uuid4().hex[:12]}"
        digest = hashlib.sha256(screenshot_bytes).hexdigest()
        now = time.time()
        path: Optional[str] = None

        if self.persist_to_disk:
            self.disk_dir.mkdir(parents=True, exist_ok=True)
            file_path = self.disk_dir / f"{screenshot_id}.png"
            with open(file_path, "wb") as f:
                f.write(screenshot_bytes)
            path = str(file_path)

        meta = ScreenshotMeta(
            screenshot_id=screenshot_id,
            timestamp=now,
            iteration=int(iteration),
            url=url or "",
            title=title or "",
            sha256=digest,
            byte_size=len(screenshot_bytes),
            in_memory=True,
            path=path,
        )

        with self._lock:
            self._metas[screenshot_id] = meta
            self._memory_bytes[screenshot_id] = screenshot_bytes
            self._memory_total_bytes += len(screenshot_bytes)
            self._evict_memory_if_needed()
            self._prune_disk_if_needed()

        return meta

    def get_meta(self, screenshot_id: str) -> Optional[ScreenshotMeta]:
        with self._lock:
            return self._metas.get(screenshot_id)

    def list_recent(self, limit: int = 20) -> List[ScreenshotMeta]:
        with self._lock:
            if limit <= 0:
                return []
            values = list(self._metas.values())
            return values[-limit:][::-1]

    def get_latest_meta(self) -> Optional[ScreenshotMeta]:
        with self._lock:
            if not self._metas:
                return None
            return next(reversed(self._metas.values()))

    def get_bytes(self, screenshot_id: str) -> Optional[bytes]:
        with self._lock:
            in_memory = self._memory_bytes.get(screenshot_id)
            if in_memory is not None:
                self._memory_bytes.move_to_end(screenshot_id)
                return in_memory

            meta = self._metas.get(screenshot_id)
            if meta is None or not meta.path:
                return None
            file_path = Path(meta.path)
            if not file_path.exists():
                return None

            data = file_path.read_bytes()
            self._memory_bytes[screenshot_id] = data
            self._memory_total_bytes += len(data)
            meta.in_memory = True
            self._evict_memory_if_needed()
            return data

    def get_latest_bytes(self) -> Optional[bytes]:
        latest = self.get_latest_meta()
        if latest is None:
            return None
        return self.get_bytes(latest.screenshot_id)

    def clear(self) -> None:
        with self._lock:
            for meta in list(self._metas.values()):
                if meta.path:
                    try:
                        Path(meta.path).unlink(missing_ok=True)
                    except Exception:
                        pass
            self._metas.clear()
            self._memory_bytes.clear()
            self._memory_total_bytes = 0

    def _evict_memory_if_needed(self) -> None:
        while self._memory_bytes and (
            len(self._memory_bytes) > self.max_in_memory_items
            or self._memory_total_bytes > self.max_in_memory_bytes
        ):
            screenshot_id, removed = self._memory_bytes.popitem(last=False)
            self._memory_total_bytes -= len(removed)
            meta = self._metas.get(screenshot_id)
            if meta:
                meta.in_memory = False

    def _prune_disk_if_needed(self) -> None:
        if not self.persist_to_disk:
            return

        disk_backed_ids = [
            screenshot_id
            for screenshot_id, meta in self._metas.items()
            if meta.path and Path(meta.path).exists()
        ]
        overflow = len(disk_backed_ids) - self.max_disk_files
        if overflow <= 0:
            return

        for screenshot_id in disk_backed_ids[:overflow]:
            meta = self._metas.get(screenshot_id)
            if not meta:
                continue
            if meta.path:
                try:
                    Path(meta.path).unlink(missing_ok=True)
                except Exception:
                    pass
                meta.path = None

            # Remove metadata only if the bytes are not retained in memory.
            if screenshot_id not in self._memory_bytes:
                self._metas.pop(screenshot_id, None)
