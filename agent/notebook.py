from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union

from models.models import NotebookEntryType
from utils.event_logger import get_event_logger

@dataclass
class NotebookEntry:
    def __init__(self, timestamp: float, task: str, iteration: Optional[int], data: Any, url: Optional[str] = None, type: NotebookEntryType = NotebookEntryType.EXTRACTION) -> None:
        self.timestamp: float = timestamp
        self.task: str = task
        self.iteration: Optional[int] = iteration
        self.data: Any = data
        self.url: Optional[str] = url
        self.type: NotebookEntryType = type

@dataclass
class Notebook:
    """Collects extracted data and task results during agent execution."""

    def __init__(self, entries: Optional[Iterable[NotebookEntry]] = None) -> None:
        self._entries: List[NotebookEntry] = list(entries or [])

    def add_extraction(
        self,
        task: str,
        data: Any,
        url: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        self._entries.append(NotebookEntry(
            timestamp=timestamp or time.time(),
            task=task,
            data=data,
            url=url,
            iteration=None,
            type=NotebookEntryType.EXTRACTION))
        get_event_logger().system_debug(f"Added extraction to notebook: {task} {data} {url}")

    def add_entry(self, entry: NotebookEntry) -> None:
        self._entries.append(entry)

    def to_list(self) -> List[NotebookEntry]:
        return self._entries

    def __iter__(self) -> Iterator[NotebookEntry]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, item):
        return self._entries[item]

    def __bool__(self) -> bool:
        return bool(self._entries)
