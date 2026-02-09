from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union

from models.models import NotebookEntryType
from utils.event_logger import get_event_logger

@dataclass
class NotebookEntry:
    def __init__(self, timestamp: float, task: str, iteration: Optional[int], subtask_turn: Optional[int], data: Any, url: Optional[str] = None, type: NotebookEntryType = NotebookEntryType.EXTRACTION) -> None:
        self.timestamp: float = timestamp
        self.task: str = task
        self.iteration: Optional[int] = iteration
        self.subtask_turn: Optional[int] = subtask_turn
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
            subtask_turn=None,
            type=NotebookEntryType.EXTRACTION))
        get_event_logger().system_debug(f"Added extraction to notebook: {task} {data} {url}")

    def add_remember(
        self,
        task: str,
        data: Any,
        timestamp: Optional[float] = None,
    ) -> None:
        self._entries.append(NotebookEntry(
            timestamp=timestamp or time.time(), 
            task=task, 
            data=data, 
            iteration=None,
            subtask_turn=None,
            type=NotebookEntryType.REMEMBER))
        get_event_logger().system_debug(f"Added remember to notebook: {task} {data}")
        return True

    def add_subtask_result(
        self,
        task_id: str,
        description: str,
        data: Any,
        entry_type: NotebookEntryType,
        subtask_turn: Optional[int] = None,
    ) -> None:
        entry: NotebookEntry = NotebookEntry(
            timestamp=time.time(), 
            task=description, 
            data=data, 
            iteration=None,
            subtask_turn=subtask_turn,
            type=entry_type)
        self._entries.append(entry)
        from utils.event_logger import get_event_logger
        event_logger = get_event_logger()
        event_logger.notebook_entry_added(task_id, entry_type, data)


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
