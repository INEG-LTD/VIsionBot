from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Iterable, Iterator, List, Optional

from models.models import NotebookEntryType
from utils.event_logger import get_event_logger


@dataclass
class NotebookEntry:
    def __init__(
        self,
        timestamp: float,
        description: str,
        data: Any,
        url: Optional[str] = None,
        iteration: Optional[int] = None,
        type: NotebookEntryType = NotebookEntryType.EXTRACTION,
    ) -> None:
        self.timestamp: float = timestamp
        self.description: str = description
        self.data: Any = data
        self.url: Optional[str] = url
        self.iteration: Optional[int] = iteration
        self.type: NotebookEntryType = type


@dataclass
class Notebook:
    """Collects extracted data during agent execution."""

    def __init__(self, entries: Optional[Iterable[NotebookEntry]] = None) -> None:
        self._entries: List[NotebookEntry] = list(entries or [])

    def add_extraction(
        self,
        description: str,
        data: Any,
        url: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        self._entries.append(NotebookEntry(
            timestamp=timestamp or time.time(),
            description=description,
            data=data,
            url=url,
            type=NotebookEntryType.EXTRACTION,
        ))
        get_event_logger().system_debug(f"Added extraction to notebook: {description} {data} {url}")

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
