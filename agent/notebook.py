from __future__ import annotations

import time
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union

from models.models import NotebookEntryType


class Notebook:
    """Collects extracted data and task results during agent execution."""

    def __init__(self, entries: Optional[Iterable[Dict[str, Any]]] = None) -> None:
        self._entries: List[Dict[str, Any]] = list(entries or [])

    def add_extraction(
        self,
        prompt: str,
        data: Any,
        url: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        self._entries.append(
            {
                "timestamp": timestamp or time.time(),
                "prompt": prompt,
                "data": data,
                "url": url,
                "type": NotebookEntryType.EXTRACTION,
            }
        )

    def add_url_extraction(
        self,
        element: str,
        url: str,
        context_url: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        self._entries.append(
            {
                "timestamp": timestamp or time.time(),
                "prompt": f"URL from {element}",
                "data": {"url": url, "element": element},
                "url": context_url,
                "type": NotebookEntryType.URL_EXTRACTION,
            }
        )

    def add_task_result(
        self,
        source: str,
        task_id: str,
        description: str,
        data: Any,
        entry_type: NotebookEntryType,
        iteration: Optional[int] = None,
    ) -> None:
        entry: Dict[str, Any] = {
            "source": source,
            "task_id": task_id,
            "description": description,
            "data": data,
            "type": entry_type,
        }
        if iteration is not None:
            entry["iteration"] = iteration
        self._entries.append(entry)

    def add_entry(self, entry: Dict[str, Any]) -> None:
        self._entries.append(entry)

    def extend(self, entries: Union["Notebook", Iterable[Dict[str, Any]]]) -> None:
        if isinstance(entries, Notebook):
            self._entries.extend(entries._entries)
        else:
            self._entries.extend(entries)

    def to_list(self) -> List[Dict[str, Any]]:
        return list(self._entries)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, item):
        return self._entries[item]

    def __bool__(self) -> bool:
        return bool(self._entries)
