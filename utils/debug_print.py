"""
Lightweight helper for controlled printing throughout the bot.

Every print site can mark whether it is meant for debug-only output or
non-debug / user-facing output. The default mode is ``PrintMode.DEBUG``,
which prints only when the global mode is set to debug, otherwise the
message is suppressed.
"""

from __future__ import annotations

import inspect
import sys
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, TextIO


class PrintMode(Enum):
    """Describes the contexts in which a message is allowed to print."""

    DEBUG = "debug"
    """Only show when the bot is running in debug mode."""

    NORMAL = "normal"
    """Show in both normal and debug modes (debug mode is more verbose)."""


_current_print_mode: PrintMode = PrintMode.DEBUG
_print_callbacks: list[Callable[["DebugPrintRecord"], None]] = []
_callback_guard = threading.local()


@dataclass(frozen=True)
class DebugPrintRecord:
    """One emitted debug-print line with lightweight origin metadata."""

    text: str
    mode: PrintMode
    thread_id: int
    module: str
    function: str



def get_print_mode() -> PrintMode:
    """Return the current print mode."""
    return _current_print_mode


def set_print_mode(mode: PrintMode) -> None:
    """Set the current print mode."""
    global _current_print_mode
    _current_print_mode = mode


def register_print_callback(callback: Callable[[DebugPrintRecord], None]) -> None:
    """Register a callback invoked for each line that passes print-mode guards."""
    if callback not in _print_callbacks:
        _print_callbacks.append(callback)


def unregister_print_callback(callback: Callable[[DebugPrintRecord], None]) -> None:
    """Remove a previously registered callback."""
    if callback in _print_callbacks:
        _print_callbacks.remove(callback)


def _should_print(message_mode: PrintMode) -> bool:
    if message_mode == PrintMode.DEBUG:
        return _current_print_mode == PrintMode.DEBUG
    return _current_print_mode in (PrintMode.DEBUG, PrintMode.NORMAL)


def _notify_callbacks(*, text: str, mode: PrintMode) -> None:
    if not text or not _print_callbacks:
        return
    if getattr(_callback_guard, "active", False):
        return

    caller_module = ""
    caller_function = ""
    frame = inspect.currentframe()
    try:
        caller = frame.f_back.f_back if frame and frame.f_back else None
        if caller is not None:
            caller_module = str(caller.f_globals.get("__name__", "") or "")
            caller_function = str(caller.f_code.co_name or "")
    except Exception:
        caller_module = ""
        caller_function = ""
    finally:
        del frame

    record = DebugPrintRecord(
        text=text,
        mode=mode,
        thread_id=threading.get_ident(),
        module=caller_module,
        function=caller_function,
    )
    _callback_guard.active = True
    try:
        for callback in list(_print_callbacks):
            try:
                callback(record)
            except Exception:
                pass
    finally:
        _callback_guard.active = False


def dprint(
    *values: Any,
    sep: str = " ",
    end: str = "\n",
    file: TextIO | None = None,
    flush: bool = False,
    mode: PrintMode = PrintMode.DEBUG,
) -> None:
    """
    Wrap ``print`` with a mode guard.

    ``mode`` determines whether the message should be visible in debug mode,
    normal mode, or both (debug includes normal). By default every message
    is considered debug-only.
    """
    if not _should_print(mode):
        return

    rendered = sep.join(str(value) for value in values)
    _notify_callbacks(text=rendered, mode=mode)

    target = file or sys.stdout
    try:
        print(*values, sep=sep, end=end, file=target, flush=flush)
    except Exception:
        # Best-effort printing; swallow any issues to avoid breaking logic.
        try:
            print(*values, sep=sep, end=end, flush=flush)
        except Exception:
            pass


__all__ = [
    "PrintMode",
    "DebugPrintRecord",
    "dprint",
    "set_print_mode",
    "get_print_mode",
    "register_print_callback",
    "unregister_print_callback",
]
