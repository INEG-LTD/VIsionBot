"""
Lightweight helper for controlled printing throughout the bot.

Every print site can mark whether it is meant for debug-only output or
non-debug / user-facing output. The default mode is ``PrintMode.DEBUG``,
which prints only when the global mode is set to debug, otherwise the
message is suppressed.
"""

from __future__ import annotations

import sys
from enum import Enum
from typing import Any, TextIO


class PrintMode(Enum):
    """Describes the contexts in which a message is allowed to print."""

    DEBUG = "debug"
    """Only show when the bot is running in debug mode."""

    NORMAL = "normal"
    """Show in both normal and debug modes (debug mode is more verbose)."""


_current_print_mode: PrintMode = PrintMode.DEBUG



def get_print_mode() -> PrintMode:
    """Return the current print mode."""
    return _current_print_mode


def set_print_mode(mode: PrintMode) -> None:
    """Set the current print mode."""
    global _current_print_mode
    _current_print_mode = mode


def _should_print(message_mode: PrintMode) -> bool:
    if message_mode == PrintMode.DEBUG:
        return _current_print_mode == PrintMode.DEBUG
    return _current_print_mode in (PrintMode.DEBUG, PrintMode.NORMAL)


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

    target = file or sys.stdout
    try:
        print(*values, sep=sep, end=end, file=target, flush=flush)
    except Exception:
        # Best-effort printing; swallow any issues to avoid breaking logic.
        try:
            print(*values, sep=sep, end=end, flush=flush)
        except Exception:
            pass


__all__ = ["PrintMode", "dprint", "set_print_mode", "get_print_mode"]
