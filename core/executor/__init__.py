"""
Action executor for browser automation.

This module exports the Executor class and supporting types.
"""
from .base import Executor, PreActionContext, PostActionContext, ScrollReason

from .ui_feedback import (
    highlight_selector,
    highlight_overlay,
    highlight_box,
    highlight_point,
    highlight_click_location,
    clear_highlight,
)

# Bind UI feedback methods to Executor class
Executor._highlight_selector = highlight_selector
Executor._highlight_overlay = highlight_overlay
Executor._highlight_box = highlight_box
Executor._highlight_point = highlight_point
Executor._highlight_click_location = highlight_click_location
Executor._clear_highlight = clear_highlight

__all__ = [
    "Executor",
    "PreActionContext",
    "PostActionContext",
    "ScrollReason",
]
