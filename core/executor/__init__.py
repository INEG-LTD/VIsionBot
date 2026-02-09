"""
Action executor for browser automation.

This module exports the Executor class and supporting types.
"""
from .base import Executor, PreActionContext, PostActionContext, ScrollReason
from .callbacks import (
    register_pre_action_callback,
    unregister_pre_action_callback,
    unregister_post_action_callback,
    track_scroll_event,
    clear_scroll_tracking,
    trigger_post_action_hooks,
)
from .ui_feedback import (
    highlight_selector,
    highlight_overlay,
    highlight_box,
    highlight_point,
    highlight_click_location,
    clear_highlight,
)

# Bind callback methods to Executor class
Executor.register_pre_action_callback = register_pre_action_callback
Executor.unregister_pre_action_callback = unregister_pre_action_callback
Executor.unregister_post_action_callback = unregister_post_action_callback
Executor.track_scroll_event = track_scroll_event
Executor.clear_scroll_tracking = clear_scroll_tracking
Executor._trigger_post_action_hooks = trigger_post_action_hooks

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
