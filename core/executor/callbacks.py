"""
Callback system for pre- and post-action hooks.
"""
from typing import Callable, Optional, Tuple

from models import ActionType, ActionStep, PageInfo, PageElements
from .base import PreActionContext, PostActionContext, ScrollReason

def clear_scroll_tracking(executor) -> None:
    """Clear scroll tracking after it's been passed to callbacks"""
    executor.last_scroll_occurred = False
    executor.last_scroll_reason = None


def unregister_post_action_callback(executor, callback: Callable[[PostActionContext], None]) -> None:
    """Remove a registered callback"""
    if callback in executor.post_action_callbacks:
        executor.post_action_callbacks.remove(callback)
        try:
            executor.event_logger.system_debug(f"Unregistered post-action callback: {callback.__name__}")
        except Exception:
            pass


def trigger_pre_action_hooks(
    executor,
    action_type: ActionType,
    step: ActionStep,
    page_info: PageInfo,
    elements: PageElements,
    coordinates: Optional[Tuple[int, int]] = None,
) -> None:
    """Execute all registered pre-action callbacks with full context"""
    if not executor.pre_action_callbacks:
        return

    # Action ID and lineage tracking removed (was ActionLedger)
    action_id = None
    action_lineage = None

    context = PreActionContext(
        action_type=action_type,
        step=step,
        page_info=page_info,
        elements=elements,
        coordinates=coordinates,
        page=executor.page,
        action_id=action_id,
        action_lineage=action_lineage,
    )

    for callback in executor.pre_action_callbacks:
        try:
            callback(context)
        except Exception as e:
            try:
                executor.event_logger.system_error(f"Pre-action callback '{callback.__name__}' error", error=e)
            except Exception:
                pass


def trigger_post_action_hooks(
    executor,
    action_type: ActionType,
    success: bool,
    step: ActionStep,
    page_info: PageInfo,
    elements: PageElements,
    coordinates: Optional[Tuple[int, int]] = None,
    error_message: Optional[str] = None,
    action_id: Optional[str] = None,
) -> None:
    """Execute all registered post-action callbacks with full context"""
    if not executor.post_action_callbacks:
        return

    # Action lineage tracking removed (was ActionLedger)
    action_lineage = None

    # Determine scroll information
    # For scroll actions, always mark as occurred with USER_ACTION reason
    # For other actions, use tracked scroll info from automatic scrolls
    if action_type == ActionType.SCROLL:
        scroll_occurred = True
        scroll_reason = ScrollReason.USER_ACTION
    else:
        scroll_occurred = executor.last_scroll_occurred
        scroll_reason = executor.last_scroll_reason

    context = PostActionContext(
        action_type=action_type,
        success=success,
        step=step,
        page_info=page_info,
        elements=elements,
        coordinates=coordinates,
        error_message=error_message,
        page=executor.page,
        action_id=action_id,
        action_lineage=action_lineage,
        scroll_occurred=scroll_occurred,
        scroll_reason=scroll_reason,
    )

    # Clear scroll tracking after passing to callbacks
    clear_scroll_tracking(executor)

    for callback in executor.post_action_callbacks:
        try:
            callback(context)
        except Exception as e:
            try:
                executor.event_logger.system_error(f"Post-action callback '{callback.__name__}' error", error=e)
            except Exception:
                pass
