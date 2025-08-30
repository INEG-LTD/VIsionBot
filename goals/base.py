"""
Base classes and types for the goal framework.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# =============================================================================
# Core Framework Types
# =============================================================================

class GoalStatus(str, Enum):
    """Status of goal evaluation"""
    PENDING = "pending"        # Goal not yet achieved
    ACHIEVED = "achieved"      # Goal successfully completed
    FAILED = "failed"          # Goal cannot be achieved
    UNKNOWN = "unknown"        # Cannot determine status


class InteractionType(str, Enum):
    """Types of browser interactions"""
    CLICK = "click"
    TYPE = "type"
    SCROLL = "scroll"
    NAVIGATION = "navigation"
    PAGE_LOAD = "page_load"
    ELEMENT_APPEAR = "element_appear"
    ELEMENT_DISAPPEAR = "element_disappear"


class EvaluationTiming(str, Enum):
    """When a goal should be evaluated relative to interactions"""
    BEFORE = "before"           # Evaluate before interaction occurs
    AFTER = "after"             # Evaluate after interaction completes
    BOTH = "both"               # Evaluate both before and after
    CONTINUOUS = "continuous"   # Evaluate continuously (polling-based)


@dataclass
class BrowserState:
    """Comprehensive snapshot of browser state at a point in time"""
    timestamp: float
    url: str
    title: str
    page_width: int
    page_height: int
    scroll_x: int
    scroll_y: int
    screenshot: Optional[bytes] = None
    dom_snapshot: Optional[str] = None
    visible_text: Optional[str] = None
    page_source: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()


@dataclass
class Interaction:
    """Record of a browser interaction"""
    timestamp: float
    interaction_type: InteractionType
    coordinates: Optional[tuple[int, int]] = None
    target_element_info: Optional[Dict[str, Any]] = None
    text_input: Optional[str] = None
    before_state: Optional[BrowserState] = None
    after_state: Optional[BrowserState] = None
    success: bool = True
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()


@dataclass
class GoalContext:
    """Complete context provided to goals for evaluation"""
    initial_state: BrowserState
    current_state: BrowserState
    all_interactions: List[Interaction] = field(default_factory=list)
    url_history: List[str] = field(default_factory=list)
    page_changes: List[BrowserState] = field(default_factory=list)
    detected_elements: List[Any] = field(default_factory=list)  # Current page elements
    session_duration: float = 0.0
    # For pre-interaction evaluation
    planned_interaction: Optional[Dict[str, Any]] = None


class GoalResult(BaseModel):
    """Result of goal evaluation"""
    status: GoalStatus
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the assessment")
    reasoning: str = Field(description="Explanation of why this status was determined")
    evidence: Dict[str, Any] = Field(default_factory=dict, description="Supporting evidence")
    next_actions: List[str] = Field(default_factory=list, description="Suggested next steps")


# =============================================================================
# Abstract Base Goal Class
# =============================================================================

class BaseGoal(ABC):
    """
    Abstract base class for all goal types.
    
    Goals monitor browser interactions and state changes to determine
    when a user's objective has been achieved.
    """
    
    # Subclasses should override this to specify when they want to be evaluated
    EVALUATION_TIMING: EvaluationTiming = EvaluationTiming.AFTER
    
    def __init__(self, description: str, **kwargs):
        self.description = description
        self.created_at = time.time()
        self.kwargs = kwargs
        self._monitoring_active = False
        self._last_evaluation: Optional[GoalResult] = None
    
    @abstractmethod
    def evaluate(self, context: GoalContext) -> GoalResult:
        """
        Evaluate whether this goal has been achieved given the current context.
        
        Args:
            context: Complete browser state and interaction history
            
        Returns:
            GoalResult with status and reasoning
        """
        pass
    
    def start_monitoring(self) -> None:
        """Called when goal monitoring begins"""
        self._monitoring_active = True
        self.on_monitoring_start()
    
    def stop_monitoring(self) -> None:
        """Called when goal monitoring ends"""
        self._monitoring_active = False
        self.on_monitoring_stop()
    
    def on_monitoring_start(self) -> None:
        """Override to perform setup when monitoring starts"""
        pass
    
    def on_monitoring_stop(self) -> None:
        """Override to perform cleanup when monitoring stops"""
        pass
    
    def on_interaction(self, interaction: Interaction) -> None:
        """
        Called after each browser interaction.
        Override to perform goal-specific tracking.
        """
        pass
    
    def on_state_change(self, old_state: BrowserState, new_state: BrowserState) -> None:
        """
        Called when browser state changes significantly.
        Override to track state-dependent goals.
        """
        pass
    
    @property
    def is_monitoring(self) -> bool:
        return self._monitoring_active
    
    @property
    def last_evaluation(self) -> Optional[GoalResult]:
        return self._last_evaluation
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}('{self.description}')"
