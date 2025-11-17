"""
Heuristic Stuck Detector - Detects when the agent is stuck or looping.

Uses lightweight heuristics to detect stuck behavior without requiring LLM calls.
When stuck is detected, the agent can be given a rewritten prompt based on
completion reasoning to help it refocus.
"""

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Dict, Any


@dataclass
class IterationSnapshot:
    """Snapshot of one iteration for stuck detection"""
    iteration: int
    action_text_norm: Optional[str]  # Normalized action text (lowercased, trimmed)
    action_type: Optional[str]  # Action type (CLICK, SCROLL, etc.)
    success: Optional[bool]  # Whether action succeeded
    url: str
    page_title: str
    url_changed: bool  # URL changed vs previous iteration
    dom_changed: bool  # DOM changed vs previous iteration
    extracted_items_count: int  # Count of extracted items (0 if not tracked)
    completion_reasoning: str  # Latest completion reasoning
    completion_confidence: float  # Latest completion confidence


@dataclass
class StuckStatus:
    """Result of stuck detection assessment"""
    is_stuck: bool
    score: float  # 0.0 to 1.0
    reason: str  # Human-readable explanation
    contributing_factors: List[str]  # List of specific issues detected


class HeuristicStuckDetector:
    """
    Heuristic-based stuck detector that analyzes iteration history.
    
    Detects patterns like:
    - Repeated same action
    - No page state changes
    - No progress on task metrics
    - Error spirals
    """
    
    def __init__(
        self,
        enabled: bool = True,
        window_size: int = 5,
        threshold: float = 0.6,
        weight_repeated_action: float = 0.15,
        weight_repetitive_action_no_change: float = 0.4,
        weight_no_state_change: float = 0.3,
        weight_no_progress: float = 0.2,
        weight_error_spiral: float = 0.2,
        weight_high_confidence_no_progress: float = 0.1,
    ):
        """
        Initialize stuck detector.
        
        Args:
            enabled: If False, detector always returns not stuck
            window_size: Number of recent iterations to analyze (default: 5)
            threshold: Score threshold to consider stuck (default: 0.6)
            weight_repeated_action: Weight for repeated action pattern (default: 0.15)
            weight_repetitive_action_no_change: Weight for repetitive actions without page change (default: 0.4)
            weight_no_state_change: Weight for no URL/DOM change pattern (default: 0.3)
            weight_no_progress: Weight for no progress on task metrics (default: 0.2)
            weight_error_spiral: Weight for error/failure spiral (default: 0.2)
            weight_high_confidence_no_progress: Weight for high confidence but no progress (default: 0.1)
        """
        self.enabled = enabled
        self.window_size = window_size
        self.threshold = threshold
        self.weight_repeated_action = weight_repeated_action
        self.weight_repetitive_action_no_change = weight_repetitive_action_no_change
        self.weight_no_state_change = weight_no_state_change
        self.weight_no_progress = weight_no_progress
        self.weight_error_spiral = weight_error_spiral
        self.weight_high_confidence_no_progress = weight_high_confidence_no_progress
        
        self.history: Deque[IterationSnapshot] = deque(maxlen=window_size)
    
    def add_iteration(self, snap: IterationSnapshot) -> None:
        """Add a new iteration snapshot to the history."""
        if self.enabled:
            self.history.append(snap)
    
    def assess(self) -> StuckStatus:
        """
        Assess if the agent is stuck based on recent history.
        
        Returns:
            StuckStatus with is_stuck, score, reason, and contributing factors
        """
        if not self.enabled:
            return StuckStatus(
                is_stuck=False,
                score=0.0,
                reason="Stuck detector disabled",
                contributing_factors=[]
            )
        
        h = list(self.history)
        if len(h) < 3:
            return StuckStatus(
                is_stuck=False,
                score=0.0,
                reason="Not enough history (need at least 3 iterations)",
                contributing_factors=[]
            )
        
        score = 0.0
        contributing_factors: List[str] = []
        
        # 1) Repeated same action text (last 3 iterations) - lower weight since repetition alone isn't necessarily stuck
        actions = [s.action_text_norm for s in h if s.action_text_norm]
        if len(actions) >= 3:
            last_3_actions = actions[-3:]
            if len(set(last_3_actions)) == 1:
                score += self.weight_repeated_action
                contributing_factors.append(
                    f"Same action repeated 3 times: '{last_3_actions[0]}'"
                )
        
        # 2) Repetitive actions without page change - actions that should change the page but don't
        # Check for page-changing action types (CLICK, TYPE, SCROLL, NAVIGATE, etc.) that didn't change the page
        page_changing_action_types = {"CLICK", "TYPE", "SCROLL", "NAVIGATE", "PRESS", "SELECT", "UPLOAD"}
        last_3 = h[-3:]
        repetitive_no_change_count = 0
        for snap in last_3:
            if snap.action_type and snap.action_type in page_changing_action_types:
                # This is a page-changing action
                if not snap.url_changed and not snap.dom_changed:
                    # Action should have changed page but didn't
                    repetitive_no_change_count += 1
        
        if repetitive_no_change_count >= 2:
            # At least 2 page-changing actions in last 3 iterations didn't change the page
            score += self.weight_repetitive_action_no_change
            contributing_factors.append(
                f"{repetitive_no_change_count} page-changing actions without page change in last 3 iterations"
            )
        
        # 3) No navigation / DOM change (last 3 iterations)
        last_3 = h[-3:]
        if all(not s.url_changed and not s.dom_changed for s in last_3):
            score += self.weight_no_state_change
            contributing_factors.append(
                "No URL or DOM change over last 3 iterations"
            )
        
        # 4) No extraction progress (if tracking extracted items)
        if len(h) >= 2:
            deltas = [
                h[i].extracted_items_count - h[i-1].extracted_items_count
                for i in range(1, len(h))
            ]
            # Check if we're tracking extractions and there's been no progress
            if h[0].extracted_items_count is not None and h[0].extracted_items_count >= 0:
                last_3_deltas = deltas[-3:] if len(deltas) >= 3 else deltas
                if all(d <= 0 for d in last_3_deltas) and any(h[i].extracted_items_count > 0 for i in range(max(0, len(h)-3), len(h))):
                    # We have extractions tracked but no increase
                    score += self.weight_no_progress
                    contributing_factors.append(
                        "No increase in extracted items over recent iterations"
                    )
        
        # 5) Error spiral (multiple failures in recent iterations)
        failures = [s for s in last_3 if s.success is False]
        if len(failures) >= 2:
            score += self.weight_error_spiral
            contributing_factors.append(
                f"{len(failures)} failures in last 3 iterations"
            )
        
        # 6) High completion confidence but still not complete
        # (indicates agent thinks it's doing the right thing but nothing changes)
        if len(h) >= 3:
            recent_confidences = [s.completion_confidence for s in last_3]
            avg_confidence = sum(recent_confidences) / len(recent_confidences)
            if avg_confidence >= 0.8 and score > 0.3:
                # High confidence but we're already seeing stuck signals
                score += self.weight_high_confidence_no_progress
                contributing_factors.append(
                    f"High completion confidence ({avg_confidence:.2f}) but no progress"
                )
        
        # Clamp score to [0, 1]
        score = min(score, 1.0)
        is_stuck = score >= self.threshold
        
        reason = "; ".join(contributing_factors) if contributing_factors else "No strong stuck signals detected"
        
        return StuckStatus(
            is_stuck=is_stuck,
            score=score,
            reason=reason,
            contributing_factors=contributing_factors
        )
    
    def clear(self) -> None:
        """Clear the history (useful for resetting between tasks)."""
        self.history.clear()
