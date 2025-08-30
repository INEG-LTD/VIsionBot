"""
Navigation Goal - Monitors page navigation and URL changes.
"""
from typing import List

from .base import BaseGoal, GoalResult, GoalStatus, EvaluationTiming, GoalContext


class NavigationGoal(BaseGoal):
    """
    Goal for page navigation tasks.
    
    This goal monitors URL changes and page content to determine if
    navigation to the correct page has occurred.
    """
    
    # NavigationGoal should be evaluated AFTER interaction since we need to see the result
    EVALUATION_TIMING = EvaluationTiming.AFTER
    
    def __init__(self, description: str, target_url_contains: List[str] = None, target_page_text: List[str] = None, **kwargs):
        super().__init__(description, **kwargs)
        self.target_url_contains = target_url_contains or []
        self.target_page_text = target_page_text or []
    
    def evaluate(self, context: GoalContext) -> GoalResult:
        """Evaluate if navigation goal has been achieved"""
        current_url = context.current_state.url.lower()
        current_text = (context.current_state.visible_text or "").lower()
        
        # Check URL contains target strings
        url_matches = []
        if self.target_url_contains:
            for target in self.target_url_contains:
                if target.lower() in current_url:
                    url_matches.append(target)
        
        # Check page contains target text
        text_matches = []
        if self.target_page_text:
            for target in self.target_page_text:
                if target.lower() in current_text:
                    text_matches.append(target)
        
        # Determine success
        url_success = not self.target_url_contains or len(url_matches) > 0
        text_success = not self.target_page_text or len(text_matches) > 0
        
        if url_success and text_success:
            return GoalResult(
                status=GoalStatus.ACHIEVED,
                confidence=0.9,
                reasoning=f"Navigation successful - URL matches: {url_matches}, Text matches: {text_matches}",
                evidence={
                    "current_url": current_url,
                    "url_matches": url_matches,
                    "text_matches": text_matches,
                    "evaluation_timing": "post_interaction"
                }
            )
        else:
            return GoalResult(
                status=GoalStatus.PENDING,
                confidence=0.8,
                reasoning=f"Navigation not complete - URL OK: {url_success}, Text OK: {text_success}",
                evidence={
                    "current_url": current_url,
                    "expected_url_contains": self.target_url_contains,
                    "expected_text_contains": self.target_page_text
                }
            )
