"""
Agent context structures used for decision-making.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Any


@dataclass
class EnvironmentState:
    """
    Comprehensive environment state snapshot for agent evaluation.
    Bundles all relevant state information for decision-making.
    """
    browser_state: Any
    interaction_history: List[Any]
    user_prompt: str
    task_start_url: str
    task_start_time: float
    current_url: str
    page_title: str
    visible_text: Optional[str] = None
    url_history: List[str] = field(default_factory=list)
    url_pointer: Optional[int] = None

    def __post_init__(self):
        if self.visible_text is None and hasattr(self.browser_state, 'visible_text'):
            self.visible_text = self.browser_state.visible_text[:2000] if self.browser_state.visible_text else ""
        if not self.url_history:
            self.url_history = []
        if self.url_pointer is None:
            self.url_pointer = len(self.url_history) - 1 if self.url_history else -1
