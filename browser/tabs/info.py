"""
TabInfo - Metadata about a browser tab.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import time
from playwright.sync_api import Page


@dataclass
class TabInfo:
    """
    Metadata about a browser tab.
    
    Attributes:
        page: Playwright Page object for this tab
        tab_id: Unique identifier for this tab
        url: Current URL of the tab
        title: Current title of the tab
        purpose: Purpose/description of this tab (e.g., "linkedin_profile", "main", "search")
        agent_id: ID of the agent that owns this tab (None if unassigned)
        created_at: Timestamp when tab was created
        last_accessed: Timestamp when tab was last accessed
        metadata: Additional context/metadata about the tab
        is_completed: Whether the tab's purpose has been completed
    """
    page: Page
    tab_id: str
    url: str
    title: str
    purpose: str
    agent_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_completed: bool = False
    
    def __post_init__(self):
        """Validate and initialize tab info"""
        if not self.tab_id:
            raise ValueError("tab_id is required")
        if not self.purpose:
            raise ValueError("purpose is required")
        if self.created_at == 0:
            self.created_at = time.time()
        if self.last_accessed == 0:
            self.last_accessed = time.time()
    
    def update_access(self) -> None:
        """Update last_accessed timestamp"""
        self.last_accessed = time.time()
    
    def mark_completed(self) -> None:
        """Mark this tab as completed"""
        self.is_completed = True
    
    def update_url(self, url: str) -> None:
        """Update the URL and refresh last_accessed"""
        self.url = url
        self.update_access()
    
    def update_title(self, title: str) -> None:
        """Update the title"""
        self.title = title
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding Page object)"""
        return {
            "tab_id": self.tab_id,
            "url": self.url,
            "title": self.title,
            "purpose": self.purpose,
            "agent_id": self.agent_id,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "metadata": self.metadata,
            "is_completed": self.is_completed
        }

