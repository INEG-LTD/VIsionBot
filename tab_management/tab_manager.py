"""
TabManager - Manages browser tabs and their lifecycle.
"""
import time
from typing import Dict, List, Optional, Callable, Any
from playwright.sync_api import BrowserContext, Page
import uuid

from .tab_info import TabInfo
from utils.event_logger import get_event_logger


class TabManager:
    """
    Manages all browser tabs and their lifecycle.
    
    Responsibilities:
    - Register and track tabs
    - Manage active tab
    - Handle tab switching
    - Detect new tabs
    - Coordinate tab closing
    """
    
    def __init__(self, browser_context: BrowserContext):
        """
        Initialize TabManager.
        
        Args:
            browser_context: Playwright BrowserContext to manage tabs for
        """
        self.browser_context = browser_context
        self.tabs: Dict[str, TabInfo] = {}  # tab_id -> TabInfo
        self.active_tab_id: Optional[str] = None
        self.tab_creation_listener: Optional[Callable[[str], None]] = None
        self._known_pages: set = set()  # Track pages we've seen to detect new ones
    
    def register_tab(
        self,
        page: Page,
        purpose: str,
        agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Register a new tab and return its ID.
        
        If the page is already registered, returns the existing tab_id and updates metadata.
        
        Args:
            page: Playwright Page object
            purpose: Purpose/description of this tab
            agent_id: Optional agent ID that owns this tab
            metadata: Optional additional metadata
        
        Returns:
            Unique tab ID (existing if page already registered, new otherwise)
        """
        page_id = id(page)
        
        # Check if this page is already registered
        if page_id in self._known_pages:
            # Find existing tab for this page
            for tab_id, tab_info in self.tabs.items():
                if id(tab_info.page) == page_id:
                    # Update purpose if it was auto_detected or if explicitly provided
                    if purpose != "auto_detected" or tab_info.purpose == "auto_detected":
                        if tab_info.purpose == "auto_detected" and purpose != "auto_detected":
                            tab_info.purpose = purpose
                    if agent_id:
                        tab_info.agent_id = agent_id
                    if metadata:
                        tab_info.metadata.update(metadata)
                    
                    # Refresh URL and title
                    try:
                        tab_info.url = page.url
                        tab_info.title = page.title()
                    except Exception:
                        pass
                    
                    try:
                        get_event_logger().tab_registered(tab_id=tab_id, purpose=tab_info.purpose, url=tab_info.url)
                    except Exception:
                        pass
                    return tab_id
        
        # Generate unique tab ID for new page
        tab_id = f"tab_{uuid.uuid4().hex[:12]}"
        
        # Get current URL and title
        try:
            url = page.url
            title = page.title()
        except Exception:
            url = "about:blank"
            title = "New Tab"
        
        # Create TabInfo
        tab_info = TabInfo(
            page=page,
            tab_id=tab_id,
            url=url,
            title=title,
            purpose=purpose,
            agent_id=agent_id,
            metadata=metadata or {}
        )
        
        # Register tab
        self.tabs[tab_id] = tab_info
        self._known_pages.add(page_id)
        
        # Set as active if it's the first tab
        if self.active_tab_id is None:
            self.active_tab_id = tab_id
        
        try:
            get_event_logger().tab_registered(tab_id=tab_id, purpose=purpose, url=url)
        except Exception:
            pass
        
        return tab_id
    
    def get_tab_info(self, tab_id: str) -> Optional[TabInfo]:
        """
        Get tab metadata by ID.
        
        Args:
            tab_id: Tab ID to look up
        
        Returns:
            TabInfo if found, None otherwise
        """
        return self.tabs.get(tab_id)
    
    def list_tabs(self, agent_id: Optional[str] = None) -> List[TabInfo]:
        """
        List all tabs, optionally filtered by agent.
        
        Args:
            agent_id: Optional agent ID to filter by
        
        Returns:
            List of TabInfo objects
        """
        if agent_id is None:
            return list(self.tabs.values())
        
        return [tab for tab in self.tabs.values() if tab.agent_id == agent_id]
    
    def get_active_tab(self) -> Optional[TabInfo]:
        """Get the currently active tab"""
        if self.active_tab_id is None:
            return None
        return self.tabs.get(self.active_tab_id)
    
    def switch_to_tab(self, tab_id: str) -> bool:
        """
        Switch to a different tab.
        
        Args:
            tab_id: Tab ID to switch to
        
        Returns:
            True if switch successful, False otherwise
        """
        if tab_id not in self.tabs:
            print(f"⚠️ Tab not found: {tab_id}")
            return False
        
        tab_info = self.tabs[tab_id]
        
        # Bring page to front in the browser
        try:
            tab_info.page.bring_to_front()
        except Exception as e:
            print(f"⚠️ Could not bring tab to front: {e}")
        
        # Update tab info with current URL/title
        try:
            tab_info.update_url(tab_info.page.url)
            tab_info.update_title(tab_info.page.title())
        except Exception:
            pass
        
        # Update last accessed
        tab_info.update_access()
        
        # Update active tab
        old_active = self.active_tab_id
        self.active_tab_id = tab_id
        
        if old_active != tab_id:
            print(f"🔀 Switched to tab: {tab_id} ({tab_info.purpose}) - {tab_info.url}")
        
        return True
    
    def close_tab(
        self,
        tab_id: str,
        switch_to: Optional[str] = None
    ) -> bool:
        """
        Close a tab, optionally switching to another first.
        
        Args:
            tab_id: Tab ID to close
            switch_to: Optional tab ID to switch to before closing
        
        Returns:
            True if close successful, False otherwise
        """
        if tab_id not in self.tabs:
            print(f"⚠️ Tab not found: {tab_id}")
            return False
        
        tab_info = self.tabs[tab_id]
        
        # Switch to another tab if specified
        if switch_to and switch_to in self.tabs and switch_to != tab_id:
            self.switch_to_tab(switch_to)
        elif self.active_tab_id == tab_id:
            # If closing active tab, switch to another if available
            other_tabs = [t for t in self.tabs.keys() if t != tab_id]
            if other_tabs:
                self.switch_to_tab(other_tabs[0])
            else:
                self.active_tab_id = None
        
        # Close the page
        try:
            tab_info.page.close()
        except Exception as e:
            print(f"⚠️ Error closing page: {e}")
        
        # Remove from registry
        del self.tabs[tab_id]
        self._known_pages.discard(id(tab_info.page))
        
        print(f"🗑️ Closed tab: {tab_id} ({tab_info.purpose})")
        
        return True
    
    def open_new_tab(
        self,
        purpose: str,
        url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Open a brand-new browser tab, register it, and optionally navigate to a URL.
        
        Args:
            purpose: Purpose/description for the new tab
            url: Optional URL to navigate to immediately after opening
            metadata: Optional metadata to attach to the TabInfo
        """
        try:
            page = self.browser_context.new_page()
        except Exception as e:
            print(f"⚠️ Failed to open new tab: {e}")
            return None
        
        tab_metadata = metadata.copy() if metadata else {}
        tab_id = self.register_tab(page=page, purpose=purpose, metadata=tab_metadata)
        
        if url:
            try:
                page.goto(url)
            except Exception as e:
                print(f"⚠️ Failed to navigate new tab {tab_id} to {url}: {e}")
        
        # Switch focus to the new tab
        self.switch_to_tab(tab_id)
        
        return tab_id
    
    def detect_new_tab(self, page: Page) -> Optional[str]:
        """
        Detect when a new tab is opened (e.g., from button click).
        
        Args:
            page: Newly opened Page object
        
        Returns:
            Tab ID if new tab detected and registered, None if already known
        """
        page_id = id(page)
        
        # Check if we've seen this page before
        if page_id in self._known_pages:
            # Find existing tab for this page
            for tab_id, tab_info in self.tabs.items():
                if id(tab_info.page) == page_id:
                    return tab_id
            return None
        
        # New page detected
        try:
            url = page.url
            title = page.title()
        except Exception:
            url = "about:blank"
            title = "New Tab"
        
        # Auto-register with generic purpose (can be updated later)
        tab_id = self.register_tab(
            page=page,
            purpose="auto_detected",
            agent_id=None,
            metadata={"auto_detected": True}
        )
        
        # Notify listener if set
        if self.tab_creation_listener:
            try:
                self.tab_creation_listener(tab_id)
            except Exception as e:
                print(f"⚠️ Error in tab creation listener: {e}")
        
        print(f"🔍 Detected new tab: {tab_id} - {url}")
        
        return tab_id
    
    def mark_tab_completed(self, tab_id: str) -> bool:
        """
        Mark a tab as completed.
        
        Args:
            tab_id: Tab ID to mark as completed
        
        Returns:
            True if successful, False otherwise
        """
        if tab_id not in self.tabs:
            return False
        
        self.tabs[tab_id].mark_completed()
        return True
    
    def update_tab_info(self, tab_id: str, **kwargs) -> bool:
        """
        Update tab information.
        
        Args:
            tab_id: Tab ID to update
            **kwargs: Fields to update (purpose, agent_id, metadata, etc.)
        
        Returns:
            True if successful, False otherwise
        """
        if tab_id not in self.tabs:
            return False
        
        tab_info = self.tabs[tab_id]
        
        # Update allowed fields
        if "purpose" in kwargs:
            tab_info.purpose = kwargs["purpose"]
        if "agent_id" in kwargs:
            tab_info.agent_id = kwargs["agent_id"]
        if "metadata" in kwargs:
            tab_info.metadata.update(kwargs["metadata"])
        if "url" in kwargs:
            tab_info.update_url(kwargs["url"])
        if "title" in kwargs:
            tab_info.update_title(kwargs["title"])
        
        return True
    
    def get_tabs_by_purpose(self, purpose: str) -> List[TabInfo]:
        """
        Get all tabs with a specific purpose.
        
        Args:
            purpose: Purpose to filter by
        
        Returns:
            List of TabInfo objects
        """
        return [tab for tab in self.tabs.values() if tab.purpose == purpose]
    
    def cleanup_orphaned_tabs(self) -> int:
        """
        Clean up tabs whose pages have been closed.
        
        Returns:
            Number of tabs cleaned up
        """
        orphaned = []
        
        for tab_id, tab_info in self.tabs.items():
            try:
                # Try to access page URL to check if still valid
                _ = tab_info.page.url
            except Exception:
                # Page is closed
                orphaned.append(tab_id)
        
        for tab_id in orphaned:
            del self.tabs[tab_id]
            if self.active_tab_id == tab_id:
                # Switch to another tab if available
                other_tabs = [t for t in self.tabs.keys() if t != tab_id]
                self.active_tab_id = other_tabs[0] if other_tabs else None
        
        if orphaned:
            print(f"🧹 Cleaned up {len(orphaned)} orphaned tab(s)")
        
        return len(orphaned)

