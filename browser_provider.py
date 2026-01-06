"""
Browser Provider Pattern for BrowserVisionBot.

This module provides an abstraction layer between the bot and browser implementations,
enabling dependency injection, easier testing, and support for remote browsers.

Example:
    >>> from browser_provider import LocalPlaywrightProvider, BrowserConfig
    >>> provider = LocalPlaywrightProvider(BrowserConfig(headless=True))
    >>> page = provider.get_page()
"""
from __future__ import annotations

import os
import uuid
from abc import ABC, abstractmethod
from typing import Optional
from playwright.sync_api import Page, Browser, Playwright, sync_playwright
# Compatible import for playwright_stealth across versions
try:
    from playwright_stealth import Stealth
    _stealth_instance = Stealth()

    def stealth_sync(page):
        return _stealth_instance.apply_stealth_sync(page)

except ImportError:
    # Fallback: disable stealth if not available
    def stealth_sync(page):
        pass  # No-op function
from pydantic import BaseModel, Field


class BrowserConfig(BaseModel):
    """Configuration for browser providers."""
    
    provider_type: str = Field(
        default="local",
        description="Browser provider type: 'local', 'remote', 'persistent', 'mock'"
    )
    
    # Local browser settings
    headless: bool = Field(
        default=False,
        description="Run browser in headless mode"
    )
    viewport_width: int = Field(
        default=1280,
        ge=100,
        description="Browser viewport width"
    )
    viewport_height: int = Field(
        default=800,
        ge=100,
        description="Browser viewport height"
    )
    user_data_dir: Optional[str] = Field(
        default=None,
        description="User data directory for persistent context"
    )
    channel: str = Field(
        default="chrome",
        description="Browser channel: 'chrome', 'chromium', 'firefox', etc."
    )
    
    # Remote browser settings
    remote_cdp_url: Optional[str] = Field(
        default=None,
        description="CDP endpoint URL for remote browser (e.g., Browserbase, Browserless)"
    )
    
    # Stealth settings
    apply_stealth: bool = Field(
        default=True,
        description="Apply stealth patches to avoid bot detection"
    )
    
    # Browser args
    extra_args: list[str] = Field(
        default_factory=lambda: [
            "--disable-dev-shm-usage",
            "--disable-blink-features=AutomationControlled",  # Required for Cloudflare bypass
            "--disable-features=VizDisplayCompositor",
            "--disable-infobars",
            "--disable-automation",
        ],
        description="Additional browser launch arguments"
    )
    
    class Config:
        arbitrary_types_allowed = True


class BrowserProvider(ABC):
    """
    Abstract base class for browser providers.
    
    Implementations must provide a way to get a Playwright Page object
    and handle cleanup.
    """
    
    def __init__(self, config: BrowserConfig):
        self.config = config
        self._page: Optional[Page] = None
        self._browser: Optional[Browser] = None
        self._playwright: Optional[Playwright] = None
    
    @abstractmethod
    def get_page(self) -> Page:
        """
        Get or create a Playwright Page object.
        
        Returns:
            Page: Playwright page ready for automation
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """
        Cleanup resources (close browser, stop playwright, etc.)
        """
        pass
    
    def is_ready(self) -> bool:
        """
        Check if the provider is ready to provide pages.
        
        Returns:
            bool: True if provider is ready
        """
        return self._page is not None and not self._page.is_closed()


class LocalPlaywrightProvider(BrowserProvider):
    """
    Default browser provider that launches a local Playwright browser.
    
    This is the standard implementation that creates a new browser instance
    with stealth patches and custom configuration.
    
    Example:
        >>> config = BrowserConfig(headless=True, viewport_width=1920)
        >>> provider = LocalPlaywrightProvider(config)
        >>> page = provider.get_page()
    """
    
    def get_page(self) -> Page:
        """Launch local browser and return page."""
        if self._page is not None and not self._page.is_closed():
            return self._page
        
        # Start Playwright
        self._playwright = sync_playwright().start()
        
        # Determine user data directory
        if self.config.user_data_dir:
            user_data_dir = self.config.user_data_dir
        else:
            # Create temporary profile
            user_data_dir = os.path.expanduser(
                f"~/Library/Application Support/Google/Chrome/Automation_{str(uuid.uuid4())[:8]}"
            )
        
        # Build browser args
        args = self.config.extra_args.copy()
        args.extend([
            f"--window-position=0,0",
            f"--window-size={self.config.viewport_width},{self.config.viewport_height}",
        ])

        # Launch persistent context
        self._browser = self._playwright.chromium.launch_persistent_context(
            viewport={
                "width": self.config.viewport_width,
                "height": self.config.viewport_height
            },
            user_data_dir=user_data_dir,
            headless=self.config.headless,
            args=args,
            channel=self.config.channel,
            ignore_default_args=["--enable-automation"],
            chromium_sandbox=True  # Keep sandbox enabled to avoid warning
        )
        
        # Get or create page
        pages = self._browser.pages
        if pages:
            self._page = pages[0]
        else:
            self._page = self._browser.new_page()
        
        # Apply stealth if enabled
        if self.config.apply_stealth:
            stealth_sync(self._page)
        
        return self._page
    
    def close(self) -> None:
        """Close browser and cleanup."""
        if self._browser:
            try:
                self._browser.close()
            except Exception:
                pass
            self._browser = None
        
        if self._playwright:
            try:
                self._playwright.stop()
            except Exception:
                pass
            self._playwright = None
        
        self._page = None


class RemoteBrowserProvider(BrowserProvider):
    """
    Browser provider that connects to a remote browser via CDP.
    
    Useful for services like Browserbase, Browserless, or custom remote browsers.
    
    Example:
        >>> config = BrowserConfig(
        ...     provider_type="remote",
        ...     remote_cdp_url="wss://connect.browserbase.com?apiKey=..."
        ... )
        >>> provider = RemoteBrowserProvider(config)
        >>> page = provider.get_page()
    """
    
    def get_page(self) -> Page:
        """Connect to remote browser and return page."""
        if self._page is not None and not self._page.is_closed():
            return self._page
        
        if not self.config.remote_cdp_url:
            raise ValueError("remote_cdp_url is required for RemoteBrowserProvider")
        
        # Start Playwright
        self._playwright = sync_playwright().start()
        
        # Connect to remote browser
        self._browser = self._playwright.chromium.connect_over_cdp(
            self.config.remote_cdp_url
        )
        
        # Get default context and page
        contexts = self._browser.contexts
        if contexts:
            context = contexts[0]
            pages = context.pages
            if pages:
                self._page = pages[0]
            else:
                self._page = context.new_page()
        else:
            # Create new context
            context = self._browser.new_context(
                viewport={
                    "width": self.config.viewport_width,
                    "height": self.config.viewport_height
                }
            )
            self._page = context.new_page()
        
        # Apply stealth if enabled
        if self.config.apply_stealth:
            stealth_sync(self._page)
        
        return self._page
    
    def close(self) -> None:
        """Disconnect from remote browser."""
        if self._browser:
            try:
                self._browser.close()
            except Exception:
                pass
            self._browser = None
        
        if self._playwright:
            try:
                self._playwright.stop()
            except Exception:
                pass
            self._playwright = None
        
        self._page = None


class PersistentContextProvider(BrowserProvider):
    """
    Browser provider that reuses an existing browser profile.
    
    Useful for maintaining sessions, cookies, and login state across runs.
    
    Example:
        >>> config = BrowserConfig(
        ...     provider_type="persistent",
        ...     user_data_dir="/path/to/chrome/profile"
        ... )
        >>> provider = PersistentContextProvider(config)
        >>> page = provider.get_page()
    """
    
    def get_page(self) -> Page:
        """Launch browser with persistent context."""
        if self._page is not None and not self._page.is_closed():
            return self._page
        
        if not self.config.user_data_dir:
            raise ValueError("user_data_dir is required for PersistentContextProvider")
        
        # Start Playwright
        self._playwright = sync_playwright().start()
        
        # Build browser args
        args = self.config.extra_args.copy()
        args.extend([
            f"--window-position=0,0",
            f"--window-size={self.config.viewport_width},{self.config.viewport_height}",
        ])

        # Launch persistent context
        self._browser = self._playwright.chromium.launch_persistent_context(
            viewport={
                "width": self.config.viewport_width,
                "height": self.config.viewport_height
            },
            user_data_dir=self.config.user_data_dir,
            headless=self.config.headless,
            args=args,
            channel=self.config.channel,
            ignore_default_args=["--enable-automation"],
            chromium_sandbox=True  # Keep sandbox enabled to avoid warning
        )
        
        # Get or create page
        pages = self._browser.pages
        if pages:
            self._page = pages[0]
        else:
            self._page = self._browser.new_page()
        
        # Apply stealth if enabled
        if self.config.apply_stealth:
            stealth_sync(self._page)
        
        return self._page
    
    def close(self) -> None:
        """Close browser but preserve profile."""
        if self._browser:
            try:
                self._browser.close()
            except Exception:
                pass
            self._browser = None
        
        if self._playwright:
            try:
                self._playwright.stop()
            except Exception:
                pass
            self._playwright = None
        
        self._page = None


class MockBrowserProvider(BrowserProvider):
    """
    Mock browser provider for testing.
    
    Returns a mock Page object that doesn't require an actual browser.
    Useful for unit tests and CI/CD pipelines.
    
    Example:
        >>> config = BrowserConfig(provider_type="mock")
        >>> provider = MockBrowserProvider(config)
        >>> page = provider.get_page()  # Returns mock page
    """
    
    def __init__(self, config: BrowserConfig, mock_page: Optional[Page] = None):
        super().__init__(config)
        self._mock_page = mock_page
    
    def get_page(self) -> Page:
        """Return mock page."""
        if self._mock_page:
            return self._mock_page
        
        # If no mock provided, raise error (user should provide mock in tests)
        raise NotImplementedError(
            "MockBrowserProvider requires a mock_page to be provided. "
            "Use: MockBrowserProvider(config, mock_page=your_mock)"
        )
    
    def close(self) -> None:
        """No-op for mock provider."""
        pass


def create_browser_provider(config: BrowserConfig) -> BrowserProvider:
    """
    Factory function to create appropriate browser provider from config.
    
    Args:
        config: Browser configuration
        
    Returns:
        BrowserProvider: Appropriate provider implementation
        
    Example:
        >>> config = BrowserConfig(provider_type="local", headless=True)
        >>> provider = create_browser_provider(config)
    """
    if config.provider_type == "local":
        return LocalPlaywrightProvider(config)
    elif config.provider_type == "remote":
        return RemoteBrowserProvider(config)
    elif config.provider_type == "persistent":
        return PersistentContextProvider(config)
    elif config.provider_type == "mock":
        return MockBrowserProvider(config)
    else:
        raise ValueError(
            f"Unknown provider_type: {config.provider_type}. "
            f"Must be one of: local, remote, persistent, mock"
        )
