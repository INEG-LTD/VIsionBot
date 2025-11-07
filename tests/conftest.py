"""
Shared pytest fixtures for all tests.
"""
import pytest
from unittest.mock import Mock, MagicMock
from playwright.sync_api import sync_playwright


@pytest.fixture(scope="session")
def browser_context():
    """Shared browser context for all tests"""
    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context(viewport={"width": 1280, "height": 800})
    yield context
    browser.close()
    playwright.stop()


@pytest.fixture
def mock_page():
    """Mock Playwright Page object"""
    page = Mock()
    page.url = "https://example.com"
    page.title.return_value = "Example Page"
    page.goto = Mock()
    page.screenshot.return_value = b"fake_screenshot"
    page.close = Mock()
    return page


@pytest.fixture
def mock_browser_context():
    """Mock browser context"""
    context = Mock()
    context.new_page.return_value = Mock()
    return context


@pytest.fixture
def tab_manager_factory(browser_context):
    """Factory for creating TabManager instances"""
    def _create():
        from tab_management import TabManager
        return TabManager(browser_context)
    return _create

