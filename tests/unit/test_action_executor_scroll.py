from types import SimpleNamespace

import pytest

from models.core_models import ActionStep, ActionType, PageInfo
from action_executor import ActionExecutor
from session_tracker import BrowserState, InteractionType


class FakePage:
    def __init__(self):
        self.url = "https://example.com"
        self._scroll_x = 0
        self._scroll_y = 0
        self.title_value = "Example"
        self.scroll_by_calls: list[tuple[int, int]] = []
        self.mouse = SimpleNamespace(click=lambda *args, **kwargs: None)
        self.viewport_size = {"width": 1280, "height": 720}

    def title(self):
        return self.title_value

    def evaluate(self, script: str, *args, **kwargs):
        normalized = script.strip()
        if "window.scrollBy(" in normalized:
            coords = normalized.split("window.scrollBy(", 1)[-1].split(")", 1)[0]
            parts = [p.strip() for p in coords.split(",")]
            if len(parts) >= 2:
                delta_x = int(float(parts[0]))
                delta_y = int(float(parts[1]))
                self._scroll_x += delta_x
                self._scroll_y += delta_y
                self.scroll_by_calls.append((delta_x, delta_y))
            return None

        if "scrollX: window.scrollX" in normalized:
            return {"scrollX": self._scroll_x, "scrollY": self._scroll_y}

        if "window.pageXOffset" in normalized and "window.pageYOffset" not in normalized:
            return self._scroll_x

        if "window.pageYOffset" in normalized:
            return self._scroll_y

        return 0


class DummySessionTracker:
    def __init__(self):
        self.recorded: list[dict] = []

    def _capture_current_state(self, include_screenshot=False):
        return BrowserState(
            timestamp=0,
            url="https://example.com",
            title="Example",
            page_width=1280,
            page_height=720,
            scroll_x=0,
            scroll_y=0,
        )

    def record_interaction(self, interaction_type: InteractionType, **kwargs):
        entry = {
            "interaction_type": interaction_type,
            **kwargs,
        }
        self.recorded.append(entry)


class DummyPageUtils:
    def __init__(self, page: FakePage):
        self.page = page
        self.last_scroll_x = 0
        self.last_scroll_y = 0

    def get_page_info(self) -> PageInfo:
        width = self.page.viewport_size["width"]
        height = self.page.viewport_size["height"]
        return PageInfo(
            width=width,
            height=height,
            scroll_x=self.page._scroll_x,
            scroll_y=self.page._scroll_y,
            url=self.page.url,
            title=self.page.title(),
            dpr=1.0,
            ss_pixel_w=width,
            ss_pixel_h=height,
            css_scale=1.0,
            doc_width=width,
            doc_height=height,
        )


class DummyUploadHandler:
    def __init__(self, page, user_messages_config=None):
        pass


class DummySelectorUtils:
    def __init__(self, page):
        pass


class DummyContextGuard:
    def __init__(self, page, _):
        pass

    @staticmethod
    def is_guarded_action(action):
        return False

    def validate(self, **kwargs):
        class Decision:
            passed = True

        return Decision()


class DummyInteractionDeduper:
    def __init__(self):
        pass


class DummyActionLedger:
    def complete_action(self, **kwargs):
        pass


class DummyEventLogger:
    def __init__(self):
        self.debug_mode = False

    def __getattr__(self, name):
        def _noop(*args, **kwargs):
            pass

        return _noop


@pytest.fixture(autouse=True)
def stub_dependencies(monkeypatch):
    import action_executor

    monkeypatch.setattr(action_executor, "UploadHandler", DummyUploadHandler)
    monkeypatch.setattr(action_executor, "SelectorUtils", DummySelectorUtils)
    monkeypatch.setattr(action_executor, "ContextGuard", DummyContextGuard)
    monkeypatch.setattr(action_executor, "InteractionDeduper", DummyInteractionDeduper)
    monkeypatch.setattr(action_executor, "ActionLedger", DummyActionLedger)
    monkeypatch.setattr("utils.event_logger.get_event_logger", lambda: DummyEventLogger())

    yield


def build_executor(fake_page: FakePage, session_tracker: DummySessionTracker):
    page_utils = DummyPageUtils(fake_page)
    return ActionExecutor(
        page=fake_page,
        session_tracker=session_tracker,
        page_utils=page_utils,
        preferred_click_method="programmatic",
    ), page_utils


def test_scroll_down_falls_back_to_window_scroll():
    fake_page = FakePage()
    session_tracker = DummySessionTracker()
    executor, page_utils = build_executor(fake_page, session_tracker)

    step = ActionStep(action=ActionType.SCROLL, scroll_direction="down")
    success = executor._execute_scroll(step)

    assert success
    assert fake_page.scroll_by_calls[-1] == (0, 300)
    last_interaction = session_tracker.recorded[-1]
    assert last_interaction["scroll_direction"] == "down"
    assert last_interaction["target_y"] == 300
    assert page_utils.last_scroll_y == 300


def test_scroll_up_moves_page():
    fake_page = FakePage()
    fake_page._scroll_y = 500
    session_tracker = DummySessionTracker()
    executor, page_utils = build_executor(fake_page, session_tracker)

    step = ActionStep(action=ActionType.SCROLL, scroll_direction="up")
    success = executor._execute_scroll(step)

    assert success
    assert fake_page.scroll_by_calls[-1] == (0, -300)
    assert fake_page._scroll_y == 200
    last_interaction = session_tracker.recorded[-1]
    assert last_interaction["scroll_direction"] == "up"
    assert last_interaction["target_y"] == 200
    assert page_utils.last_scroll_y == 200
