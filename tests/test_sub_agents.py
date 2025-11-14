import pytest
from types import SimpleNamespace

from agent.agent_controller import AgentController, SubAgentPolicyLevel


class DummyPage:
    def __init__(self, url: str = "https://example.com", title: str = "Example"):
        self.url = url
        self._title = title

    def title(self) -> str:
        return self._title


class DummyTab:
    def __init__(self, tab_id: str, purpose: str, agent_id: str = None, completed: bool = False):
        self.tab_id = tab_id
        self.purpose = purpose
        self.agent_id = agent_id
        self.is_completed = completed
        self.url = f"https://example.com/{tab_id}"
        self._title = f"Tab {tab_id}"
        self.page = DummyPage(url=self.url, title=self._title)

    def update_url(self, url: str) -> None:
        self.url = url

    def update_title(self, title: str) -> None:
        self._title = title


class DummyTabManager:
    def __init__(self, tabs, active_tab_id: str):
        self._tabs = {tab.tab_id: tab for tab in tabs}
        self._active_tab_id = active_tab_id

    def list_tabs(self):
        return list(self._tabs.values())

    def get_active_tab(self):
        return self._tabs[self._active_tab_id]

    def get_tab_info(self, tab_id: str):
        return self._tabs.get(tab_id)

    def switch_to_tab(self, tab_id: str) -> bool:  # pragma: no cover - not used but expected by controller
        if tab_id in self._tabs:
            self._active_tab_id = tab_id
            return True
        return False

    def close_tab(self, tab_id: str, switch_to: str = None) -> bool:  # pragma: no cover
        if tab_id in self._tabs:
            del self._tabs[tab_id]
            if switch_to and switch_to in self._tabs:
                self._active_tab_id = switch_to
            return True
        return False


class DummySubAgentController:
    def __init__(self, history=None):
        self._history = history or []
        self.sub_agents = {}

    def get_execution_history(self):
        return list(self._history)


class DummyBot:
    def __init__(self, tab_manager=None):
        self.tab_manager = tab_manager
        self.started = True
        self.page = DummyPage()
        self.goal_monitor = SimpleNamespace(url_history=[], url_pointer=None, interaction_history=[])

    def switch_to_page(self, page):  # pragma: no cover
        self.page = page


@pytest.fixture
def controller_without_tabs():
    bot = DummyBot(tab_manager=None)
    controller = AgentController(bot)
    controller.sub_agent_controller = None
    return controller


@pytest.fixture
def controller_with_tabs():
    tabs = [
        DummyTab("main", "Main workflow"),
        DummyTab("research", "Research workspace"),
        DummyTab("alt", "Alternate source"),
    ]
    manager = DummyTabManager(tabs, active_tab_id="main")
    bot = DummyBot(tab_manager=manager)
    controller = AgentController(bot)
    controller.sub_agent_controller = DummySubAgentController()
    controller._query_sub_agent_policy_llm = lambda *args, **kwargs: (
        "single_threaded",
        0.0,
        "Default single-threaded policy for tests."
    )
    return controller


def test_policy_single_when_controller_unavailable(controller_without_tabs):
    level, score, rationale = controller_without_tabs._compute_sub_agent_policy("do something", "")
    assert level == SubAgentPolicyLevel.SINGLE_THREADED
    assert score <= 0.1
    assert "unavailable" in rationale.lower()


def test_policy_override_respected(controller_with_tabs):
    controller = controller_with_tabs
    controller._sub_agent_policy_override = SubAgentPolicyLevel.PARALLELIZED
    controller._query_sub_agent_policy_llm = lambda *args, **kwargs: ("single_threaded", 0.2, "LLM would prefer single, but override wins.")
    level, score, rationale = controller._compute_sub_agent_policy("any task", "")
    assert level == SubAgentPolicyLevel.PARALLELIZED
    assert score == pytest.approx(1.0)
    assert "override" in rationale.lower()


def test_policy_promotes_parallel_for_research_tasks(controller_with_tabs):
    controller = controller_with_tabs
    controller._sub_agent_policy_override = None
    controller._query_sub_agent_policy_llm = lambda *args, **kwargs: ("parallelized", 0.85, "LLM requests aggressive delegation.")
    level, score, rationale = controller._compute_sub_agent_policy(
        "research market trends and gather sources",
        "blocked by login, need parallel lookup"
    )
    assert level == SubAgentPolicyLevel.PARALLELIZED
    assert score >= 0.7
    assert "delegation" in rationale.lower() or "parallel" in rationale.lower()


def test_policy_prefers_single_for_focus_tasks(controller_with_tabs):
    controller = controller_with_tabs
    # Mark secondary tabs as already assigned/completed to remove unassigned options
    controller.bot.tab_manager._tabs["research"].agent_id = "sub_123"
    controller.bot.tab_manager._tabs["alt"].agent_id = "sub_456"
    controller.bot.tab_manager._tabs["alt"].is_completed = False
    controller._query_sub_agent_policy_llm = lambda *args, **kwargs: ("single_threaded", 0.3, "LLM recommends staying focused.")
    level, score, rationale = controller._compute_sub_agent_policy(
        "fill out the onboarding form and submit it",
        "waiting for user data entry"
    )
    assert level == SubAgentPolicyLevel.SINGLE_THREADED
    assert score <= 0.31
    assert "focused" in rationale.lower() or "stay" in rationale.lower()
