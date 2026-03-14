from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_tool_module():
    return import_module("job_application_app.process_focused_job_tool")


class _FakePage:
    def __init__(self, *, url: str, title: str, search_box_value: str = ""):
        self.url = url
        self._title = title
        self._search_box_value = search_box_value

    def title(self) -> str:
        return self._title

    def evaluate(self, script: str):
        return self._search_box_value


def test_process_focused_job_detects_search_context_mismatch_before_extraction() -> None:
    tool_module = _load_tool_module()
    page = _FakePage(
        url="https://www.google.com/search?q=Jobs%20at%20Jobster&udm=8",
        title="Jobs at Jobster - Google Search",
        search_box_value="Jobs at Jobster",
    )
    ctx = SimpleNamespace(
        page=page,
        memory_store=None,
        event_logger=None,
        sandbox_policy=None,
        runtime_state=None,
    )
    args = tool_module.ProcessFocusedJobArgs(
        job_profile="junior it developer",
        search_query="junior it developer in england",
        reasoning="Verify the currently focused job is on the original search results page.",
    )

    original_extract = tool_module._extract_focused_job
    try:
        def _unexpected_extract(*_args, **_kwargs):
            raise AssertionError("_extract_focused_job should not run after search context mismatch")

        tool_module._extract_focused_job = _unexpected_extract
        outcome = tool_module.process_focused_job(ctx, args)
    finally:
        tool_module._extract_focused_job = original_extract

    assert outcome.output.success is True
    assert outcome.output.data["reason"] == "search_context_mismatch"
    assert outcome.output.data["search_context_valid"] is False
    assert outcome.output.data["expected_search_query"] == "junior it developer in england"
    assert outcome.output.data["observed_search_query"] == "Jobs at Jobster"


def test_process_focused_job_allows_minor_google_query_rewrites() -> None:
    tool_module = _load_tool_module()
    page = _FakePage(
        url="https://www.google.com/search?q=junior+developer+england&udm=8",
        title="junior developer england - Google Search",
        search_box_value="junior developer england",
    )
    ctx = SimpleNamespace(page=page)
    args = tool_module.ProcessFocusedJobArgs(
        job_profile="junior it developer",
        search_query="junior it developer in england",
        reasoning="Allow reasonable Google query rewrites while preserving the original search intent.",
    )

    mismatch = tool_module._detect_search_context_mismatch(ctx, args)

    assert mismatch is None
