from agent.action_planner import ActionPlanner
from agent.memory import NarrativeMemory
from agent.notebook import Notebook


def _planner(**kwargs) -> ActionPlanner:
    return ActionPlanner(
        user_prompt="test mission",
        memory_store=NarrativeMemory(browser=None),
        **kwargs,
    )


def test_parse_next_hint_compact_candidate_envelope() -> None:
    raw = (
        '{"s":"c","f":"click","a":{"element_id":42},"oi":42,'
        '"c":0.87,"r":"target remains visible","id":"h42"}'
    )

    candidate, status, reason, confidence = ActionPlanner._parse_next_hint_envelope(raw)

    assert status == "candidate"
    assert reason == "target remains visible"
    assert confidence == 0.87
    assert candidate is not None
    assert candidate.function_name == "click"
    assert candidate.function_arguments["element_id"] == 42
    assert candidate.target_signature is not None
    assert candidate.target_signature.overlay_index == 42
    assert candidate.candidate_id == "h42"


def test_parse_next_hint_legacy_envelope_still_supported() -> None:
    raw = (
        '{"status":"candidate","function_name":"type_text","function_arguments":{"text":"hello"},'
        '"overlay_index":7,"confidence":0.9,"reason":"input still focused"}'
    )

    candidate, status, reason, confidence = ActionPlanner._parse_next_hint_envelope(raw)

    assert status == "candidate"
    assert reason == "input still focused"
    assert confidence == 0.9
    assert candidate is not None
    assert candidate.function_name == "type_text"
    assert candidate.target_signature is not None
    assert candidate.target_signature.overlay_index == 7


def test_notebook_delta_context_only_sends_new_entries() -> None:
    notebook = Notebook()
    notebook.add_extraction(description="first", data="A" * 320, url="https://example.com")
    notebook.add_extraction(description="second", data="B" * 320, url="https://example.com")

    planner = _planner(
        previous_response_id="resp_1",
        notebook_last_sent_index=1,
    )
    rendered = planner._format_notebook(notebook)

    assert "NOTEBOOK (New Since Last Turn)" in rendered
    assert "1. first" not in rendered
    assert "2. second" in rendered
    assert "..." in rendered


def test_notebook_delta_context_omits_block_when_nothing_new() -> None:
    notebook = Notebook()
    notebook.add_extraction(description="only", data={"k": "v"}, url="https://example.com")

    planner = _planner(
        previous_response_id="resp_2",
        notebook_last_sent_index=1,
    )
    rendered = planner._format_notebook(notebook)

    assert rendered == ""

