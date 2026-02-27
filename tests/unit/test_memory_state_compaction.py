from agent.memory import MemoryState, NarrativeMemory


def _state(**overrides) -> MemoryState:
    data = {
        "timestamp": 1.0,
        "url": "https://example.com",
        "title": "Example",
        "page_width": 1280,
        "page_height": 720,
        "scroll_x": 0,
        "scroll_y": 0,
        "visible_text": "",
        "text_hash": "abc123",
        "text_len": 42,
        "screenshot": None,
    }
    data.update(overrides)
    return MemoryState(**data)


def test_state_dict_uses_compact_text_fields() -> None:
    memory = NarrativeMemory(browser=None)
    state = _state(visible_text="full page text that should not be persisted")

    payload = memory._state_to_dict(state)

    assert payload["text_hash"] == "abc123"
    assert payload["text_len"] == 42
    assert "visible_text" not in payload


def test_meaningful_change_detects_text_hash_difference() -> None:
    memory = NarrativeMemory(browser=None)
    before = _state(text_hash="aaaa", text_len=100)
    after = _state(text_hash="bbbb", text_len=100)

    assert memory._has_meaningful_change(before, after) is True


def test_meaningful_change_supports_legacy_visible_text_fallback() -> None:
    memory = NarrativeMemory(browser=None)
    before = _state(text_hash="", text_len=0, visible_text="one")
    after = _state(text_hash="", text_len=0, visible_text="two")

    assert memory._has_meaningful_change(before, after) is True

