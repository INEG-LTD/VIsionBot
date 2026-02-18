from agent.action_tools import ACTION_TOOLS


def _find_tool(tools, name):
    for tool in tools:
        fn = tool.get("function", {})
        if fn.get("name") == name:
            return tool
    raise AssertionError(f"Tool {name} not found")


def test_think_contains_loop_fields():
    think_props = _find_tool(ACTION_TOOLS, "think")["function"]["parameters"]["properties"]

    # Core fields
    assert "reasoning" in think_props
    assert "next_action" in think_props

    # Loop fields
    assert "loop_count" in think_props
    assert "loop_description" in think_props

    # next_action should include loop-related values
    enum_values = think_props["next_action"]["enum"]
    assert "start_loop" in enum_values
    assert "advance" in enum_values
    assert "end_loop" in enum_values
    assert "done" in enum_values
    assert "continue" in enum_values
    assert "stuck" in enum_values


def test_no_mark_progress_or_revise_target():
    """mark_progress and revise_target tools should not exist."""
    tool_names = {t["function"]["name"] for t in ACTION_TOOLS}
    assert "mark_progress" not in tool_names
    assert "revise_target" not in tool_names
