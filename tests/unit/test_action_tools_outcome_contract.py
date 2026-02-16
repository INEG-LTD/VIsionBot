from agent.action_tools import ACTION_TOOLS, PLANNING_TOOLS


def _find_tool(tools, name):
    for tool in tools:
        fn = tool.get("function", {})
        if fn.get("name") == name:
            return tool
    raise AssertionError(f"Tool {name} not found")


def test_plan_next_contains_required_fields():
    tool = _find_tool(PLANNING_TOOLS, "plan_next")
    props = tool["function"]["parameters"]["properties"]

    assert "required_tools_for_completion" in props
    assert "final_answer_draft" in props
    assert "task" in props
    assert "target" in props

    # Removed outcome fields should not be present
    assert "primary_outcome_id" not in props
    assert "outcome_updates" not in props
    assert "final_answer_evidence_ids" not in props


def test_think_and_mark_progress_no_evidence_outcome():
    think_props = _find_tool(ACTION_TOOLS, "think")["function"]["parameters"]["properties"]
    progress_props = _find_tool(ACTION_TOOLS, "mark_progress")["function"]["parameters"]["properties"]

    # evidence_outcome should be removed
    assert "evidence_outcome" not in think_props
    assert "evidence_outcome" not in progress_props

    # Core fields should remain
    assert "reasoning" in think_props
    assert "next_action" in think_props
    assert "description" in progress_props
