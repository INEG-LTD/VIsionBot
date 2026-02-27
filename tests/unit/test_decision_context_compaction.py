from agent.prompts import DecisionContext, render_decision_context


def test_render_decision_context_compacts_memory_ids() -> None:
    context = DecisionContext(
        action_iteration=7,
        mission="find pricing",
        current_url="https://example.com/pricing",
        page_title="Pricing",
        executed_memory_ids=["m11", "m12"],
        executed_memory_older_count=9,
        reflection_memory_ids=["m13"],
        reflection_memory_older_count=2,
        budget_spent=7,
        budget_remaining=13,
        budget_total=20,
        budget_phase="normal",
        low_budget_mode=False,
    )

    rendered = render_decision_context(context)

    assert "Recent executed-action memory IDs: m11, m12 (+9 older)" in rendered
    assert "Recent reflection memory IDs: m13 (+2 older)" in rendered

