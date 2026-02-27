from agent.speculative_hints import HintCandidate, hydrate_candidate_to_action_step


def _hydrate(candidate: HintCandidate):
    return hydrate_candidate_to_action_step(
        candidate,
        budget_spent=1,
        budget_remaining=9,
        budget_total=10,
    )


def test_hydrate_generates_human_narrative_for_click() -> None:
    candidate = HintCandidate(
        candidate_id="c1",
        function_name="click",
        function_arguments={
            "description": "Continue button",
            "element_type": "button",
        },
        confidence=0.9,
    )

    step = _hydrate(candidate)
    narrative = str((step.function_arguments or {}).get("narrative", ""))

    assert narrative.startswith("I'm clicking")
    assert "Continue button" in narrative
    assert "validated predicted next action" not in narrative.lower()


def test_hydrate_generates_tool_specific_narrative_for_open_url() -> None:
    candidate = HintCandidate(
        candidate_id="c2",
        function_name="open_url",
        function_arguments={"url": "https://example.com/billing"},
        confidence=0.92,
    )

    step = _hydrate(candidate)
    narrative = str((step.function_arguments or {}).get("narrative", ""))

    assert narrative == "I'm opening https://example.com/billing."


def test_hydrate_preserves_existing_narrative() -> None:
    candidate = HintCandidate(
        candidate_id="c3",
        function_name="open_url",
        function_arguments={
            "url": "https://example.com",
            "narrative": "I'm opening the billing page to continue checkout.",
        },
        confidence=0.88,
    )

    step = _hydrate(candidate)
    narrative = str((step.function_arguments or {}).get("narrative", ""))

    assert narrative == "I'm opening the billing page to continue checkout."
