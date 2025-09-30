"""Baseline checks for generic semantic target handling."""
from utils.semantic_targets import build_semantic_target, semantic_score_element


def test_build_semantic_target_has_terms():
    target = build_semantic_target("click the submit button")
    assert target is not None
    assert target.role in {None, "button"}
    assert target.primary_terms or target.required_terms or target.context_terms


def test_semantic_score_prefers_matching_element():
    target = build_semantic_target("press the submit button")
    assert target is not None

    submit_element = {
        "tagName": "button",
        "role": "button",
        "textContent": "Submit",
        "contextText": "Complete form",
        "ariaLabel": "Submit",
    }
    cancel_element = {
        "tagName": "button",
        "role": "button",
        "textContent": "Cancel",
        "contextText": "Abort form",
        "ariaLabel": "Cancel",
    }

    submit_score = semantic_score_element(submit_element, target)
    cancel_score = semantic_score_element(cancel_element, target)

    assert submit_score is not None
    assert cancel_score is None or submit_score > cancel_score
