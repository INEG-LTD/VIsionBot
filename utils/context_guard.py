"""Context guard helper for validating element surroundings before interaction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from playwright.sync_api import Page

from ai_utils import answer_question_with_vision
from goals.element_analyzer import ElementAnalyzer
from models import ActionStep, VisionPlan, PageInfo
from models.core_models import ActionType, DetectedElement
from vision_utils import get_gemini_box_2d_center_pixels


@dataclass
class GuardDecision:
    """Outcome of a guard validation."""

    passed: bool
    reason: str = ""
    cached: bool = False


class ContextGuard:
    """Validates that an element satisfies required surrounding context before interaction."""

    def __init__(self, page: Page, element_analyzer: Optional[ElementAnalyzer] = None):
        self.page = page
        self.element_analyzer = element_analyzer or ElementAnalyzer(page)
        self._cache: Dict[Tuple[str, int, str], GuardDecision] = {}

    def reset_cache(self) -> None:
        """Clear cached guard results."""
        self._cache.clear()

    def validate(
        self,
        *,
        step: ActionStep,
        plan: VisionPlan,
        page_info: PageInfo,
        guard_text: str,
    ) -> GuardDecision:
        guard_text = (guard_text or "").strip()
        if not guard_text:
            return GuardDecision(True)

        overlay_index = step.overlay_index
        if overlay_index is None:
            reason = "No overlay index present for guarded action"
            return GuardDecision(False, reason=reason)

        cache_key = (guard_text, overlay_index, getattr(page_info, "url", ""))
        if cache_key in self._cache:
            decision = self._cache[cache_key]
            return GuardDecision(decision.passed, reason=decision.reason, cached=True)

        detected = self._find_detected_element(plan, overlay_index)
        if not detected or not getattr(detected, "box_2d", None):
            reason = "Detected element missing for overlay"
            decision = GuardDecision(False, reason=reason)
            self._cache[cache_key] = decision
            return decision

        center_x, center_y = get_gemini_box_2d_center_pixels(
            detected.box_2d,
            page_info.width or page_info.ss_pixel_w or 0,
            page_info.height or page_info.ss_pixel_h or 0,
        )

        try:
            screenshot = self.page.screenshot(type="jpeg", quality=45, full_page=False)
        except Exception:
            screenshot = self._capture_context_image(detected, page_info)
        if not screenshot:
            reason = "Failed to capture context image"
            decision = GuardDecision(False, reason=reason)
            self._cache[cache_key] = decision
            return decision

        element_info = self.element_analyzer.analyze_element_at_coordinates(center_x, center_y)
        summary_parts = []
        if element_info:
            text = (element_info.get("text") or element_info.get("innerText") or "").strip()
            if text:
                summary_parts.append(f"Text: {text[:120]}")
            tag_name = element_info.get("tagName") or element_info.get("elementType")
            if tag_name:
                summary_parts.append(f"Tag: {tag_name}")
            attributes = element_info.get("attributes") or {}
            aria = attributes.get("aria-label") or attributes.get("aria_label")
            if aria:
                summary_parts.append(f"Aria: {aria[:80]}")
        element_summary = "; ".join(summary_parts) if summary_parts else "(no additional element summary)"

        question = (
            "You are verifying the correctness of an automation target.\n"
            f"Required surrounding context: {guard_text}\n"
            f"Element summary: {element_summary}\n"
            "Does the highlighted region satisfy the required surrounding context?"
        )

        answer = answer_question_with_vision(question, screenshot)
        if answer is None:
            reason = "Vision model returned no decision"
            decision = GuardDecision(False, reason=reason)
        elif answer is True:
            decision = GuardDecision(True)
        else:
            reason = "Vision model indicated context mismatch"
            decision = GuardDecision(False, reason=reason)

        self._cache[cache_key] = decision
        return decision

    @staticmethod
    def is_guarded_action(action_type: ActionType) -> bool:
        return action_type in {
            ActionType.CLICK,
            ActionType.TYPE,
            ActionType.HANDLE_SELECT,
            ActionType.HANDLE_UPLOAD,
            ActionType.HANDLE_DATETIME,
        }

    def _find_detected_element(self, plan: VisionPlan, overlay_index: int) -> Optional[DetectedElement]:
        elements = getattr(plan.detected_elements, "elements", None) or []
        for element in elements:
            if getattr(element, "overlay_number", None) == overlay_index:
                return element
        return None

    def _capture_context_image(self, detected: DetectedElement, page_info: PageInfo) -> Optional[bytes]:
        box = getattr(detected, "box_2d", None)
        if not box or len(box) != 4:
            return None

        y_min, x_min, y_max, x_max = box
        width_px = max(page_info.width or page_info.ss_pixel_w or 1, 1)
        height_px = max(page_info.height or page_info.ss_pixel_h or 1, 1)

        x_min_px = int(max(0, min(x_min / 1000.0 * width_px, width_px)))
        x_max_px = int(max(0, min(x_max / 1000.0 * width_px, width_px)))
        y_min_px = int(max(0, min(y_min / 1000.0 * height_px, height_px)))
        y_max_px = int(max(0, min(y_max / 1000.0 * height_px, height_px)))

        # Ensure minimum size and add padding for surrounding context
        pad_x = max(20, int(0.15 * max(1, x_max_px - x_min_px)))
        pad_y = max(20, int(0.15 * max(1, y_max_px - y_min_px)))

        clip_x = max(0, x_min_px - pad_x)
        clip_y = max(0, y_min_px - pad_y)
        clip_w = min(width_px - clip_x, (x_max_px - x_min_px) + pad_x * 2)
        clip_h = min(height_px - clip_y, (y_max_px - y_min_px) + pad_y * 2)

        if clip_w <= 0 or clip_h <= 0:
            return None

        try:
            return self.page.screenshot(
                type="jpeg",
                quality=50,
                full_page=False,
                clip={
                    "x": clip_x,
                    "y": clip_y,
                    "width": clip_w,
                    "height": clip_h,
                },
            )
        except Exception:
            return None
