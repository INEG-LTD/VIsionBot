"""
Plan generation utilities extracted from BrowserVisionBot.
"""
from __future__ import annotations

from typing import Optional, List, Dict, Any
import re
from datetime import datetime

from pydantic import BaseModel, Field

from ai_utils import generate_model
from goals.base import BaseGoal
from models import VisionPlan, PageElements
from models.core_models import DetectedElement, PageSection, ActionStep, PageInfo, ActionType
from utils.semantic_targets import SemanticTarget


class OverlaySelection(BaseModel):
    """Response schema for NL overlay selection."""

    overlay_index: Optional[int] = Field(default=None)
    confidence: float = Field(default=0.0)
    reasoning: Optional[str] = Field(default=None)


class PlanGenerator:
    """Generates VisionPlan objects from page context and overlays."""

    def __init__(
        self,
        *,
        include_detailed_elements: bool = True,
        max_detailed_elements: int = 400,
    ) -> None:
        self.include_detailed_elements = include_detailed_elements
        self.max_detailed_elements = max_detailed_elements

    # -------------------- Public API --------------------
    def create_plan(
        self,
        *,
        goal_description: str,
        additional_context: str,
        detected_elements: PageElements,
        page_info: PageInfo,
        screenshot: bytes,
        active_goal: Optional[BaseGoal] = None,
        retry_goal: Optional[BaseGoal] = None,
        page: Any = None,
        command_history: Optional[List[str]] = None,
        dedup_context: Optional[Dict[str, Any]] = None,
        target_context_guard: Optional[str] = None,
    ) -> Optional[VisionPlan]:
        """Create an action plan using the already-detected elements list."""
        print(f"[PlanGen] create_plan goal='{goal_description}' url={page_info.url}\n")

        # Vision-only fast paths are handled in create_plan_with_element_indices where
        # high-confidence overlay data is available. When detection is skipped we fall
        # back directly to LLM planning.

        goal_context = self._build_goal_context_block(
            goal_description, 
            active_goal, 
            page_info, 
            page, 
            retry_goal, 
            command_history or []
        )

        dedup_block, dedup_instruction_lines = self._build_dedup_context_block(dedup_context)
        instructions = [
            "Focus ONLY on the ACTIVE GOAL DESCRIPTIONS above. Prioritize fields or targets marked as pending.",
            "Do not re-complete fields already marked as completed (✅).",
            "Use specialized actions for select/upload/datetime/press/back/forward/stop when appropriate.",
            "Return 1-3 action steps that move directly toward finishing the goal.",
        ]
        if target_context_guard:
            instructions.append(
                f"Only include action steps targeting elements whose immediate surrounding context satisfies: {target_context_guard}."
            )
        if dedup_instruction_lines:
            instructions.extend(dedup_instruction_lines)
        instructions_text = "\n".join(f"{idx}. {line}" for idx, line in enumerate(instructions, start=1))
        dedup_section = f"\n{dedup_block}\n" if dedup_block else ""

        system_prompt = f"""
        You are a web automation assistant. Create a plan based on the ACTIVE GOAL DESCRIPTIONS below.

        Current page: {page_info.url}
        DETECTED ELEMENTS: {[elem.model_dump() for elem in detected_elements.elements]}
        {goal_context}{dedup_section}
        {f"REQUIRED CONTEXT GUARD: {target_context_guard}" if target_context_guard else ""}

        CRITICAL INSTRUCTIONS:
        {instructions_text}
        """

        try:
            plan: VisionPlan = generate_model(
                prompt="Create a plan to achieve the goal using the detected elements.",
                model_object_type=VisionPlan,
                reasoning_level="medium",
                system_prompt=system_prompt,
                image=screenshot,
            )
            return plan
        except Exception as e:
            print(f"❌ Error creating plan: {e}")
            return None

    def create_plan_with_element_indices(
        self,
        *,
        goal_description: str,
        additional_context: str,
        element_data: List[Dict[str, Any]],
        screenshot_with_overlays: bytes,
        page_info: PageInfo,
        command_history: Optional[List[str]] = None,
        active_goal: Optional[BaseGoal] = None,
        retry_goal: Optional[BaseGoal] = None,
        page: Any = None,
        interpretation_mode: str = "literal",
        semantic_hint: Optional[SemanticTarget] = None,
        dedup_context: Optional[Dict[str, Any]] = None,
        target_context_guard: Optional[str] = None,
    ) -> Optional[VisionPlan]:
        """Create a plan leveraging overlay indices to refer to elements precisely."""
        print(f"[PlanGen] create_plan_with_element_indices goal='{goal_description}' url={page_info.url}")

        element_data = element_data or []

        selected_overlay = None
        if interpretation_mode.lower() != "literal":
            try:
                selected_overlay = self._select_overlay_with_language(
                    goal_description,
                    element_data,
                    semantic_hint=semantic_hint,
                )
            except Exception as e:
                print(f"[PlanGen][NL] overlay selection failed: {e}")

        if selected_overlay is not None:
            overlay_data = next(
                (elem for elem in element_data if elem.get("index") == selected_overlay),
                None,
            )
            if not overlay_data:
                print(f"[PlanGen][NL] overlay #{selected_overlay} missing from element data; ignoring")
                selected_overlay = None
            else:
                is_clickable = self._infer_clickable(overlay_data)
                area_ratio = self._normalized_area_ratio(overlay_data)
                if not is_clickable:
                    print(
                        f"[PlanGen][NL] overlay #{selected_overlay} discarded (non-clickable tag='{overlay_data.get('tagName')}' role='{overlay_data.get('role')}')"
                    )
                    selected_overlay = None
                elif area_ratio > 0.25:
                    print(
                        f"[PlanGen][NL] overlay #{selected_overlay} discarded (area_ratio={area_ratio:.3f} too large for precise click)"
                    )
                    selected_overlay = None

        preferred_overlay = selected_overlay if selected_overlay is not None else None

        # Natural-language overlay selection is retained only for diagnostics.
        # Execution always proceeds to full planning to ensure context-aware decisions.

        # Build goal description block with real-time goal analyses
        # Build available elements list
        elements_list_lines: List[str] = []

        detailed_list = self._build_detailed_element_context(element_data, page_info, f"{goal_description} {additional_context}")
        elements_list_lines.append("\nDETAILED ELEMENTS (top candidates):\n" + "\n".join(detailed_list))

        dedup_block, dedup_instruction_lines = self._build_dedup_context_block(dedup_context)
        instructions = [
            "Treat user instructions as casual guidance from a teammate; choose the action that a careful human would naturally take to fulfil the request.",
            "Assume that everything needed to fulfil the goal is visible on the page.",
            "Do not assume a secondary action is needed to fulfil the goal. Eg. If the goal is to click a button and the button is, do not click a secondary button you think is needed to fulfil the goal.",
            "Use the overlay numbers exactly when referring to targets.",
            "Prefer elements that match the active goal descriptions and pending tasks.",
            "Use specialized actions when appropriate:\n           - HANDLE_SELECT, HANDLE_UPLOAD, HANDLE_DATETIME, PRESS, BACK, FORWARD, STOP",
            "Return 1-3 precise action steps.",
        ]
        if preferred_overlay is not None:
            instructions.append(
                "If a VISION PRESELECTED TARGET is supplied, choose it unless it clearly conflicts with the goal."
            )
        if target_context_guard:
            instructions.append(
                f"Reject any overlay whose surrounding content does not satisfy: {target_context_guard}."
            )
        if dedup_instruction_lines:
            instructions.extend(dedup_instruction_lines)
        instructions_text = "\n".join(f"{idx}. {line}" for idx, line in enumerate(instructions, start=1))
        dedup_section = f"\n{dedup_block}\n" if dedup_block else ""

        print(f"DEDUP BLOCK: {dedup_block}")

        system_prompt = f"""
        You are a web automation assistant. Create a plan using the numbered overlays.

        Current page: {page_info.url}

        AVAILABLE ELEMENTS (numbered overlays):
        {chr(10).join(elements_list_lines[:500])}
        {dedup_section}

        INSTRUCTIONS:
        {instructions_text}

        EXAMPLES (Good vs Bad interpretations):
        - User: "click the accept cookies button" → Do: click the visible "Accept"/"Accept all" button in the cookie banner. Don't: click links like "Privacy policy" inside the banner.
        - User: "click the first link in the list" → Do: choose the first item within the current numbered list. Don't: click unrelated navigation links.
        - User: "type my email into the login form" → Do: type into the email/username textbox inside the login form. Don't: type into the search bar or password field.
        - User: "select United States from the country dropdown" → Do: open the country select and choose "United States". Don't: select a different country or change an unrelated dropdown.
        - User: "upload my resume" → Do: use the upload field nearest the resume label. Don't: click random file inputs or open download links.
        - User: "set the meeting time to tomorrow at 3pm" → Do: fill the datetime/date field with the requested value. Don't: edit other fields or leave the value unchanged.
        - User: "open the pricing page" → Do: click the navigation/tab item for Pricing. Don't: open ads or unrelated pages.
        - User: "press submit" → Do: click the primary submit/save button for the active form. Don't: press cancel/reset or navigate away.
        - User: "scroll to the bottom" → Do: issue a scroll-down action until near the page footer. Don't: navigate to a new page or scroll upward.
        - User: "check the terms and conditions box" → Do: tick the terms checkbox required for submission. Don't: toggle newsletter/marketing checkboxes instead.
        
        Ensure the overlay index of the elements to interact with in the action steps match the overlay index's of the elements presented in the reasoning.
        """

        try:
            plan: VisionPlan = generate_model(
                prompt=f"Create a plan to achieve the goal: '{goal_description}' using the numbered elements.",
                model_object_type=VisionPlan,
                reasoning_level="medium",
                system_prompt=system_prompt,
                image=screenshot_with_overlays,
            )
            if plan and plan.action_steps:
                # if preferred_overlay is not None:
                #     self._apply_preferred_overlay(plan, preferred_overlay, element_data)
                detected_elements = self.convert_indices_to_elements(plan.action_steps, element_data)
                plan.detected_elements = detected_elements
                print(f"✅ Plan created with {len(detected_elements.elements)} detected elements")
            return plan
        except Exception as e:
            print(f"❌ Error creating plan with element indices: {e}")
            return None

    # -------------------- Helpers --------------------
    def _build_dedup_context_block(self, dedup_context: Optional[Dict[str, Any]]) -> tuple[str, List[str]]:
        """Prepare a textual summary of interacted elements for prompt injection."""

        if not dedup_context or not dedup_context.get("avoid_duplicates"):
            return "", []

        raw_entries = dedup_context.get("entries") or dedup_context.get("interactions") or []
        if not raw_entries:
            return "", []

        quantity = dedup_context.get("quantity")
        entries = list(raw_entries)
        if isinstance(quantity, int) and quantity >= 0:
            if quantity == 0:
                return "", []
            entries = entries[-quantity:]

        def _resolve_position(entry_data: Dict[str, Any], element_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            position = entry_data.get("position")
            if isinstance(position, dict) and position.get("x") is not None and position.get("y") is not None:
                return position

            coords = (
                element_data.get("normalizedCoords")
                or element_data.get("box2d")
                or element_data.get("box_2d")
            )
            if isinstance(coords, (list, tuple)) and len(coords) >= 4:
                try:
                    y_min, x_min, y_max, x_max = [float(coords[i]) for i in range(4)]
                    x_center = (x_min + x_max) / 2.0
                    y_center = (y_min + y_max) / 2.0
                    return {
                        "x": round(x_center, 2),
                        "y": round(y_center, 2),
                        "reference": "normalized",
                    }
                except (TypeError, ValueError):
                    pass

            rect = element_data.get("rect")
            if isinstance(rect, dict):
                try:
                    x = float(rect.get("x", 0.0)) + float(rect.get("width", 0.0)) / 2.0
                    y = float(rect.get("y", 0.0)) + float(rect.get("height", 0.0)) / 2.0
                    return {
                        "x": round(x, 1),
                        "y": round(y, 1),
                        "reference": "pixel",
                    }
                except (TypeError, ValueError):
                    pass

            return None

        lines: List[str] = []
        for entry in entries:
            element = entry.get("element") or {}
            signature = entry.get("signature") or ""
            short_sig = signature[:12] if signature else "unknown"
            text_source = (
                element.get("text")
                or element.get("description")
                or element.get("ariaLabel")
                or ""
            )
            text = str(text_source).strip()
            text = re.sub(r"\s+", " ", text)
            if len(text) > 80:
                text = f"{text[:77]}..."
            text = text.replace('"', "'")

            tag = element.get("tagName") or element.get("element_type") or element.get("role") or "element"
            overlay = element.get("overlayIndex")
            action = (entry.get("interaction_type") or "unknown").lower()
            timestamp = entry.get("timestamp")
            seen = "unknown"
            if isinstance(timestamp, (int, float)) and timestamp > 0:
                try:
                    seen = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    seen = str(timestamp)

            position = _resolve_position(entry, element)
            pos_str = "pos=unknown"
            if position and position.get("x") is not None and position.get("y") is not None:
                reference = (position.get("reference") or position.get("type") or "normalized").lower()
                x_val = position.get("x")
                y_val = position.get("y")
                try:
                    if reference.startswith("pixel"):
                        pos_str = f"pos(px)=({int(round(float(x_val)))},{int(round(float(y_val)))})"
                    else:
                        pos_str = f"pos(norm)=({round(float(x_val), 2)},{round(float(y_val), 2)})"
                except (TypeError, ValueError):
                    pos_str = "pos=unknown"

            lines.append(
                f"- sig={short_sig} | action={action} | overlay={overlay if overlay is not None else 'n/a'} | {pos_str} | tag={tag} | seen={seen} | text=\"{text}\""
            )

        if not lines:
            return "", []

        block = "PREVIOUS INTERACTIONS TO AVOID:\n" + "\n".join(lines)
        instructions = [
            "Avoid selecting any element whose visible text exactly matches (case-sensitive) an entry listed under PREVIOUS INTERACTIONS TO AVOID. For example, if 'Accept Cookies' was previously clicked, do not select another element with text 'Accept Cookies'.",
            "Do not assign actions to any element listed under PREVIOUS INTERACTIONS TO AVOID. For example, if a 'Submit' button was previously clicked at position (100,200), do not click that same button again.",
            "If every viable element has already been interacted with, return a scroll down plan.",
        ]
        return block, instructions

    def _build_goal_context_block(
        self,
        goal_description: str,
        active_goal: Optional[BaseGoal],
        page_info: PageInfo,
        page: Any,
        retry_goal: Optional[BaseGoal],
        command_history: List[str],
    ) -> str:
        """Assemble a textual block describing current goals and recent commands."""

        lines: List[str] = []
        lines.append("ACTIVE GOAL DESCRIPTIONS:")
        if goal_description:
            lines.append(f"- Primary request: {goal_description}")

        if active_goal:
            desc = getattr(active_goal, "description", "") or "(no description)"
            lines.append(f"- Active goal: {desc}")
            if getattr(active_goal, "retry_count", 0):
                retries = getattr(active_goal, "retry_count", 0)
                max_retries = getattr(active_goal, "max_retries", 0)
                lines.append(f"  • Retry state: {retries}/{max_retries}")

        if retry_goal and retry_goal is not active_goal:
            retry_desc = getattr(retry_goal, "description", "") or "(no description)"
            lines.append(f"- Pending retry goal: {retry_desc}")

        if command_history:
            recent = command_history[-3:]
            lines.append("RECENT COMMAND HISTORY:")
            for cmd in reversed(recent):
                lines.append(f"  • {cmd}")

        try:
            url_line = page_info.url if page_info else getattr(page, "url", lambda: "")()
            if url_line:
                lines.append(f"CURRENT PAGE: {url_line}")
        except Exception:
            pass

        return "\n" + "\n".join(lines) + "\n"

    def _apply_preferred_overlay(
        self,
        plan: VisionPlan,
        preferred_overlay: int,
        element_data: List[Dict[str, Any]],
    ) -> None:
        """Prioritise the vision-selected overlay and drop conflicting clicks."""

        try:
            overlay_index = int(preferred_overlay)
        except Exception:
            return

        if not any(elem.get("index") == overlay_index for elem in element_data):
            return

        preferred_data = next((elem for elem in element_data if elem.get("index") == overlay_index), None)
        preferred_area = self._normalized_area_ratio(preferred_data) if preferred_data else 0.0

        filtered_steps: List[ActionStep] = []
        primary_assigned = False

        for step in plan.action_steps:
            if step.action != ActionType.CLICK:
                filtered_steps.append(step)
                continue

            if not primary_assigned:
                original_index = step.overlay_index
                current_data = next(
                    (elem for elem in element_data if elem.get("index") == original_index),
                    None,
                )
                current_area = self._normalized_area_ratio(current_data) if current_data else 1_000.0

                step.overlay_index = overlay_index
                step.x = None
                step.y = None
                filtered_steps.append(step)
                primary_assigned = True

                print(
                    f"[PlanGen] prioritising overlay #{overlay_index} for first click"
                    f" (prev_index={original_index}, preferred_area={preferred_area:.4f}, prev_area={current_area:.4f})"
                )
                continue

            if step.overlay_index == overlay_index:
                filtered_steps.append(step)
                continue

            print(
                f"[PlanGen] dropping secondary click targeting overlay #{step.overlay_index}"
                f" in favour of preferred overlay #{overlay_index}"
            )

        plan.action_steps = filtered_steps

    def _select_overlay_with_language(
        self,
        instruction: str,
        element_data: List[Dict[str, Any]],
        *,
        semantic_hint: Optional[SemanticTarget] = None,
        max_samples: int = 40,
    ) -> Optional[int]:
        if not element_data:
            return None

        # Prioritise likely clickable overlays and discard huge structural regions
        filtered: List[Dict[str, Any]] = []
        for elem in element_data:
            idx = elem.get("index")
            if idx is None:
                continue
            if not self._infer_clickable(elem):
                continue
            if self._normalized_area_ratio(elem) > 0.3:
                continue
            filtered.append(elem)

        if not filtered:
            return None

        goal_tokens = self._extract_goal_tokens(instruction)
        hint_terms = self._extract_hint_terms(semantic_hint)
        goal_token_str = ", ".join(goal_tokens) if goal_tokens else "none"

        scored_candidates: List[Dict[str, Any]] = []
        for elem in filtered:
            score_hint, token_hits, hint_hits = self._score_overlay_for_instruction(
                goal_tokens,
                hint_terms,
                elem,
            )
            elem_copy = dict(elem)
            elem_copy["_score_hint"] = score_hint
            elem_copy["_token_hits"] = token_hits
            elem_copy["_hint_hits"] = hint_hits
            scored_candidates.append(elem_copy)

        scored_candidates.sort(key=lambda e: e.get("_score_hint", 0), reverse=True)

        samples: List[str] = []
        for elem in scored_candidates[:max_samples]:
            idx = elem.get("index")
            role = (elem.get("role") or elem.get("tagName") or "").lower()
            tag = (elem.get("tagName") or "").lower()
            text = (elem.get("textContent") or "").strip()
            aria = (elem.get("ariaLabel") or "").strip()
            area = self._normalized_area_ratio(elem) * 100
            score_hint = int(elem.get("_score_hint", 0))
            token_hits = elem.get("_token_hits") or []
            hint_hits = elem.get("_hint_hits") or []

            text_snip = text[:80]
            aria_snip = aria[:60] if aria else "-"
            match_str = ",".join(token_hits) if token_hits else "-"
            hint_str = ",".join(hint_hits) if hint_hits else "-"

            samples.append(
                f"- #{idx} score={score_hint} role={role or 'unknown'} tag={tag or 'unknown'} area={area:.2f}% "
                f"text='{text_snip}' aria='{aria_snip}' matches={match_str} hint_hits={hint_str}"
            )

        prompt = (
            "You select the overlay index that best fulfills the browsing instruction.\n"
            f"Instruction: \"{instruction}\"\n"
            f"Instruction terms to align with: {goal_token_str}.\n"
        )

        if semantic_hint:
            prompt += (
                f"Target role preference: {semantic_hint.role or 'any'}.\n"
                f"Must-have terms: {', '.join(semantic_hint.required_terms or []) or 'none'}.\n"
                f"Key content terms: {', '.join(semantic_hint.primary_terms or []) or 'none'}.\n"
                f"Helpful context terms: {', '.join(semantic_hint.context_terms or []) or 'none'}.\n"
            )

        prompt += (
            "Guidance:\n"
            "- Choose concise clickable controls (buttons/links) that directly execute the request.\n"
            "- Prefer overlays whose visible text or aria labels contain several instruction or must-have terms.\n"
            "- Reject overlays that describe entire job cards, long descriptions, or containers that lack a primary action.\n"
            "Return JSON {\"overlay_index\": number|null, \"confidence\": number, \"reasoning\": string}.\n"
            "Overlays:\n" + "\n".join(samples)
        )

        try:
            selection = generate_model(
                prompt=prompt,
                model_object_type=OverlaySelection,
                reasoning_level="medium",
            )
        except Exception as e:
            print(f"[PlanGen][NL] LLM selection error: {e}")
            return None

        if not selection or selection.overlay_index is None:
            return None

        print(
            f"[PlanGen][NL] overlay #{selection.overlay_index} chosen (confidence={selection.confidence:.2f})"
        )
        try:
            return int(selection.overlay_index)
        except Exception:
            return None

    def _normalized_area_ratio(self, element: Dict[str, Any]) -> float:
        box = element.get("normalizedCoords") or []
        if len(box) != 4:
            return 0.0
        height = max(box[2] - box[0], 0)
        width = max(box[3] - box[1], 0)
        return (height * width) / 1_000_000.0

    def _extract_goal_tokens(self, instruction: str) -> List[str]:
        stop_words = {
            "the",
            "a",
            "an",
            "to",
            "for",
            "please",
            "click",
            "tap",
            "button",
            "press",
            "open",
            "show",
            "select",
        }
        tokens = [
            t
            for t in re.split(r"[^a-z0-9]+", (instruction or "").lower())
            if t and len(t) > 1 and t not in stop_words
        ]
        return list(dict.fromkeys(tokens))[:20]

    def _extract_hint_terms(self, semantic_hint: Optional[SemanticTarget]) -> List[str]:
        if not semantic_hint:
            return []
        terms = [
            *(semantic_hint.required_terms or []),
            *(semantic_hint.primary_terms or []),
            *(semantic_hint.context_terms or []),
        ]
        normalized = [t.strip().lower() for t in terms if t]
        return list(dict.fromkeys(normalized))[:20]

    def _score_overlay_for_instruction(
        self,
        goal_tokens: List[str],
        hint_terms: List[str],
        element: Dict[str, Any],
    ) -> tuple[int, List[str], List[str]]:
        if not element:
            return 0, [], []

        text = (element.get("textContent") or "").lower()
        aria = (element.get("ariaLabel") or "").lower()
        context = (element.get("contextText") or "").lower()
        combined = " ".join(part for part in [text, aria, context] if part)
        tag = (element.get("tagName") or "").lower()
        role = (element.get("role") or "").lower()
        class_name = (element.get("className") or "").lower()
        etype = (element.get("type") or "").lower()
        area_ratio = self._normalized_area_ratio(element)

        score = 0
        token_hits: List[str] = []
        for tok in goal_tokens:
            if tok in text:
                score += 6
                token_hits.append(tok)
            elif tok in aria:
                score += 4
                token_hits.append(tok)
            elif tok in context:
                score += 2
                token_hits.append(tok)

        hint_hits: List[str] = []
        for term in hint_terms:
            if term in combined:
                hint_hits.append(term)
                score += 2

        if tag in {"button", "a"}:
            score += 5
        elif role in {"button", "link"}:
            score += 4
        elif "btn" in class_name or etype in {"submit", "button"}:
            score += 2

        if area_ratio < 0.01:
            score += 3
        elif area_ratio > 0.15:
            score -= 5
        elif area_ratio > 0.08:
            score -= 3
        elif area_ratio > 0.04:
            score -= 1

        text_len = len(text)
        if text_len > 160:
            score -= 4
        elif text_len > 110:
            score -= 2

        if len(context) > 400:
            score -= 3

        if not token_hits and goal_tokens:
            score -= 3

        return score, token_hits, hint_hits

    def convert_indices_to_elements(self, action_steps: List[ActionStep], element_data: List[Dict[str, Any]]) -> PageElements:
        detected_elements = []
        used_indices = set()
        for step in action_steps:
            if step.overlay_index is not None and step.overlay_index not in used_indices:
                matching_data = next((elem for elem in element_data if elem.get('index') == step.overlay_index), None)
                if matching_data:
                    label = matching_data.get('description') or matching_data.get('textContent') or f"Element {step.overlay_index}"
                    box = matching_data.get('normalizedCoords') or matching_data.get('box_2d') or [0, 0, 0, 0]
                    element = DetectedElement(
                        element_label=label,
                        description=label,
                        element_type=self._infer_element_type(matching_data),
                        is_clickable=self._infer_clickable(matching_data),
                        box_2d=box,
                        section=PageSection.CONTENT,
                        confidence=matching_data.get('confidence', 0.9),
                        overlay_number=step.overlay_index,
                    )
                    detected_elements.append(element)
                    used_indices.add(step.overlay_index)
        return PageElements(elements=detected_elements)

    def _rank_elements(self, element_data: List[Dict[str, Any]], goal_text: str) -> List[Dict[str, Any]]:
        def score_elem(e: Dict[str, Any]) -> int:
            s = 0
            txt = (e.get('textContent') or e.get('description') or "").lower()
            tag = (e.get('tagName') or '').lower()
            # Prioritize links/buttons for click/navigation intents
            if any(w in goal_text.lower() for w in ["click", "open", "visit", "go to", "navigate", "link", "button"]):
                if tag in ("a", "button"):
                    s += 3
            # Simple text match boost
            for tok in ["login", "sign in", "submit", "search", "apply", "next", "continue"]:
                if tok in txt:
                    s += 2
            # Slightly prefer items with aria-labels
            if (e.get('ariaLabel') or '').strip():
                s += 1
            return -s  # sort ascending by negative to get high score first
        return sorted(list(element_data or []), key=score_elem)

    def _build_element_detail_line(self, elem: Dict[str, Any], page_info: PageInfo) -> str:
        idx = elem.get('index')
        desc = elem.get('description') or elem.get('textContent') or ''
        tag = (elem.get('tagName') or '').lower()
        href = elem.get('href') or ''
        aria = elem.get('ariaLabel') or ''
        coords = elem.get('normalizedCoords') or []
        extra = []
        if tag:
            extra.append(f"tag={tag}")
        if href:
            extra.append(f"href={href}")
        if aria:
            extra.append(f"aria={aria}")
        return f"#{idx}: {desc}  [{' '.join(map(str, coords))}]  {' '.join(extra)}"

    def _build_detailed_element_context(self, element_data: List[Dict[str, Any]], page_info: PageInfo, goal_text: str) -> List[str]:
        ranked = self._rank_elements(element_data, goal_text)
        top_n = max(1, min(self.max_detailed_elements, len(ranked)))
        lines = []
        for e in ranked[:top_n]:
            lines.append(self._build_element_detail_line(e, page_info))
        return lines

    def _infer_element_type(self, element_data: Dict[str, Any]) -> str:
        tag = (element_data.get('tagName') or '').lower()
        if tag in ('a', 'button'):
            return 'button' if tag == 'button' else 'link'
        if tag in ('input', 'textarea'):
            t = (element_data.get('type') or '').lower()
            return t or 'input'
        if tag == 'select':
            return 'select'
        return tag or 'element'

    def _infer_clickable(self, element_data: Dict[str, Any]) -> bool:
        tag = (element_data.get('tagName') or '').lower()
        if tag in ('a', 'button'):
            return True
        role = (element_data.get('role') or '').lower()
        if role in ('button', 'link', 'tab'):
            return True
        onclick = element_data.get('onclick')
        return bool(onclick)
