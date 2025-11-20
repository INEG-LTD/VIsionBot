"""
Plan generation utilities extracted from BrowserVisionBot.
"""
from __future__ import annotations

from typing import Optional, List, Dict, Any
import re
from datetime import datetime

from ai_utils import generate_model, generate_text
from typing import Any
from models import VisionPlan, PageElements
from models.core_models import DetectedElement, PageSection, ActionStep, PageInfo, ActionType
from utils.semantic_targets import SemanticTarget
from utils.intent_parsers import parse_action_intent
from utils.event_logger import get_event_logger


class PlanGenerator:
    """Generates VisionPlan objects from page context and overlays."""

    def __init__(
        self,
        *,
        include_detailed_elements: bool = True,
        max_detailed_elements: int = 400,
        max_steps: Optional[int] = None,
        merge_overlay_selection: bool = False,
        return_overlay_only: bool = False,
        overlay_selection_max_samples: Optional[int] = None,
    ) -> None:
        self.include_detailed_elements = include_detailed_elements
        self.max_detailed_elements = max_detailed_elements
        self.max_steps = max_steps  # Maximum number of steps to generate (None = no limit, uses default 1-3)
        self.merge_overlay_selection = merge_overlay_selection
        self.return_overlay_only = return_overlay_only
        if overlay_selection_max_samples is not None and overlay_selection_max_samples <= 0:
            overlay_selection_max_samples = None
        self.overlay_selection_max_samples = overlay_selection_max_samples

    # -------------------- Public API --------------------
    def create_plan(
        self,
        *,
        goal_description: str,
        additional_context: str,
        detected_elements: PageElements,
        page_info: PageInfo,
        screenshot: bytes,
        active_goal: Optional[Any] = None,
        retry_goal: Optional[Any] = None,
        page: Any = None,
        command_history: Optional[List[str]] = None,
        dedup_context: Optional[Dict[str, Any]] = None,
        target_context_guard: Optional[str] = None,
        max_steps: Optional[int] = None,
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
        # Determine max_steps: use method parameter if provided, otherwise use instance default, otherwise default to 3
        effective_max_steps = max_steps if max_steps is not None else (self.max_steps if self.max_steps is not None else 3)
        
        force_click_only = self._is_click_goal(goal_description, active_goal)

        instructions = [
            "Focus ONLY on the ACTIVE GOAL DESCRIPTIONS above. Prioritize fields or targets marked as pending.",
            "Do not re-complete fields already marked as completed (✅).",
            "Use specialized actions for select/upload/datetime/press/back/forward/stop when appropriate.",
        ]
        if effective_max_steps == 1:
            instructions.append("Return EXACTLY 1 action step that directly accomplishes the goal. Do NOT add preparatory steps like clicking to focus or additional actions. Only do exactly what the action demands.")
        else:
            instructions.append(f"Return 1-{effective_max_steps} action steps that move directly toward finishing the goal.")
        if target_context_guard:
            instructions.append(
                f"Only include action steps targeting elements whose immediate surrounding context satisfies: {target_context_guard}."
            )
        if force_click_only:
            instructions.append(
                "The active goal is a CLICK. Return exactly one CLICK action targeting the requested element. Do NOT include TYPE, SCROLL, or any other action types."
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
        {f"ADDITIONAL CONTEXT: {additional_context}" if additional_context else ""}

        CRITICAL INSTRUCTIONS:
        {instructions_text}
        """

        try:
            user_prompt = "Create a plan to achieve the goal using the detected elements."
            if additional_context:
                user_prompt += f"\n\nAdditional context to consider: {additional_context}"
                
            plan: VisionPlan = generate_model(
                prompt=user_prompt,
                model_object_type=VisionPlan,
                system_prompt=system_prompt,
                image=screenshot,
            )
            if plan and plan.action_steps:
                # Enforce max_steps limit strictly
                effective_max_steps = max_steps if max_steps is not None else (self.max_steps if self.max_steps is not None else None)
                if effective_max_steps is not None and len(plan.action_steps) > effective_max_steps:
                    print(f"[PlanGen] ⚠️ Plan generated {len(plan.action_steps)} steps, limiting to {effective_max_steps} step(s) as required")
                    plan.action_steps = plan.action_steps[:effective_max_steps]
                    # Update reasoning to reflect the limitation
                    plan.reasoning = f"(Limited to {effective_max_steps} step(s)) {plan.reasoning}"
                if force_click_only:
                    self._enforce_click_actions(plan)
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
        active_goal: Optional[Any] = None,
        retry_goal: Optional[Any] = None,
        page: Any = None,
        interpretation_mode: str = "literal",
        semantic_hint: Optional[SemanticTarget] = None,
        dedup_context: Optional[Dict[str, Any]] = None,
        target_context_guard: Optional[str] = None,
        max_steps: Optional[int] = None,
    ) -> Optional[VisionPlan]:
        """Create a plan leveraging overlay indices to refer to elements precisely."""
        print(f"[PlanGen] create_plan_with_element_indices goal='{goal_description}' url={page_info.url}")
        print(f"[PlanGen] Received {len(element_data) if element_data else 0} element(s) for planning")

        element_data = element_data or []
        force_click_only = self._is_click_goal(goal_description, active_goal)
        action_intent = parse_action_intent(goal_description) if self.return_overlay_only else None
        overlay_only_supported_actions = {"click", "type", "select", "upload", "datetime"}
        overlay_only_candidate = (
            action_intent is not None
            and action_intent.action in overlay_only_supported_actions
        )
        
        selected_overlay = None
        overlay_selection = None
        should_run_selection = (
            interpretation_mode.lower() != "literal"
            and (
                not self.merge_overlay_selection
                or overlay_only_candidate
            )
        )
        if should_run_selection:
            try:
                overlay_selection = self._select_overlay_with_language(
                    goal_description,
                    element_data,
                    semantic_hint=semantic_hint,
                    screenshot=screenshot_with_overlays,
                )
            except Exception as e:
                print(f"[PlanGen][NL] overlay selection failed: {e}")

        if overlay_selection is not None:
            selected_overlay = overlay_selection.overlay_index

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

        # Log how many elements will be included in detailed context
        detailed_list = self._build_detailed_element_context(element_data, page_info, f"{goal_description} {additional_context}")
        print(f"[PlanGen] Including {len(detailed_list)} element(s) in detailed context (max: {self.max_detailed_elements})")
        elements_list_lines.append("\nDETAILED ELEMENTS (top candidates):\n" + "\n".join(detailed_list))

        dedup_block, dedup_instruction_lines = self._build_dedup_context_block(dedup_context)
        # Determine max_steps: use method parameter if provided, otherwise use instance default, otherwise default to 3
        effective_max_steps = max_steps if max_steps is not None else (self.max_steps if self.max_steps is not None else 3)

        instructions = [
            "Treat user instructions as casual guidance from a teammate; choose the action that a careful human would naturally take to fulfil the request.",
            "Assume that everything needed to fulfil the goal is visible on the page.",
            "Do not assume a secondary action is needed to fulfil the goal. Eg. If the goal is to click a button and the button is, do not click a secondary button you think is needed to fulfil the goal.",
            "Do not click elements to 'trigger' or 'open' other content unless explicitly requested. Focus only on what is currently visible and directly actionable.",
            "If the goal can be achieved by analyzing or working with what's currently visible, do not generate click actions to navigate elsewhere.",
            "Use the overlay numbers exactly when referring to targets.",
            "Prefer elements that match the active goal descriptions and pending tasks.",
            "Use specialized actions when appropriate:\n           - HANDLE_SELECT, HANDLE_UPLOAD, HANDLE_DATETIME, PRESS, BACK, FORWARD, STOP",
        ]
        if effective_max_steps == 1:
            instructions.append("Return EXACTLY 1 action step that directly accomplishes the goal. Do NOT add preparatory steps like clicking to focus, typing, or additional actions. Only do exactly what the action demands. For example, if the goal is 'type: yahoo finance', only return a TYPE action - do NOT add a CLICK action to focus first.")
        else:
            instructions.append(f"Return 1-{effective_max_steps} precise action steps.")
        if preferred_overlay is not None:
            instructions.append(
                "If a VISION PRESELECTED TARGET is supplied, choose it unless it clearly conflicts with the goal."
            )
        elif self.merge_overlay_selection and interpretation_mode.lower() != "literal":
            instructions.append(
                "Select the best overlay directly within your action plan; reference its number explicitly in the returned step."
            )
        if target_context_guard:
            instructions.append(
                f"Reject any overlay whose surrounding content does not satisfy: {target_context_guard}."
            )
        if force_click_only:
            instructions.append(
                "This goal requires a CLICK. Return exactly one CLICK action that targets the requested element. Do NOT output TYPE or other action types."
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
        {f"ADDITIONAL CONTEXT: {additional_context}" if additional_context else ""}

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

        if self.return_overlay_only and overlay_only_candidate:
            # Use overlay selection result when available; otherwise request selection now.
            selection = overlay_selection
            if selection is None and interpretation_mode.lower() != "literal":
                try:
                    selection = self._select_overlay_with_language(
                        goal_description,
                        element_data,
                        semantic_hint=semantic_hint,
                        screenshot=screenshot_with_overlays,
                    )
                except Exception as e:
                    print(f"[PlanGen][NL] overlay-only selection failed: {e}")

            if selection and selection.overlay_index is not None:
                overlay_idx = selection.overlay_index
                action_type = {
                    "click": ActionType.CLICK,
                    "type": ActionType.TYPE,
                    "select": ActionType.HANDLE_SELECT,
                    "upload": ActionType.HANDLE_UPLOAD,
                    "datetime": ActionType.HANDLE_DATETIME,
                }[action_intent.action]

                step_kwargs: Dict[str, Any] = {}
                if action_intent.action == "type":
                    if not action_intent.value:
                        print("[PlanGen] Overlay-only TYPE command missing value, falling back to full plan.")
                        selection = None
                    else:
                        step_kwargs["text_to_type"] = action_intent.value
                elif action_intent.action == "select":
                    if not action_intent.value:
                        print("[PlanGen] Overlay-only SELECT command missing option, falling back to full plan.")
                        selection = None
                    else:
                        step_kwargs["select_option_text"] = action_intent.value
                elif action_intent.action == "upload":
                    if not action_intent.value:
                        print("[PlanGen] Overlay-only UPLOAD command missing file path, falling back to full plan.")
                        selection = None
                    else:
                        step_kwargs["upload_file_path"] = action_intent.value
                elif action_intent.action == "datetime":
                    if not action_intent.value:
                        print("[PlanGen] Overlay-only DATETIME command missing value, falling back to full plan.")
                        selection = None
                    else:
                        step_kwargs["datetime_value"] = action_intent.value

                if selection is not None:
                    action_step = ActionStep(
                        action=action_type,
                        overlay_index=overlay_idx,
                        **step_kwargs,
                    )
                detected_elements = self.convert_indices_to_elements(
                    [action_step],
                    element_data,
                )
                reasoning_suffix = ""
                if action_intent.action == "type":
                    reasoning_suffix = f" and type '{action_intent.value}'"
                elif action_intent.action == "select":
                    reasoning_suffix = f" and select option '{action_intent.value}'"
                elif action_intent.action == "upload":
                    reasoning_suffix = f" and upload file '{action_intent.value}'"
                elif action_intent.action == "datetime":
                    reasoning_suffix = f" and set value '{action_intent.value}'"

                target_hint = action_intent.target_text or ""
                reason_prefix = selection.reasoning or f"Selected overlay #{overlay_idx}"
                reasoning = f"{reason_prefix}{reasoning_suffix}"
                if target_hint:
                    reasoning += f" for target '{target_hint}'"

                confidence = max(0.0, min(1.0, selection.confidence or 0.5))
                print(
                    f"[PlanGen] Overlay-only plan generated {action_type.value.upper()} on overlay #{overlay_idx} "
                    f"(confidence={confidence:.2f})"
                )
                return VisionPlan(
                    detected_elements=detected_elements,
                    action_steps=[action_step],
                    reasoning=reasoning,
                    confidence=confidence,
                )

            print("[PlanGen] ❌ Overlay-only planning could not determine a target; falling back to full plan.")

        try:
            user_prompt = f"Create a plan to achieve the goal: '{goal_description}' using the numbered elements."
            if additional_context:
                user_prompt += f"\n\nAdditional context to consider: {additional_context}"
            
            plan: VisionPlan = generate_model(
                prompt=user_prompt,
                model_object_type=VisionPlan,
                system_prompt=system_prompt,
                image=screenshot_with_overlays,
            )
            if plan and plan.action_steps:
                # Enforce max_steps limit strictly
                effective_max_steps = max_steps if max_steps is not None else (self.max_steps if self.max_steps is not None else None)
                if effective_max_steps is not None and len(plan.action_steps) > effective_max_steps:
                    print(f"[PlanGen] ⚠️ Plan generated {len(plan.action_steps)} steps, limiting to {effective_max_steps} step(s) as required")
                    plan.action_steps = plan.action_steps[:effective_max_steps]
                    # Update reasoning to reflect the limitation
                    plan.reasoning = f"(Limited to {effective_max_steps} step(s)) {plan.reasoning}"
                
                # if preferred_overlay is not None:
                #     self._apply_preferred_overlay(plan, preferred_overlay, element_data)
                detected_elements = self.convert_indices_to_elements(plan.action_steps, element_data)
                plan.detected_elements = detected_elements
                print(f"✅ Plan created with {len(detected_elements.elements)} detected elements")
                if force_click_only:
                    self._enforce_click_actions(plan)
            return plan
        except Exception as e:
            print(f"❌ Error creating plan with element indices: {e}")
            return None

    # -------------------- Helpers --------------------
    def select_best_overlay(
        self,
        instruction: str,
        element_data: List[Dict[str, Any]],
        *,
        semantic_hint: Optional[SemanticTarget] = None,
        screenshot: Optional[bytes] = None,
        max_samples: Optional[int] = None,
        model: Optional[str] = None,
    ) -> Optional[int]:
        """
        Lightweight helper that asks the LLM to pick a single overlay number.
        Returns the overlay index as an int, or None if selection fails.
        
        Args:
            model: Optional model name to use for this selection. If None, uses default model.
        """
        selection = self._select_overlay_with_language(
            instruction,
            element_data,
            semantic_hint=semantic_hint,
            screenshot=screenshot,
            max_samples=max_samples,
            model=model,
        )
        if selection is None:
            return None
        if isinstance(selection, int):
            return selection
        if isinstance(selection, str):
            try:
                return int(selection.strip())
            except Exception:
                return None
        # Fallback for legacy OverlaySelection object
        overlay_idx = getattr(selection, "overlay_index", None)
        if overlay_idx is None:
            return None
        try:
            return int(overlay_idx)
        except Exception:
            return None
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
        active_goal: Optional[Any],
        page_info: PageInfo,
        page: Any,
        retry_goal: Optional[Any],
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

    def _is_click_goal(self, goal_description: str, active_goal: Optional[Any]) -> bool:
        # Goal system removed - always return False
        if False and active_goal and hasattr(active_goal, '__class__') and active_goal.__class__.__name__ == "ClickGoal":
            return True
        text = (goal_description or "").strip().lower()
        if not text:
            return False
        click_leads = ("click", "press", "tap")
        if text.split(":", 1)[0] in click_leads:
            return True
        return any(text.startswith(f"{prefix} ") or text.startswith(f"{prefix}:") for prefix in click_leads)

    def _enforce_click_actions(self, plan: VisionPlan) -> None:
        if not plan or not plan.action_steps:
            return

        adjusted_steps: List[ActionStep] = []
        mutated = False

        for step in plan.action_steps:
            if step.action == ActionType.CLICK:
                adjusted_steps.append(step)
                continue

            mutated = True
            if step.overlay_index is None:
                print("[PlanGen] ⚠️ Non-click step without overlay removed during click-only enforcement.")
                continue

            adjusted_steps.append(
                ActionStep(
                    action=ActionType.CLICK,
                    overlay_index=step.overlay_index,
                    x=step.x,
                    y=step.y,
                    text_to_type=None,
                    wait_time_ms=step.wait_time_ms,
                    scroll_direction=step.scroll_direction,
                    keys_to_press=None,
                    select_option_text=None,
                    datetime_value=None,
                    upload_file_path=None,
                    url=step.url,
                )
            )

        if not adjusted_steps:
            print("[PlanGen] ⚠️ Click-only enforcement produced no actionable steps; clearing plan.")
            plan.action_steps = []
            return

        if mutated:
            print("[PlanGen] 🔧 Adjusted plan to enforce CLICK-only actions.")
            plan.action_steps = adjusted_steps
            plan.reasoning = (plan.reasoning or "") + " [click-only enforcement]"

    def _select_overlay_with_language(
        self,
        instruction: str,
        element_data: List[Dict[str, Any]],
        *,
        semantic_hint: Optional[SemanticTarget] = None,
        screenshot: Optional[bytes] = None,
        max_samples: Optional[int] = None,
        model: Optional[str] = None,
    ) -> Optional[int]:
        if not element_data:
            return None

        # Prioritise likely clickable overlays and discard huge structural regions
        goal_tokens = self._extract_goal_tokens(instruction)
        hint_terms = self._extract_hint_terms(semantic_hint)
        goal_token_str = ", ".join(goal_tokens) if goal_tokens else "none"

        if max_samples is None:
            max_samples = self.overlay_selection_max_samples
        if max_samples is not None and max_samples <= 0:
            max_samples = None

        samples: List[str] = []
        candidate_lines: List[str] = []
        if element_data:
            try:
                get_event_logger().plan_overlay_candidates([])  # Will be populated below
            except Exception:
                pass
        iterable = element_data if max_samples is None else element_data[:max_samples]
        for elem in iterable:
            idx = elem.get("index")
            role = (elem.get("role") or elem.get("tagName") or "").lower()
            tag = (elem.get("tagName") or "").lower()
            text = (elem.get("textContent") or "").strip()
            aria = (elem.get("ariaLabel") or "").strip()
            
            # Get additional helpful attributes
            placeholder = (elem.get("placeholder") or "").strip()
            name = (elem.get("name") or "").strip()
            elem_id = (elem.get("id") or "").strip()
            elem_type = (elem.get("type") or "").strip()
            value = (elem.get("value") or "").strip()
            href = (elem.get("href") or "").strip()
            
            # Increase snippet lengths for better context
            text_snip = text.strip()[:100]  # Increased from 60
            aria_snip = aria.strip()[:60]   # Increased from 40
            
            # Skip elements with no useful text/aria/placeholder
            if not text_snip and not aria_snip and not placeholder:
                continue
                
            role_str = role or "unknown"
            tag_str = tag or "unknown"
            if role_str == tag_str:
                role_str = ""
            
            # Build comprehensive description
            parts = []
            parts.append(f"#{idx} tag={tag_str}")
            if role_str:
                parts.append(f"role={role_str}")
            if elem_type:
                parts.append(f"type={elem_type}")
            if aria_snip:
                parts.append(f'aria="{aria_snip}"')
            if text_snip:
                parts.append(f'txt="{text_snip}"')
            if placeholder:
                parts.append(f'placeholder="{placeholder[:40]}"')
            if name:
                parts.append(f'name="{name[:30]}"')
            if elem_id:
                parts.append(f'id="{elem_id[:30]}"')
            if value and len(value) < 40:
                parts.append(f'value="{value}"')
            if href and len(href) < 60:
                parts.append(f'href="{href}"')
            
            log_line = f"  • {' '.join(parts)}"
            candidate_lines.append(log_line)
            samples.append(f"- {' '.join(parts)}")
        
        if candidate_lines:
            try:
                get_event_logger().plan_overlay_candidates(candidate_lines)
            except Exception:
                pass

        prompt = (
            "You are given a page screenshot and a list of numbered overlay summaries.\n"
            "Your job is to pick the overlay number that best satisfies the browsing instruction.\n"
            "If NONE of the overlays clearly match the instruction (their text/aria labels do not correspond to the requested control), "
            "respond with 0 to indicate that there is no suitable element.\n"
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
            "- Prefer overlays whose visible text, aria labels, or placeholders contain several instruction or must-have terms.\n"
            "- For search/input tasks, look for elements with type='text', type='search', role='combobox', or 'search' in aria/placeholder.\n"
            "- For typing tasks, prioritize input/textarea elements with matching placeholder, name, or aria labels.\n"
            "- Reject overlays that describe entire job cards, long descriptions, or containers that lack a primary action.\n"
            "- If you cannot find any overlay that reasonably matches the instruction, respond with 0.\n"
            "\n"
            "Examples:\n"
            "- Instruction: 'type into search box' → Choose: textarea role=combobox aria='Search' or input type='search'\n"
            "- Instruction: 'enter email' → Choose: input type='email' or input placeholder='Email'\n"
            "- Instruction: 'click submit button' → Choose: button txt='Submit' or input type='submit'\n"
            "\n"
            "Respond with ONLY the overlay number as an integer (e.g., 5), or 0 if there is no match. No explanation, no JSON.\n"
            "Overlays:\n" + "\n".join(samples)
        )

        try:
            raw_response = generate_text(
                prompt=prompt,
                system_prompt=(
                    "You select the overlay index that best fits the instruction.\n"
                    "If no overlay is a reasonable match, reply with 0.\n"
                    "Reply with just the index as an integer (e.g., 5 or 0)."
                ),
                image=screenshot,
                model=model,
            )
        except Exception as e:
            print(f"[PlanGen][NL] LLM selection error: {e}")
            return None

        if not raw_response:
            return None

        match = re.search(r"\d+", str(raw_response))
        if not match:
            return None

        overlay_index = int(match.group())
        if overlay_index == 0:
            try:
                get_event_logger().plan_overlay_chosen(overlay_index=0, raw_response=raw_response)
            except Exception:
                pass
            return None

        try:
            get_event_logger().plan_overlay_chosen(overlay_index=overlay_index, raw_response=raw_response)
        except Exception:
            pass
        return overlay_index

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
        # Filter out elements with invalid coordinates first
        valid_elements = []
        for e in element_data:
            coords = e.get('normalizedCoords', [])
            if len(coords) == 4:
                y_min, x_min, y_max, x_max = coords
                # Check if coordinates are within valid range (0-1000)
                if (0 <= y_min <= 1000 and 0 <= x_min <= 1000 and 
                    0 <= y_max <= 1000 and 0 <= x_max <= 1000 and
                    y_min < y_max and x_min < x_max):
                    valid_elements.append(e)
                else:
                    print(f"[PlanGen] Filtering out element #{e.get('index')} with invalid coordinates: {coords}")
            else:
                print(f"[PlanGen] Filtering out element #{e.get('index')} with missing coordinates")
        
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
        
        return sorted(valid_elements, key=score_elem)

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
