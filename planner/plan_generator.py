"""
Plan generation utilities extracted from BrowserVisionBot.
"""
from __future__ import annotations

from typing import Optional, List, Dict, Any

from pydantic import BaseModel

from ai_utils import generate_model
from models import VisionPlan, PageElements
from models.core_models import DetectedElement, PageSection, ActionStep, ActionType, PageInfo
from utils.memory_store import stable_sig


class PlanGenerator:
    """Generates VisionPlan objects from page context and overlays."""

    def __init__(
        self,
        *,
        include_detailed_elements: bool = True,
        max_detailed_elements: int = 20,
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
        active_goals: Optional[List[Any]] = None,
        retry_goals: Optional[List[Any]] = None,
        page: Any = None,
    ) -> Optional[VisionPlan]:
        """Create an action plan using the already-detected elements list."""
        print(f"Creating plan for goal: {goal_description}\n")

        goal_context = self._build_goal_context_block(
            goal_description, active_goals or [], page_info, page, retry_goals or []
        )

        system_prompt = f"""
        You are a web automation assistant. Create a plan based on the ACTIVE GOAL DESCRIPTIONS below.

        Current page: {page_info.url}
        DETECTED ELEMENTS: {[elem.model_dump() for elem in detected_elements.elements]}
        {goal_context}

        CRITICAL INSTRUCTIONS:
        1. Focus ONLY on the ACTIVE GOAL DESCRIPTIONS above. Prioritize fields or targets marked as pending.
        2. Do not re-complete fields already marked as completed (✅).
        3. Use specialized actions for select/upload/datetime/press/back/forward/stop when appropriate.
        4. Return 1-3 action steps that move directly toward finishing the goal.
        """

        try:
            plan: VisionPlan = generate_model(
                prompt="Create a plan to achieve the goal using the detected elements.",
                model_object_type=VisionPlan,
                reasoning_level="low",
                system_prompt=system_prompt,
                model="gpt-5-nano",
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
        relevant_overlay_indices: Optional[List[int]] = None,
        active_goals: Optional[List[Any]] = None,
        retry_goals: Optional[List[Any]] = None,
        page: Any = None,
        dedup_mode: str = "off",
    ) -> Optional[VisionPlan]:
        """Create a plan leveraging overlay indices to refer to elements precisely."""
        # Build goal description block with real-time goal analyses
        goal_context = self._build_goal_context_block(
            goal_description, active_goals or [], page_info, page, retry_goals or []
        )

        # Heuristic refinement for link/navigation intents
        if relevant_overlay_indices:
            refined = self.refine_overlays_by_goal(element_data, relevant_overlay_indices, goal_description)
            if refined:
                relevant_overlay_indices = refined

        # Build available elements list
        elements_list_lines = []
        filtered_indices = set(relevant_overlay_indices or [])
        if filtered_indices:
            filtered = [e for e in element_data if e.get('index') in filtered_indices]
        else:
            filtered = list(element_data)

        # Optional: rank elements and deduplicate by signature if prompt implies
        ranked = self._rank_elements(filtered, f"{goal_description} {additional_context}")

        # Dedup if requested by prompt or policy
        do_dedup = self._should_dedup_for(dedup_mode, goal_description, additional_context)
        seen_sigs: set[str] = set()
        for elem in ranked:
            if do_dedup:
                sig = self._signature_for_candidate(elem)
                if sig and sig in seen_sigs:
                    continue
                if sig:
                    seen_sigs.add(sig)
            elements_list_lines.append(self._build_element_detail_line(elem, page_info))

        if self.include_detailed_elements and element_data:
            detailed_list = self._build_detailed_element_context(element_data, page_info, f"{goal_description} {additional_context}")
            elements_list_lines.append("\nDETAILED ELEMENTS (top candidates):\n" + "\n".join(detailed_list[: self.max_detailed_elements]))

        excluded_indices = None
        if filtered_indices:
            excluded_indices = [e['index'] for e in element_data if e.get('index') not in filtered_indices]
        if excluded_indices and do_dedup:
            print(f"🧹 Dedup active; excluding {len(excluded_indices)} overlays from prompt")

        system_prompt = f"""
        You are a web automation assistant. Create a plan using the numbered overlays.

        Current page: {page_info.url}
        {goal_context}

        AVAILABLE ELEMENTS (numbered overlays):
        {chr(10).join(elements_list_lines[:200])}

        INSTRUCTIONS:
        1. Use the overlay numbers exactly when referring to targets.
        2. Prefer elements that match the active goal descriptions and pending tasks.
        3. Use specialized actions when appropriate:
           - HANDLE_SELECT, HANDLE_UPLOAD, HANDLE_DATETIME, PRESS, BACK, FORWARD, STOP
        4. Return 1-3 precise action steps.
        """

        try:
            plan: VisionPlan = generate_model(
                prompt="Create a plan to achieve the goal using the numbered elements.",
                model_object_type=VisionPlan,
                reasoning_level="minimal",
                system_prompt=system_prompt,
                model="gpt-5-nano",
                image=screenshot_with_overlays,
            )
            if plan and plan.action_steps:
                detected_elements = self.convert_indices_to_elements(plan.action_steps, element_data)
                plan.detected_elements = detected_elements
                print(f"✅ Plan created with {len(detected_elements.elements)} detected elements")
            return plan
        except Exception as e:
            print(f"❌ Error creating plan with element indices: {e}")
            return None

    # -------------------- Helpers --------------------
    def _build_goal_context_block(
        self,
        goal_description: str,
        active_goals: List[Any],
        page_info: PageInfo,
        page: Any,
        retry_goals: List[Any],
    ) -> str:
        goal_descriptions: List[str] = []
        if active_goals:
            print("📋 Gathering goal descriptions for plan generation...")
            for goal in active_goals:
                try:
                    # Construct a basic context for the goal to describe itself
                    from goals.base import GoalContext, BrowserState
                    basic_context = GoalContext(
                        initial_state=BrowserState(
                            timestamp=0, url=page_info.url, title=page_info.title,
                            page_width=page_info.doc_width, page_height=page_info.doc_height,
                            scroll_x=0, scroll_y=0,
                        ),
                        current_state=BrowserState(
                            timestamp=0, url=page_info.url, title=page_info.title,
                            page_width=page_info.doc_width, page_height=page_info.doc_height,
                            scroll_x=0, scroll_y=0,
                        ),
                        page_reference=page,
                    )
                    goal_desc = goal.get_description(basic_context)
                    goal_descriptions.append(goal_desc)
                    print(f"   📝 {goal.__class__.__name__}: {goal_desc[:100]}...")
                except Exception as e:
                    print(f"   ⚠️ Error getting description for {getattr(goal, '__class__', type(goal)).__name__}: {e}")
                    goal_descriptions.append(f"{getattr(goal, '__class__', type(goal)).__name__}: {getattr(goal, 'description', '')}")

        retry_context = ""
        if retry_goals:
            retry_info = []
            for goal in retry_goals:
                retry_info.append(f"- {goal.__class__.__name__}: Retry attempt {goal.retry_count}/{goal.max_retries}")
                if getattr(goal, 'retry_reason', None):
                    retry_info.append(f"  Reason: {goal.retry_reason}")
            retry_context = f"""
        
        RETRY CONTEXT:
        The following goals have requested retries due to previous failures:
        {chr(10).join(retry_info)}
        
        IMPORTANT: This is a retry attempt. The previous plan failed because the goals detected issues.
        Make sure to:
        - Look for different elements that might match the goal requirements
        - Consider alternative approaches to achieve the same goal
        - Be more careful about element selection and targeting
        - Address the specific failure reasons mentioned above
        """
            print(f"🔄 Retry context included in plan generation: {len(retry_goals)} goals requesting retry")

        goal_context = ""
        if goal_descriptions:
            goal_context = f"""
                                USER GOAL: {goal_description}
                                
                                ACTIVE GOAL DESCRIPTIONS:
                                {chr(10).join(f"- {desc}" for desc in goal_descriptions)}
                                
                                IMPORTANT: Use the goal descriptions above to understand exactly what needs to be accomplished. 
                                These descriptions contain real-time analysis of the current page state and goal progress.
                                Pay special attention to:
                                - Which specific fields need to be filled (don't fill unnecessary fields)
                                - Current completion status of fields (✅ = completed, ⏳ = needs filling)
                                - Any validation errors that need to be addressed (❌ indicators)
                                - Specific targets for click or navigation goals
                                - Progress metrics (📊 indicators show completion ratios)
                                {retry_context}
                            """
        return goal_context

    def _signature_for_candidate(self, elem: Dict[str, Any]) -> str:
        try:
            tag = str((elem.get('tagName') or '')).lower()
            href = str((elem.get('href') or '')).strip().lower()
            txt = str((elem.get('textContent') or elem.get('description') or '')).strip().lower()
            aria = str((elem.get('ariaLabel') or '')).strip().lower()
            eid = str((elem.get('id') or '')).strip().lower()
            role = str((elem.get('role') or '')).strip().lower()
            sig_text = "|".join([tag, href, txt, aria, eid, role])
            return stable_sig(sig_text)
        except Exception:
            return ""

    def _should_dedup_for(self, dedup_mode: str, goal_description: str, additional_context: str) -> bool:
        mode = (dedup_mode or "off").strip().lower()
        if mode == "on":
            return True
        if mode != "auto":
            return False
        # Auto: only when prompt explicitly requests no repeats/unique processing
        txt = f"{goal_description}\n{additional_context}".lower()
        keywords = [
            "no duplicates", "avoid duplicates", "without duplicates", "skip duplicates", "deduplicate", "de-duplicate",
            "don't repeat", "do not repeat", "no repeat", "no repeats", "without repeating", "not repeating",
            "each only once", "only once", "one time each", "each item once", "each listing once", "unique", "distinct",
            "do not click the same", "don't click the same", "do not re-open", "don't re-open",
            "do not click again", "don't click again", "no duplicate clicks", "avoid duplicate clicks",
        ]
        if any(k in txt for k in keywords):
            return True
        has_duplicate_word = any(w in txt for w in ["duplicate", "duplicates", "dupe", "dup"])
        has_action_word = any(w in txt for w in ["click", "open", "visit", "select", "tap"])
        if has_duplicate_word and has_action_word:
            return True
        if ("do not click" in txt or "don't click" in txt) and any(w in txt for w in ["same", "again", "duplicate"]):
            return True
        return False

    def convert_indices_to_elements(self, action_steps: List[ActionStep], element_data: List[Dict[str, Any]]) -> PageElements:
        detected_elements = []
        used_indices = set()
        for step in action_steps:
            if step.target_element_index is not None and step.target_element_index not in used_indices:
                matching_data = next((elem for elem in element_data if elem['index'] == step.target_element_index), None)
                if matching_data:
                    element = DetectedElement(
                        element_label=f"Element {step.target_element_index}",
                        description=matching_data['description'],
                        element_type=self._infer_element_type(matching_data),
                        is_clickable=self._infer_clickable(matching_data),
                        box_2d=matching_data['normalizedCoords'],
                        section=PageSection.CONTENT,
                        confidence=0.9,
                        overlay_number=step.target_element_index,
                    )
                    detected_elements.append(element)
                    used_indices.add(step.target_element_index)
                    print(f"  ✅ Converted element #{step.target_element_index}: {element.element_label}")
        return PageElements(elements=detected_elements)

    def _rank_elements(self, element_data: List[Dict[str, Any]], goal_text: str) -> List[Dict[str, Any]]:
        def score_elem(e: Dict[str, Any]) -> int:
            s = 0
            txt = (e.get('textContent') or e.get('description') or "").lower()
            tag = (e.get('tagName') or '').lower()
            # Prioritize links/buttons for click/navigation intents
            if any(w in goal_text.lower() for w in ["click", "open", "visit", "go to", "navigate", "link", "button"]):
                if tag in ("a", "button"): s += 3
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

    def refine_overlays_by_goal(
        self, element_data: List[Dict[str, Any]], indices: Optional[List[int]], goal_text: str
    ) -> Optional[List[int]]:
        """Refine overlay indices based on goal text (domain/phrase matching)."""
        try:
            if not element_data:
                return indices
            goal_l = (goal_text or "").lower()
            import re as _re
            domain = None
            m = _re.search(r"([a-z0-9][a-z0-9-]*\.[a-z]{2,})(?!\w)", goal_l)
            if m:
                domain = m.group(1)
            wants_link = 'link' in goal_l or 'open' in goal_l or 'visit' in goal_l
            exact_phrase = None
            qm = _re.search(r"['\"]([^'\"]+)['\"]", goal_l)
            if qm:
                exact_phrase = qm.group(1).strip()
            index_set = set(indices) if indices else set([e['index'] for e in element_data])
            scored = []
            for elem in element_data:
                idx = elem.get('index')
                if idx not in index_set:
                    continue
                tag = (elem.get('tagName') or '').lower()
                href = (elem.get('href') or '').lower()
                txt = (elem.get('textContent') or elem.get('description') or '').lower()
                score = 0
                if wants_link and tag == 'a':
                    score += 2
                if domain and domain in href:
                    score += 3
                if exact_phrase and exact_phrase in txt:
                    score += 2
                for tok in ["company", "careers", "pricing", "about", "contact", "apply"]:
                    if tok in goal_l and tok in txt:
                        score += 1
                scored.append((score, idx))
            scored.sort(reverse=True)
            if not scored:
                return indices
            # Keep top 40 for safety to avoid over-pruning
            top = [idx for (_s, idx) in scored[:40]]
            return top
        except Exception:
            return indices
