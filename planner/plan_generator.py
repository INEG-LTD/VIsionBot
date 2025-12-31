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
        include_textless_overlays: bool = False,
        **kwargs,
    ) -> None:
        self.include_detailed_elements = include_detailed_elements
        self.max_detailed_elements = max_detailed_elements
        self.max_steps = max_steps  # Maximum number of steps to generate (None = no limit, uses default 1-3)
        self.merge_overlay_selection = merge_overlay_selection
        self.return_overlay_only = return_overlay_only
        self.include_textless_overlays = include_textless_overlays
        if overlay_selection_max_samples is not None and overlay_selection_max_samples <= 0:
            overlay_selection_max_samples = None
        self.overlay_selection_max_samples = overlay_selection_max_samples


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
        base_knowledge: Optional[List[str]] = None,
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
            base_knowledge=base_knowledge,
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
    def _select_overlay_with_language(
        self,
        instruction: str,
        element_data: List[Dict[str, Any]],
        *,
        semantic_hint: Optional[SemanticTarget] = None,
        screenshot: Optional[bytes] = None,
        max_samples: Optional[int] = None,
        model: Optional[str] = None,
        base_knowledge: Optional[List[str]] = None,
    ) -> Optional[int]:
        if not element_data:
            return None

        # Prioritise likely clickable overlays and discard huge structural regions
        goal_tokens = self._extract_goal_tokens(instruction)
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
            field_label = (elem.get("fieldLabel") or "").strip()  # Associated label/question for form elements
            select_options = elem.get("selectOptions") or ""  # Available options for select elements
            
            # Increase snippet lengths for better context
            text_snip = text.strip()[:100]  # Increased from 60
            aria_snip = aria.strip()[:60]   # Increased from 40

            # Get surrounding context from parent elements (collected during overlay creation)
            context_text = (elem.get("contextText") or "").strip()

            # Skip elements with no useful text/aria/placeholder
            # Exception: Always include select elements even if they have no text (they have options)
            # When include_textless_overlays is enabled, also include elements with surrounding context
            is_select = tag == "select" or role in ("combobox", "listbox")
            has_identifying_info = text_snip or aria_snip or placeholder or is_select
            has_context = context_text and len(context_text) > 10  # Require meaningful context

            if not has_identifying_info and not (self.include_textless_overlays and has_context):
                continue
                
            role_str = role or "unknown"
            tag_str = tag or "unknown"
            if role_str == tag_str:
                role_str = ""
            
            # Build comprehensive description
            parts = []
            parts.append(f"#{idx} tag={tag_str}")
            
            # Only mark as SELECT_FIELD if it's a select element AND has actual options
            # This ensures we only use select: action for elements that actually have options to choose from
            if is_select and select_options:
                parts.append("SELECT_FIELD")  # Prominent marker for select elements with options
            
            if role_str:
                parts.append(f"role={role_str}")
            if elem_type:
                parts.append(f"type={elem_type}")
            # Add field label/question context for form elements (especially important for radios/checkboxes)
            if field_label:
                parts.append(f'field="{field_label[:80]}"')  # Include the question/label context
            if aria_snip:
                parts.append(f'aria="{aria_snip}"')
            if text_snip:
                parts.append(f'txt="{text_snip}"')
            if placeholder:
                parts.append(f'placeholder="{placeholder[:40]}"')
            # Prominently display select options if available
            if select_options:
                # Extract options from the selectOptions string (format: " [options: opt1, opt2, ...]")
                parts.append(f'options={select_options.strip()}')
            if name:
                parts.append(f'name="{name[:30]}"')
            if elem_id:
                parts.append(f'id="{elem_id[:30]}"')
            if value and len(value) < 40:
                parts.append(f'value="{value}"')
            if href and len(href) < 60:
                parts.append(f'href="{href}"')

            # Add surrounding context for elements that lack their own identifying information
            if not has_identifying_info and self.include_textless_overlays and has_context:
                # Include relevant context from surrounding elements
                context_snip = context_text[:150]  # Limit context length
                parts.append(f'context="{context_snip}"')

            log_line = f"  • {' '.join(parts)}"
            candidate_lines.append(log_line)
            samples.append(f"- {' '.join(parts)}")
        
        if candidate_lines:
            try:
                get_event_logger().plan_overlay_candidates(candidate_lines)
            except Exception:
                pass

        # Build base knowledge section if provided
        base_knowledge_section = ""
        if base_knowledge:
            base_knowledge_section = "\n\nBASE KNOWLEDGE (Custom Rules):\n"
            for i, knowledge in enumerate(base_knowledge, 1):
                base_knowledge_section += f"{i}. {knowledge}\n"

        prompt = (
            "You are given a page screenshot and a list of numbered overlay summaries.\n"
            "Your job is to pick the overlay number that best satisfies the browsing instruction.\n"
            "If NONE of the overlays clearly match the instruction (their text/aria labels do not correspond to the requested control), "
            "respond with 0 to indicate that there is no suitable element.\n"
            f'Instruction: "{instruction}"\n'
            f"Instruction terms to align with: {goal_token_str}.\n"
            f"{base_knowledge_section}"
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
            "- ⚠️ CRITICAL DROPDOWN/SELECT HANDLING: For ANY dropdown/select task, you MUST look for elements marked with SELECT_FIELD or showing 'options=' in their description. These elements show available options in the 'options=' field. When you see SELECT_FIELD or options= in an element description, this indicates a dropdown/select field that REQUIRES using 'select:' action (NOT 'click:'). IMPORTANT: Only elements with SELECT_FIELD or options= should use 'select:' action. If an element is tag=select, role=combobox, or role=listbox but does NOT have options= or SELECT_FIELD, the agent will use 'click:' instead.\n"
            "- ⚠️ CRITICAL: When an element shows 'options=[options: Option1, Option2, ...]' in its description, the agent MUST use one of those exact options. Never invent or make up option values that are not in the list. The options= field shows the ONLY valid choices.\n"
            "- When you see SELECT_FIELD or options=, this indicates a dropdown/select field that requires using 'select:' action (not 'click:'). If you see tag=select, role=combobox, or role=listbox WITHOUT options= or SELECT_FIELD, the element does not have options available and should use 'click:' instead.\n"
            "- ⚠️ CRITICAL FILE UPLOAD HANDLING: For ANY file upload/attachment task, you MUST look for elements with type='file', tag=input (type=file), or elements showing upload/attach/browse indicators. When you see type=file, tag=input (type=file), or upload/attach/browse in an element description, this indicates a file upload control that REQUIRES using 'upload:' action (NOT 'click:'). If the instruction mentions uploading, attaching, browsing for files, or selecting files, you MUST choose an element with type=file or upload/attach indicators.\n"
            "- Reject overlays that describe entire job cards, long descriptions, or containers that lack a primary action.\n"
            "- If you cannot find any overlay that reasonably matches the instruction, respond with 0.\n"
            "\n"
            "Examples:\n"
            "- Instruction: 'type into search box' → Choose: textarea role=combobox aria='Search' or input type='search'\n"
            "- Instruction: 'enter email' → Choose: input type='email' or input placeholder='Email'\n"
            "- Instruction: 'click submit button' → Choose: button txt='Submit' or input type='submit'\n"
            "- Instruction: 'select country' or 'choose country' or 'pick country' or 'fill country dropdown' → Choose: SELECT_FIELD tag=select options=[options: USA, Canada, UK...] (MUST have SELECT_FIELD or options=)\n"
            "- Instruction: 'fill dropdown' or 'pick option' or 'select option' → Choose: SELECT_FIELD tag=select with options= field showing available choices (MUST have SELECT_FIELD or options=)\n"
            "- Instruction: 'click dropdown' or 'open dropdown' → If it's a dropdown/select field with SELECT_FIELD or options=, choose that element (the agent will use select: action, not click:). If it's tag=select without options=, choose it anyway (agent will use click:)\n"
            "- Instruction: 'upload file' or 'attach file' or 'browse for file' or 'choose file' or 'select file' → Choose: tag=input (type=file) or element with type=file (MUST have type=file)\n"
            "- Instruction: 'upload resume' or 'attach document' → Choose: tag=input (type=file) or element showing upload/attach indicators (MUST have type=file or upload/attach text)\n"
            "- Instruction: 'click upload button' or 'click file input' → If it's a file upload control (has type=file or tag=input (type=file)), still choose that element (the agent will use upload: action, not click:)\n"
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

    def convert_indices_to_elements(self, action_steps: List[ActionStep], element_data: List[Dict[str, Any]]) -> PageElements:
        detected_elements = []
        used_indices = set()
        for step in action_steps:
            if step.overlay_index is not None and step.overlay_index not in used_indices:
                matching_data = next((elem for elem in element_data if elem.get('index') == step.overlay_index), None)
                if matching_data:
                    description = matching_data.get('description') or matching_data.get('textContent') or f"Element {step.overlay_index}"
                    
                    # Try to get a CSS selector for element_label (used by handlers)
                    css_selector = None
                    if matching_data.get('id'):
                        css_selector = f"#{matching_data['id']}"
                    elif matching_data.get('name'):
                        css_selector = f'[name="{matching_data["name"]}"]'
                    elif matching_data.get('cssSelector'):
                        css_selector = matching_data['cssSelector']
                    
                    # Use CSS selector as label if available, otherwise use description
                    label = css_selector or description
                    
                    box = matching_data.get('normalizedCoords') or matching_data.get('box_2d') or [0, 0, 0, 0]
                    element = DetectedElement(
                        element_label=label,
                        description=description,
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
