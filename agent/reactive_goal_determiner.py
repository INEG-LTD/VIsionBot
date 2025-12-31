"""
Reactive Goal Determiner - Determines the next action based on current viewport and state.

Step 2: LLM-based reactive goal determination that decides what to do RIGHT NOW
based on what's visible in the viewport, not pre-planning.
"""

from typing import Optional, List, ClassVar, Set, Union, Dict, Any
from pydantic import BaseModel, Field, field_validator
import re

from session_tracker import BrowserState, Interaction
from agent.completion_contract import EnvironmentState
from ai_utils import (
    generate_model,
    ReasoningLevel,
    get_default_agent_model,
    get_default_agent_reasoning_level,
)
from utils.event_logger import get_event_logger


class NextAction(BaseModel):
    """Structured LLM response for next action determination"""
    action: str = Field(description="The specific action to take RIGHT NOW in proper command format. For CLICK commands: MUST include the element TYPE (button, link, div, input, etc.) AND be descriptive - e.g., 'click: Google Search button', 'click: first article link titled 'Introduction'', 'click: Accept all cookies button', 'click: search suggestion 'yahoo finance' link'. For TYPE commands: MUST be descriptive with element type - e.g., 'type: John Doe in name input field'. For PRESS commands: MUST be brief - just the key name (e.g., 'press: Enter', 'press: Escape'). NEVER use vague terms like 'first element', 'that button', 'the field', or ambiguous text without element type for click/type commands.")
    reasoning: str = Field(description="Why this action is appropriate given the current viewport")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence that this is the right next action")
    expected_outcome: str = Field(description="What should happen after executing this action")
    needs_exploration: bool = Field(description="True if the action requires exploring/searching for elements not visible in current viewport (e.g., scrolling to find a field)")

    VALID_COMMANDS: ClassVar[Set[str]] = {
        "click",
        "type",
        "press",
        "scroll",
        "extract",
        "defer",
        "navigate",
        "back",
        "forward",
        "subagents",
        "form",
        "select",
        "upload",
        "datetime",
        "stop",
        "open",
        "handle_datetime",
        "mini_goal",
    }

    @field_validator("action", mode="before")
    @classmethod
    def _normalize_action(cls, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("action must be a string")
        action = value.strip()
        if not action:
            raise ValueError("action cannot be empty")

        action = cls._ensure_colon(action)
        command, body = action.split(":", 1)
        command = command.strip().lower()
        body = body.strip()

        if command not in cls.VALID_COMMANDS:
            raise ValueError(f"Unsupported action command: '{command}'")

        if command == "click":
            body = cls._normalize_click_body(body)
            action = f"click: {body}"
        elif command == "type":
            body = cls._normalize_type_body(body)
            action = f"type: {body}"
        elif command == "press":
            body = cls._normalize_press_body(body)
            action = f"press: {body}"
        elif command == "scroll":
            body = cls._normalize_scroll_body(body)
            action = f"scroll: {body}"
        elif command == "extract":
            if not body:
                raise ValueError("extract command requires a description")
            action = f"extract: {body}"
        elif command == "defer":
            action = "defer" if not body else f"defer: {body}"
        elif command == "navigate":
            if not body:
                raise ValueError("navigate command requires a target URL or site")
            action = f"navigate: {body}"
        elif command in {"back", "forward"}:
            if not body:
                body = "1"
            if not body.isdigit():
                raise ValueError(f"{command} command requires numeric steps")
            action = f"{command}: {body}"
        elif command == "subagents":
            if not body:
                raise ValueError("subagents command requires a mode")
            action = f"subagents: {body}"
        elif command == "mini_goal":
            if not body:
                raise ValueError("mini_goal command requires an instruction")
            action = f"mini_goal: {body}"
        elif command in {"form", "select", "upload", "datetime", "stop", "open", "handle_select", "handle_datetime"}:
            if not body:
                raise ValueError(f"{command} command requires additional detail")
            action = f"{command}: {body}"

        return action

    @staticmethod
    def _ensure_colon(text: str) -> str:
        if ":" in text:
            head, tail = text.split(":", 1)
            return f"{head.strip()}: {tail.strip()}"
        match = re.match(r"^(?P<cmd>\w+)\s+(?P<body>.+)$", text)
        if not match:
            raise ValueError("action must contain a command and description")
        cmd = match.group("cmd")
        body = match.group("body").strip()
        if cmd.lower() == "click" and body.lower().startswith("on "):
            body = body[3:].strip()
        return f"{cmd}: {body}"

    @staticmethod
    def _normalize_click_body(body: str) -> str:
        if not body:
            raise ValueError("click command requires a target")
        if body.lower().startswith("on "):
            body = body[3:].strip()
        if not re.search(r"\b(button|link|tab|checkbox|radio|option|div|input|icon|item)\b", body, flags=re.IGNORECASE):
            if not body.lower().endswith("link"):
                body = f"{body} link"
        return body.strip()

    @staticmethod
    def _normalize_type_body(body: str) -> str:
        if not body:
            raise ValueError("type command requires text and target")
        lowered = body.lower()
        if ":" in body and not re.search(r"\b(in|into)\b", lowered):
            pass  # assume already structured
        elif not re.search(r"\b(in|into)\b", lowered):
            raise ValueError("type command must specify target field using 'in'")
        if not re.search(r"\b(field|input|area|box|textbox)\b", lowered):
            body = re.sub(r"\b(in|into)\b\s*", lambda m: f"{m.group(0)}input field ", body, count=1, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", body).strip()

    @staticmethod
    def _normalize_press_body(body: str) -> str:
        key = body.strip()
        if not key:
            raise ValueError("press command requires a key")
        if re.search(r"\s", key):
            raise ValueError("press command should only contain a single key")
        return key

    @staticmethod
    def _normalize_scroll_body(body: str) -> str:
        direction = body.strip().lower()
        allowed = {"up", "down", "left", "right"}
        if direction not in allowed:
            raise ValueError("scroll command must be one of: up, down, left, right")
        return direction


class ReactiveGoalDeterminer:
    """
    Determines the next action reactively based on:
    - Current viewport (what's visible on screen)
    - Environment state (interactions, scroll position, etc.)
    - User prompt (what we're trying to achieve)
    
    This is reactive - it only determines ONE action to take NOW, not a sequence.
    """
    
    def __init__(
        self,
        user_prompt: str,
        base_knowledge: Optional[List[str]] = None,
        *,
        model_name: Optional[str] = None,
        reasoning_level: Union[ReasoningLevel, str, None] = None,
        image_detail: str = "low",
        interaction_summary_limit: Optional[int] = None,
        include_overlays_in_agent_context: bool = True,
    ):
        """
        Initialize the reactive goal determiner.
        
        Args:
            user_prompt: The user's high-level goal
            base_knowledge: Optional list of knowledge rules/instructions that guide the agent's behavior.
                           Example: ["just press enter after you've typed a search term into a search field"]
            image_detail: Image detail level for vision API ("low" for faster, "high" for more accurate).
                         Default: "low" for better performance.
            interaction_summary_limit: Max interactions to include in the prompt.
                                       None means include all interactions. Default: None.
        """
        self.user_prompt = user_prompt
        self.base_knowledge = base_knowledge or []
        self.model_name = model_name or get_default_agent_model()
        if reasoning_level is None:
            reasoning = ReasoningLevel.coerce(get_default_agent_reasoning_level())
        else:
            reasoning = ReasoningLevel.coerce(reasoning_level)
        self.reasoning_level: ReasoningLevel = reasoning
        self.image_detail = image_detail
        self._system_prompt_cache: dict[bool, str] = {}  # Cache system prompts by is_exploring
        self.interaction_summary_limit = interaction_summary_limit
        self.include_overlays_in_agent_context = include_overlays_in_agent_context
    
    def determine_next_action(
        self,
        environment_state: EnvironmentState,
        screenshot: bytes,
        is_exploring: bool = False,
        viewport_snapshot: Optional[BrowserState] = None,
        failed_actions: Optional[List[str]] = None,
        ineffective_actions: Optional[List[str]] = None,
        overlay_data: Optional[List[Dict[str, Any]]] = None
    ) -> tuple[Optional[str], bool, Optional[str]]:
        """
        Determine the single next action to take based on current state.
        
        Args:
            environment_state: Current environment state
            screenshot: Current screenshot (viewport or full-page depending on is_exploring)
            is_exploring: If True, we're in exploration mode and screenshot is full-page
            viewport_snapshot: Optional viewport snapshot for comparison when exploring
            overlay_data: Optional list of overlay element data with descriptions, SELECT_FIELD markers, options, etc.
            
        Returns:
            Tuple of (action_command, needs_exploration, reasoning):
            - action_command: Action to take (e.g., "type: John Doe in name field") or None
              NOTE: In exploration mode, this MUST be "scroll: up" or "scroll: down"
            - needs_exploration: True if next iteration should use full-page screenshot
            - reasoning: Why this action was chosen (for inclusion in interaction history)
        """
        try:
            # Identify what element we're looking for if in exploration mode
            missing_element = None
            if is_exploring:
                remaining_tasks = self._identify_remaining_tasks(environment_state.interaction_history)
                if remaining_tasks:
                    missing_element = remaining_tasks
            
            # Generate single action directly
            response = self._generate_single_action(
                environment_state,
                screenshot,
                is_exploring,
                missing_element,
                viewport_snapshot,
                failed_actions or [],
                ineffective_actions or [],
                overlay_data
            )
            
            if not response:
                print("⚠️ No action generated")
                return None, True, None  # Return None and indicate exploration is needed
            
            action = response.action
            reasoning = response.reasoning
            
            # In exploration mode, validate that only scroll commands are returned
            if is_exploring:
                if not action or not action.startswith("scroll:"):
                    print(f"⚠️ Exploration mode returned invalid command: {action}")
                    print("   Forcing scroll command based on reasoning...")
                    # Try to extract scroll direction from reasoning
                    reasoning_lower = response.reasoning.lower()
                    if "above" in reasoning_lower or "up" in reasoning_lower:
                        action = "scroll: up"
                    elif "below" in reasoning_lower or "down" in reasoning_lower:
                        action = "scroll: down"
                    else:
                        # Default to down if unclear
                        action = "scroll: down"
                    print(f"   Converted to: {action}")
            
            try:
                get_event_logger().action_determined(
                    action=action,
                    reasoning=response.reasoning,
                    confidence=response.confidence,
                    expected_outcome=response.expected_outcome
                )
            except Exception:
                pass
            if response.needs_exploration and not is_exploring:
                print("   🔍 Needs exploration: element not visible in viewport")
            
            return action, response.needs_exploration, reasoning
            
        except Exception as e:
            print(f"⚠️ ReactiveGoalDeterminer error: {e}")
            import traceback
            traceback.print_exc()
            return None, False, None
    
    def _generate_single_action(
        self,
        environment_state: EnvironmentState,
        screenshot: bytes,
        is_exploring: bool,
        missing_element: Optional[str],
        viewport_snapshot: Optional[BrowserState],
        failed_actions: List[str] = None,
        ineffective_actions: List[str] = None,
        overlay_data: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[NextAction]:
        """
        Generate a single action.
        
        Returns:
            NextAction or None if no action can be determined
        """
        try:
            system_prompt = self._build_system_prompt(is_exploring)
            user_prompt_text = self._build_action_prompt(
                environment_state, 
                is_exploring, 
                missing_element,
                viewport_snapshot,
                failed_actions or [],
                ineffective_actions or [],
                overlay_data
            )
            action = generate_model(
                prompt=user_prompt_text,
                model_object_type=NextAction,
                system_prompt=system_prompt,
                image=screenshot,
                image_detail=self.image_detail,
                model=self.model_name,
                reasoning_level=self.reasoning_level,
            )
            return action
        except Exception as e:
            print(f"⚠️ Error generating action: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    
    def _build_system_prompt(self, is_exploring: bool = False) -> str:
        """Build system prompt for reactive action determination"""
        # Use cached prompt if available
        if is_exploring in self._system_prompt_cache:
            return self._system_prompt_cache[is_exploring]
        
        screenshot_note = "full-page screenshot (entire page)" if is_exploring else "viewport screenshot (currently visible area)"
        
        exploration_rules = """
🔍 EXPLORATION MODE (Full-Page View):
- Job: Determine scroll direction to bring target element into viewport
- Commands: Only "scroll: up" or "scroll: down"
- Logic: Target above current viewport → scroll up; Target below → scroll down
- Exit: When element enters viewport, you'll switch to interaction mode
""" if is_exploring else ""
        
        # Build base knowledge section if provided
        base_knowledge_section = ""
        if self.base_knowledge:
            base_knowledge_section = "\n\nBASE KNOWLEDGE (Custom Rules):\n"
            for i, knowledge in enumerate(self.base_knowledge, 1):
                base_knowledge_section += f"{i}. {knowledge}\n"
        
        prompt = f"""
You determine the NEXT SINGLE ACTION based on current state ({screenshot_note}).

{exploration_rules}
{base_knowledge_section}

ACTION RULES:
1. SELECT FIELDS (with SELECT_FIELD or options=): Use "select: <exact-option> in <field>"
   - Must use an option from the options= list (never invent values)
   - If tag=select but NO options= shown → use "click:" (options not loaded)

2. FILE UPLOADS (type=file or upload/attach/browse): Use "upload: [file] in <target>"
   - With file: "upload: resume.pdf in file input"
   - Without file: "upload: file input" (opens picker)
   - Never invent filenames

3. TEXT INPUTS: Fields auto-clear before typing - type complete desired value
   - "type: john@example.com in email input field" (not just "john")

4. EXTRACTION: Use "extract: <what>" when user wants data retrieved
   - Keywords: extract, get, find, collect, list, show, display

5. COMMAND FORMAT:
   - Click: Must include element type (button/link/div) - "click: Submit button"
   - Type: Must include field type - "type: John in name input field"
   - Press: Just key name - "press: Enter"

6. AVOID REPEATING: Failed actions mentioned in prompt should not be repeated

7. DEFER: Use "defer:" when user requests manual control or captcha appears
   - If "defer" already shows "resumed" in interactions, proceed (don't defer again)

COMMANDS:
{'' if is_exploring else '''- extract: <what> - Extract visible data from page
- click: <type> <description> - Interact with element (must specify type: button/link/etc)
- type: <text> in <field-type> <name> - Enter text (field auto-clears first)
- select: <option> in <dropdown> - Pick option (only if element has SELECT_FIELD/options=)
- upload: [file] in <target> - Handle file uploads (file optional, opens picker if omitted)
- press: <key> - Press single key (Enter/Escape/Tab)
- navigate: <url> - Direct navigation (prefer back/forward if in history)
- back: [n] / forward: [n] - Navigate history (default n=1)
- defer: [msg|seconds] - Pause for user (use when captcha/manual input needed)
- scroll: <up|down> - Move viewport
- form: <description> - Fill entire form
- datetime: <value> in <picker> - Set date/time
- mini_goal: <instruction> - Focus on complex sub-task'''}
{'- scroll: <up|down> - ONLY command available in exploration mode' if is_exploring else ''}

EXAMPLES:
✅ "select: Canada in country dropdown" | Element: options=[Canada, USA, UK]
❌ "click: Canada option" | Should use select: for dropdowns with options=

✅ "upload: resume.pdf in file input" | User provided file
❌ "click: file input" | Should use upload: for type=file elements

✅ "type: john@example.com in email input field" | Complete value + field type
❌ "type: john in field" | Incomplete value + vague target

✅ "click: Submit button" | Specific type + description
❌ "click: button" | Too vague

{'In exploration mode: Only return scroll commands to bring target into viewport' if is_exploring else 'If element is visible, interact with it. Set needs_exploration=True only if NO valid action possible.'}
"""
        # Cache the prompt
        self._system_prompt_cache[is_exploring] = prompt
        return prompt
    
    def _build_action_prompt(
        self, 
        state: EnvironmentState, 
        is_exploring: bool = False, 
        missing_element: Optional[str] = None,
        viewport_snapshot: Optional[BrowserState] = None,
        failed_actions: List[str] = None,
        ineffective_actions: List[str] = None,
        overlay_data: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Build prompt for determining next action"""
        
        # Summarize what's been done
        interaction_summary = self._summarize_interactions(state.interaction_history)
        
        # Extract what still needs to be done from user prompt and interactions
        remaining_tasks = self._identify_remaining_tasks(state.interaction_history)
        
        # Check for pending select fields that need an option to be selected
        pending_select_info = self._check_pending_select_fields(state.interaction_history)
        
        # Add failed and ineffective actions context
        ineffective_actions_context = ""
        all_ineffective = []
        
        if failed_actions:
            all_ineffective.extend([(action, "failed") for action in failed_actions])
        if ineffective_actions:
            all_ineffective.extend([(action, "succeeded but no change") for action in ineffective_actions])
        
        if all_ineffective:
            ineffective_actions_context = "\n⚠️ IMPORTANT: The following actions were recently tried and did NOT yield any page change (same URL, same DOM state):\n"
            for i, (action, reason) in enumerate(all_ineffective, 1):
                ineffective_actions_context += f"   {i}. {action} ({reason})\n"
            ineffective_actions_context += "\nDO NOT suggest these same actions again. They were ineffective. Try a different approach or different element.\n"
        
        exploration_context = ""
        if is_exploring:
            if missing_element:
                exploration_context = f"\n🔍 EXPLORATION MODE: You are specifically looking for: {missing_element}\n"
            elif remaining_tasks:
                exploration_context = f"\n🔍 EXPLORATION MODE: You are searching for elements needed to complete: {remaining_tasks}\n"
            else:
                exploration_context = "\n🔍 EXPLORATION MODE: Use the full-page screenshot to locate the target element.\n"
        
        nav_summary = self._summarize_navigation_history(
            getattr(state, "url_history", []),
            getattr(state, "url_pointer", None)
        )
        
        # Format overlay information if available
        # Only include overlay context if configured to do so
        overlay_context = ""
        if overlay_data and self.include_overlays_in_agent_context:
            # Filter to only interactive/actionable elements (similar to what plan_generator does)
            # Focus on elements that are likely to be interacted with
            relevant_elements = []
            for elem in overlay_data:
                idx = elem.get("index")
                tag = (elem.get("tagName") or "").lower()
                role = (elem.get("role") or "").lower()
                text = (elem.get("textContent") or "").strip()
                aria = (elem.get("ariaLabel") or "").strip()
                placeholder = (elem.get("placeholder") or "").strip()
                select_options = elem.get("selectOptions") or ""
                
                # Include elements that are interactive or have useful text
                is_select = tag == "select" or role in ("combobox", "listbox")
                is_native_select = tag == "select"  # Native <select> elements
                has_content = text or aria or placeholder or is_select
                
                if has_content:
                    # Build description similar to plan_generator format
                    parts = []
                    parts.append(f"#{idx} tag={tag}")
                    
                    # Highlight SELECT_FIELD for native selects (always) or custom selects with options
                    # Native <select> elements should always use select: action, even if options aren't shown
                    # because the select handler can extract options from the DOM
                    if is_native_select or (is_select and select_options):
                        parts.append("SELECT_FIELD")
                    
                    if role and role != tag:
                        parts.append(f"role={role}")
                    
                    if placeholder:
                        parts.append(f'placeholder="{placeholder[:40]}"')
                    
                    # Include select options prominently
                    if select_options:
                        parts.append(f'options={select_options.strip()}')
                    
                    if text:
                        parts.append(f'txt="{text[:60]}"')
                    elif aria:
                        parts.append(f'aria="{aria[:60]}"')
                    
                    description = " ".join(parts)
                    
                    # Add relationship/grouping information
                    related_elements = elem.get("relatedElements", [])
                    group_size = elem.get("groupSize", 0)
                    if related_elements and group_size > 1:
                        # Show which other elements are in the same group
                        related_indices = [str(r) for r in related_elements[:5]]  # Limit to 5 for brevity
                        if len(related_elements) > 5:
                            related_indices.append(f"... ({group_size} total in group)")
                        group_info = f"[GROUP: elements #{', '.join(related_indices)} belong to same question/group]"
                        parts.append(group_info)
                        description = " ".join(parts)
                    
                    # Add recommended action hint for elements with options
                    action_hint = ""
                    if is_native_select or (is_select and select_options):
                        # Extract first real option (skip placeholder like "Please select one")
                        options_text = select_options.strip()
                        if options_text.startswith("[options:"):
                            # Parse options from format: [options: Option1, Option2, ...]
                            options_list = options_text[9:].rstrip("]").split(",")
                            # Find first non-placeholder option
                            for opt in options_list:
                                opt_clean = opt.strip()
                                if opt_clean and opt_clean.lower() not in ["please select one", "select one", "--", ""]:
                                    action_hint = f" → Recommended: select: {opt_clean} in {tag} field"
                                    break
                        if not action_hint:
                            action_hint = f" → Recommended: select: <option> in {tag} field"
                    
                    if action_hint:
                        description += action_hint
                    
                    relevant_elements.append(description)
            
            if relevant_elements:
                # Limit to most relevant elements (prioritize select fields, inputs, buttons)
                # Show up to 30 elements to give good context without overwhelming
                overlay_context = "\n\nAVAILABLE INTERACTIVE ELEMENTS (numbered overlays visible in screenshot):\n"
                overlay_context += "These elements are marked with numbered red overlays in the screenshot.\n"
                overlay_context += "Pay special attention to elements marked with SELECT_FIELD - these are dropdown/select fields that require 'select:' action.\n"
                overlay_context += "Elements showing 'options=' have available options listed - use one of those exact options when selecting.\n"
                overlay_context += "Elements with '→ Recommended: select:' show a suggested action format - follow this format when interacting with that element.\n"
                overlay_context += "Elements with '[GROUP: elements #X, #Y, ...]' belong to the same question/group - clicking one when another in the group is already selected may be ineffective.\n\n"
                for elem_desc in relevant_elements[:30]:  # Limit to 30 most relevant
                    overlay_context += f"  • {elem_desc}\n"
                if len(relevant_elements) > 30:
                    overlay_context += f"  ... and {len(relevant_elements) - 30} more elements\n"
        
        # Simplified user prompt - only essential context (rules/examples are in system prompt)
        prompt = f"""
Determine the NEXT SINGLE ACTION to take based on:

USER GOAL: "{self.user_prompt}"

CURRENT STATE:
- URL: {state.current_url}
- Page Title: {state.page_title}
- Scroll Position: Y={state.browser_state.scroll_y}, X={state.browser_state.scroll_x}
- Viewport: {state.browser_state.page_width}x{state.browser_state.page_height}
- {"Screenshot: FULL-PAGE (entire page visible)" if is_exploring else "Screenshot: VIEWPORT ONLY (currently visible area)"}
- Visible Text: {state.visible_text[:300]}...

NAVIGATION HISTORY:
{nav_summary}
{("- Viewport shows content from Y={viewport_snapshot.scroll_y} to Y={viewport_snapshot.scroll_y + viewport_snapshot.page_height}" if is_exploring and viewport_snapshot else "")}
{("- Target element to find: {missing_element}" if is_exploring and missing_element else "")}

WHAT'S BEEN DONE:
{interaction_summary}

{ineffective_actions_context if ineffective_actions_context else ""}
{exploration_context if exploration_context else ""}
{pending_select_info if pending_select_info else ""}
{overlay_context if overlay_context else ""}
{"WHAT STILL NEEDS TO BE DONE:" if remaining_tasks else ""}
{remaining_tasks if remaining_tasks else ""}

Look at the screenshot and determine the single next action to progress toward the goal.
"""
        return prompt
    
    def _identify_remaining_tasks(self, interactions: List[Interaction]) -> Optional[str]:
        """
        Identify what still needs to be done by comparing user prompt with interaction history.
        Returns a description of remaining tasks.
        """
        # Extract form fields from user prompt
        prompt_lower = self.user_prompt.lower()
        
        # Pattern to match "field: value" or "field: value,"
        field_pattern = r'(\w+)\s*:\s*([^,]+)'
        requested_fields = {}
        for match in re.finditer(field_pattern, prompt_lower):
            field_name = match.group(1).strip()
            field_value = match.group(2).strip()
            requested_fields[field_name] = field_value
        
        # Check what's been filled from interaction history
        filled_fields = set()
        for interaction in interactions:
            if interaction.text_input:
                text_lower = interaction.text_input.lower()
                # Check if any requested field value matches what was typed
                for field_name, field_value in requested_fields.items():
                    if field_value.lower() in text_lower or text_lower in field_value.lower():
                        filled_fields.add(field_name)
                        break
        
        # Determine what's still missing
        if not requested_fields:
            return None
        
        remaining = []
        for field_name in requested_fields.keys():
            if field_name not in filled_fields:
                remaining.append(field_name)
        
        if remaining:
            return ", ".join([f"{field} field" for field in remaining])
        return None
    
    def _summarize_interactions(self, interactions) -> str:
        """Summarize what interactions have occurred"""
        if not interactions:
            return "No interactions yet."
        
        summary_parts = []
        limit = self.interaction_summary_limit
        interactions_to_summarize = interactions if not limit or limit <= 0 else interactions[-limit:]
        for i, interaction in enumerate(interactions_to_summarize, 1):
            interaction_type = interaction.interaction_type.value
            summary = f"{i}. {interaction_type}"
            
            # Special handling for defer interactions
            if interaction.interaction_type.value == "defer":
                if interaction.text_input:
                    if interaction.text_input == "resumed":
                        summary += " - resumed (control returned to agent)"
                    else:
                        summary += f" - {interaction.text_input}"
                if interaction.reasoning:
                    summary += f" ({interaction.reasoning[:50]})"
                summary_parts.append(summary)
                continue
            
            if interaction.text_input:
                summary += f" - entered: '{interaction.text_input[:30]}'"
            if interaction.target_element_info:
                element_desc = interaction.target_element_info.get('description', '')[:40]
                if element_desc:
                    summary += f" on: {element_desc}"
            
            summary_parts.append(summary)
        
        return "\n".join(summary_parts) if summary_parts else "No interactions."

    def _check_pending_select_fields(self, interactions: List[Interaction]) -> Optional[str]:
        """
        Check if there's a pending select field that needs an option to be selected.
        Returns a formatted string with information about the pending select field and available options.
        """
        if not interactions:
            return None
        
        # Check the most recent interactions for pending select fields
        # Look at the last few interactions (in case the select was opened recently)
        for interaction in reversed(interactions[-5:]):  # Check last 5 interactions
            if interaction.interaction_type.value == "select":
                target_info = interaction.target_element_info or {}
                if target_info.get("pending_select") and target_info.get("available_options"):
                    available_options = target_info.get("available_options", [])
                    select_field = target_info.get("select_field_description", "select field")
                    
                    # Format available options for display
                    option_texts = [opt.get('text', '') or opt.get('label', '') or opt.get('value', '') 
                                   for opt in available_options if opt.get('text') or opt.get('label') or opt.get('value')]
                    
                    if option_texts:
                        options_display = ', '.join(option_texts[:15])
                        if len(option_texts) > 15:
                            options_display += f" (and {len(option_texts) - 15} more)"
                        
                        return (
                            f"\n🎯 PENDING SELECT FIELD:\n"
                            f"A select field '{select_field}' was recently opened but no option was selected.\n"
                            f"Available options: {options_display}\n"
                            f"Your next action should be to select an appropriate option using: 'select: <option text>'\n"
                            f"Choose the option that best matches the user's goal.\n"
                        )
        
        return None
    
    def _summarize_navigation_history(self, url_history: List[str], url_pointer: Optional[int]) -> str:
        """Provide a concise navigation summary for the prompt."""
        if not url_history:
            return "No navigation history recorded yet."

        total = len(url_history)
        pointer = url_pointer if url_pointer is not None and 0 <= url_pointer < total else total - 1
        pointer = max(0, pointer)

        prev_url = url_history[pointer - 1] if pointer > 0 else None
        next_url = url_history[pointer + 1] if pointer < total - 1 else None

        start_idx = max(0, total - 3)  # Reduced for speed
        lines = []
        for idx in range(start_idx, total):
            marker = " (current)" if idx == pointer else ""
            lines.append(f"{idx}: {url_history[idx]}{marker}")

        history_block = "\n    ".join(lines)
        prev_line = f"Previous page (back target): {prev_url}" if prev_url else "Previous page (back target): none"
        next_line = f"Next page (forward target): {next_url}" if next_url else "Next page (forward target): none"

        return (
            f"Total pages visited: {total}\n"
            f"Current history index: {pointer}\n"
            f"{prev_line}\n"
            f"{next_line}\n"
            f"Recent history (oldest → newest):\n    {history_block}"
        )

