"""
Reactive Goal Determiner - Determines the next action based on current viewport and state.

Step 2: LLM-based reactive goal determination that decides what to do RIGHT NOW
based on what's visible in the viewport, not pre-planning.
"""

from typing import Optional, List, ClassVar, Set, Union, Dict, Any
from pydantic import BaseModel, Field, field_validator
import re

from session_tracker import Interaction
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
    action: str = Field(description="The specific action to take RIGHT NOW in proper command format. For CLICK commands: MUST include the element TYPE (button, link, div, input, etc.) AND be descriptive - e.g., 'click: Google Search button', 'click: first article link titled 'Introduction'', 'click: Accept all cookies button', 'click: search suggestion 'yahoo finance' link'. For TYPE commands: Use format 'type: <text> : <field>' with colon separator - e.g., 'type: John Doe : name input field', 'type: john@example.com : email input field'. For PRESS commands: MUST be brief - just the key name (e.g., 'press: Enter', 'press: Escape'). NEVER use vague terms like 'first element', 'that button', 'the field', or ambiguous text without element type for click/type commands.")
    reasoning: str = Field(description="Why this action is appropriate given the current viewport")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence that this is the right next action")
    expected_outcome: str = Field(description="What should happen after executing this action")
    overlay_index: Optional[int] = Field(default=None, description="The numbered overlay index of the element to interact with (e.g., 25 for overlay #25). Only applicable for click, type, and select commands. Leave empty for scroll, navigate, press, or other non-element actions.")

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
        "ask",  # Ask user for help/clarification
        "complete",  # Agent signals task completion
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
        """
        Validate and normalize type command body.

        Accepted formats:
        - New format: "<text> : <field>" - colon separator
        - Legacy format: "<text> in <field>" or "<text> into <field>"
        """
        if not body:
            raise ValueError("type command requires text and target")

        lowered = body.lower()

        # Check for new colon separator format: "text : field"
        if " : " in body:
            # New format with colon separator - validate and normalize
            parts = body.split(" : ", 1)
            if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
                raise ValueError("type command with ':' separator must be 'text : field'")
            # Already in correct format
            return re.sub(r"\s+", " ", body).strip()

        # Legacy format: Check for "in" or "into"
        if re.search(r"\b(in|into)\b", lowered):
            # Has "in/into" - validate field keyword present
            if not re.search(r"\b(field|input|area|box|textbox)\b", lowered):
                # Add "input field" after the preposition
                body = re.sub(r"\b(in|into)\b\s*", lambda m: f"{m.group(0)}input field ", body, count=1, flags=re.IGNORECASE)
            return re.sub(r"\s+", " ", body).strip()

        # No separator found - invalid format
        raise ValueError("type command must use format 'text : field' or 'text in field'")

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
        include_visible_text_in_agent_context: bool = False,
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
        self._system_prompt_cache: dict[str, str] = {}  # Cache system prompts
        self.interaction_summary_limit = interaction_summary_limit
        self.include_overlays_in_agent_context = include_overlays_in_agent_context
        self.include_visible_text_in_agent_context = include_visible_text_in_agent_context
    
    def determine_next_action(
        self,
        environment_state: EnvironmentState,
        screenshot: bytes,
        failed_actions: Optional[List[str]] = None,
        ineffective_actions: Optional[List[str]] = None,
        overlay_data: Optional[List[Dict[str, Any]]] = None
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Determine the single next action to take based on current state.
        
        Args:
            environment_state: Current environment state
            screenshot: Current screenshot (viewport only)
            overlay_data: Optional list of overlay element data with descriptions.
            
        Returns:
            Tuple of (action_command, reasoning, overlay_index):
            - action_command: Action to take (e.g., "type: John Doe in name field") or None
            - reasoning: Why this action was chosen (for inclusion in interaction history)
            - overlay_index: Optional overlay number if agent provided one (e.g., 25 for #25)
        """
        try:
            # Generate single action directly
            response = self._generate_single_action(
                environment_state,
                screenshot,
                failed_actions or [],
                ineffective_actions or [],
                overlay_data
            )
            
            if not response:
                print("⚠️ No action generated")
                return None, None, None

            action = response.action
            reasoning = response.reasoning
            overlay_index = response.overlay_index

            try:
                get_event_logger().action_determined(
                    action=action,
                    reasoning=response.reasoning,
                    confidence=response.confidence,
                    expected_outcome=response.expected_outcome
                )
            except Exception:
                pass

            return action, reasoning, overlay_index

        except Exception as e:
            print(f"⚠️ ReactiveGoalDeterminer error: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def _generate_single_action(
        self,
        environment_state: EnvironmentState,
        screenshot: bytes,
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
            system_prompt = self._build_system_prompt()
            user_prompt_text = self._build_action_prompt(
                environment_state, 
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
    
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for reactive action determination"""
        # Use cached prompt if available
        if "default" in self._system_prompt_cache:
            return self._system_prompt_cache["default"]
        
        # Build base knowledge section if provided
        base_knowledge_section = ""
        if self.base_knowledge:
            base_knowledge_section = "\n\nBASE KNOWLEDGE (Custom Rules):\n"
            for i, knowledge in enumerate(self.base_knowledge, 1):
                base_knowledge_section += f"{i}. {knowledge}\n"
        
        prompt = f"""
You determine the NEXT SINGLE ACTION based on current viewport screenshot.

{base_knowledge_section}

⚠️ IMPORTANT - PRIORITIZE USER GUIDANCE & RULES:
1. CUSTOM RULES (Base Knowledge): Always follow instructions in "BASE KNOWLEDGE".
2. USER CLARIFICATIONS (Interaction History): Check "User-provided clarifications" section. If a previous answer addresses the current situation, you MUST act on it instead of asking again.
3. ASK FOR HELP WHEN STUCK: Use "ask: <question>" ONLY IF no previous guidance covers the current blocker. Do not repeat questions if you have an answer.

Use "ask: <question>" immediately when:
- A dropdown/autocomplete shows "no results" or doesn't match expected values AND no clarification exists
- You've tried the same element 2+ times without success  
- The provided data doesn't work (e.g., city not in location options)
- You're unsure what value to use or how to proceed
- An element behaves unexpectedly

Examples:
- "ask: The location field shows 'no results' for Liverpool. What location should I use instead?"
- "ask: I've tried selecting the date but the picker won't accept it. How should I proceed?"
- "ask: The form requires a field I don't have data for. What should I enter?"

ACTION RULES:
1. FILE UPLOADS (type=file or upload/attach/browse): Use "upload: [file] in <target>"
   - With file: "upload: resume.pdf in file input"
   - Without file: "upload: file input" (opens picker)
   - Never invent filenames

2. TEXT INPUTS: Fields auto-clear before typing - type complete desired value
   - Format: "type: <text> : <field>" - Use colon to separate text from target
   - Example: "type: john@example.com : email input field" (not "type: john")

3. EXTRACTION: Use "extract: <what>" when user wants data retrieved

4. COMMAND FORMAT:
   - Click: Must include element type (button/link/div) - "click: Submit button"
   - Type: Use colon separator - "type: John Doe : name input field"
   - Press: Just key name - "press: Enter"

5. AVOID REPEATING: Failed actions should not be repeated - ASK for help instead

6. DEFER: Use "defer:" when user requests manual control or captcha appears

7. COMPLETION: Use "complete: <reasoning>" when you've successfully accomplished the user's goal
   - Only call when task is truly finished
   - Provide clear reasoning explaining what was accomplished
   - Example: "complete: Successfully submitted job application. All required fields filled and form submitted."

COMMANDS:
- complete: <reasoning> - CALL WHEN TASK FINISHED - Explain what was accomplished
- ask: <question> - ASK USER when stuck, confused, or element not working as expected
- click: <type> <description> - Interact with element (must specify type: button/link/etc)
- type: <text> : <field> - Enter text (use colon separator, field auto-clears first)
- select: <option> in <dropdown> - Pick option from dropdown
- upload: [file] in <target> - Handle file uploads (file optional, opens picker if omitted)
- press: <key> - Press single key (Enter/Escape/Tab)
- scroll: <up|down> - Move viewport
- extract: <what> - Extract visible data from page
- navigate: <url> - Direct navigation
- back: [n] / forward: [n] - Navigate history
- defer: [msg|seconds] - Pause for user (captcha/manual input)
- form: <description> - Fill entire form
- datetime: <value> in <picker> - Set date/time
- mini_goal: <instruction> - Focus on complex sub-task

DECISION MAKING:
- If user's goal is accomplished, use "complete: <reasoning>" immediately
- Interact with visible elements. If target not visible, use scroll commands
- If something isn't working after 1-2 attempts, use "ask:" to get user guidance
"""
        # Cache the prompt
        self._system_prompt_cache["default"] = prompt
        return prompt
    
    def _build_action_prompt(
        self, 
        state: EnvironmentState, 
        failed_actions: List[str] = None,
        ineffective_actions: List[str] = None,
        overlay_data: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Build prompt for determining next action"""
        
        # Summarize what's been done
        interaction_summary = self._summarize_interactions(state.interaction_history)
        
        # Extract what still needs to be done from user prompt and interactions
        remaining_tasks = self._identify_remaining_tasks(state.interaction_history)
        
        # Add failed and ineffective actions context
        ineffective_actions_context = ""
        all_ineffective = []
        
        if failed_actions:
            all_ineffective.extend([(action, "failed") for action in failed_actions])
        if ineffective_actions:
            all_ineffective.extend([(action, "succeeded but no change") for action in ineffective_actions])
        
        if all_ineffective:
            ineffective_actions_context = "\n⚠️ ACTIONS THAT DIDN'T WORK - Consider using 'ask:' for help:\n"
            for i, (action, reason) in enumerate(all_ineffective, 1):
                ineffective_actions_context += f"   {i}. {action} ({reason})\n"
            ineffective_actions_context += "\nDO NOT repeat these. If you've tried 2+ different approaches without success, use 'ask: <your question>' to get user guidance.\n"
        
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
                
                # Include elements that are interactive or have useful text
                is_select = tag == "select" or role in ("combobox", "listbox")
                has_content = text or aria or placeholder or is_select
                
                if has_content:
                    # Build description similar to plan_generator format
                    parts = []
                    parts.append(f"#{idx} tag={tag}")
                    
                    if role and role != tag:
                        parts.append(f"role={role}")
                    
                    if placeholder:
                        parts.append(f'placeholder="{placeholder[:40]}"')
                    
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
                    
                    relevant_elements.append(description)
            
            if relevant_elements:
                # Limit to most relevant elements (prioritize inputs, buttons)
                # Show up to 30 elements to give good context without overwhelming
                overlay_context = "\n\nAVAILABLE INTERACTIVE ELEMENTS (numbered overlays visible in screenshot):\n"
                overlay_context += "These elements are marked with numbered red overlays in the screenshot.\n"
                overlay_context += "IMPORTANT: When your action targets one of these elements (click, type, select, upload), you MUST provide the overlay_index (just the number, e.g., 25 for #25).\n"
                overlay_context += "Elements with '[GROUP: elements #X, #Y, ...]' belong to the same question/group - clicking one when another in the group is already selected may be ineffective.\n\n"
                for elem_desc in relevant_elements[:30]:  # Limit to 30 most relevant
                    overlay_context += f"  • {elem_desc}\n"
                if len(relevant_elements) > 30:
                    overlay_context += f"  ... and {len(relevant_elements) - 30} more elements\n"
        
        # Build visible text context if enabled
        visible_text_context = ""
        if self.include_visible_text_in_agent_context and state.visible_text:
            visible_text_context = f"- Visible Text: {state.visible_text[:300]}...\n"

        # Simplified user prompt - only essential context (rules/examples are in system prompt)
        prompt = f"""
Determine the NEXT SINGLE ACTION to take based on:

USER GOAL: "{self.user_prompt}"

CURRENT STATE:
- URL: {state.current_url}
- Page Title: {state.page_title}
- Scroll Position: Y={state.browser_state.scroll_y}, X={state.browser_state.scroll_x}
- Viewport: {state.browser_state.page_width}x{state.browser_state.page_height}
- Screenshot: VIEWPORT ONLY (currently visible area)
{visible_text_context}
NAVIGATION HISTORY:
{nav_summary}

WHAT'S BEEN DONE:
{interaction_summary}

{ineffective_actions_context if ineffective_actions_context else ""}
{overlay_context if overlay_context else ""}
{"WHAT STILL NEEDS TO BE DONE:" if remaining_tasks else ""}
{remaining_tasks if remaining_tasks else ""}

COMPLETION CHECK:
Before taking another action, ask yourself: "Is the user's goal fully accomplished?"
- If YES: Use "complete: <reasoning>" explaining what was accomplished
- If NO: Determine the next action to progress toward the goal

Look at the screenshot and determine the single next action (or complete if done).
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

