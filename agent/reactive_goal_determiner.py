"""
Reactive Goal Determiner - Determines the next viewport-aware plan based on current state.

Step 2: LLM-based reactive goal determination that builds a sequence of actions you can execute
before the viewport changes, relying only on what is visible.
"""

from typing import Optional, List, Union, Dict, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict
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
from history import HistoryManager
from utils.overlay_description import describe_overlay_element, overlay_element_metadata


VALID_ACTION_COMMANDS = {
    "click", "type", "press", "scroll", "extract", "extract_url", "get_url",
    "defer", "navigate", "back", "forward", "subagents", "form", "select",
    "upload", "datetime", "stop", "open", "handle_datetime", "mini_goal",
    "ask", "complete",
}


def _parse_action(text: str) -> tuple[str, str]:
    """Parse action text into (command, body). Raises ValueError if invalid."""
    text = text.strip()
    if not text:
        raise ValueError("action cannot be empty")

    if ":" in text:
        cmd, body = text.split(":", 1)
        return cmd.strip().lower(), body.strip()

    # Try to parse "command args" format
    match = re.match(r"^(\w+)\s+(.+)$", text)
    if match:
        cmd, body = match.groups()
        # Handle "click on X" -> "click X"
        if cmd.lower() == "click" and body.lower().startswith("on "):
            body = body[3:]
        return cmd.lower(), body.strip()

    # Single word command (defer, stop, forward, back)
    if text.isalpha():
        return text.lower(), ""

    raise ValueError("action must be 'command: target' or 'command target'")


def _normalize_body(cmd: str, body: str) -> str:
    """Normalize the body based on command type."""
    if cmd == "click":
        if not body:
            raise ValueError("click requires a target")
        body = body[3:].strip() if body.lower().startswith("on ") else body
        # Add element type hint if missing
        if not re.search(r"\b(button|link|tab|checkbox|radio|option|div|input|icon|item)\b", body, re.I):
            if not body.lower().endswith("link"):
                body = f"{body} link"
        return body.strip()

    if cmd == "type":
        if not body:
            raise ValueError("type requires text and target")
        # Accept "text : field" or "text in/into field" format
        if " : " in body:
            parts = body.split(" : ", 1)
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                return re.sub(r"\s+", " ", body).strip()
        if re.search(r"\b(in|into)\b", body, re.I):
            return re.sub(r"\s+", " ", body).strip()
        raise ValueError("type must use 'text : field' or 'text in field'")

    if cmd == "press":
        key = body.strip()
        if not key or " " in key:
            raise ValueError("press requires a single key")
        return key

    if cmd == "scroll":
        direction = body.strip().lower()
        if direction not in {"up", "down", "left", "right"}:
            raise ValueError("scroll must be up, down, left, or right")
        return direction

    if cmd in {"back", "forward"}:
        return body if body and body.isdigit() else "1"

    if cmd in {"extract", "extract_url", "get_url", "navigate", "subagents",
               "mini_goal", "form", "select", "upload", "datetime", "open",
               "handle_datetime"}:
        if not body:
            raise ValueError(f"{cmd} requires additional detail")
        return body

    # Commands with optional body: defer, stop, complete, ask
    return body


class ActionStep(BaseModel):
    """One viewport-safe action"""
    # Allow extra attributes (like overlay_metadata) without including them in the schema
    model_config = ConfigDict(extra='allow')

    action: str = Field(
        description="Command to execute on the current viewport (e.g., 'click: Submit button')."
    )
    overlay_index: Optional[int] = Field(
        default=None,
        description="Overlay index from the reference table for the element referenced in this step."
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Explain why this action is needed and how it advances the goal without leaving the current viewport."
    )
    overlay_description: Optional[str] = Field(
        default=None,
        description="Canonical description of the overlay element so it can be matched across iterations."
    )
    # Note: overlay_metadata is handled as an extra attribute, not a Pydantic field
    # This prevents it from appearing in the JSON schema sent to Gemini

    def __init__(self, **data):
        super().__init__(**data)
        # Ensure overlay_metadata always exists, even if not provided
        if not hasattr(self, 'overlay_metadata'):
            object.__setattr__(self, 'overlay_metadata', None)

    def __setattr__(self, name, value):
        """Allow setting overlay_metadata after initialization."""
        if name == 'overlay_metadata':
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)

    @field_validator("action", mode="before")
    @classmethod
    def _normalize_action(cls, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("action must be a string")

        cmd, body = _parse_action(value)
        if cmd not in VALID_ACTION_COMMANDS:
            raise ValueError(f"Unsupported command: '{cmd}'")

        body = _normalize_body(cmd, body)
        return f"{cmd}: {body}" if body else cmd


class ActionPlan(BaseModel):
    """A sequential plan of actions that can be executed before the viewport changes."""
    steps: List[ActionStep] = Field(description="Actions that can be executed sequentially on the current viewport.")
    reasoning: str = Field(description="Why this sequence of steps achieves progress right now.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in this plan for the current viewport.")
    expected_outcome: str = Field(description="What should happen after executing the plan.")

    @field_validator("steps")
    @classmethod
    def _validate_steps(cls, value: List[ActionStep]) -> List[ActionStep]:
        if not value:
            raise ValueError("Action plan must contain at least one step.")
        return value


class ReactiveGoalDeterminer:
    """
    Determines a viewport-safe action plan based on:
    - Current viewport (what's visible on screen)
    - Environment state (interactions, scroll position, etc.)
    - User prompt (what we're trying to achieve)

    This is reactive - it determines an ordered list of actions that can be executed
    without waiting for new UI elements to appear.
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
        history_manager: Optional[HistoryManager] = None,
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
        self.history_manager = history_manager
    
    def determine_action_plan(
        self,
        environment_state: EnvironmentState,
        screenshot: bytes,
        failed_actions: Optional[List[str]] = None,
        ineffective_actions: Optional[List[str]] = None,
        overlay_data: Optional[List[Dict[str, Any]]] = None,
        notebook: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[ActionPlan]:
        """
        Determine an ordered action plan that can be executed before the viewport changes.

        Args:
            environment_state: Current environment state
            screenshot: Current screenshot (viewport only)
            overlay_data: Optional list of overlay element data with descriptions.
            notebook: Agent's notebook with previously extracted data.

        Returns:
            An ActionPlan describing the steps to take, or None if no plan could be generated.
        """
        try:
            plan = self._generate_action_plan(
                environment_state,
                screenshot,
                failed_actions or [],
                ineffective_actions or [],
                overlay_data,
                notebook or []
            )
            
            if not plan:
                print("⚠️ No action plan generated")
                return None

            return plan

        except Exception as e:
            print(f"⚠️ ReactiveGoalDeterminer error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _generate_action_plan(
        self,
        environment_state: EnvironmentState,
        screenshot: bytes,
        failed_actions: List[str] = None,
        ineffective_actions: List[str] = None,
        overlay_data: Optional[List[Dict[str, Any]]] = None,
        notebook: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[ActionPlan]:
        """
        Generate an ordered action plan.

        Returns:
            ActionPlan or None if no plan can be determined
        """
        try:
            system_prompt = self._build_system_prompt()
            user_prompt_text = self._build_action_prompt(
                environment_state,
                failed_actions or [],
                ineffective_actions or [],
                overlay_data,
                notebook or []
            )
            plan = generate_model(
                prompt=user_prompt_text,
                model_object_type=ActionPlan,
                system_prompt=system_prompt,
                image=screenshot,
                image_detail=self.image_detail,
                model=self.model_name,
                reasoning_level=self.reasoning_level,
            )
            if plan and overlay_data:
                overlay_desc_map = {}
                overlay_metadata_map = {}
                for elem in overlay_data:
                    idx = elem.get("index")
                    if idx is None:
                        continue
                    overlay_desc_map[idx] = describe_overlay_element(elem)
                    overlay_metadata_map[idx] = overlay_element_metadata(elem)
                for step in plan.steps:
                    if step.overlay_index is not None:
                        if desc := overlay_desc_map.get(step.overlay_index):
                            step.overlay_description = desc
                        if metadata := overlay_metadata_map.get(step.overlay_index):
                            step.overlay_metadata = metadata
            if plan:
                try:
                    steps_summary = "\n".join(
                        f"{idx}. {step.action} (overlay {step.overlay_index or 'N'}) - {step.reasoning or 'No reasoning provided.'}"
                        for idx, step in enumerate(plan.steps, 1)
                    )
                    get_event_logger().plan_generated(
                        plan_reasoning=plan.reasoning,
                        confidence=plan.confidence,
                        expected_outcome=plan.expected_outcome,
                        steps_summary=steps_summary,
                    )
                except Exception:
                    pass
            return plan
        except Exception as e:
            print(f"⚠️ Error generating action plan: {e}")
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
You determine the NEXT VIEWPORT-AWARE PLAN based on the current screenshot.

CRITICAL - VISUAL-FIRST DECISION MAKING:
You MUST make decisions based on what you SEE in the screenshot, NOT based on the overlay list.
1. Look at the screenshot to understand the page and identify what element to interact with
2. Use the visual appearance, text, position, and context to decide your action
3. ONLY THEN match your chosen element to an overlay ID from the reference list
The overlay list is purely for IDENTIFICATION - it tells you which number corresponds to which element.
Never let the overlay numbers influence WHAT you decide to do - only use them to reference your visual choice.

MODAL/POPUP AWARENESS:
After clicking a button (especially "Apply", "Submit", "Continue"), CHECK if a modal, popup, or new form appeared:
- If you see a darkened/grayed background with a centered panel, a MODAL has opened
- If new form fields appeared that weren't there before, interact with THOSE instead
- NEVER click the same button repeatedly - if it didn't work, something else is now in focus
- If an overlay/modal is blocking the page, interact with the modal content FIRST
- Look for close buttons (X), form fields, or action buttons INSIDE any visible modal

{base_knowledge_section}

- Before choosing the next plan, review your previous step: was it successful, unsuccessful, or uncertain? Use the interaction summary and history context to remember what changed.
- Note any blockers, failed actions, or missing inputs so they stay visible in future steps, and avoid repeating actions that already failed.
- Define a precise next goal that stays aligned with the user prompt, then let that goal guide your chosen action.
- If you still need clarification or data, frame it as an `ask:` question before attempting more actions.

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

VIEWPORT PLAN RULES:
- Stay anchored to what you can see; do not reference elements that are not in the current screenshot.
- When an element is partially off-screen, include a scroll step before interacting with it and treat that scroll as part of the same plan.
- Stop the plan as soon as the next logical step would require additional viewport content (autocomplete, modal, navigation, etc.).
- Each plan should include 3-6 steps to keep execution tight.

ACTION RULES:
1. FILE UPLOADS (type=file or upload/attach/browse): Use "upload: [file] in <target>"
   - With file: "upload: resume.pdf in file input"
   - Without file: "upload: file input" (opens picker)
   - Never invent filenames

2. TEXT INPUTS: Fields auto-clear before typing - type complete desired value
   - Format: "type: <text> : <field>" - Use colon to separate text from target
   - Example: "type: john@example.com : email input field" (not "type: john")

3. EXTRACTION - CRITICAL DISTINCTION:
   - "extract: <what>" - Extract DATA/TEXT from page (job title, company name, price, description, etc.)
     Example: "extract: job title and company name"
   - "extract_url: <target>" - Extract URL/LINK from element (apply button, job link, etc.)
     Example: "extract_url: Apply button"
   - ⚠️ NEVER use "extract:" for URLs - ALWAYS use "extract_url:" for links/buttons
   - If user asks for URL/link/href → use "extract_url:", NOT "extract:"

4. COMMAND FORMAT:
   - Click: Must include element type (button/link/div) - "click: Submit button"
   - Type: Use colon separator - "type: John Doe : name input field"
   - Press: Just key name - "press: Enter"
   - Extract data: "extract: job title and company"
   - Extract URL: "extract_url: Apply button" (NOT "extract: url from apply button")

5. AVOID REPEATING: Failed actions should not be repeated - ASK for help instead

6. DEFER: Use "defer:" when user requests manual control or captcha appears

7. COMPLETION: Use "complete: <reasoning>" when you've successfully accomplished the user's goal
   - CRITICAL: If a NOTEBOOK section appears below, CHECK IT FIRST before planning more actions
   - If the notebook already contains the data the user asked for, use "complete:" IMMEDIATELY
   - Do NOT extract the same data twice - if it's in the notebook, the task is DONE
   - Provide clear reasoning explaining what was accomplished
   - Example: "complete: Successfully extracted 10 job listings. Data includes job titles, companies, and locations."

COMMANDS:
- complete: <reasoning> - CALL WHEN TASK FINISHED - Check notebook first! If data is there, use this
- ask: <question> - ASK USER when stuck, confused, or element not working as expected
- click: <type> <description> - Interact with element (must specify type: button/link/etc)
- type: <text> : <field> - Enter text (use colon separator, field auto-clears first)
- select: <option> in <dropdown> - Pick option from dropdown
- upload: [file] in <target> - Handle file uploads (file optional, opens picker if omitted)
- press: <key> - Press single key (Enter/Escape/Tab)
- scroll: <up|down> - Move viewport
- extract: <what> - Extract visible data from page
- extract_url: <target> - Get URL from element (button/link/div with href or data-url)
- get_url: <target> - Alias for extract_url
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

        # Add sequential task counting rules for NLP-based progress tracking
        # Cache the prompt
        self._system_prompt_cache["default"] = prompt
        return prompt

    def _get_history_block(self) -> str:
        if not self.history_manager:
            return ""
        return self.history_manager.history_block(limit=self.interaction_summary_limit)
    
    def _build_action_prompt(
        self,
        state: EnvironmentState,
        failed_actions: List[str] = None,
        ineffective_actions: List[str] = None,
        overlay_data: Optional[List[Dict[str, Any]]] = None,
        notebook: Optional[List[Dict[str, Any]]] = None
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
        if self.include_overlays_in_agent_context:
            count = len(overlay_data) if overlay_data else 0
            if get_event_logger().debug_mode:
                print(f"[Debug] overlay_data available: {count} elements")
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
                overlay_context = "\n\nELEMENT REFERENCE TABLE (for matching your visual choice to an ID):\n"
                overlay_context += "DO NOT use this list to decide what to do - use the SCREENSHOT for that.\n"
                overlay_context += "This table helps you find the overlay_index for the element you've already chosen visually.\n"
                overlay_context += "After deciding which element to interact with based on the screenshot, find it here and provide its overlay_index.\n"
                overlay_context += "Elements with '[GROUP: ...]' belong to the same question/group.\n\n"
                for elem_desc in relevant_elements[:30]:  # Limit to 30 most relevant
                    overlay_context += f"  • {elem_desc}\n"
                if len(relevant_elements) > 30:
                    overlay_context += f"  ... and {len(relevant_elements) - 30} more elements\n"
        
        # Build visible text context if enabled
        visible_text_context = ""
        if self.include_visible_text_in_agent_context and state.visible_text:
            visible_text_context = f"- Visible Text: {state.visible_text[:300]}...\n"

        # Simplified user prompt - only essential context (rules/examples are in system prompt)
        history_block = self._get_history_block()
        history_prefix = f"{history_block}\n\n" if history_block else ""

        prompt = f"""
{history_prefix}Determine a viewport-aware plan of steps that can be executed without introducing new UI elements:

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

WHAT'S BEEN DONE (DO NOT REPEAT THESE):
{interaction_summary}

⚠️ CRITICAL: Review the list above. If your intended action matches something already done (especially clicks),
the page state has likely CHANGED. Look for NEW elements like modals, forms, or popups that appeared as a result.

{ineffective_actions_context if ineffective_actions_context else ""}
{overlay_context if overlay_context else ""}
{"WHAT STILL NEEDS TO BE DONE:" if remaining_tasks else ""}
{remaining_tasks if remaining_tasks else ""}
{self._format_notebook(notebook) if notebook else ""}
COMPLETION CHECK (DO THIS FIRST):
1. IF there is a NOTEBOOK section above with extracted data:
   - Compare the extracted data against the USER GOAL
   - If the data satisfies what the user asked for → use "complete:" IMMEDIATELY
   - Example: If user asked for "job listings" and notebook has job listings → DONE
2. ONLY if the notebook is empty OR data doesn't match the goal:
   - Plan the next actions to achieve the goal
- NEVER extract the same data twice - check the notebook first!

PLAN GUIDELINES:
- Choose a sequential plan of 3-6 steps that can all be executed without leaving the current viewport.
- Only include actions whose targets are fully visible; if you spot a useful element that is clipped or outside the viewport, add a scroll step before interacting with it so nothing is half-hidden.
- Each step must include its own reasoning explaining how it moves the task forward while relying only on visible UI.
- Do NOT plan for autocomplete suggestions, dropdown entries, or modals unless they are already visible in the screenshot.
- Plan for the full set of actions you can safely execute now (e.g., typing into a field and then pressing Enter) and stop once the next action would require new UI content (modal, suggestion list, navigation, etc.).

RESPOND IN THIS JSON SHAPE:

For COMPLETION (when notebook has the requested data or goal is achieved):
{{
  "steps": [
    {{"action": "complete: Successfully extracted 10 job listings including titles, companies, and locations.", "reasoning": "The notebook contains the job data the user requested. Task is done."}}
  ],
  "reasoning": "The notebook already contains the extracted job listings. No further actions needed.",
  "confidence": 0.95,
  "expected_outcome": "Task complete. User has the requested data."
}}

For ACTIONS (when more work is needed):
{{
  "steps": [
    {{"action": "type: john@example.com : email input field", "overlay_index": 12, "reasoning": "Enter the recipient address that is currently visible and required."}},
    {{"action": "press: Tab", "reasoning": "Move focus to the next input that's already visible."}}
  ],
  "reasoning": "Explain how these steps progress toward the goal.",
  "confidence": 0.0-1.0,
  "expected_outcome": "Describe what should happen after executing the plan."
}}

overlay_index is only needed when targeting a visible element. Skip it for complete/press/scroll/defer actions.
"""
        return prompt
    
    def _format_notebook(self, notebook: List[Dict[str, Any]]) -> str:
        """Format notebook entries for inclusion in the prompt."""
        if not notebook:
            return ""

        lines = [
            "",
            "=" * 60,
            "🛑 STOP - CHECK THIS BEFORE PLANNING MORE ACTIONS:",
            "=" * 60,
            f"USER GOAL: \"{self.user_prompt}\"",
            "",
            "EXTRACTED DATA IN NOTEBOOK:"
        ]

        for i, entry in enumerate(notebook[-5:], 1):  # Show last 5 entries
            prompt = entry.get("prompt", "unknown")
            data = entry.get("data", {})

            # Show actual data samples so LLM can verify extraction matches user goal
            if isinstance(data, dict) and "items" in data:
                items = data["items"]
                lines.append(f"  Entry {i}: [{prompt}] - {len(items)} items")
                # Show first 2-3 items as samples
                for j, item in enumerate(items[:3]):
                    item_str = str(item)
                    if len(item_str) > 150:
                        item_str = item_str[:150] + "..."
                    lines.append(f"    Sample {j+1}: {item_str}")
                if len(items) > 3:
                    lines.append(f"    ... and {len(items) - 3} more items")
            else:
                data_str = str(data)
                if len(data_str) > 300:
                    data_str = data_str[:300] + "..."
                lines.append(f"  Entry {i}: [{prompt}] - {data_str}")

        lines.extend([
            "",
            "=" * 60,
            "⚠️ DECISION REQUIRED:",
            "  - Does the notebook data above satisfy the USER GOAL?",
            "  - If YES: Use 'complete: Successfully extracted [X items/data]. Summary: [brief description]'",
            "  - If NO: Explain what's missing and plan next action",
            "  - DO NOT extract the same data again!",
            "=" * 60,
            ""
        ])

        return "\n".join(lines)

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
                element_desc = interaction.target_element_info.get('description', '')[:50]
                if element_desc:
                    summary += f" on: {element_desc}"
            # Add reasoning for click actions to help agent understand what happened
            if interaction.interaction_type.value == "click" and interaction.reasoning:
                summary += f" (reason: {interaction.reasoning[:40]}...)"
            
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
