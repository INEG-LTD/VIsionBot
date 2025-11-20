"""
Reactive Goal Determiner - Determines the next action based on current viewport and state.

Step 2: LLM-based reactive goal determination that decides what to do RIGHT NOW
based on what's visible in the viewport, not pre-planning.
"""

from typing import Optional, List, ClassVar, Set, Union
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
        "handle_select",
        "handle_upload",
        "handle_datetime",
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
        elif command in {"form", "select", "upload", "datetime", "stop", "open", "handle_select", "handle_upload", "handle_datetime"}:
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
    ):
        """
        Initialize the reactive goal determiner.
        
        Args:
            user_prompt: The user's high-level goal
            base_knowledge: Optional list of knowledge rules/instructions that guide the agent's behavior.
                           Example: ["just press enter after you've typed a search term into a search field"]
            image_detail: Image detail level for vision API ("low" for faster, "high" for more accurate).
                         Default: "low" for better performance.
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
    
    def determine_next_action(
        self,
        environment_state: EnvironmentState,
        screenshot: bytes,
        is_exploring: bool = False,
        viewport_snapshot: Optional[BrowserState] = None,
        failed_actions: Optional[List[str]] = None,
        ineffective_actions: Optional[List[str]] = None
    ) -> tuple[Optional[str], bool]:
        """
        Determine the single next action to take based on current state.
        
        Args:
            environment_state: Current environment state
            screenshot: Current screenshot (viewport or full-page depending on is_exploring)
            is_exploring: If True, we're in exploration mode and screenshot is full-page
            viewport_snapshot: Optional viewport snapshot for comparison when exploring
            
        Returns:
            Tuple of (action_command, needs_exploration):
            - action_command: Action to take (e.g., "type: John Doe in name field") or None
              NOTE: In exploration mode, this MUST be "scroll: up" or "scroll: down"
            - needs_exploration: True if next iteration should use full-page screenshot
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
                ineffective_actions or []
            )
            
            if not response:
                print("⚠️ No action generated")
                return None, True  # Return None and indicate exploration is needed
            
            action = response.action
            
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
            
            return action, response.needs_exploration
            
        except Exception as e:
            print(f"⚠️ ReactiveGoalDeterminer error: {e}")
            import traceback
            traceback.print_exc()
            return None, False
    
    def _generate_single_action(
        self,
        environment_state: EnvironmentState,
        screenshot: bytes,
        is_exploring: bool,
        missing_element: Optional[str],
        viewport_snapshot: Optional[BrowserState],
        failed_actions: List[str] = None,
        ineffective_actions: List[str] = None
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
                ineffective_actions or []
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
CRITICAL EXPLORATION MODE RULES:
- You are in EXPLORATION MODE - you can see the FULL PAGE screenshot
- Your ONLY job is to determine scroll direction (up or down) to bring the target element into viewport
- You MUST ONLY return scroll commands: "scroll: up" or "scroll: down"
- DO NOT return type/click commands - the element is NOT in the viewport yet
- Look at where the target element is in the full-page screenshot
- Compare it to what's currently visible in the viewport (based on scroll position)
- If target is ABOVE current viewport → return "scroll: up"
- If target is BELOW current viewport → return "scroll: down"
- Once the element is visible in viewport, you will exit exploration mode and can interact with it
""" if is_exploring else ""
        
        # Build base knowledge section if provided
        base_knowledge_section = ""
        if self.base_knowledge:
            base_knowledge_section = "\n\nBASE KNOWLEDGE (Rules that guide your behavior):\n"
            for i, knowledge in enumerate(self.base_knowledge, 1):
                base_knowledge_section += f"{i}. {knowledge}\n"
            base_knowledge_section += "\nIMPORTANT: Apply these base knowledge rules when determining actions. They override general assumptions.\n"
        
        prompt = f"""
You are determining the NEXT SINGLE ACTION to take based on the current state.

Your job is to look at the screenshot ({screenshot_note}) and decide:
"What should I do RIGHT NOW to progress toward the user's goal?"

{exploration_rules}
{base_knowledge_section}
CRITICAL RULES:
1. **EXTRACTION PRIORITY: If the user's goal involves extracting, getting, finding, collecting, listing, showing, displaying, or retrieving data from the page, you MUST use "extract:" commands. After any necessary navigation/interaction is complete and the data is visible, your next action MUST be an "extract:" command.**
   - Examples: "extract the price" → "extract: price"
   - Examples: "get the stock price" → "extract: stock price"  
   - Examples: "find the current price" → "extract: current price"
   - Examples: "collect product information" → "extract: product information"
   - Examples: "list the top 5 stocks" → "extract: top 5 stocks with prices and percentage change"
   - Examples: "show me the prices" → "extract: prices"
   - Examples: "display the results" → "extract: results"
   - Examples: "what are the top stocks?" → "extract: top stocks with prices and percentage change"
   - **CRITICAL: If data is already visible on the page and the user wants it extracted, use "extract:" immediately. Do not click/type/scroll unless needed to make the data visible first.**
2. {"You can see the FULL PAGE in this screenshot - use it to determine scroll direction ONLY" if is_exploring else "You can ONLY see what's in the current viewport screenshot - nothing below or above"}
3. Determine ONE action at a time - be reactive, not pre-planned
4. {"You are in exploration mode - you MUST ONLY return scroll commands until the target element is visible in viewport" if is_exploring else "If elements are visible in the viewport, suggest actions for them. Only indicate exploration is needed if NO valid action can be determined."}
5. {"Only return scroll commands (scroll: up or scroll: down) - do NOT return type/click commands" if is_exploring else "Only suggest actions for elements clearly visible in the screenshot. If you can determine a valid action, return it and set needs_exploration=False."}
6. **CRITICAL: If the prompt mentions failed actions that didn't yield any change, DO NOT suggest those same actions again. Try a different element or approach.**
7. **HANDOFFS: When the user explicitly asks to pause, give control to them, or resolve something manually (captcha, MFA, legal acknowledgement, etc.), respond with a `defer:` command instead of continuing automation.**
8. **NAVIGATION HISTORY: Use the navigation history summary to decide when to issue `back:` or `forward:` commands. If the goal requires revisiting a previous page or moving ahead in session history, prefer these commands over retyping URLs.**
9. Format actions as executable commands:
   - **For EXTRACT commands: HIGHEST PRIORITY when user wants data** - Use "extract: <description>" when the user prompt contains extraction keywords (extract, get, find, note, collect, gather, retrieve, pull, fetch, list, show, display, return, output, print, read, scan, capture, obtain, acquire, present, report, summarize, detail, enumerate, itemize, catalog, record, document, save, export, download, copy, quote, cite) and the data is (or will be) visible. Examples: "extract: price", "extract: stock price", "extract: current and after market price", "extract: top 5 stocks with prices and percentage change"
   - **For CLICK commands: MUST include element TYPE (button, link, div, input, etc.) AND be specific** - e.g., "click: Google Search button", "click: first article link titled 'Introduction'", "click: search suggestion 'yahoo finance' link", "click: Accept all cookies button". NEVER use vague terms like 'first element', 'that button', or ambiguous text without element type like "click: search suggestion 'yahoo finance'" (must specify: link, button, div, etc.).
   - **For TYPE commands: MUST include element type** - e.g., "type: John Doe in name input field", "type: john@example.com in email input field". NEVER use vague terms like 'the field' or 'it'.
   - **For PRESS commands: MUST be brief** - just the key name (e.g., "press: Enter", "press: Escape", "press: Tab"). Do NOT add descriptions or context.
10. **IMPORTANT: needs_exploration should ONLY be True when you cannot determine ANY valid action. If you can see actionable elements, return an action and set needs_exploration=False.**

AVAILABLE COMMANDS:
- extract: <description>  (e.g., "extract: product price", "extract: article title", "extract: current and after market price")
  **PRIORITY COMMAND**: Use this when user wants to extract/collect data from the page.
  **When to use**: 
    - If data is already visible → use "extract:" immediately
    - If data needs page interaction first → do that, then use "extract:"
  **Examples**:
    - "extract: stock price" (when price is visible)
    - "extract: current and after market price" (when both prices are visible)
    - "extract: product name and price" (when product info is visible)
  This triggers automatic extraction using the bot's extract function.
- defer: <optional message or seconds>  (e.g., "defer", "defer: 15", "defer: give control to the user")
  Use this when automation must pause and hand control to the user.
  - "defer" → pause indefinitely until the user resumes
  - "defer: 10" → pause for 10 seconds, then resume automatically
  - "defer: take over" → pause and show the provided message to the user
  Only defer when the user explicitly requests manual control or human input is required (captcha, multi-factor auth, etc.).
- subagents: <mode>  (e.g., "subagents: single", "subagents: parallel", "subagents: reset")
  Override the adaptive sub-agent utilization policy when it is clearly misaligned with the task or the user explicitly requests a change.
  - "single"/"single-threaded"/"off"/"conservative" → minimize or block spawning.
  - "parallel"/"aggressive"/"max" → encourage eager spawning for independent work.
  - "reset"/"default"/"auto"/"adaptive" → clear overrides and return to adaptive mode.
  Only issue this command when the current policy is blocking progress or after the user asks for a different parallelism mode.
- scroll: <direction>  (e.g., "scroll: down", "scroll: up") - {"Use this ONLY in exploration mode" if is_exploring else "Use if needed elements aren't visible"}
{"- DO NOT use type/click commands in exploration mode - element is not in viewport yet" if is_exploring else ""}
{"- type: <value> in <specific field description>  (e.g., \"type: John Doe in name field\", \"type: john@example.com in email input field\") - Only use when NOT in exploration mode" if not is_exploring else ""}
{"   * GOOD: \"type: John Doe in name field\"" if not is_exploring else ""}
{"   * GOOD: \"type: john@example.com in email input field\"" if not is_exploring else ""}
{"   * BAD: \"type: John Doe in field\" (too vague)" if not is_exploring else ""}
{"   * BAD: \"type: John Doe in it\" (unclear)" if not is_exploring else ""}
{"- click: <specific element description WITH element type>  (e.g., \"click: Google Search button\", \"click: first article link titled 'Introduction to Python'\", \"click: Accept all cookies button\", \"click: search suggestion 'yahoo finance' link\") - Only use when NOT in exploration mode" if not is_exploring else ""}
{"   * CRITICAL: Always include element TYPE (button, link, div, input, etc.) in click commands" if not is_exploring else ""}
{"   * GOOD: \"click: Google Search button\" (includes type: button)" if not is_exploring else ""}
{"   * GOOD: \"click: first article link titled 'Introduction to Python'\" (includes type: link)" if not is_exploring else ""}
{"   * GOOD: \"click: search suggestion 'yahoo finance' link\" (includes type: link)" if not is_exploring else ""}
{"   * GOOD: \"click: Accept all cookies button\" (includes type: button)" if not is_exploring else ""}
{"   * BAD: \"click: search suggestion 'yahoo finance'\" (missing element type - is it a link? button? div?)" if not is_exploring else ""}
{"   * BAD: \"click: first element\" (too vague - what element? what type?)" if not is_exploring else ""}
{"   * BAD: \"click: button\" (too vague - which button?)" if not is_exploring else ""}
{"   * BAD: \"click: that button\" (unclear reference)" if not is_exploring else ""}
{"   IMPORTANT: If the user's goal contains ordinal information (first, second, third, etc.), PRESERVE IT and make it specific WITH element type:" if not is_exploring else ""}
{"   - \"click: first article\" → \"click: first article link titled '<title>'\" or \"click: first article link in the list\" (NOT \"click: first article\" - must include type)" if not is_exploring else ""}
{"   - \"click: second button\" → \"click: second submit button\" or \"click: second button labeled '<text>'\" (already includes type: button)" if not is_exploring else ""}
{"   - \"click: third link\" → \"click: third link titled '<text>'\" or \"click: third navigation link\" (already includes type: link)" if not is_exploring else ""}
{"   - \"click: search suggestion 'yahoo finance'\" → \"click: search suggestion 'yahoo finance' link\" or \"click: search suggestion 'yahoo finance' button\" (must specify type)" if not is_exploring else ""}
{"- navigate: <url or site>  (e.g., \"navigate: https://news.ycombinator.com\") - Only use when the navigation history summary shows no way back/forward and no other tab already has the required page. Prefer taking advantage of history or existing tabs first." if not is_exploring else ""}
{"- press: <key>  (e.g., \"press: Enter\", \"press: Escape\", \"press: Tab\") - MUST be brief, just the key name. Do NOT add descriptions or context." if not is_exploring else ""}
{"   * GOOD: \"press: Enter\"" if not is_exploring else ""}
{"   * GOOD: \"press: Escape\"" if not is_exploring else ""}
{"   * BAD: \"press: Enter in the search input field\" (too descriptive - press commands should be brief)" if not is_exploring else ""}
{"   * BAD: \"press: Enter to search\" (too descriptive - press commands should be brief)" if not is_exploring else ""}
{"- form: <description>  (e.g., \"form: complete checkout form\") - Use when the task explicitly refers to filling an entire form or multiple related fields. Keep description precise." if not is_exploring else ""}
{"- select: <option description>  (e.g., \"select: United States option in country dropdown\") - Use for dropdowns or select menus when a specific option must be chosen." if not is_exploring else ""}
{"- upload: <file description>  (e.g., \"upload: resume.pdf using file input\") - Use when the workflow requires attaching a local file and the upload control is visible." if not is_exploring else ""}
{"- datetime: <value and field>  (e.g., \"datetime: 2024-06-01 in start date picker\") - Use for date/time pickers that need structured input." if not is_exploring else ""}
{"- back: <steps>  (e.g., \"back: 1\") and forward: <steps>  - Use when the navigation history summary shows the desired page is behind/ahead or when the user explicitly requests it. Defaults to 1 step if omitted." if not is_exploring else ""}
{"   * Confirm the target using the history summary (previous page, next page) before issuing back/forward." if not is_exploring else ""}
{"- focus/subfocus: <region description>  - Use to narrow the vision focus to a specific section before interacting. 'subfocus' zooms further inside the current focus." if not is_exploring else ""}
{"- undofocus/undo:  - Use to restore the previous focus region when the narrowed view is no longer helpful." if not is_exploring else ""}

{"EXPLORATION MODE (Full-Page Visible):" if is_exploring else "VIEWPORT-FIRST APPROACH:"}
- Look at the screenshot carefully
- {"Identify where the target element is in the full-page screenshot" if is_exploring else "Identify what's visible and what's needed"}
{"- Compare target element position with current viewport position (scroll_y tells you current position)" if is_exploring else ""}
{"- If target is ABOVE viewport → return \"scroll: up\"" if is_exploring else "- If target element is visible → interact with it"}
{"- If target is BELOW viewport → return \"scroll: down\"" if is_exploring else "- If target element is NOT visible → scroll to reveal it"}
- Don't plan ahead - just decide the immediate next action
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
        ineffective_actions: List[str] = None
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
        for i, interaction in enumerate(interactions[-3:], 1):  # Last 3 (reduced for speed)
            interaction_type = interaction.interaction_type.value
            summary = f"{i}. {interaction_type}"
            
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

