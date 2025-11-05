"""
Reactive Goal Determiner - Determines the next action based on current viewport and state.

Step 2: LLM-based reactive goal determination that decides what to do RIGHT NOW
based on what's visible in the viewport, not pre-planning.
"""

from typing import Optional, List
from pydantic import BaseModel, Field
import re

from goals.base import BrowserState, Interaction
from agent.completion_contract import EnvironmentState
from ai_utils import generate_model


class NextAction(BaseModel):
    """Structured LLM response for next action determination"""
    action: str = Field(description="The specific action to take RIGHT NOW in proper command format. For CLICK commands: MUST include the element TYPE (button, link, div, input, etc.) AND be descriptive - e.g., 'click: Google Search button', 'click: first article link titled 'Introduction'', 'click: Accept all cookies button', 'click: search suggestion 'yahoo finance' link'. For TYPE commands: MUST be descriptive with element type - e.g., 'type: John Doe in name input field'. For PRESS commands: MUST be brief - just the key name (e.g., 'press: Enter', 'press: Escape'). NEVER use vague terms like 'first element', 'that button', 'the field', or ambiguous text without element type for click/type commands.")
    reasoning: str = Field(description="Why this action is appropriate given the current viewport")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence that this is the right next action")
    expected_outcome: str = Field(description="What should happen after executing this action")
    needs_exploration: bool = Field(description="True if the action requires exploring/searching for elements not visible in current viewport (e.g., scrolling to find a field)")


class ReactiveGoalDeterminer:
    """
    Determines the next action reactively based on:
    - Current viewport (what's visible on screen)
    - Environment state (interactions, scroll position, etc.)
    - User prompt (what we're trying to achieve)
    
    This is reactive - it only determines ONE action to take NOW, not a sequence.
    """
    
    def __init__(self, user_prompt: str, base_knowledge: Optional[List[str]] = None):
        """
        Initialize the reactive goal determiner.
        
        Args:
            user_prompt: The user's high-level goal
            base_knowledge: Optional list of knowledge rules/instructions that guide the agent's behavior.
                           Example: ["just press enter after you've typed a search term into a search field"]
        """
        self.user_prompt = user_prompt
        self.base_knowledge = base_knowledge or []
    
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
            
            print(f"💡 Next action determined: {action}")
            print(f"   Reasoning: {response.reasoning}")
            print(f"   Confidence: {response.confidence:.2f}")
            print(f"   Expected outcome: {response.expected_outcome}")
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
                image_detail="high"
            )
            return action
        except Exception as e:
            print(f"⚠️ Error generating action: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    
    def _build_system_prompt(self, is_exploring: bool = False) -> str:
        """Build system prompt for reactive action determination"""
        
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
        
        return f"""
You are determining the NEXT SINGLE ACTION to take based on the current state.

Your job is to look at the screenshot ({screenshot_note}) and decide:
"What should I do RIGHT NOW to progress toward the user's goal?"

{exploration_rules}
{base_knowledge_section}
CRITICAL RULES:
1. {"You can see the FULL PAGE in this screenshot - use it to determine scroll direction ONLY" if is_exploring else "You can ONLY see what's in the current viewport screenshot - nothing below or above"}
2. Determine ONE action at a time - be reactive, not pre-planned
3. {"You are in exploration mode - you MUST ONLY return scroll commands until the target element is visible in viewport" if is_exploring else "If elements are visible in the viewport, suggest actions for them. Only indicate exploration is needed if NO valid action can be determined."}
4. {"Only return scroll commands (scroll: up or scroll: down) - do NOT return type/click commands" if is_exploring else "Only suggest actions for elements clearly visible in the screenshot. If you can determine a valid action, return it and set needs_exploration=False."}
5. **CRITICAL: If the prompt mentions failed actions that didn't yield any change, DO NOT suggest those same actions again. Try a different element or approach.**
6. Format actions as executable commands:
   - **For CLICK commands: MUST include element TYPE (button, link, div, input, etc.) AND be specific** - e.g., "click: Google Search button", "click: first article link titled 'Introduction'", "click: search suggestion 'yahoo finance' link", "click: Accept all cookies button". NEVER use vague terms like 'first element', 'that button', or ambiguous text without element type like "click: search suggestion 'yahoo finance'" (must specify: link, button, div, etc.).
   - **For TYPE commands: MUST include element type** - e.g., "type: John Doe in name input field", "type: john@example.com in email input field". NEVER use vague terms like 'the field' or 'it'.
   - **For PRESS commands: MUST be brief** - just the key name (e.g., "press: Enter", "press: Escape", "press: Tab"). Do NOT add descriptions or context.
7. **IMPORTANT: needs_exploration should ONLY be True when you cannot determine ANY valid action. If you can see actionable elements, return an action and set needs_exploration=False.**

AVAILABLE COMMANDS:
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
{"   IMPORTANT: For navigation tasks, use \"click: <specific link or button description>\" instead of navigate commands. Be specific about what link/button to click." if not is_exploring else ""}
{"- press: <key>  (e.g., \"press: Enter\", \"press: Escape\", \"press: Tab\") - MUST be brief, just the key name. Do NOT add descriptions or context." if not is_exploring else ""}
{"   * GOOD: \"press: Enter\"" if not is_exploring else ""}
{"   * GOOD: \"press: Escape\"" if not is_exploring else ""}
{"   * BAD: \"press: Enter in the search input field\" (too descriptive - press commands should be brief)" if not is_exploring else ""}
{"   * BAD: \"press: Enter to search\" (too descriptive - press commands should be brief)" if not is_exploring else ""}

{"EXPLORATION MODE (Full-Page Visible):" if is_exploring else "VIEWPORT-FIRST APPROACH:"}
- Look at the screenshot carefully
- {"Identify where the target element is in the full-page screenshot" if is_exploring else "Identify what's visible and what's needed"}
{"- Compare target element position with current viewport position (scroll_y tells you current position)" if is_exploring else ""}
{"- If target is ABOVE viewport → return \"scroll: up\"" if is_exploring else "- If target element is visible → interact with it"}
{"- If target is BELOW viewport → return \"scroll: down\"" if is_exploring else "- If target element is NOT visible → scroll to reveal it"}
- Don't plan ahead - just decide the immediate next action
"""
    
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
            ineffective_actions_context = f"\n⚠️ IMPORTANT: The following actions were recently tried and did NOT yield any page change (same URL, same DOM state):\n"
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
        
        # Build base knowledge section if provided
        base_knowledge_context = ""
        if self.base_knowledge:
            base_knowledge_context = "\n\nBASE KNOWLEDGE (Apply these rules when determining actions):\n"
            for i, knowledge in enumerate(self.base_knowledge, 1):
                base_knowledge_context += f"{i}. {knowledge}\n"
            base_knowledge_context += "\nThese base knowledge rules should guide your action selection. Apply them when relevant.\n"
        
        prompt = f"""
Determine the NEXT SINGLE ACTION to take based on:

USER GOAL:
"{self.user_prompt}"
{base_knowledge_context}
CURRENT STATE:
- URL: {state.current_url}
- Page Title: {state.page_title}
- Scroll Position: Y={state.browser_state.scroll_y}, X={state.browser_state.scroll_x}
- Viewport: {state.browser_state.page_width}x{state.browser_state.page_height}
- {"Screenshot: FULL-PAGE (you can see entire page layout)" if is_exploring else "Screenshot: VIEWPORT ONLY (currently visible area)"}
- Visible Text: {state.visible_text[:500]}...
{("- CURRENT VIEWPORT INFO: The viewport currently shows content from approximately Y={viewport_snapshot.scroll_y} to Y={viewport_snapshot.scroll_y + viewport_snapshot.page_height}" if is_exploring and viewport_snapshot else "")}
{("- TARGET ELEMENT: You need to find: {missing_element}" if is_exploring and missing_element else "")}

WHAT'S BEEN DONE:
{interaction_summary}

{ineffective_actions_context if ineffective_actions_context else ""}
{exploration_context if exploration_context else ""}{"WHAT STILL NEEDS TO BE DONE:" if remaining_tasks else ""}
{remaining_tasks if remaining_tasks else ""}

INSTRUCTIONS:
1. Look at the screenshot - {"it shows the FULL PAGE" if is_exploring else "it shows ONLY the current viewport (what's visible now)"}
2. Determine ONE action to take RIGHT NOW based on:
   - What's visible in the screenshot
   - What still needs to be done (user goal vs. what's been done)
   - What would make progress toward the goal
   {"- In exploration mode: Look for the specific element(s) mentioned above in the full-page screenshot" if is_exploring else ""}
   {"- If target element is in the screenshot but not in viewport, determine scroll direction (up/down) based on element position" if is_exploring else ""}
   {"- **IMPORTANT: Apply the BASE KNOWLEDGE rules above when they are relevant to the current situation**" if self.base_knowledge else ""}

3. Format the action as an executable command:
   **For CLICK commands: MUST include element TYPE (button, link, div, input, etc.) AND be descriptive**
   - NEVER use vague terms: "first element", "that button", "the field", "it", "this"
   - NEVER use ambiguous text without element type: "click: search suggestion 'yahoo finance'" (must specify: link, button, div, etc.)
   - ALWAYS include element type: button, link, div, input, etc.
   - ALWAYS be specific: include button text, field label, article title, or other identifying details
   
   Examples:
   - If you see a field that needs filling: "type: <value> in <specific field description>"
     * GOOD: "type: John Doe in name input field" (includes type: input field)
     * GOOD: "type: john@example.com in email input field" (includes type: input field)
     * BAD: "type: John Doe in field" (too vague - what type of field?)
   
   - If you see a button that needs clicking: "click: <specific button description>" (already includes type: button)
     * GOOD: "click: Google Search button" (includes type: button)
     * GOOD: "click: Accept all cookies button" (includes type: button)
     * GOOD: "click: first article link titled 'Introduction to Python'" (includes type: link)
     * GOOD: "click: search suggestion 'yahoo finance' link" (includes type: link)
     * BAD: "click: search suggestion 'yahoo finance'" (missing element type - is it a link? button? div?)
     * BAD: "click: first article" (missing element type - is it a link? div? button?)
     * BAD: "click: first element" (too vague - what element? what type?)
     * BAD: "click: button" (too vague - which button?)
   
   - **CRITICAL: Preserve ordinal information AND include element TYPE for click commands**:
     * If user says "first article" → use "click: first article link titled '<title>'" or "click: first article link in the list" (must include type: link) NOT "click: first article" (missing type) or "click: first element" (too vague)
     * If user says "second button" → use "click: second submit button" or "click: second button labeled '<text>'" (already includes type: button)
     * If user says "third link" → use "click: third link titled '<text>'" or "click: third navigation link" (already includes type: link)
     * If user says "search suggestion 'yahoo finance'" → use "click: search suggestion 'yahoo finance' link" or "click: search suggestion 'yahoo finance' button" (must specify type) NOT "click: search suggestion 'yahoo finance'" (missing type)
     * Ordinals: first, second, third, fourth, fifth, last, etc.
   
   **For PRESS commands: MUST be brief**
   - Just the key name: "press: Enter", "press: Escape", "press: Tab"
   - Do NOT add descriptions, context, or field names
   - GOOD: "press: Enter"
   - BAD: "press: Enter in the search input field" (too descriptive)
   - BAD: "press: Enter to search" (too descriptive)
   
   {("- If target element is above current position: scroll up" if is_exploring else "- If needed element isn't visible, use scroll commands")}
   {("- If target element is below current position: scroll down" if is_exploring else "")}

4. Return the action command with clear reasoning.
5. **CRITICAL: Set needs_exploration=True ONLY when you cannot determine ANY valid action** (i.e., when no actionable element is visible and you cannot proceed without scrolling first). 
   - If you can determine a valid action (even if it's not perfect), set needs_exploration=False and return that action
   - Only set needs_exploration=True when you genuinely cannot determine what to do next without exploring (scrolling) the page
   - If you return None or cannot determine an action, set needs_exploration=True to trigger exploration mode

What is the single next action to take?
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
        for i, interaction in enumerate(interactions[-5:], 1):  # Last 5
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

