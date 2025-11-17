"""
Completion Contract - LLM-based completion evaluation for agentic mode.

Step 2: Evaluates task completion using LLM with full environment state.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

from goals.base import BrowserState, Interaction, InteractionType
from ai_utils import (
    generate_model,
    ReasoningLevel,
    get_default_agent_model,
    get_default_agent_reasoning_level,
)


class CompletionYesNo(BaseModel):
    """Lightweight yes/no response for completion check"""
    model_config = ConfigDict(extra="forbid")
    is_complete: bool = Field(description="True if the task is complete, False otherwise")


class CompletionEvaluation(BaseModel):
    """Structured LLM response for completion evaluation"""
    is_complete: bool = Field(description="Whether the overall task is complete")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the assessment")
    reasoning: str = Field(description="Detailed explanation of why task is/isn't complete")
    evidence: Optional[str] = Field(
        default=None,
        description="Supporting evidence as JSON string (URL patterns matched, text found, parameters satisfied, etc.)"
    )
    remaining_steps: List[str] = Field(
        default_factory=list,
        description="If not complete, what steps likely remain"
    )


@dataclass
class EnvironmentState:
    """
    Comprehensive environment state snapshot for LLM evaluation.
    Bundles all relevant state information.
    """
    browser_state: BrowserState
    interaction_history: List[Interaction]
    user_prompt: str
    task_start_url: str  # Initial URL when task started
    task_start_time: float  # Timestamp when task started
    current_url: str
    page_title: str
    visible_text: Optional[str] = None  # First 2000 chars of page text
    url_history: List[str] = field(default_factory=list)  # Navigation history
    url_pointer: Optional[int] = None  # Current position within navigation history
    
    def __post_init__(self):
        if self.visible_text is None:
            self.visible_text = self.browser_state.visible_text[:2000] if self.browser_state.visible_text else ""
        if not self.url_history:
            self.url_history = []
        if self.url_pointer is None:
            self.url_pointer = len(self.url_history) - 1 if self.url_history else -1


class CompletionContract:
    """
    LLM-based completion contract that evaluates task completion
    based on full environment state.
    
    Step 2: Uses LLM to determine if the overall task is complete.
    """
    
    def __init__(
        self,
        user_prompt: str,
        allow_partial_completion: bool = False,
        show_task_completion_reason: bool = False,
        strict_mode: bool = False,
        *,
        model_name: Optional[str] = None,
        reasoning_level: Union[ReasoningLevel, str, None] = None,
    ):
        self.user_prompt = user_prompt
        self.allow_partial_completion = allow_partial_completion
        self.show_task_completion_reason = show_task_completion_reason
        self.strict_mode = strict_mode
        self.model_name = model_name or get_default_agent_model()
        if reasoning_level is None:
            reasoning = ReasoningLevel.coerce(get_default_agent_reasoning_level())
        else:
            reasoning = ReasoningLevel.coerce(reasoning_level)
        self.reasoning_level: ReasoningLevel = reasoning
    
    def evaluate(
        self,
        environment_state: EnvironmentState,
        screenshot: Optional[bytes] = None,
        user_inputs: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[bool, str, CompletionEvaluation]:
        """
        Evaluates if the task is complete using LLM with full environment context.
        
        Uses a two-stage approach:
        1. Quick yes/no check (fast)
        2. Full evaluation with reasoning (only if complete or show_task_completion_reason=True)
        
        Args:
            environment_state: Complete state snapshot
            screenshot: Optional screenshot for vision-based evaluation
            
        Returns:
            (is_complete, reasoning, full_evaluation)
        """
        try:
            # First, do a quick yes/no check
            # Note: quick check doesn't include user inputs for speed, but full evaluation will
            is_complete_quick = self._query_completion_yes_no(environment_state, screenshot)
            
            # If not complete and not showing reason, return early with minimal reasoning
            if not is_complete_quick and not self.show_task_completion_reason:
                return (
                    False,
                    "Task not complete",
                    CompletionEvaluation(
                        is_complete=False,
                        confidence=0.0,
                        reasoning="Task not complete",
                        evidence=None,
                        remaining_steps=[]
                    )
                )
            
            # Task is complete OR show_task_completion_reason is True - do full evaluation
            # Build comprehensive context for LLM
            system_prompt = self._build_system_prompt()
            user_prompt_text = self._build_evaluation_prompt(environment_state, user_inputs or [])
            
            # Call LLM with structured schema
            evaluation = generate_model(
                prompt=user_prompt_text,
                model_object_type=CompletionEvaluation,
                system_prompt=system_prompt,
                image=screenshot,
                image_detail="high" if screenshot else "low",  # Use high detail for completion checks
                model=self.model_name,
                reasoning_level=self.reasoning_level,
            )
            
            return (
                evaluation.is_complete,
                evaluation.reasoning,
                evaluation
            )
            
        except Exception as e:
            print(f"⚠️ CompletionContract evaluation error: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: conservative - assume not complete
            fallback_eval = CompletionEvaluation(
                is_complete=False,
                confidence=0.0,
                reasoning=f"Evaluation error: {str(e)}",
                evidence=str({"error": str(e)}),
                remaining_steps=["Fix evaluation error"]
            )
            return False, fallback_eval.reasoning, fallback_eval
    
    def _query_completion_yes_no(
        self,
        environment_state: EnvironmentState,
        screenshot: Optional[bytes] = None
    ) -> bool:
        """
        Lightweight yes/no check to determine if task is complete.
        Returns True if task is complete, False otherwise.
        """
        prompt = self._build_completion_yes_no_prompt(environment_state)
        try:
            result = generate_model(
                prompt=prompt,
                model_object_type=CompletionYesNo,
                system_prompt=self._build_completion_yes_no_system_prompt(),
                image=screenshot,
                image_detail="low",  # Use low detail for quick check
                model=self.model_name,
                reasoning_level=ReasoningLevel.LOW,  # Use low reasoning for quick check
            )
            return result.is_complete
        except Exception as e:
            print(f"⚠️ Failed to evaluate completion yes/no check via LLM: {e}")
            # Fallback: assume not complete (conservative)
            return False
    
    @staticmethod
    def _build_completion_yes_no_system_prompt() -> str:
        return """You are making a quick yes/no decision: is the task complete?

A task is complete when:
- The user's main request has been fulfilled
- Required data has been extracted (if extraction was requested)
- Target page has been reached (if navigation was requested)
- Success indicators are visible (if form submission was requested)

A task is NOT complete when:
- Still on wrong page or intermediate page
- Required interactions haven't been performed
- Data extraction hasn't occurred (if extraction was requested)
- Task is clearly in progress

Respond with just is_complete (true/false)."""
    
    def _build_completion_yes_no_prompt(self, state: EnvironmentState) -> str:
        """Build a lightweight prompt for yes/no completion decision"""
        interaction_summary = self._summarize_interactions(state.interaction_history)
        
        return f"""
QUICK COMPLETION CHECK
======================

USER PROMPT: "{self.user_prompt}"

CURRENT STATE:
- URL: {state.current_url}
- Page Title: {state.page_title}
- Visible Text: {state.visible_text[:500] if state.visible_text else "None"}...

INTERACTIONS: {interaction_summary[:300]}...

QUESTION: Is the task complete? (true/false only)
"""
    
    def _build_system_prompt(self) -> str:
        """Build system prompt that guides the LLM on how to evaluate completion"""
        partial_guidance = ""
        if self.allow_partial_completion:
            partial_guidance = """
- Partial completion is allowed for this evaluation. If the majority of the user's requested deliverables have been satisfied and missing items are minor or clearly unobtainable, you may mark the task complete but explicitly note any remaining gaps in the reasoning.
"""
        return f"""
You are evaluating whether a web automation task has been successfully completed.

Your task is to determine if the user's request has been fulfilled based on:
1. The original user prompt and intent
2. Current browser state (URL, page content, visible elements)
3. Interaction history (what actions were taken)
4. Visual evidence from screenshots (if provided)

Evaluation Guidelines:
- Be conservative: only mark complete when you have high confidence the task is done
- Consider the intent: a "navigate" task completes when the target page loads
- **CRITICAL for extraction tasks: If the user prompt involves extracting, getting, finding, or collecting data, the task is ONLY complete if an EXTRACT interaction has been successfully performed AND the extracted data matches what was requested. Do NOT mark complete if data is just visible on the page - extraction must have actually occurred.**
- Consider data collection: a "collect info" task completes when required data is visible/collected AND extraction has been performed
- Consider form submission: a "submit" or "login" task completes when:
  * Confirmation appears or success indicators are visible, OR
  * Navigation to a new page occurs (URL changed), OR  
  * The final action (clicking submit/login button) was performed - if the user's goal explicitly states "click the login button" or "click submit", and that action has been executed, consider the task complete even if there's no immediate page change
  **STRICT MODE: If strict_mode is enabled, ONLY check if the explicitly requested actions have been executed. Do NOT infer that login/submit must succeed - if the user asked to "click the login button", consider it complete after clicking, regardless of success/failure or error messages.**
  
  **STRICT MODE EXAMPLES:**
  - User goal: "click the login button"
    * Strict mode: Task is COMPLETE after the login button is clicked, even if a "Wrong password!" error appears
    * Non-strict mode: Task is NOT complete if login fails (would infer success is required)
  
  - User goal: "Fill in username 'exampleuser2' and when I say 'continue', click the login button"
    * Strict mode: Task is COMPLETE if:
      ✓ Username was filled in
      ✓ "continue" command was received (check user inputs/interaction history)
      ✓ Login button was clicked
    * Do NOT check if password was correct, do NOT check if login succeeded, do NOT consider error messages as failure
  
  - User goal: "navigate to https://example.com and click submit"
    * Strict mode: Task is COMPLETE if:
      ✓ Navigation to the URL occurred (URL matches)
      ✓ Submit button was clicked
    * Do NOT check if form submission succeeded, do NOT check for validation errors
  
  **Key principle: In strict mode, completion = all explicitly requested actions were performed, regardless of outcomes, errors, or success indicators.**
- Look for explicit success indicators: confirmation messages, success pages, completion banners
- Consider navigation patterns: unexpected navigation away from target may indicate failure
- Be aware of intermediate states: don't mark complete during multi-step workflows unless truly finished
- **Note: Also consider the interaction history - if the user requested specific actions and they appear to have been executed successfully, this can support completion even if page state hasn't changed yet**
{partial_guidance}

Your evaluation should be:
- Precise: Base conclusions on concrete evidence
- Context-aware: Understand the difference between progress and completion
- Explainable: Provide clear reasoning for your assessment

Return your evaluation as structured JSON with:
- is_complete: boolean indicating completion
- confidence: float 0-1 indicating confidence level
- reasoning: detailed explanation
- evidence: supporting facts (URL patterns, text found, parameters checked)
- remaining_steps: (DEPRECATED - leave empty) This field is not used. Actions are determined reactively.
"""
    
    def _build_evaluation_prompt(self, state: EnvironmentState, user_inputs: List[Dict[str, Any]] = None) -> str:
        """Build detailed prompt with all environment context"""
        
        # Summarize interactions
        interaction_summary = self._summarize_interactions(state.interaction_history)
        
        # URL/navigation history
        nav_summary = self._summarize_navigation(
            state.url_history,
            state.task_start_url,
            state.current_url,
            state.url_pointer
        )
        
        partial_note = ""
        if self.allow_partial_completion:
            partial_note = "\nPARTIAL COMPLETION ENABLED: If substantial progress (e.g., most requested extractions or subtasks) is evident, you may mark the task complete while noting any remaining gaps.\n"
        prompt = f"""
Evaluate if the following task has been completed:

IMPORTANT: You are viewing a VIEWPORT SNAPSHOT (what's currently visible on screen), not the full page.
- Only elements visible in the current viewport are accessible
- If required elements are not visible, you must suggest scrolling first
- The screenshot shows ONLY what's currently visible on screen

ORIGINAL USER PROMPT:
"{self.user_prompt}"

CURRENT BROWSER STATE (VIEWPORT):
- URL: {state.current_url}
- Page Title: {state.page_title}
- Scroll Position: Y={state.browser_state.scroll_y}, X={state.browser_state.scroll_x}
- Viewport Size: {state.browser_state.page_width}x{state.browser_state.page_height}
- Visible Text (first 2000 chars): {state.visible_text}

NAVIGATION HISTORY:
{nav_summary}

INTERACTIONS PERFORMED:
{interaction_summary}
{f"""
USER-PROVIDED INPUTS (from defer_input commands):
{chr(10).join([f"- {entry.get('prompt', 'Input')}: \"{entry.get('response', '')}\"" for entry in reversed(user_inputs[-3:])])}
""" if user_inputs else ""}

TASK PROGRESS:
- Started at: {state.task_start_url}
- Started at time: {state.task_start_time}
- Current time: {state.browser_state.timestamp}
- Session duration: {state.browser_state.timestamp - state.task_start_time:.1f} seconds
{partial_note}

Based on this comprehensive state, determine:
1. Has the user's request been fulfilled?
2. What evidence supports your conclusion?

IMPORTANT: Your job is ONLY to evaluate if the task is complete. Do NOT plan ahead or list remaining steps.
The agent will determine what to do next reactively based on the current viewport state.

Consider:
- Whether the current page/state matches what would be expected after task completion
- Whether all required interactions have been performed (review the INTERACTIONS PERFORMED section)
- Whether the user's explicitly stated final action (if any) has been executed
- Whether success indicators are present
- Whether error messages indicate a problem or are transient/disappearing
- Whether the task reached a natural completion point vs. being in progress
{f"""
{"="*60}
STRICT MODE IS ENABLED - EVALUATION RULES:
{"="*60}
The user prompt has been rewritten to explicitly state completion criteria.

Your job: Check if all actions explicitly listed in the user prompt have been performed.
- Review the INTERACTIONS PERFORMED section
- Review the USER-PROVIDED INPUTS section for conditional commands (e.g., "continue")
- DO NOT check outcomes, error messages, success indicators, or page navigation unless explicitly required

The rewritten prompt explicitly states what constitutes completion - follow it exactly.
{"="*60}
""" if self.strict_mode else ""}
"""
        return prompt
    
    def _summarize_interactions(self, interactions: List[Interaction]) -> str:
        """Summarize interaction history for LLM"""
        if not interactions:
            return "No interactions yet."
        
        summary_parts = []
        extraction_count = 0
        successful_extractions = []
        
        for i, interaction in enumerate(interactions[-10:], 1):  # Last 10 interactions
            interaction_type = interaction.interaction_type.value
            summary = f"{i}. {interaction_type}"
            
            # Special handling for extraction interactions
            if interaction.interaction_type == InteractionType.EXTRACT:
                extraction_count += 1
                if interaction.extraction_prompt:
                    summary += f" - prompt: '{interaction.extraction_prompt[:100]}'"
                if interaction.success and interaction.extracted_data:
                    # Show key extracted data (limit size for prompt)
                    import json
                    try:
                        data_str = json.dumps(interaction.extracted_data, indent=2)[:200]
                        summary += f"\n   ✅ Successfully extracted: {data_str}..."
                        successful_extractions.append({
                            'prompt': interaction.extraction_prompt,
                            'data': interaction.extracted_data
                        })
                    except:
                        summary += f"\n   ✅ Successfully extracted data"
                elif not interaction.success:
                    summary += f"\n   ❌ Failed: {interaction.error_message or 'Unknown error'}"
            else:
                # Standard interaction details
                if interaction.coordinates:
                    summary += f" at ({interaction.coordinates[0]}, {interaction.coordinates[1]})"
                if interaction.text_input:
                    summary += f" with text: '{interaction.text_input[:50]}'"
                if interaction.target_element_info:
                    element_desc = interaction.target_element_info.get('description', '')[:50]
                    if element_desc:
                        summary += f" on element: {element_desc}"
                # Add effectiveness indicators
                if not interaction.success:
                    summary += f" ❌ FAILED"
                elif interaction.before_state and interaction.after_state:
                    # Check if page changed (effective action)
                    url_changed = interaction.before_state.url != interaction.after_state.url
                    # Check DOM change by comparing dom_snapshot if available
                    dom_changed = False
                    if interaction.before_state.dom_snapshot and interaction.after_state.dom_snapshot:
                        dom_changed = interaction.before_state.dom_snapshot != interaction.after_state.dom_snapshot
                    elif url_changed:
                        # If URL changed, DOM likely changed too
                        dom_changed = True
                    if not url_changed and not dom_changed:
                        summary += f" ⚠️ (no page change)"
                    else:
                        summary += f" ✅ (effective)"
            
            summary_parts.append(summary)
        
        if len(interactions) > 10:
            summary_parts.append(f"... and {len(interactions) - 10} more interactions")
        
        # Add extraction summary at the end if there were extractions
        if extraction_count > 0:
            summary_parts.append(f"\n📊 EXTRACTION SUMMARY:")
            summary_parts.append(f"   Total extraction attempts: {extraction_count}")
            summary_parts.append(f"   Successful extractions: {len(successful_extractions)}")
            if successful_extractions:
                summary_parts.append(f"   Extracted data:")
                for ext in successful_extractions[-3:]:  # Last 3 extractions
                    summary_parts.append(f"      - '{ext['prompt']}': {str(ext['data'])[:100]}...")
        
        return "\n".join(summary_parts) if summary_parts else "No interactions."
    
    def _summarize_navigation(
        self,
        url_history: List[str],
        start_url: str,
        current_url: str,
        url_pointer: Optional[int]
    ) -> str:
        """Summarize navigation patterns including back/forward availability"""
        if not url_history:
            return f"Started at: {start_url}\nCurrently at: {current_url}"

        total = len(url_history)
        pointer = url_pointer if url_pointer is not None and 0 <= url_pointer < total else total - 1
        back_available = pointer > 0
        forward_available = pointer < total - 1

        start_idx = max(0, total - 5)
        lines = []
        for idx in range(start_idx, total):
            marker = " (current)" if idx == pointer else ""
            lines.append(f"{idx}: {url_history[idx]}{marker}")

        history_block = "\n    ".join(lines)
        return (
            f"Pages visited: {total}\n"
            f"Back available: {'yes' if back_available else 'no'}\n"
            f"Forward available: {'yes' if forward_available else 'no'}\n"
            f"Recent history (oldest → newest):\n    {history_block}"
        )

