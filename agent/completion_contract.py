"""
Completion Contract - LLM-based completion evaluation for agentic mode.

Step 2: Evaluates task completion using LLM with full environment state.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

from session_tracker import BrowserState, Interaction, InteractionType
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
        interaction_summary_limit: Optional[int] = None,
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
        self.interaction_summary_limit = interaction_summary_limit
    
    def evaluate(
        self,
        environment_state: EnvironmentState,
        screenshot: Optional[bytes] = None,
        user_inputs: Optional[List[Dict[str, Any]]] = None,
        user_prompt: Optional[str] = None
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
            # Stage 1: Yes/No check with prompt override
            is_complete_quick = self._query_completion_yes_no(environment_state, screenshot, user_prompt=user_prompt)
            
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
            
            # Stage 2: Full evaluation with reasoning
            system_prompt = self._build_system_prompt()
            user_prompt_text = self._build_evaluation_prompt(environment_state, user_inputs or [], user_prompt=user_prompt)
            
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
        screenshot: Optional[bytes] = None,
        user_prompt: Optional[str] = None
    ) -> bool:
        """
        Lightweight yes/no check to determine if task is complete.
        Returns True if task is complete, False otherwise.
        """
        prompt = self._build_completion_yes_no_prompt(environment_state, user_prompt=user_prompt)
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
    
    def _build_completion_yes_no_prompt(self, state: EnvironmentState, user_prompt: Optional[str] = None) -> str:
        """Build a lightweight prompt for yes/no completion decision"""
        interaction_summary = self._summarize_interactions(state.interaction_history)
        active_prompt = user_prompt if user_prompt else self.user_prompt
        
        return f'''
QUICK COMPLETION CHECK
======================

USER PROMPT: "{active_prompt}"

CURRENT STATE:
- URL: {state.current_url}
- Page Title: {state.page_title}
- Visible Text: {state.visible_text[:500] if state.visible_text else "None"}...

INTERACTIONS: {interaction_summary[:300]}...

QUESTION: Is the task complete? (true/false only)
'''
    
    def _build_system_prompt(self, user_prompt: Optional[str] = None) -> str:
        """Build system prompt that guides the LLM on how to evaluate completion"""
        active_prompt = user_prompt if user_prompt else self.user_prompt
        partial_guidance = "- Partial completion allowed: Mark complete if most deliverables satisfied, noting gaps in reasoning.\n" if self.allow_partial_completion else ""
        
        return f"""
TASK EVALUATION
===============
USER PROMPT: "{active_prompt}"

Determine if the automation task is complete based on:
1. Original user intent
2. Current browser state (URL, page content, visible elements)
3. Interaction history (actions taken)
4. Visual evidence (screenshot)

COMPLETION CRITERIA:
- Navigation: Complete when target page loads
- Link clicks: Complete when on destination page (don't require returning to original page)
  Example: "go to X and click Y" → Done when clicked Y and viewing Y's page
- Extraction: Complete ONLY if EXTRACT interaction succeeded with matching data
  Example: "get price" → Done only after successful EXTRACT, not just visible data
- Forms/Login: Complete when confirmation visible, URL changed, OR final action performed
{partial_guidance}
{self._get_strict_mode_guidance() if self.strict_mode else ""}

EVALUATION APPROACH:
- Conservative: High confidence required
- Precise: Base on concrete evidence
- Context-aware: Progress ≠ completion
- Consider interaction timeline as sequence leading to current state

Return JSON:
- is_complete: boolean
- confidence: float (0-1)
- reasoning: detailed explanation
- evidence: supporting facts
- remaining_steps: (DEPRECATED - leave empty)
"""
    
    def _get_strict_mode_guidance(self) -> str:
        """Get strict mode guidance text"""
        return """
STRICT MODE ENABLED:
- Only verify explicitly requested actions were performed
- Ignore outcomes, errors, success indicators (unless explicitly required)
- Examples:
  * "click login" → Done after click, regardless of success
  * "fill X then click Y" → Done after both actions, regardless of validation
"""
    
    def _build_evaluation_prompt(self, state: EnvironmentState, user_inputs: List[Dict[str, Any]] = None, user_prompt: Optional[str] = None) -> str:
        """Build detailed prompt with all environment context"""
        active_prompt = user_prompt if user_prompt else self.user_prompt
        
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
        prompt = f'''
EVALUATE TASK COMPLETION

USER PROMPT: "{active_prompt}"

CURRENT STATE (Viewport Only):
URL: {state.current_url}
Title: {state.page_title}
Scroll: Y={state.browser_state.scroll_y}
Visible: {state.visible_text[:500] if state.visible_text else "None"}...

NAVIGATION:
{nav_summary}

INTERACTIONS (chronological, current state = result of these):
{interaction_summary}
{f"""
USER INPUTS:
{chr(10).join([f'- {entry.get("prompt", "Input")}: "{entry.get("response", "")}"' for entry in reversed(user_inputs[-3:])])}
""" if user_inputs else ""}

DURATION: {state.browser_state.timestamp - state.task_start_time:.1f}s
{partial_note}

REVIEW CHECKLIST:
1. Interaction timeline (sequence → current state)
2. Link clicks: If clicked on page X, URL changed to Y, now on Y = complete
3. Extractions: Must have EXTRACT interaction (visible ≠ extracted)
4. Defer: "resumed" means handoff complete, proceed with next step
5. Success indicators: Confirmations, URL changes, completion state
6. Required actions: All explicitly requested actions performed
{f"""
{'='*40}
STRICT MODE: Check only if explicitly requested actions performed.
Ignore outcomes/errors unless explicitly required.
{'='*40}
""" if self.strict_mode else ""}

Has the user's request been fulfilled? What evidence supports this?
'''
        return prompt
    
    def _summarize_interactions(self, interactions: List[Interaction]) -> str:
        """Summarize interaction history for LLM"""
        if not interactions:
            return "No interactions yet."
        
        summary_parts = []
        extraction_count = 0
        successful_extractions = []
        
        limit = self.interaction_summary_limit
        interactions_to_summarize = interactions if not limit or limit <= 0 else interactions[-limit:]
        
        # Only show full details for last 3 interactions, summarize older ones
        recent_threshold = len(interactions_to_summarize) - 3
        
        for i, interaction in enumerate(interactions_to_summarize, 1):
            is_recent = i > recent_threshold
            interaction_type = interaction.interaction_type.value
            summary = f"{i}. {interaction_type}"
            
            # For recent interactions, show full context
            if is_recent:
                # Show where the action originated from
                if interaction.before_state and interaction.before_state.url:
                    summary += f" (on: {interaction.before_state.url[:50]}...)"
                
                # Add reasoning (truncated for token efficiency)
                if interaction.reasoning:
                    summary += f"\n   Why: {interaction.reasoning[:150]}"  # Truncate to 150 chars
            else:
                # For older interactions, just show success/failure indicator
                if not interaction.success:
                    summary += " ✗"
                elif interaction.before_state and interaction.after_state:
                    if interaction.before_state.url != interaction.after_state.url:
                        summary += " ✓ (nav)"
                    else:
                        summary += " ✓"
            
            # Special handling for navigation interactions
            if interaction.interaction_type == InteractionType.NAVIGATION:
                if interaction.target_element_info:
                    direction = interaction.target_element_info.get('direction', '')
                    from_url = interaction.target_element_info.get('from', '')
                    to_url = interaction.target_element_info.get('to', '')
                    if direction and from_url and to_url:
                        summary += f"\n   Navigation: {direction} from {from_url[:50]}... to {to_url[:50]}..."
                    elif interaction.navigation_url:
                        summary += f"\n   Navigated to: {interaction.navigation_url[:60]}..."
                # Check URL change from before/after state
                if interaction.before_state and interaction.after_state:
                    if interaction.before_state.url != interaction.after_state.url:
                        summary += f"\n   ✅ URL changed: {interaction.before_state.url[:50]}... → {interaction.after_state.url[:50]}..."
                    else:
                        summary += f"\n   ⚠️ No URL change (still on {interaction.before_state.url[:50]}...)"
                if not interaction.success:
                    summary += f"\n   ❌ FAILED: {interaction.error_message or 'Unknown error'}"
                summary_parts.append(summary)
                continue  # Skip standard handling for navigation
            
            # Special handling for defer interactions
            if interaction.interaction_type == InteractionType.DEFER:
                if interaction.text_input:
                    if interaction.text_input == "resumed":
                        summary += f" - resumed (control returned to agent)"
                        if interaction.reasoning:
                            summary += f"\n   {interaction.reasoning}"
                    else:
                        summary += f" - {interaction.text_input}"
                        if interaction.reasoning:
                            summary += f"\n   Why: {interaction.reasoning}"
                else:
                    if interaction.reasoning:
                        summary += f"\n   Why: {interaction.reasoning}"
                summary_parts.append(summary)
                continue  # Skip standard handling for defer
            
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
                    if url_changed:
                        # Explicitly show URL change - this is critical for link-clicking tasks
                        summary += f" ✅ (navigated from {interaction.before_state.url[:50]}... to {interaction.after_state.url[:50]}...)"
                    elif dom_changed:
                        summary += f" ✅ (page changed)"
                    else:
                        summary += f" ⚠️ (no page change)"
            
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

