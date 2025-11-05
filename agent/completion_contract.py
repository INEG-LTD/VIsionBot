"""
Completion Contract - LLM-based completion evaluation for agentic mode.

Step 2: Evaluates task completion using LLM with full environment state.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field

from goals.base import BrowserState, Interaction
from ai_utils import generate_model


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
    
    def __post_init__(self):
        if self.visible_text is None:
            self.visible_text = self.browser_state.visible_text[:2000] if self.browser_state.visible_text else ""
        if not self.url_history:
            self.url_history = []


class CompletionContract:
    """
    LLM-based completion contract that evaluates task completion
    based on full environment state.
    
    Step 2: Uses LLM to determine if the overall task is complete.
    """
    
    def __init__(self, user_prompt: str):
        self.user_prompt = user_prompt
    
    def evaluate(
        self,
        environment_state: EnvironmentState,
        screenshot: Optional[bytes] = None
    ) -> Tuple[bool, str, CompletionEvaluation]:
        """
        Evaluates if the task is complete using LLM with full environment context.
        
        Args:
            environment_state: Complete state snapshot
            screenshot: Optional screenshot for vision-based evaluation
            
        Returns:
            (is_complete, reasoning, full_evaluation)
        """
        try:
            # Build comprehensive context for LLM
            system_prompt = self._build_system_prompt()
            user_prompt_text = self._build_evaluation_prompt(environment_state)
            
            # Call LLM with structured schema
            evaluation = generate_model(
                prompt=user_prompt_text,
                model_object_type=CompletionEvaluation,
                system_prompt=system_prompt,
                image=screenshot,
                image_detail="high" if screenshot else "low"  # Use high detail for completion checks
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
                evidence={"error": str(e)},
                remaining_steps=["Fix evaluation error"]
            )
            return False, fallback_eval.reasoning, fallback_eval
    
    def _build_system_prompt(self) -> str:
        """Build system prompt that guides the LLM on how to evaluate completion"""
        return """
You are evaluating whether a web automation task has been successfully completed.

Your task is to determine if the user's request has been fulfilled based on:
1. The original user prompt and intent
2. Current browser state (URL, page content, visible elements)
3. Interaction history (what actions were taken)
4. Visual evidence from screenshots (if provided)

Evaluation Guidelines:
- Be conservative: only mark complete when you have high confidence the task is done
- Consider the intent: a "navigate" task completes when the target page loads
- Consider data collection: a "collect info" task completes when required data is visible/collected
- Consider form submission: a "submit" task completes when confirmation appears or success indicators are visible
- Look for explicit success indicators: confirmation messages, success pages, completion banners
- Consider navigation patterns: unexpected navigation away from target may indicate failure
- Be aware of intermediate states: don't mark complete during multi-step workflows unless truly finished

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
    
    def _build_evaluation_prompt(self, state: EnvironmentState) -> str:
        """Build detailed prompt with all environment context"""
        
        # Summarize interactions
        interaction_summary = self._summarize_interactions(state.interaction_history)
        
        # URL/navigation history
        nav_summary = self._summarize_navigation(
            state.url_history,
            state.task_start_url,
            state.current_url
        )
        
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

TASK PROGRESS:
- Started at: {state.task_start_url}
- Started at time: {state.task_start_time}
- Current time: {state.browser_state.timestamp}
- Session duration: {state.browser_state.timestamp - state.task_start_time:.1f} seconds

Based on this comprehensive state, determine:
1. Has the user's request been fulfilled?
2. What evidence supports your conclusion?

IMPORTANT: Your job is ONLY to evaluate if the task is complete. Do NOT plan ahead or list remaining steps.
The agent will determine what to do next reactively based on the current viewport state.

Consider:
- Whether the current page/state matches what would be expected after task completion
- Whether all required interactions have been performed
- Whether success indicators are present
- Whether the task reached a natural completion point vs. being in progress
"""
        return prompt
    
    def _summarize_interactions(self, interactions: List[Interaction]) -> str:
        """Summarize interaction history for LLM"""
        if not interactions:
            return "No interactions yet."
        
        summary_parts = []
        for i, interaction in enumerate(interactions[-10:], 1):  # Last 10 interactions
            interaction_type = interaction.interaction_type.value
            summary = f"{i}. {interaction_type}"
            
            if interaction.coordinates:
                summary += f" at ({interaction.coordinates[0]}, {interaction.coordinates[1]})"
            if interaction.text_input:
                summary += f" with text: '{interaction.text_input[:50]}'"
            if interaction.target_element_info:
                element_desc = interaction.target_element_info.get('description', '')[:50]
                if element_desc:
                    summary += f" on element: {element_desc}"
            
            summary_parts.append(summary)
        
        if len(interactions) > 10:
            summary_parts.append(f"... and {len(interactions) - 10} more interactions")
        
        return "\n".join(summary_parts) if summary_parts else "No interactions."
    
    def _summarize_navigation(self, url_history: List[str], start_url: str, current_url: str) -> str:
        """Summarize navigation patterns"""
        if not url_history:
            return f"Started at: {start_url}\nCurrently at: {current_url}"
        
        nav_path = " → ".join(url_history[-5:])  # Last 5 URLs
        if len(url_history) > 5:
            nav_path = f"... → {nav_path}"
        
        return f"Navigation path: {nav_path}"

