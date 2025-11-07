"""
TabDecisionEngine - LLM-based decision making for tab management.

Phase 2: Tab Decision Making
"""
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field
from playwright.sync_api import Page

from .tab_manager import TabManager
from .tab_info import TabInfo
from ai_utils import generate_model


class TabAction(str, Enum):
    """Actions that can be taken on tabs"""
    SWITCH = "switch"  # Switch to a different tab
    CLOSE = "close"  # Close a tab
    CONTINUE = "continue"  # Continue on current tab
    SPAWN_SUB_AGENT = "spawn_sub_agent"  # Spawn a sub-agent for a tab (Phase 3)


class TabDecision(BaseModel):
    """
    Decision made by TabDecisionEngine about tab management.
    """
    action: TabAction = Field(description="The action to take")
    target_tab_id: Optional[str] = Field(None, description="Target tab ID for switch/close actions")
    reasoning: str = Field(description="Explanation of why this decision was made")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the decision (0.0 to 1.0)")
    should_take_action: bool = Field(True, description="Whether this decision should take action immediately (True) or just be informational (False)")


class TabDecisionEngine:
    """
    LLM-based engine for making tab management decisions.
    
    Analyzes current context, available tabs, and user goals to decide:
    - Whether to switch to a different tab
    - Whether to close a tab
    - Whether to continue on current tab
    - Whether to spawn a sub-agent (Phase 3)
    """
    
    def __init__(self, tab_manager: TabManager, model_name: str = "gpt-5-mini"):
        """
        Initialize TabDecisionEngine.
        
        Args:
            tab_manager: TabManager instance to query tabs from
            model_name: LLM model to use for decisions
        """
        self.tab_manager = tab_manager
        self.model_name = model_name
    
    def make_decision(
        self,
        current_tab_id: str,
        user_prompt: str,
        current_action: Optional[str] = None,
        task_context: Optional[Dict[str, Any]] = None
    ) -> TabDecision:
        """
        Make a decision about tab management based on current context.
        
        Args:
            current_tab_id: ID of the currently active tab
            user_prompt: Current user prompt/goal
            current_action: Current action being considered
            task_context: Additional context about the task
        
        Returns:
            TabDecision with the recommended action
        """
        # Get current tab info
        current_tab = self.tab_manager.get_tab_info(current_tab_id)
        if not current_tab:
            # Fallback: continue on current tab if we can't find it
            return TabDecision(
                action=TabAction.CONTINUE,
                reasoning="Current tab not found in TabManager",
                confidence=0.5,
                should_take_action=False
            )
        
        # Refresh current tab info with latest URL/title
        try:
            current_tab.update_url(current_tab.page.url)
            current_tab.update_title(current_tab.page.title())
        except Exception:
            pass
        
        # Get all available tabs and refresh their info
        all_tabs = self.tab_manager.list_tabs()
        for tab in all_tabs:
            try:
                tab.update_url(tab.page.url)
                tab.update_title(tab.page.title())
            except Exception:
                pass
        
        # If only one tab, continue on it
        if len(all_tabs) <= 1:
            return TabDecision(
                action=TabAction.CONTINUE,
                reasoning="Only one tab available, continuing on current tab",
                confidence=1.0,
                should_take_action=False
            )
        
        # Build context for LLM
        tabs_summary = self._build_tabs_summary(all_tabs, current_tab_id)
        
        # Build decision prompt
        decision_prompt = self._build_decision_prompt(
            current_tab=current_tab,
            tabs_summary=tabs_summary,
            user_prompt=user_prompt,
            current_action=current_action,
            task_context=task_context
        )
        
        # Get LLM decision
        try:
            decision = generate_model(
                prompt=decision_prompt,
                model_object_type=TabDecision,
                system_prompt=self._build_system_prompt(),
                model=self.model_name
            )
            
            # Validate decision
            if decision.action == TabAction.SWITCH or decision.action == TabAction.CLOSE:
                if not decision.target_tab_id:
                    # LLM didn't provide target, default to continue
                    decision.action = TabAction.CONTINUE
                    decision.reasoning += " (No target tab specified, defaulting to continue)"
                    decision.confidence = 0.3
                elif decision.target_tab_id not in [t.tab_id for t in all_tabs]:
                    # Invalid target tab
                    decision.action = TabAction.CONTINUE
                    decision.reasoning += f" (Target tab {decision.target_tab_id} not found, defaulting to continue)"
                    decision.confidence = 0.3
            
            return decision
            
        except Exception as e:
            print(f"⚠️ Error making tab decision: {e}")
            # Fallback: continue on current tab
            return TabDecision(
                action=TabAction.CONTINUE,
                reasoning=f"Error in decision engine: {str(e)}",
                confidence=0.2,
                should_take_action=False
            )
    
    def _build_tabs_summary(self, tabs: List[TabInfo], current_tab_id: str) -> str:
        """Build a summary of all tabs for the LLM"""
        lines = []
        for tab in tabs:
            is_current = " (CURRENT)" if tab.tab_id == current_tab_id else ""
            status = " [COMPLETED]" if tab.is_completed else ""
            agent_info = f" [Agent: {tab.agent_id}]" if tab.agent_id else ""
            
            lines.append(
                f"- Tab {tab.tab_id}: {tab.purpose} - {tab.url} - {tab.title}{is_current}{status}{agent_info}"
            )
        
        return "\n".join(lines)
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for tab decision making"""
        return """You are a tab management decision engine for a web automation agent.

Your role is to analyze the current context and decide the best tab management action:
- SWITCH: Switch to a different tab that is more relevant to the current task
- CLOSE: Close a tab that is no longer needed or has completed its purpose
- CONTINUE: Continue working on the current tab
- SPAWN_SUB_AGENT: Spawn a sub-agent to work on another tab (Phase 3). Use this when:
  * Work needs to happen on another tab while the main agent continues on current tab
  * You need to gather information from another tab without leaving the current one
  * Parallel work is needed (e.g., "research this candidate" - spawn sub-agent for LinkedIn while main agent continues)

CRITICAL RULES:
1. Only switch tabs if another tab is clearly more relevant to the current task
2. Only close tabs if they are completed or clearly unnecessary
3. Default to CONTINUE if the current tab is appropriate
4. Consider tab purposes, URLs, and completion status
5. Be conservative - don't switch/close unnecessarily
6. If switching/closing, you MUST provide a valid target_tab_id
7. Confidence should reflect how certain you are about the decision
8. Set should_take_action=True only if the action should happen immediately (for SWITCH/CLOSE), False for CONTINUE (informational only)"""
    
    def _build_decision_prompt(
        self,
        current_tab: TabInfo,
        tabs_summary: str,
        user_prompt: str,
        current_action: Optional[str],
        task_context: Optional[Dict[str, Any]]
    ) -> str:
        """Build the decision prompt for the LLM"""
        prompt_parts = [
            "TAB MANAGEMENT DECISION",
            "=" * 80,
            "",
            f"CURRENT TAB:",
            f"  ID: {current_tab.tab_id}",
            f"  Purpose: {current_tab.purpose}",
            f"  URL: {current_tab.url}",
            f"  Title: {current_tab.title}",
            f"  Completed: {current_tab.is_completed}",
            f"  Agent: {current_tab.agent_id or 'None'}",
            "",
            f"ALL AVAILABLE TABS:",
            tabs_summary,
            "",
            f"USER PROMPT/GOAL: {user_prompt}",
        ]
        
        if current_action:
            prompt_parts.append(f"CURRENT ACTION: {current_action}")
        
        if task_context:
            prompt_parts.append(f"TASK CONTEXT: {task_context}")
        
        prompt_parts.extend([
            "",
            "DECISION REQUIRED:",
            "Based on the context above, decide what tab management action to take.",
            "Consider:",
            "- Is the current tab appropriate for the task?",
            "- Would another tab be more relevant?",
            "- Are there completed tabs that should be closed?",
            "- Does work need to happen on another tab?",
            "",
            "Return a TabDecision with:",
            "- action: SWITCH, CLOSE, CONTINUE, or SPAWN_SUB_AGENT",
            "- target_tab_id: Required if action is SWITCH or CLOSE",
            "- reasoning: Clear explanation",
            "- confidence: 0.0 to 1.0",
            "- should_take_action: True if action should happen immediately (SWITCH/CLOSE), False for CONTINUE (informational)"
        ])
        
        return "\n".join(prompt_parts)

