from __future__ import annotations
import re
from enum import Enum
from typing import List, Dict, Optional, Any, Callable, TYPE_CHECKING
from pydantic import BaseModel

if TYPE_CHECKING:
    from vision_bot import BrowserVisionBot
    from agent.agent_controller import AgentController
    from agent.task_result import TaskResult

class MiniGoalMode(Enum):
    AUTONOMY = "autonomy"
    SCRIPTED = "scripted"

class MiniGoalTrigger(BaseModel):
    """Defines conditions that trigger a mini goal"""
    action_type: Optional[str] = None  # e.g., "click"
    target_regex: Optional[str] = None # regex matching the element description or label
    selector: Optional[str] = None     # explicit selector match
    observation_regex: Optional[str] = None # regex matching any text in the viewport

    def matches_action(self, action: str) -> bool:
        """Check if a determined action matches this trigger"""
        if not self.action_type and not self.target_regex:
            return False
        
        parts = action.split(":", 1)
        act_type = parts[0].strip().lower()
        act_target = parts[1].strip() if len(parts) > 1 else ""

        print(f"🔍 Checking trigger match for action: '{action}' against trigger (type={self.action_type}, regex={self.target_regex})")
        if self.action_type and self.action_type.lower() != act_type:
            return False
        
        if self.target_regex and not re.search(self.target_regex, act_target, re.IGNORECASE):
            return False
            
        print(f"🎯 TRIGGER MATCHED! action='{action}'")
        return True

    def matches_observation(self, visible_text: str) -> bool:
        """Check if the current page observation matches this trigger"""
        if not self.observation_regex or not visible_text:
            return False
        return bool(re.search(self.observation_regex, visible_text, re.IGNORECASE))

class MiniGoalScriptContext:
    """Context passed to scripted mini goal handlers"""
    def __init__(self, bot: BrowserVisionBot, controller: AgentController, action_step: Optional[Any] = None, action: Optional[str] = None):
        self.bot = bot
        self.controller = controller
        self.action_step = action_step
        self.action = action

    def ask_question(self, query: str) -> str:
        """Uses the agent's current context to answer a question via LLM"""
        from ai_utils import generate_text

        # Capture current viewport state for context
        snapshot = self.controller._capture_snapshot(full_page=False)

        prompt = f"""
        You are a scripted helper for an automation agent.
        A custom script is asking you a question about the current page state to help it complete a mini-goal.

        CONTEXT:
        URL: {snapshot.url}
        Page Title: {snapshot.title}
        Visible Text (partial): {snapshot.visible_text[:2000] if snapshot.visible_text else "None"}

        QUESTION FROM SCRIPT:
        {query}

        Answer the question concisely based ONLY on the provided context and the current screenshot.
        """

        return generate_text(
            prompt=prompt,
            image=snapshot.screenshot,
            model=self.controller.agent_model_name,
            reasoning_level="low"
        )

    def ask_question_structured(self, query: str, model_class: type) -> Any:
        """Uses the agent's current context to answer a question via LLM and return structured data"""
        from ai_utils import generate_model
        from pydantic import BaseModel

        if not issubclass(model_class, BaseModel):
            raise ValueError("model_class must be a Pydantic BaseModel subclass")

        # Capture current viewport state for context
        snapshot = self.controller._capture_snapshot(full_page=False)

        prompt = f"""
        You are a scripted helper for an automation agent.
        A custom script is asking you a question about the current page state to help it complete a mini-goal.

        CONTEXT:
        URL: {snapshot.url}
        Page Title: {snapshot.title}
        Visible Text (partial): {snapshot.visible_text[:2000] if snapshot.visible_text else "None"}

        QUESTION FROM SCRIPT:
        {query}

        Analyze the current page state and provide your answer in the requested structured format.
        """

        return generate_model(
            prompt=prompt,
            model_object_type=model_class,
            image=snapshot.screenshot,
            model=self.controller.agent_model_name,
            reasoning_level="low"
        )

class MiniGoalManager:
    """Manages the registration and execution of mini goals"""
    
    def __init__(self, bot: BrowserVisionBot):
        self.bot = bot
        self.registry: List[Dict[str, Any]] = []
        self.recursion_limit = 3

    def register_mini_goal(
        self, 
        trigger: MiniGoalTrigger, 
        mode: MiniGoalMode, 
        handler: Optional[Callable[[MiniGoalScriptContext], None]] = None,
        instruction_override: Optional[str] = None
    ):
        """Register a new mini goal trigger and handler"""
        self.registry.append({
            "trigger": trigger,
            "mode": mode,
            "handler": handler,
            "instruction_override": instruction_override
        })

    def find_matching_goal(self, action: Optional[str] = None, visible_text: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Find a registered mini goal that matches the current action or observation"""
        print(f"🔍 find_matching_goal: registry_size={len(self.registry)}, action='{action}', text_len={len(visible_text) if visible_text else 0}")
        for entry in self.registry:
            trigger = entry["trigger"]
            if action and trigger.matches_action(action):
                return entry
            if visible_text and trigger.matches_observation(visible_text):
                return entry
        return None

    def execute_scripted(self, entry: Dict[str, Any], controller: AgentController, action_step: Optional[Any] = None, action: Optional[str] = None):
        """Execute a scripted mini goal"""
        handler = entry["handler"]
        if not handler:
            print("⚠️ Scripted mini goal triggered but no handler provided")
            return

        context = MiniGoalScriptContext(self.bot, controller, action_step, action)
        print("🎭 Executing Scripted Mini Goal...")
        handler(context)
        print("✅ Scripted Mini Goal finished")

    def execute_autonomous(self, entry: Dict[str, Any], controller: AgentController, action: str) -> TaskResult:
        """Execute an autonomous mini goal using a sub-agent loop"""
        instruction = entry.get("instruction_override") or f"Complete the following interaction: {action}"
        
        print(f"🤖 Starting Autonomous Mini Goal: {instruction}")
        
        # Use existing SubAgentController if available, or create a temporary one
        if not controller.sub_agent_controller:
            from agent.sub_agent_controller import SubAgentController
            controller.sub_agent_controller = SubAgentController(
                self.bot,
                controller.agent_context,
                controller_factory=controller._spawn_child_controller,
                track_ineffective_actions=controller.detect_ineffective_actions,
                allow_partial_completion=controller.allow_partial_completion
            )

        # Spawn a sub-agent on the current tab
        current_tab_id = self.bot.tab_manager.get_active_tab().tab_id if self.bot.tab_manager else "main"
        sub_agent_id = controller.sub_agent_controller.spawn_sub_agent(
            tab_id=current_tab_id,
            instruction=instruction
        )
        
        if not sub_agent_id:
            from agent.task_result import TaskResult
            return TaskResult(success=False, reasoning="Failed to spawn sub-agent for mini goal")

        # Execute the sub-agent
        # Note: We should ideally pass a flag to block navigation, but we'll implement that in AgentController
        result_dict = controller.sub_agent_controller.execute_sub_agent(sub_agent_id)
        
        from agent.task_result import TaskResult
        return TaskResult(
            success=result_dict.get("success", False),
            reasoning=result_dict.get("reasoning", ""),
            confidence=result_dict.get("confidence", 0.0),
            evidence=result_dict.get("evidence", {})
        )
