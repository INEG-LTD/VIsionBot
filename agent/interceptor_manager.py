from __future__ import annotations
import re
from enum import Enum
from typing import List, Dict, Optional, Any, Callable, TYPE_CHECKING
from pydantic import BaseModel
from utils.debug_print import dprint, PrintMode

if TYPE_CHECKING:
    from core.browser import Browser
    from agent.agent_controller import Agent

class InterceptorMode(Enum):
    SCRIPTED = "scripted"

class Interceptor(BaseModel):
    """Defines conditions that trigger an interceptor."""
    action_type: Optional[str] = None  # e.g., "click"
    target_regex: Optional[str] = None # regex matching the element description or label
    selector: Optional[str] = None     # explicit selector match
    observation_regex: Optional[str] = None # regex matching any text in the viewport

    def matches_action(self, action: str, debug: bool = False) -> bool:
        """Check if a determined action matches this trigger"""
        if not self.action_type and not self.target_regex:
            return False
        
        parts = action.split(":", 1)
        act_type = parts[0].strip().lower()
        act_target = parts[1].strip() if len(parts) > 1 else ""

        if debug:
            dprint(
                f"🔍 Checking trigger match for action: '{action}' against trigger "
                f"(type={self.action_type}, regex={self.target_regex})"
            )
        if self.action_type and self.action_type.lower() != act_type:
            return False
        
        if self.target_regex and not re.search(self.target_regex, act_target, re.IGNORECASE):
            return False
            
        if debug:
            dprint(f"🎯 TRIGGER MATCHED! action='{action}'")
        return True

    def matches_observation(self, visible_text: str) -> bool:
        """Check if the current page observation matches this trigger."""
        if not self.observation_regex or not visible_text:
            return False
        return bool(re.search(self.observation_regex, visible_text, re.IGNORECASE))

class InterceptorContext:
    """Context passed to scripted interceptor handlers."""
    def __init__(self, browser: Browser, controller: Agent, action_step: Optional[Any] = None, action: Optional[str] = None):
        self.browser = browser
        self.controller = controller
        self.action_step = action_step
        self.action = action

    def ask_question(self, query: str) -> str:
        """Uses the agent's current context to answer a question via LLM."""
        from lib.ai import generate_text

        # Capture current viewport state for context
        snapshot = self.controller._capture_snapshot(full_page=False)

        prompt = f"""
        You are a scripted helper for an automation agent.
        A custom script is asking you a question about the current page state to help it complete an interceptor.

        CONTEXT:
        URL: {snapshot.url}
        Page Title: {snapshot.title}
        Visible Text: {snapshot.visible_text if snapshot.visible_text else "None"}

        QUESTION FROM SCRIPT:
        {query}

        Answer the question concisely based ONLY on the provided context and the current screenshot.
        """

        return generate_text(
            prompt=prompt,
            image=snapshot.screenshot,
            model=self.controller.agent_model_name,
        )

    def ask_question_structured(self, query: str, model_class: type) -> Any:
        """Uses the agent's current context to answer a question via LLM and return structured data."""
        from lib.ai import generate_model
        from pydantic import BaseModel

        if not issubclass(model_class, BaseModel):
            raise ValueError("model_class must be a Pydantic BaseModel subclass")

        # Capture current viewport state for context
        snapshot = self.controller._capture_snapshot(full_page=False)

        prompt = f"""
        You are a scripted helper for an automation agent.
        A custom script is asking you a question about the current page state to help it complete an interceptor.

        CONTEXT:
        URL: {snapshot.url}
        Page Title: {snapshot.title}
        Visible Text (partial): {snapshot.visible_text if snapshot.visible_text else "None"}

        QUESTION FROM SCRIPT:
        {query}

        Analyze the current page state and provide your answer in the requested structured format.
        """

        return generate_model(
            prompt=prompt,
            model_object_type=model_class,
            image=snapshot.screenshot,
            model=self.controller.agent_model_name,
        )

class InterceptorManager:
    """Manages the registration and execution of interceptors."""
    
    def __init__(self, browser: Browser):
        self.browser = browser
        self.registry: List[Dict[str, Any]] = []
        self.recursion_limit = 3

    def register_interceptor(
        self, 
        trigger: Interceptor, 
        mode: InterceptorMode, 
        handler: Optional[Callable[[InterceptorContext], None]] = None
    ):
        """Register a new interceptor trigger and handler."""
        self.registry.append({
            "trigger": trigger,
            "mode": mode,
            "handler": handler,
        })

    def find_matching_interceptor(
        self,
        action: Optional[str] = None,
        visible_text: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Find a registered interceptor that matches the current action or observation."""
        # Check if debug mode is enabled
        debug_mode = (
            hasattr(self.bot, 'event_logger') and 
            hasattr(self.bot.event_logger, 'debug_mode') and 
            self.bot.event_logger.debug_mode
        )
        
        if debug_mode:
            dprint(
                f"🔍 find_matching_interceptor: registry_size={len(self.registry)}, "
                f"action='{action}', text_len={len(visible_text) if visible_text else 0}"
            )
        for entry in self.registry:
            trigger = entry["trigger"]
            if action and trigger.matches_action(action, debug=debug_mode):
                return entry
            if visible_text and trigger.matches_observation(visible_text):
                return entry
        return None

    def execute_scripted(
        self,
        entry: Dict[str, Any],
        controller: Agent,
        action_step: Optional[Any] = None,
        action: Optional[str] = None,
    ):
        """Execute a scripted interceptor."""
        handler = entry["handler"]
        if not handler:
            dprint("⚠️ Scripted interceptor triggered but no handler provided")
            return

        context = InterceptorContext(self.browser, controller, action_step, action)
        dprint("🎭 Executing Scripted Interceptor...")
        handler(context)
        dprint("✅ Scripted Interceptor finished")
