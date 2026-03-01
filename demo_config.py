"""demo_config.py — Agent config, callbacks, and interceptors.

Edit this file to change:
  - Model selection (agent_model, command_model)
  - Max iterations per mission (max_actions_per_mission)
  - Browser settings (headless, stealth, provider_type)
  - Callbacks: on_user_question, on_data_reported
  - Interceptors (e.g., the dropdown handler)
"""

from __future__ import annotations

import os
from time import sleep
from typing import Optional

from pydantic import BaseModel

from agent.agent_controller import Agent
from agent.interceptor_manager import Interceptor, InterceptorContext, InterceptorMode
from browser.provider import BrowserConfig
from core.config import (
    Config,
    DebugConfig,
    ExecutionConfig,
    ModelConfig,
    ToolPolicyConfig,
)
from lib.ai import ReasoningLevel

os.environ.setdefault("PLAYWRIGHT_CHROMIUM_DISABLE_CRASHPAD", "1")

# ---------------------------------------------------------------------------
# Starting URL — where the browser opens before a mission begins
# ---------------------------------------------------------------------------

STARTING_URL = "https://example.com"

# ---------------------------------------------------------------------------
# Config — edit model names, iteration limits, browser options here
# ---------------------------------------------------------------------------

config = Config(
    model=ModelConfig(
        agent_model="gpt-5-mini",
        command_model="gpt-5-mini",
        agent_reasoning_level=ReasoningLevel.HIGH,
    ),
    execution=ExecutionConfig(
        max_actions_per_mission=600,
        max_actions_per_plan=3,
        # Effect-governed tool policy profile:
        # - FULL, WEB_SAFE, LOCKED_DOWN
        tool_policy=ToolPolicyConfig(
            preset="WEB_SAFE",
            mode="ENFORCE",
        ),
        wait_for_load_before_iteration=True,
        wait_for_load_state="domcontentloaded",
        wait_for_load_timeout_ms=5000,
        # Memory: send the most recent 50% of entries on long missions
        memory_narrative_recent_percent=0.5,
        speculative_hints_enabled=True,
        speculative_hints_min_confidence=0.75,
    ),
    logging=DebugConfig(
        debug_mode=False,
        show_overlay_candidates=True,
        show_llm_costs=False,
        save_screenshots=True,
    ),
    browser=BrowserConfig(
        provider_type="local",
        headless=False,
        apply_stealth=True,
        # JPEG at quality 80: ~60-80% smaller than PNG, saves ~0.5s LLM latency per step
        screenshot_format="jpeg",
        screenshot_quality=80,
    ),
)

# ---------------------------------------------------------------------------
# Callbacks — called from the agent worker thread
# ---------------------------------------------------------------------------


def on_user_question(question: str, context: dict) -> str:
    """Called when the agent uses the ask tool to request user input."""
    print(f"\n❓ Agent asks: {question}")
    try:
        return input("   Your answer: ").strip()
    except (KeyboardInterrupt, EOFError):
        return ""


def on_data_reported(payload: str, context: dict) -> None:
    """Called when the agent calls write_data / report."""
    print("\n📦 Agent reported data:")
    print(payload)
    url = context.get("current_url", "")
    if url:
        print(f"URL: {url}")


# ---------------------------------------------------------------------------
# Interceptors — hook into specific action types before they execute
# ---------------------------------------------------------------------------


def setup_interceptors(agent: Agent) -> None:
    """Register action interceptors on an agent instance."""

    class DropdownSelection(BaseModel):
        recommended_option: str
        confidence: float = 1.0
        reasoning: Optional[str] = None

    class IsDropdownVisible(BaseModel):
        is_visible: bool

    dropdown_trigger_select = Interceptor(
        action_type="select",
        target_regex=r"(?i)dropdown|select|combobox",
    )
    dropdown_trigger_click = Interceptor(
        action_type="click",
        target_regex=r"(?i)dropdown|select|combobox",
    )

    def select_dropdown_handler(context: InterceptorContext) -> None:
        try:
            current_action = context.action
            if not current_action:
                return

            action_part = ""
            if current_action.startswith("select:"):
                action_part = current_action[7:].strip()
            elif current_action.startswith("click:"):
                action_part = current_action[6:].strip()

            action_part = (
                action_part.replace("dropdown", "")
                .replace("combobox", "")
                .replace("select", "")
                .strip()
            )

            selection_info: DropdownSelection = context.ask_question_structured(
                f'What is the best option to select for the field "{action_part}"?',
                DropdownSelection,
            )
            if selection_info.confidence < 0.3:
                return

            dropdown_visible: IsDropdownVisible = context.ask_question_structured(
                f'Is the dropdown "{action_part}" currently visible?',
                IsDropdownVisible,
            )
            if not dropdown_visible.is_visible:
                agent.action_executor.act(
                    f"type: {selection_info.recommended_option} in {action_part}"
                )
                sleep(5)
            agent.action_executor.act(f"click: {selection_info.recommended_option}")
        except Exception:
            pass

    agent.register_interceptor(
        trigger=dropdown_trigger_click,
        mode=InterceptorMode.SCRIPTED,
        handler=select_dropdown_handler,
    )
    agent.register_interceptor(
        trigger=dropdown_trigger_select,
        mode=InterceptorMode.SCRIPTED,
        handler=select_dropdown_handler,
    )
