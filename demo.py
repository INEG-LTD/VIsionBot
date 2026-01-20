"""
Browser Demo Script
=============================

This is the primary demonstration of Browser capabilities.
It showcases the key features of the framework with minimal logging output.

WHAT THIS DEMO SHOWS:
--------------------
1. ✨ Visual Thinking Indicator - Flashing blue border on browser while agent thinks
2. 🎯 Interceptors System - Autonomous dropdown selection, error recovery, file uploads
3. 🤖 Clean Event Output - Shows only iteration number, reasoning, and actions
4. 💬 User Interaction - Agent can ask questions when stuck (ask: command)
5. ⚙️ Full Configuration - Example of all major configuration options

QUICK START:
-----------
1. Set your task and URL at the bottom of this file (line ~677)
2. Adjust the model configuration if needed (line ~614)
3. Run: python demo.py

KEY FEATURES DEMONSTRATED:
-------------------------
- Agent iteration loop with visual feedback
- Interceptor handlers for common UI patterns (dropdowns, errors, uploads)
- Custom event callbacks for clean console output
- Browser thinking border effect (flashing blue)
- Base knowledge injection for task-specific instructions
- User question callbacks for human-in-the-loop interaction

CUSTOMIZATION:
-------------
- Task & URL: See bottom of file (execute_mission call)
- Models: See config.model section
- Max iterations: See config.execution.max_attempts
- Visual effects: Set config.logging.debug_mode=True to disable border
- Interceptors: See setup_interceptors() function

This demo is production-ready and can be adapted for your own automation tasks.
"""
from time import sleep
import sys
import threading
import os

from pydantic import BaseModel
from pathlib import Path
from browser.provider import BrowserConfig
from core.config import Config, ModelConfig, ExecutionConfig, ElementConfig, DebugConfig, UserMessagesConfig
from core.config import ActFunctionConfig
from lib.ai import ReasoningLevel
from core.browser import Browser
from utils.event_logger import BotEvent, EventType
from agent.interceptor_manager import Interceptor, InterceptorMode, InterceptorContext
from utils.select_option_utils import SelectOptionError
import random
from prompt_toolkit import HTML, print_formatted_text as print

os.environ.setdefault("PLAYWRIGHT_CHROMIUM_DISABLE_CRASHPAD", "1")

def type_text_sequentially(page, text: str, delay: int = None):
    if delay is None:
        delay = random.randint(50, 150)
    page.keyboard.type(text, delay=delay)
    print(f"Typed text: {text}")

_spinner_active = False
_spinner_thread = None
_completion_shown = False

def _show_spinner():
    global _spinner_active
    spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    i = 0
    while _spinner_active:
        sys.stdout.write(f'\r{spinner_chars[i % len(spinner_chars)]} Thinking...')
        sys.stdout.flush()
        sleep(0.1)
        i += 1
    sys.stdout.write('\r' + ' ' * 20 + '\r')
    sys.stdout.flush()

def _start_spinner():
    global _spinner_active, _spinner_thread
    _spinner_active = True
    _spinner_thread = threading.Thread(target=_show_spinner, daemon=True)
    _spinner_thread.start()

def _stop_spinner():
    global _spinner_active
    _spinner_active = False
    if _spinner_thread:
        _spinner_thread.join(timeout=0.2)

class ThinkingBorderManager:
    
    _JS_INIT = """
    (function() {
        if (window.__agentThinkingBorder) return;
        window.__agentThinkingBorder = {
            overlay: null,
            blockingOverlay: null,
            init: function() {
                if (this.overlay) return;
                const style = document.createElement('style');
                style.textContent = `
                    @keyframes agent-thinking-pulse {
                        0%, 100% { box-shadow: inset 0 0 60px 20px rgba(59, 130, 246, 0.6); }
                        50% { box-shadow: inset 0 0 80px 30px rgba(59, 130, 246, 0.8); }
                    }
                    @keyframes agent-thinking-fadeout {
                        from { opacity: 1; }
                        to { opacity: 0; }
                    }
                `;
                document.head.appendChild(style);

                this.overlay = document.createElement('div');
                this.overlay.style.cssText = 'position:fixed;top:0;left:0;right:0;bottom:0;pointer-events:none;z-index:2147483647;display:none;opacity:0;';
                document.body.appendChild(this.overlay);

                this.blockingOverlay = document.createElement('div');
                this.blockingOverlay.style.cssText = 'position:fixed;top:0;left:0;right:0;bottom:0;background:transparent;pointer-events:auto;z-index:2147483646;display:none;cursor:not-allowed;';
                document.body.appendChild(this.blockingOverlay);
            },
            start: function() {
                this.init();
                this.overlay.style.display = 'block';
                this.overlay.style.opacity = '1';
                this.overlay.style.animation = 'agent-thinking-pulse 1.5s ease-in-out infinite';
            },
            stop: function() {
                if (!this.overlay) return;
                this.overlay.style.animation = 'agent-thinking-fadeout 0.5s ease-out forwards';
                setTimeout(() => { this.overlay.style.display = 'none'; }, 500);
            },
            enableBlocking: function() {
                this.init();
                this.blockingOverlay.style.display = 'block';
                document.documentElement.style.overflow = 'hidden';
                document.body.style.overflow = 'hidden';
            },
            disableBlocking: function() {
                if (!this.blockingOverlay) return;
                this.blockingOverlay.style.display = 'none';
                document.documentElement.style.overflow = '';
                document.body.style.overflow = '';
            }
        };
    })();
    """

    def __init__(self, bot: Browser):
        self.bot = bot
        self._enabled = not bot.config.logging.debug_mode
        self._last_page_id = None

    def _ensure_init(self):
        if not self._enabled or not self.bot.page:
            return

        page_id = id(self.bot.page)
        needs_init = self._last_page_id != page_id

        if not needs_init:
            try:
                needs_init = self.bot.page.evaluate(
                    "() => !window.__agentThinkingBorder || !window.__agentThinkingBorder.overlay"
                )
            except Exception:
                needs_init = True

        if needs_init:
            try:
                self.bot.page.evaluate(self._JS_INIT)
            except Exception:
                pass

        self._last_page_id = page_id

    def start(self):
        if not self._enabled: return
        self._ensure_init()
        try:
            self.bot.page.evaluate("if(window.__agentThinkingBorder) window.__agentThinkingBorder.start();")
        except Exception:
            pass

    def stop(self):
        if not self._enabled: return
        try:
            self.bot.page.evaluate("if(window.__agentThinkingBorder) window.__agentThinkingBorder.stop();")
        except Exception:
            pass

    def enable_blocking(self):
        if not self._enabled: return
        self._ensure_init()
        try:
            self.bot.page.evaluate("if(window.__agentThinkingBorder) window.__agentThinkingBorder.enableBlocking();")
        except Exception:
            pass
        try:
            self.start()
        except Exception:
            pass

    def disable_blocking(self):
        if not self._enabled: return
        try:
            self.bot.page.evaluate("if(window.__agentThinkingBorder) window.__agentThinkingBorder.disableBlocking();")
        except Exception:
            pass

def apply_thinking_border(bot: Browser):
    manager = ThinkingBorderManager(bot)
    bot._thinking_border_manager = manager

    if hasattr(bot.page, "on"):
        def _handle_frame_navigation(frame):
            try:
                if frame != bot.page.main_frame:
                    return
                manager._last_page_id = None
                manager.start()
            except Exception:
                pass
        try:
            bot.page.on("framenavigated", _handle_frame_navigation)
        except Exception:
            pass

    from agent import Agent
    from core.browser import Browser

    original_capture = Agent._capture_snapshot
    def patched_capture(self, *args, **kwargs):
        manager.start()
        return original_capture(self, *args, **kwargs)
    Agent._capture_snapshot = patched_capture

    original_act = Browser.act
    def patched_act(self, *args, **kwargs):
        manager.stop()
        manager.disable_blocking()
        try:
            return original_act(self, *args, **kwargs)
        finally:
            manager.enable_blocking()
    Browser.act = patched_act

    original_extract = Browser.extract
    def patched_extract(self, *args, **kwargs):
        manager.stop()
        manager.disable_blocking()
        try:
            return original_extract(self, *args, **kwargs)
        finally:
            manager.enable_blocking()
    Browser.extract = patched_extract

    original_run = Agent.run_execute_task
    def patched_run(self, *args, **kwargs):
        manager.enable_blocking()
        try:
            result = original_run(self, *args, **kwargs)
            return result
        finally:
            manager.disable_blocking()
    Agent.run_execute_task = patched_run

    return manager

def _convert_to_first_person(text: str) -> str:
    if not text:
        return text

    text = text.strip()

    first_person_starters = ('i ', 'i\'', 'i\'m', 'i\'ll', 'i\'ve', 'i\'d', 'i see', 'i need', 'i should', 'i will', 'i can')
    if text.lower().startswith(first_person_starters):
        return text

    if text.startswith('The '):
        text = 'I see ' + text.lower()
    elif text.startswith('To '):
        text = 'I need ' + text.lower()
    elif text and text[0].isupper():
        text = 'I ' + text[0].lower() + text[1:]
    else:
        text = 'I ' + text

    if text:
        text = text[0].upper() + text[1:]

    return text

def _format_action_first_person(action: str) -> str:
    if not action:
        return action

    parts = action.split(':', 1)
    if len(parts) != 2:
        return f"I will now {action}"

    action_type = parts[0].strip().upper()
    target = parts[1].strip()

    if action_type == "ASK":
        return "I will now ask the user a question"
    return f"I will now [{action_type}] >{target}<"

def setup_interceptors(bot: Browser):
    
    from typing import Optional
    class DropdownSelection(BaseModel):
        recommended_option: str
        confidence: float = 1.0
        reasoning: Optional[str] = None

    class IsDropdownVisible(BaseModel):
        is_visible: bool
    dropdown_trigger_select = Interceptor(
        action_type="select",
        target_regex=r"(?i)dropdown|select|combobox"
    )
    dropdown_trigger_click = Interceptor(
        action_type="click",
        target_regex=r"(?i)dropdown|select|combobox"
    )

    def select_dropdown_handler(context: InterceptorContext):
        print("🎯 Running dropdown selection interceptor...")

        try:
            current_action = context.action

            if not current_action:
                print("❌ No current action available for dropdown selection")
                return

            print(f"📍 Action to execute: {current_action}")

            action_part = ""

            if current_action.startswith('select:'):
                action_part = current_action[7:].strip()
                print(f"🎯 Need to select: '{action_part}'")
                action_part = action_part.replace("dropdown", "").replace("combobox", "").replace("select", "")
            elif current_action.startswith('click:'):
                action_part = current_action[6:].strip()
                print(f"🎯 Clicked on: '{action_part}'")
                action_part = action_part.replace("dropdown", "").replace("combobox", "").replace("select", "")

            analysis_prompt = f"""
                Based on the current page state, what is the best option to select for this select field with the placeholder: "{action_part}"?
                Consider the overall task context and what would be the most logical selection.
                """
            selection_info: DropdownSelection = context.ask_question_structured(
                analysis_prompt,
                DropdownSelection
            )

            dropdown_prompt = f"""
                Based on the current page state, is the dropdown with the placeholder: "{action_part}" visible?
                """
            dropdown_visible: IsDropdownVisible = context.ask_question_structured(
                dropdown_prompt,
                IsDropdownVisible
            )

            print(f"🤖 AI Analysis: Select '{selection_info.recommended_option}'")
            print(f"   Confidence: {selection_info.confidence}")

            if selection_info.confidence < 0.3:
                print("⚠️ AI confidence too low, skipping selection")
                return

            if not dropdown_visible.is_visible:
                bot.act(f"type: {selection_info.recommended_option} in {action_part}")
                sleep(5)
            bot.act(f"click: {selection_info.recommended_option}")
        except SelectOptionError as e:
            print(f"❌ Select option error: {e}")
        except Exception as e:
            print(f"❌ Unexpected error in dropdown handler: {e}")

    bot.register_interceptor(
        trigger=dropdown_trigger_click,
        mode=InterceptorMode.SCRIPTED,
        handler=select_dropdown_handler
    )
    bot.register_interceptor(
        trigger=dropdown_trigger_select,
        mode=InterceptorMode.SCRIPTED,
        handler=select_dropdown_handler
    )

    def error_recovery_handler(context: InterceptorContext):
        print("🚨 Error detected, running recovery interceptor...")

        error_details = context.ask_question(
            "An error message appeared on the page. What type of error is this and how should I handle it? "
            "Consider whether it's a validation error, network error, or user input error."
        )

        if "network" in error_details.lower():
            context.bot.page.reload()
        elif "login" in error_details.lower() or "auth" in error_details.lower():
            context.ask_question("The user needs to log in. Should I navigate to the login page or ask them for credentials?")
        else:
            context.bot.page.evaluate("""
                const errors = document.querySelectorAll('.error, .alert-danger, [class*="error"]');
                errors.forEach(el => el.style.backgroundColor = 'yellow');
            """)

    error_trigger = Interceptor(
        observation_regex=r"(?i)error|failed|invalid|please try again|something went wrong"
    )

    bot.register_interceptor(
        trigger=error_trigger,
        mode=InterceptorMode.SCRIPTED,
        handler=error_recovery_handler
    )

    def file_upload_handler(context: InterceptorContext):
        print("📁 File upload interceptor activated...")

        upload_requirements = context.ask_question(
            "What type of file should be uploaded here? Consider file format, size limits, "
            "and any specific naming conventions or content requirements."
        )

        print(f"🤖 Upload requirements: {upload_requirements}")
        upload_state = context.bot.page.evaluate("""
            () => {
                const fileInputs = document.querySelectorAll('input[type="file"]');
                const dragZones = document.querySelectorAll('[class*="drop"], [class*="upload"]');
                return {
                    fileInputs: fileInputs.length,
                    dragZones: dragZones.length,
                    hasProgress: !!document.querySelector('[class*="progress"], .upload-progress')
                };
            }
        """)

        if upload_state['fileInputs'] > 0:
            context.ask_question(
                f"I found {upload_state['fileInputs']} file input fields. "
                "Should I ask the user to select a file, or do they want me to use a test file?"
            )
        elif upload_state['dragZones'] > 0:
            context.ask_question(
                "This appears to be a drag-and-drop upload interface. "
                "Should I guide the user through the drag-and-drop process?"
            )

    upload_trigger = Interceptor(
        action_type="click",
        target_regex=r"(?i)upload.*file|choose.*file|select.*file|browse"
    )

    bot.register_interceptor(
        trigger=upload_trigger,
        mode=InterceptorMode.SCRIPTED,
        handler=file_upload_handler
    )

def create_event_callback(bot, debug_mode: bool = True):
    def simple_event_callback(event: BotEvent):
        if event.event_type == EventType.AGENT_ITERATION:
            iteration = event.details.get('iteration', '?')
            max_iterations = event.details.get('max_iterations', '?')
            print(HTML(f"\n<b>∞ Iteration {iteration}/{max_iterations}</b>"))
            _start_spinner()

        elif event.event_type == EventType.ACTION_DETERMINED:
            _stop_spinner()

            action = event.details.get('action', 'Unknown action')
            reasoning = event.details.get('reasoning', '')
            plan_step = event.details.get('plan_step')
            pre_generated_iter = event.details.get('pre_generated_iteration')

            if reasoning:
                first_person_reasoning = _convert_to_first_person(reasoning)
                print(HTML(f"<gray>> Here's what the agent is thinking: {first_person_reasoning}</gray>"))
            if pre_generated_iter is not None:
                plan_note = f"Pre-generated step {plan_step or '?'} from iteration {pre_generated_iter} (reusing a cached plan)."
                print(f"    🧠 {plan_note}")
            first_person_action = _format_action_first_person(action)
            print(f"    ⚡ {first_person_action}")

        elif event.event_type == EventType.AGENT_COMPLETE and event.details.get('success', False):
            reasoning = event.details.get('reasoning', '')
            confidence = event.details.get('confidence')
            if reasoning:
                first_person_reasoning = _convert_to_first_person(reasoning)
                print("\n✅ The task has been completed!")
                print(f"📝 Reasoning: {first_person_reasoning}")
                if confidence is not None:
                    print(f"🎯 Confidence: {confidence:.2f}")
    
    return simple_event_callback

def ask_user_for_help(question: str, context: dict) -> str | None:
    print(f"\n❓ Agent asks: {question}")
    print(f"   (Press Enter to skip, or type your answer)")

    try:
        answer = input("   Your answer: ").strip()
        if answer:
            return answer
        return None
    except (KeyboardInterrupt, EOFError):
        return None

user_data_path = Path.cwd() / ".browser_data"
user_data_path.mkdir(parents=True, exist_ok=True)
crashpad_path = user_data_path / "crashpad"
crashpad_path.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("CRASHPAD_DATABASE", str(crashpad_path))
os.environ.setdefault("CRASHPAD_METRICS", str(crashpad_path))

config = Config(
    model=ModelConfig(
        agent_model="groq/meta-llama/llama-4-maverick-17b-128e-instruct",
        command_model="gpt-5-mini",
        reasoning_level=ReasoningLevel.HIGH
    ),
    execution=ExecutionConfig(
        max_attempts=30,
        # max_actions_per_plan=1,
        track_ineffective_actions=False,
        wait_for_load_before_turn=True,
        wait_for_load_state="networkidle",
        wait_for_load_timeout_ms=5000,
        auto_complete_extract_commands=False
    ),
    elements=ElementConfig(
        overlay_mode="all",
        include_textless_overlays=True,
        selection_fallback_model="gemini/gemini-2.5-flash-lite",
        selection_retry_attempts=2,
        include_overlays_in_agent_context=True,
    ),
    logging=DebugConfig(
        debug_mode=True,
        show_overlay_candidates=False,
    ),
    browser=BrowserConfig(
        provider_type="local",
        headless=False,
        apply_stealth=True,
    ),
    act_function=ActFunctionConfig(
        enable_target_context_guard=False,
        enable_modifier=True,
        enable_additional_context=True
    ),
    user_messages=UserMessagesConfig(
        file_upload_prompt="⏸️ Please select the file you would like to upload. Press [Enter] when done...",
        file_upload_interrupted="⚠️ Upload interrupted. Please try again."
    )
)
bot = Browser(config=config)
setup_interceptors(bot)

bot.event_logger.register_callback(create_event_callback(bot, debug_mode=config.logging.debug_mode))
bot.start()
bot.page.goto("https://news.ycombinator.com/")

apply_thinking_border(bot)
result = bot.execute_mission(
    " click and open the 5th article webpage and give me a summary of the article",
    base_knowledge=[
        "Clicking an article will open a new webpage"
    ],
    # base_knowledge=[
    #     "You must click the 'Jobs' tab button before clicking a job listing"
    #     "You must press enter after typing in a search field"
    #     "Don't click jobs you have already clicked"
    #     "To extract the necessary information, you must click the job listing and then extract the job title and company name and then extract the url from the apply button"
    #     "Don't click the apply button, just extract the job title, url and company name",
    #     """
    #     This is what you should do when you are on the job listing page:
    #     For each job listing:
    #         - Click a job listing
    #             - A right side bar should appear with the job listing details
    #             - The side bar should have an Apply button, it might have multiple Apply buttons
    #         - Extract the job title (eg Doctor), company name (eg NHS)
    #         - Extract the URL from the first Apply button in the right side bar
    #         - Close the side bar after extracting the URL
    #             - If the side bar is still visible, keep attempting to close it
    #         - If you are on the 5th job listing, you are done, otherwise:
    #             - Scroll down if the other job listings are not visible
    #             - Click the next job listing and repeat the process
    #     """
    # ],
    show_completion_reasoning_every_iteration=False,
    user_question_callback=ask_user_for_help,
)

if result.success:
    print(f"\n✅ Task completed! Confidence: {result.confidence:.2f}")
    print(f"Reasoning: {result.reasoning}")

    if result.extracted_data:
        print("\n📊 Extracted Data:")
        for prompt, data in result.extracted_data.items():
            print(f"  {prompt}: {data}")
else:
    print(f"\n❌ Task failed: {result.reasoning}")

input("Press Enter to continue...")
